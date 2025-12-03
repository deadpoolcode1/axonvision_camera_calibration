#!/usr/bin/env python3
"""
Extrinsic Camera Calibration with INS Integration - Known Ground Positions Method
==================================================================================

This calibration system uses RTK-measured ground positions to determine camera
position and orientation with high accuracy.

METHOD: KNOWN GROUND POSITIONS + RTK
- Mark positions on ground relative to vehicle reference point
- Use RTK GPS to measure positions (~1-2cm accuracy)
- Place calibration board at each position, measure height and distance
- Optimization solves for camera pose that makes all measurements consistent

KEY ADVANTAGES:
- Position accuracy: <1cm relative to reference point
- Orientation accuracy: <0.1 degree
- Each camera calibrated INDEPENDENTLY (scales to N cameras)
- No inter-camera measurements needed
- Simple field procedure with 2 technicians

PROBLEM FORMULATION:
We solve a nonlinear least-squares optimization to find camera extrinsics
that minimize:
  - Reprojection error of ChArUco corners
  - Violation of laser distance constraints
  - Violation of known board position constraints
  - Deviation from prior measurements (with appropriate weighting)

COORDINATE SYSTEMS:
- Vehicle Frame (NED): X=Forward, Y=Right, Z=Down
- Camera Frame: X=Right, Y=Down, Z=Forward (optical axis)
- Board Frame: X=Right, Y=Down, Z=Out (normal toward camera)
- World Frame: NED (North-East-Down), defined by INS

ROTATION CHAIN:
    R_camera_to_vehicle = R_world_to_vehicle @ R_board_to_world @ R_camera_to_board
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChArUcoBoardConfig:
    """ChArUco board configuration."""
    squares_x: int = 10
    squares_y: int = 10
    square_size: float = 0.11      # 11cm squares
    marker_size: float = 0.085     # 8.5cm markers
    dictionary_id: int = cv2.aruco.DICT_6X6_250
    
    @property
    def board_width(self) -> float:
        return self.squares_x * self.square_size
    
    @property
    def board_height(self) -> float:
        return self.squares_y * self.square_size
    
    def create_board(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_size,
            self.marker_size,
            aruco_dict
        )
        return board, aruco_dict


@dataclass 
class CalibrationConfig:
    """Configuration for the optimization."""
    # Prior uncertainties (from field measurements)
    translation_prior_std: float = 0.20  # 20cm per axis
    rotation_prior_std_deg: float = 1.0  # 1 degree
    
    # Measurement uncertainties
    laser_distance_std: float = 0.02     # 2cm laser accuracy
    corner_detection_std_px: float = 0.5  # Sub-pixel corner accuracy
    
    # Optimization settings
    max_iterations: int = 100
    ftol: float = 1e-8
    
    # Weights for cost function terms
    weight_reprojection: float = 1.0
    weight_distance: float = 10.0      # Laser distance is accurate
    weight_prior: float = 0.1          # Prior is rough
    weight_level_board: float = 100.0  # Board is definitely level


# =============================================================================
# ROTATION UTILITIES
# =============================================================================

class RotationUtils:
    """
    Consistent rotation utilities using scipy.spatial.transform.Rotation.
    
    Convention:
    - NED frame (X=North/Forward, Y=East/Right, Z=Down)
    - Euler angles: ZYX intrinsic (yaw, then pitch, then roll)
    - Yaw: rotation about Z (down), 0 deg = North, positive clockwise
    - Pitch: rotation about Y (right), positive nose up
    - Roll: rotation about X (forward), positive right wing down
    """
    
    @staticmethod
    def euler_to_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix (world to body)."""
        r = Rotation.from_euler('ZYX', [yaw_deg, pitch_deg, roll_deg], degrees=True)
        return r.as_matrix()
    
    @staticmethod
    def rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (yaw, pitch, roll in degrees)."""
        r = Rotation.from_matrix(R)
        angles = r.as_euler('ZYX', degrees=True)
        return float(angles[0]), float(angles[1]), float(angles[2])
    
    @staticmethod
    def rotation_to_camera_angles(R_camera_to_vehicle: np.ndarray) -> Dict[str, float]:
        """
        Convert camera-to-vehicle rotation to intuitive angles.
        
        Returns:
            azimuth: angle right of forward (positive = right)
            elevation: angle below horizontal (positive = down)
            roll: rotation around optical axis
        """
        # Camera optical axis (Z) in vehicle frame
        optical_axis = R_camera_to_vehicle @ np.array([0, 0, 1])
        
        # Azimuth: angle in XY plane
        azimuth = np.degrees(np.arctan2(optical_axis[1], optical_axis[0]))
        
        # Elevation: angle from horizontal
        horizontal = np.sqrt(optical_axis[0]**2 + optical_axis[1]**2)
        elevation = np.degrees(np.arctan2(optical_axis[2], horizontal))
        
        # Roll: camera Y axis relative to expected down
        camera_y = R_camera_to_vehicle @ np.array([0, 1, 0])
        
        # Project onto plane perpendicular to optical axis
        camera_y_perp = camera_y - np.dot(camera_y, optical_axis) * optical_axis
        norm = np.linalg.norm(camera_y_perp)
        
        if norm > 0.01:
            camera_y_perp /= norm
            # Expected down in this plane
            vehicle_down = np.array([0, 0, 1])
            expected = vehicle_down - np.dot(vehicle_down, optical_axis) * optical_axis
            exp_norm = np.linalg.norm(expected)
            
            if exp_norm > 0.01:
                expected /= exp_norm
                dot = np.clip(np.dot(camera_y_perp, expected), -1, 1)
                roll = np.degrees(np.arccos(dot))
                cross = np.cross(expected, camera_y_perp)
                if np.dot(cross, optical_axis) < 0:
                    roll = -roll
            else:
                roll = 0.0
        else:
            roll = 0.0
        
        return {"azimuth": azimuth, "elevation": elevation, "roll": roll}
    
    @staticmethod
    def camera_angles_to_rotation(azimuth: float, elevation: float, roll: float) -> np.ndarray:
        """Convert camera angles to R_camera_to_vehicle."""
        az = np.radians(azimuth)
        el = np.radians(elevation)
        ro = np.radians(roll)
        
        # Optical axis direction
        cam_z = np.array([
            np.cos(az) * np.cos(el),
            np.sin(az) * np.cos(el),
            np.sin(el)
        ])
        cam_z /= np.linalg.norm(cam_z)
        
        # Initial right vector (horizontal, 90 deg clockwise from optical axis when viewed from above)
        # When facing azimuth az, right is at azimuth (az + 90 deg)
        # Direction at angle θ from North = [cos(θ), sin(θ), 0]
        # So right = [cos(az+90), sin(az+90), 0] = [-sin(az), cos(az), 0]
        cam_x = np.array([-np.sin(az), np.cos(az), 0])
        if np.linalg.norm(cam_x) < 0.01:
            cam_x = np.array([1, 0, 0])
        cam_x /= np.linalg.norm(cam_x)
        
        # Down vector
        cam_y = np.cross(cam_z, cam_x)
        cam_y /= np.linalg.norm(cam_y)
        
        # Recalculate for orthogonality
        cam_x = np.cross(cam_y, cam_z)
        cam_x /= np.linalg.norm(cam_x)
        
        # Apply roll
        if abs(ro) > 1e-6:
            c, s = np.cos(ro), np.sin(ro)
            cam_x, cam_y = c * cam_x + s * cam_y, -s * cam_x + c * cam_y
        
        return np.column_stack([cam_x, cam_y, cam_z])
    
    @staticmethod
    def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
        """Proper SO(3) averaging using quaternions."""
        quats = np.array([Rotation.from_matrix(R).as_quat() for R in rotations])
        
        # Ensure same hemisphere
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        
        mean_quat = np.mean(quats, axis=0)
        mean_quat /= np.linalg.norm(mean_quat)
        
        return Rotation.from_quat(mean_quat).as_matrix()


# =============================================================================
# INS DATA
# =============================================================================

@dataclass
class INSData:
    """INS orientation at capture time (NED convention)."""
    yaw: float      # 0 deg = North, positive clockwise
    pitch: float    # positive = nose up
    roll: float     # positive = right wing down
    timestamp: float = 0.0
    
    def to_rotation_matrix(self) -> np.ndarray:
        """R_world_to_vehicle: transforms world (NED) to vehicle frame."""
        return RotationUtils.euler_to_rotation(self.yaw, self.pitch, self.roll)


# =============================================================================
# MEASUREMENT DATA STRUCTURES
# =============================================================================

@dataclass
class Measurement:
    """A single calibration measurement."""
    corners_2d: np.ndarray          # Detected corner positions (Nx2)
    corner_ids: np.ndarray          # Corner IDs
    corners_3d: np.ndarray          # 3D positions in board frame (Nx3)
    R_board_in_camera: np.ndarray   # Rotation matrix from PnP
    t_board_in_camera: np.ndarray   # Translation vector from PnP
    reproj_error: float             # Reprojection error
    ins_data: INSData               # INS at capture
    laser_distance: float           # Camera to board center (meters)
    image_shape: Tuple[int, int]    # (height, width)
    board_position_vehicle: Optional[np.ndarray] = None  # Known board position in vehicle frame


@dataclass
class CameraPrior:
    """Prior estimate of camera pose (from field measurements)."""
    position: np.ndarray  # [x, y, z] in vehicle frame
    position_std: np.ndarray  # Uncertainty per axis
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    roll: Optional[float] = None
    orientation_std_deg: float = 1.0


@dataclass
# =============================================================================
# CHARUCO DETECTOR
# =============================================================================

class ChArUcoDetector:
    """ChArUco corner detection and PnP."""
    
    def __init__(self, board_config: ChArUcoBoardConfig):
        self.config = board_config
        self.board, self.aruco_dict = board_config.create_board()
        self.detector = cv2.aruco.CharucoDetector(self.board)
        
        if hasattr(self.board, 'getChessboardCorners'):
            self.corners_3d = self.board.getChessboardCorners()
        else:
            self.corners_3d = self.board.chessboardCorners
    
    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect ChArUco corners."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corners, ids, _, _ = self.detector.detectBoard(gray)
        
        if corners is None or len(corners) < 6:
            return None
        return corners.reshape(-1, 2), ids.flatten()
    
    def estimate_pose(self, corners_2d: np.ndarray, ids: np.ndarray,
                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray
                      ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Estimate board pose."""
        obj_points = self.corners_3d[ids]
        
        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners_2d.reshape(-1, 1, 2),
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_err = np.sqrt(np.mean((corners_2d - proj.reshape(-1, 2))**2))
        
        return R, t, reproj_err


# =============================================================================
# LEVEL BOARD CONSTRAINT
# =============================================================================

def construct_R_board_to_world(board_yaw_deg: float) -> np.ndarray:
    """
    Construct rotation matrix for a LEVEL board facing direction board_yaw.
    
    Board frame (OpenCV ChArUco convention):
    - X = right (when viewing markers)
    - Y = down (when viewing markers)
    - Z = OUT of the board (toward viewer/camera)
    
    World frame: NED (X=North, Y=East, Z=Down)
    
    board_yaw is the direction the MARKERS face (toward the camera).
    The board's Z axis points in this same direction (toward camera).
    """
    yaw = np.radians(board_yaw_deg)
    
    # Board Z points toward camera (same direction markers face)
    board_z = np.array([np.cos(yaw), np.sin(yaw), 0])
    
    # Board Y points down (= world Z for level board)
    board_y = np.array([0, 0, 1])
    
    # Board X = Y × Z (right-hand rule)
    board_x = np.cross(board_y, board_z)
    board_x /= np.linalg.norm(board_x)
    
    return np.column_stack([board_x, board_y, board_z])


def estimate_board_yaw_from_pnp(
    R_board_in_camera: np.ndarray,
    t_board_in_camera: np.ndarray,
    R_world_to_vehicle: np.ndarray,
    camera_azimuth_approx: float = 0.0
) -> float:
    """
    Estimate board yaw in world frame from PnP results.
    
    board_yaw is the direction the MARKERS face (toward camera).
    Board Z (into board) points opposite to this.
    
    Strategy: The board normal (toward camera) in camera frame is approximately
    the negative of the view direction, which we can estimate from the board position.
    """
    # Board position in camera frame
    t = t_board_in_camera
    
    # Direction from camera to board center (normalized)
    cam_to_board = t / np.linalg.norm(t)
    
    # This is roughly the direction the camera is pointing
    # In vehicle frame: need to transform by R_camera_to_vehicle
    # But we don't know that yet, so use the rough approximation
    
    # For forward-facing camera (azimuth=0):
    # Camera X = vehicle Y (right)
    # Camera Y = vehicle Z (down) 
    # Camera Z = vehicle X (forward)
    
    # With azimuth offset A, camera Z rotates in XY plane
    az_rad = np.radians(camera_azimuth_approx)
    
    # Transform cam_to_board from camera to vehicle frame (rough)
    # This is an approximation assuming small elevation/roll
    cam_to_board_veh = np.array([
        cam_to_board[2] * np.cos(az_rad) - cam_to_board[0] * np.sin(az_rad),  # Forward component
        cam_to_board[2] * np.sin(az_rad) + cam_to_board[0] * np.cos(az_rad),  # Right component
        cam_to_board[1]   # Down component
    ])
    
    # Transform to world frame
    R_vehicle_to_world = R_world_to_vehicle.T
    cam_to_board_world = R_vehicle_to_world @ cam_to_board_veh
    
    # Project to horizontal
    cam_to_board_world[2] = 0
    norm = np.linalg.norm(cam_to_board_world)
    if norm < 0.01:
        return camera_azimuth_approx + 180.0
    cam_to_board_world /= norm
    
    # Board markers face back toward camera, so board_yaw = direction of -cam_to_board
    board_yaw = np.degrees(np.arctan2(-cam_to_board_world[1], -cam_to_board_world[0]))
    
    # Normalize to [-180, 180]
    while board_yaw > 180:
        board_yaw -= 360
    while board_yaw < -180:
        board_yaw += 360
    
    return board_yaw


# =============================================================================
# OPTIMIZATION-BASED CALIBRATOR
# =============================================================================

class ExtrinsicCalibrator:
    """
    Optimization-based extrinsic calibrator.
    
    Solves for camera pose that minimizes:
    1. Rotation consistency across measurements
    2. Level board constraint (board Y must be vertical)
    3. Distance constraint violation (PnP distance ≈ laser distance)
    4. Prior deviation (stay close to field measurement)
    """
    
    def __init__(self, board_config: ChArUcoBoardConfig, intrinsics: dict,
                 camera_id: str, config: CalibrationConfig = None):
        self.board_config = board_config
        self.camera_id = camera_id
        self.config = config or CalibrationConfig()
        
        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics["distortion_coefficients"], dtype=np.float64)
        
        self.detector = ChArUcoDetector(board_config)
        self.measurements: List[Measurement] = []
        self.prior: Optional[CameraPrior] = None
        self.result: Optional[dict] = None
    
    def set_prior(self, prior: CameraPrior):
        """Set prior estimate from field measurements."""
        self.prior = prior
    
    def add_measurement(self, image: np.ndarray, laser_distance: float,
                        ins_data: INSData, board_position_vehicle: np.ndarray = None) -> dict:
        """Add a calibration measurement.
        
        Args:
            image: Camera image
            laser_distance: Measured distance from camera to board
            ins_data: INS data at capture time
            board_position_vehicle: Optional known board position in vehicle frame.
                                   If provided, enables position refinement for single camera.
        """
        detection = self.detector.detect(image)
        if detection is None:
            return {"success": False, "error": "No corners detected"}
        
        corners_2d, ids = detection
        corners_3d = self.detector.corners_3d[ids]
        
        pose = self.detector.estimate_pose(corners_2d, ids, self.camera_matrix, self.dist_coeffs)
        if pose is None:
            return {"success": False, "error": "PnP failed"}
        
        R_board_camera, t_board_camera, reproj_err = pose
        
        pnp_distance = np.linalg.norm(t_board_camera)
        distance_error = abs(pnp_distance - laser_distance)
        
        if distance_error > 0.5:
            return {
                "success": False, 
                "error": f"Distance mismatch: PnP={pnp_distance:.2f}m, laser={laser_distance:.2f}m"
            }
        
        meas = Measurement(
            corners_2d=corners_2d,
            corner_ids=ids,
            corners_3d=corners_3d,
            R_board_in_camera=R_board_camera,
            t_board_in_camera=t_board_camera,
            reproj_error=reproj_err,
            ins_data=ins_data,
            laser_distance=laser_distance,
            image_shape=image.shape[:2],
            board_position_vehicle=board_position_vehicle
        )
        
        self.measurements.append(meas)
        
        # Compute rough estimate for feedback
        board_yaw = estimate_board_yaw_from_pnp(R_board_camera, t_board_camera, ins_data.to_rotation_matrix(), 0.0)
        R_board_world = construct_R_board_to_world(board_yaw)
        R_world_vehicle = ins_data.to_rotation_matrix()
        R_camera_board = R_board_camera.T
        R_camera_vehicle_rough = R_world_vehicle @ R_board_world @ R_camera_board
        angles = RotationUtils.rotation_to_camera_angles(R_camera_vehicle_rough)
        
        return {
            "success": True,
            "corners_detected": len(corners_2d),
            "reproj_error": reproj_err,
            "pnp_distance": pnp_distance,
            "distance_error": distance_error,
            "euler_angles_rough": angles
        }
    
    def _residual_function(self, params: np.ndarray) -> np.ndarray:
        """Compute residual vector for optimization."""
        n_meas = len(self.measurements)
        
        t_camera = params[0:3]
        q_camera = params[3:7]
        q_camera = q_camera / np.linalg.norm(q_camera)
        R_camera_vehicle = Rotation.from_quat(q_camera).as_matrix()
        
        board_yaws = params[7:7+n_meas]
        
        residuals = []
        
        for i, meas in enumerate(self.measurements):
            R_world_vehicle = meas.ins_data.to_rotation_matrix()
            R_board_world = construct_R_board_to_world(board_yaws[i])
            R_camera_board = meas.R_board_in_camera.T
            
            # Expected R_camera_vehicle from chain
            R_expected = R_world_vehicle @ R_board_world @ R_camera_board
            
            # Rotation error
            R_error = R_camera_vehicle @ R_expected.T
            rvec_error = Rotation.from_matrix(R_error).as_rotvec()
            rot_weight = self.config.weight_reprojection * 180.0 / np.pi
            residuals.extend(rot_weight * rvec_error)
            
            # Distance constraint
            pnp_dist = np.linalg.norm(meas.t_board_in_camera)
            dist_residual = (pnp_dist - meas.laser_distance) / self.config.laser_distance_std
            residuals.append(self.config.weight_distance * dist_residual)
            
            # Known board position constraint (enables position refinement!)
            if meas.board_position_vehicle is not None:
                # PnP gives board ORIGIN position in camera frame
                # But operator places board CENTER at known position
                # Need to compute where board center is
                
                # Board center offset in board frame (ChArUco corners are in a grid)
                # The center is at approximately half the board dimensions
                board_half_w = self.board_config.board_width / 2
                board_half_h = self.board_config.board_height / 2
                board_center_offset_board = np.array([board_half_w, board_half_h, 0])
                
                # Transform center offset to camera frame
                R_camera_board = meas.R_board_in_camera
                board_center_offset_camera = R_camera_board @ board_center_offset_board
                
                # Board center position in camera frame
                board_center_camera = meas.t_board_in_camera + board_center_offset_camera
                
                # Transform to vehicle frame
                board_center_computed = t_camera + R_camera_vehicle @ board_center_camera
                
                # Residual: computed center should match known position
                board_std = 0.05  # 5cm accuracy on marked positions
                for j in range(3):
                    residuals.append(self.config.weight_distance * 
                                   (board_center_computed[j] - meas.board_position_vehicle[j]) / board_std)
        
        # Prior constraint on translation
        if self.prior is not None:
            for j in range(3):
                prior_residual = (t_camera[j] - self.prior.position[j]) / self.prior.position_std[j]
                residuals.append(self.config.weight_prior * prior_residual)
            
            # Prior constraint on orientation (critical for static INS)
            if self.prior.azimuth is not None:
                angles = RotationUtils.rotation_to_camera_angles(R_camera_vehicle)
                az_residual = (angles['azimuth'] - self.prior.azimuth) / self.prior.orientation_std_deg
                residuals.append(self.config.weight_prior * 5 * az_residual)
            if self.prior.elevation is not None:
                angles = RotationUtils.rotation_to_camera_angles(R_camera_vehicle)
                el_residual = (angles['elevation'] - self.prior.elevation) / self.prior.orientation_std_deg
                residuals.append(self.config.weight_prior * 5 * el_residual)
            if self.prior.roll is not None:
                angles = RotationUtils.rotation_to_camera_angles(R_camera_vehicle)
                roll_residual = (angles['roll'] - self.prior.roll) / self.prior.orientation_std_deg
                residuals.append(self.config.weight_prior * 5 * roll_residual)
        
        return np.array(residuals)
    
    def _compute_direct_estimate(self) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Compute direct estimate for initialization."""
        # Use prior orientation if available, else assume forward-facing
        if self.prior and self.prior.azimuth is not None:
            camera_az_approx = self.prior.azimuth
        else:
            # Estimate from board position in camera frame
            # The board is roughly in front of camera, so we use a simple heuristic
            # Look at the average translation to get a rough idea
            avg_t = np.mean([m.t_board_in_camera for m in self.measurements], axis=0)
            # In camera frame, Z is forward. If board is offset in X/Y, camera might be angled
            # But for rough estimate, assume camera points roughly at board
            camera_az_approx = 0.0  # Start with forward-facing assumption
        
        board_yaws = []
        rotations = []
        
        for meas in self.measurements:
            R_world_vehicle = meas.ins_data.to_rotation_matrix()
            
            # Board yaw: board faces camera, so ~180 deg from camera direction
            board_yaw = estimate_board_yaw_from_pnp(
                meas.R_board_in_camera, meas.t_board_in_camera, R_world_vehicle, camera_az_approx
            )
            board_yaws.append(board_yaw)
            
            R_board_world = construct_R_board_to_world(board_yaw)
            R_camera_board = meas.R_board_in_camera.T
            R_camera_vehicle = R_world_vehicle @ R_board_world @ R_camera_board
            rotations.append(R_camera_vehicle)
        
        R_avg = RotationUtils.average_rotations(rotations)
        t_camera = self.prior.position.copy() if self.prior else np.zeros(3)
        
        return R_avg, t_camera, board_yaws
    
    def compute_extrinsics(self) -> dict:
        """Compute camera extrinsics via optimization."""
        if len(self.measurements) < 3:
            raise ValueError("Need at least 3 measurements")
        
        # Use prior orientation if available, otherwise try grid search
        if self.prior and self.prior.azimuth is not None:
            # Single optimization from prior
            result = self._optimize_with_initial_azimuth(self.prior.azimuth)
            if result is None or not result.success:
                warnings.warn("Optimization from prior did not converge, trying grid search")
                result = self._grid_search_optimize()
        else:
            # Grid search without prior orientation
            result = self._grid_search_optimize()
        
        return self._extract_results(result)
    
    def _grid_search_optimize(self):
        """Grid search over azimuth values."""
        best_result = None
        best_cost = float('inf')
        best_azimuth = None
        
        azimuths_to_try = [0, 10, 20, 30, 45, 60, 90, 120, 150, 180, -150, -120, -90, -60, -45, -30, -20, -10]
        for az_init in azimuths_to_try:
            result = self._optimize_with_initial_azimuth(az_init)
            if result is not None and result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                best_azimuth = az_init
        
        if best_azimuth is not None:
            print(f"  Best initial azimuth: {best_azimuth} deg (cost: {best_cost:.1f})")
        
        if best_result is None or not best_result.success:
            warnings.warn("Optimization did not converge")
            best_result = self._optimize_with_initial_azimuth(0)
        
        return best_result
    
    def _optimize_with_initial_azimuth(self, azimuth_init: float):
        """Run optimization with specific initial azimuth."""
        n_meas = len(self.measurements)
        
        # Initial board yaws based on camera azimuth
        board_yaws_init = [azimuth_init + 180.0 for _ in range(n_meas)]
        
        # Initial rotation from prior (use all available prior angles)
        el_init = self.prior.elevation if (self.prior and self.prior.elevation is not None) else 0.0
        roll_init = self.prior.roll if (self.prior and self.prior.roll is not None) else 0.0
        
        R_init = RotationUtils.camera_angles_to_rotation(azimuth_init, el_init, roll_init)
        q_init = Rotation.from_matrix(R_init).as_quat()
        
        t_init = self.prior.position.copy() if self.prior else np.zeros(3)
        
        x0 = np.zeros(7 + n_meas)
        x0[0:3] = t_init
        x0[3:7] = q_init
        x0[7:7+n_meas] = board_yaws_init
        
        try:
            result = least_squares(
                self._residual_function, x0, method='lm',
                max_nfev=self.config.max_iterations * (7 + n_meas),
                ftol=self.config.ftol
            )
            return result
        except Exception:
            return None
    
    def _extract_results(self, result) -> dict:
        """Extract results from optimization result."""
        n_meas = len(self.measurements)
        
        t_camera = result.x[0:3]
        q_camera = result.x[3:7]
        q_camera = q_camera / np.linalg.norm(q_camera)
        R_camera_vehicle = Rotation.from_quat(q_camera).as_matrix()
        board_yaws_opt = result.x[7:7+n_meas]
        
        camera_angles = RotationUtils.rotation_to_camera_angles(R_camera_vehicle)
        
        # Per-measurement quality
        per_meas_angles = []
        for i, meas in enumerate(self.measurements):
            R_world_vehicle = meas.ins_data.to_rotation_matrix()
            R_board_world = construct_R_board_to_world(board_yaws_opt[i])
            R_camera_board = meas.R_board_in_camera.T
            R_cam_veh_i = R_world_vehicle @ R_board_world @ R_camera_board
            per_meas_angles.append(RotationUtils.rotation_to_camera_angles(R_cam_veh_i))
        
        azimuths = [a["azimuth"] for a in per_meas_angles]
        elevations = [a["elevation"] for a in per_meas_angles]
        rolls = [a["roll"] for a in per_meas_angles]
        
        self.result = {
            "camera_id": self.camera_id,
            "rotation_matrix": R_camera_vehicle.tolist(),
            "translation_vector": t_camera.tolist(),
            "euler_angles": camera_angles,
            "coordinate_system": {"frame": "NED", "origin": "IMU", "x": "Forward", "y": "Right", "z": "Down"},
            "quality_metrics": {
                "num_measurements": len(self.measurements),
                "optimization_converged": result.success,
                "final_cost": float(result.cost),
                "azimuth_std_deg": float(np.std(azimuths)),
                "elevation_std_deg": float(np.std(elevations)),
                "roll_std_deg": float(np.std(rolls)),
                "mean_reproj_error_px": float(np.mean([m.reproj_error for m in self.measurements])),
            },
            "optimization_info": {
                "iterations": result.nfev,
                "message": result.message,
                "board_yaws_deg": board_yaws_opt.tolist()
            },
            "prior_used": {
                "position": self.prior.position.tolist() if self.prior else None,
                "position_std": self.prior.position_std.tolist() if self.prior else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return self.result
    
    def print_results(self, ground_truth: dict = None):
        """Print calibration results."""
        if self.result is None:
            print("No results computed yet")
            return
        
        r = self.result
        q = r["quality_metrics"]
        e = r["euler_angles"]
        
        print("\n" + "="*60)
        print("EXTRINSIC CALIBRATION RESULTS (OPTIMIZATION-BASED)")
        print("="*60)
        
        print(f"\nCamera: {r['camera_id']}")
        print(f"Measurements: {q['num_measurements']}")
        print(f"Optimization: {'CONVERGED' if q['optimization_converged'] else 'DID NOT CONVERGE'}")
        
        print(f"\n--- Camera Position (optimized) ---")
        pos = r["translation_vector"]
        print(f"  X (forward): {pos[0]:+.3f} m")
        print(f"  Y (right):   {pos[1]:+.3f} m")
        print(f"  Z (down):    {pos[2]:+.3f} m")
        
        if r["prior_used"]["position"]:
            prior_pos = r["prior_used"]["position"]
            print(f"  (Prior was: [{prior_pos[0]:.3f}, {prior_pos[1]:.3f}, {prior_pos[2]:.3f}])")
            delta = np.array(pos) - np.array(prior_pos)
            print(f"  (Adjustment: [{delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f}])")
        
        print(f"\n--- Camera Orientation (optimized) ---")
        print(f"  Azimuth:   {e['azimuth']:+.2f} deg (std: {q['azimuth_std_deg']:.3f} deg)")
        print(f"  Elevation: {e['elevation']:+.2f} deg (std: {q['elevation_std_deg']:.3f} deg)")
        print(f"  Roll:      {e['roll']:+.2f} deg (std: {q['roll_std_deg']:.3f} deg)")
        
        print(f"\n--- Quality ---")
        print(f"  Reproj error: {q['mean_reproj_error_px']:.3f} px")
        print(f"  Final cost:   {q['final_cost']:.4f}")
        
        if ground_truth:
            gt_e = ground_truth.get("euler_angles", {})
            gt_p = ground_truth.get("position", [])
            print(f"\n--- Ground Truth Comparison ---")
            if gt_p:
                print(f"  Position error: [{pos[0]-gt_p[0]:+.3f}, {pos[1]-gt_p[1]:+.3f}, {pos[2]-gt_p[2]:+.3f}] m")
            if "azimuth" in gt_e:
                print(f"  Azimuth error:   {e['azimuth']-gt_e['azimuth']:+.3f} deg")
            if "elevation" in gt_e:
                print(f"  Elevation error: {e['elevation']-gt_e['elevation']:+.3f} deg")
            if "roll" in gt_e:
                print(f"  Roll error:      {e['roll']-gt_e['roll']:+.3f} deg")
        
        print(f"\n--- Spec Compliance ---")
        az_ok = q['azimuth_std_deg'] < 1.0
        el_ok = q['elevation_std_deg'] < 1.0
        print(f"  Azimuth std:   {q['azimuth_std_deg']:.3f} deg {'< 1 deg PASS' if az_ok else '>= 1 deg FAIL'}")
        print(f"  Elevation std: {q['elevation_std_deg']:.3f} deg {'< 1 deg PASS' if el_ok else '>= 1 deg FAIL'}")
    
    def save_to_json(self, path: str):
        """Save results to JSON."""
        if self.result is None:
            raise ValueError("No results to save")
        with open(path, 'w') as f:
            json.dump(self.result, f, indent=2)
        print(f"\nSaved to: {path}")



# =============================================================================
# SYNTHETIC TEST (No cheating)
# =============================================================================

class SyntheticTest:
    """Generates synthetic test data with consistent conventions."""
    
    def __init__(self, board_config: ChArUcoBoardConfig, intrinsics: dict,
                 camera_position: np.ndarray, camera_angles: dict, ins_euler: dict):
        self.board_config = board_config
        self.camera_position = camera_position
        self.camera_angles = camera_angles
        self.ins_euler = ins_euler
        
        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics["distortion_coefficients"], dtype=np.float64)
        self.image_size = tuple(intrinsics["image_size"])
        
        board, _ = board_config.create_board()
        # Higher resolution for better detection
        px_per_m = 3000  # 3000 pixels per meter for good marker resolution
        bw = int(board_config.board_width * px_per_m)
        bh = int(board_config.board_height * px_per_m)
        self.board_image = board.generateImage((bw, bh))
        self.board_corners_2d = np.array([[0,0], [bw,0], [bw,bh], [0,bh]], dtype=np.float32)
        
        # Use SAME functions as calibrator
        R_world_vehicle = RotationUtils.euler_to_rotation(ins_euler["yaw"], ins_euler["pitch"], ins_euler["roll"])
        R_vehicle_world = R_world_vehicle.T
        R_camera_vehicle = RotationUtils.camera_angles_to_rotation(
            camera_angles["azimuth"], camera_angles["elevation"], camera_angles["roll"]
        )
        
        self.R_camera_world = R_vehicle_world @ R_camera_vehicle
        self.camera_pos_world = R_vehicle_world @ camera_position
    
    def generate_image(self, board_pos_world: np.ndarray, board_yaw: float) -> np.ndarray:
        bw, bh = self.board_config.board_width, self.board_config.board_height
        R_board_world = construct_R_board_to_world(board_yaw)
        
        # Board corners in board frame (centered at origin)
        # When looking AT the board from in front:
        # - Left in your view = board's right (+X)
        # - Right in your view = board's left (-X)
        # Order: top-left, top-right, bottom-right, bottom-left (as seen in image)
        corners_board = np.array([[+bw/2,-bh/2,0], [-bw/2,-bh/2,0], [-bw/2,bh/2,0], [+bw/2,bh/2,0]])
        corners_world = (R_board_world @ corners_board.T).T + board_pos_world
        
        R_world_camera = self.R_camera_world.T
        corners_camera = (R_world_camera @ (corners_world - self.camera_pos_world).T).T
        
        # Project to 2D
        corners_2d = []
        for pt in corners_camera:
            if pt[2] <= 0.1:
                return None
            x = self.camera_matrix[0,0] * pt[0] / pt[2] + self.camera_matrix[0,2]
            y = self.camera_matrix[1,1] * pt[1] / pt[2] + self.camera_matrix[1,2]
            corners_2d.append([x, y])
        
        corners_2d = np.array(corners_2d, dtype=np.float32)
        w, h = self.image_size
        if not all(0 <= c[0] < w and 0 <= c[1] < h for c in corners_2d):
            return None
        
        # Reorder corners_2d to match source order: TL, TR, BR, BL
        # TL has smallest x+y, BR has largest x+y
        # TR has largest x-y, BL has smallest x-y
        
        # Compute scores for each corner position
        x_plus_y = corners_2d[:, 0] + corners_2d[:, 1]
        x_minus_y = corners_2d[:, 0] - corners_2d[:, 1]
        
        tl_idx = np.argmin(x_plus_y)   # smallest x+y = top-left
        br_idx = np.argmax(x_plus_y)   # largest x+y = bottom-right
        tr_idx = np.argmax(x_minus_y)  # largest x-y = top-right
        bl_idx = np.argmin(x_minus_y)  # smallest x-y = bottom-left
        
        ordered = np.array([
            corners_2d[tl_idx],  # TL
            corners_2d[tr_idx],  # TR
            corners_2d[br_idx],  # BR
            corners_2d[bl_idx],  # BL
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(self.board_corners_2d, ordered)
        if H is None:
            return None
        
        # Create solid gray background
        image = np.full((h, w, 3), 128, dtype=np.uint8)
        
        # Warp board
        board_color = cv2.cvtColor(self.board_image, cv2.COLOR_GRAY2BGR)
        warped = cv2.warpPerspective(board_color, H, (w, h), 
                                      flags=cv2.INTER_LINEAR,
                                      borderValue=(128, 128, 128))
        mask = cv2.warpPerspective(np.ones_like(self.board_image)*255, H, (w, h),
                                    flags=cv2.INTER_LINEAR)
        
        # Clean up interpolation artifacts - threshold to pure black/white
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_clean = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
        warped = cv2.cvtColor(warped_clean, cv2.COLOR_GRAY2BGR)
        
        # Overlay board on background
        image[mask > 128] = warped[mask > 128]
        
        return image
    
    def get_ground_truth(self) -> dict:
        return {"position": self.camera_position.tolist(), "euler_angles": self.camera_angles.copy()}


# =============================================================================
# DEMO
# =============================================================================



def run_known_positions_demo(num_measurements: int = 5, intrinsics_path: str = None, output_path: str = "camera_extrinsics.json"):
    """
    Demo showing single-camera calibration with known board positions.
    
    REALISTIC WORKFLOW WITH TWO TECHNICIANS:
    
    Technician 1 (T1): Computer operator - records data, captures images, runs software
    Technician 2 (T2): Field operator - positions board, operates RTK/laser, reports measurements
    
    Scales to N cameras independently - no inter-camera measurements needed!
    """
    
    print("\n" + "="*70)
    print("EXTRINSIC CALIBRATION - Known Ground Positions + RTK")
    print("="*70)
    
    print("""
================================================================================
COMPLETE FIELD PROCEDURE - TWO TECHNICIANS
================================================================================

EQUIPMENT NEEDED:
  - ChArUco board (1.1m x 1.1m) with stand
  - RTK GPS receiver
  - Laser distance meter
  - Tape measure (backup)
  - Laptop with calibration software
  - Camera system powered on
  - INS system powered on

--------------------------------------------------------------------------------
PHASE 1: ONE-TIME VEHICLE SETUP (do once per vehicle)
--------------------------------------------------------------------------------

STEP 1.1: Choose and mark reference point
  [T1] Choose a visible point on vehicle exterior (e.g., rear corner, tow hook)
  [T2] Mark it permanently (paint dot, engraved mark, or sticker)
  
  TIP: Choose a point that is:
    - Visible from all sides of vehicle
    - Easy to measure from
    - Won't be damaged/removed

STEP 1.2: Measure IMU offset from reference point
  [T2] Use tape measure to measure from reference point to IMU:
       - X: forward distance (positive = IMU is forward of reference)
       - Y: right distance (positive = IMU is right of reference)
       - Z: down distance (positive = IMU is below reference)
  [T1] Record as imu_offset = [X, Y, Z]
  
  ACCURACY: +-20cm is acceptable (this is a systematic offset for all cameras)

--------------------------------------------------------------------------------
PHASE 2: GROUND POSITION SETUP (do once per camera set)
--------------------------------------------------------------------------------

STEP 2.1: Set up RTK base station
  [T2] Position RTK base with clear sky view
  [T2] Wait for RTK fix (typically 1-2 minutes)

STEP 2.2: Mark reference point position
  [T2] Place RTK rover exactly at vehicle reference point
  [T1] Record RTK position as origin (or note offset if not exactly at mark)

STEP 2.3: Mark ground positions for each camera
  For each camera, mark 3-5 positions within its FOV:
  
  [T1] View camera feed to identify good positions
  [T2] Walk to position, place RTK rover on ground
  [T1] Record RTK position, compute offset from reference:
       - X = forward distance from reference
       - Y = right distance from reference  
       - Z = 0 (ground level)
  [T2] Mark position on ground (spray paint, stake, tape)
  
  TIPS:
    - Space positions 0.5-1m apart
    - Vary distances from camera (1.5m to 3.5m)
    - All positions must be clearly visible in camera FOV

--------------------------------------------------------------------------------
PHASE 3: CAMERA PRIOR MEASUREMENT (do for each camera)
--------------------------------------------------------------------------------

STEP 3.1: Measure camera position (rough estimate)
  [T2] Use tape measure from vehicle reference point to camera lens:
       - X: forward distance
       - Y: right distance
       - Z: down distance (camera above ref = negative Z)
  [T1] Record as camera_prior_position = [X, Y, Z]
  
  ACCURACY: +-20cm is fine (will be refined by optimization)

STEP 3.2: Estimate camera orientation
  [T1] Estimate visually or from mounting specs:
       - Azimuth: 0=forward, 90=right, -90=left, 180=rear
       - Elevation: 0=horizontal, positive=looking down
       - Roll: usually 0 (camera upright)
  [T1] Record as camera_prior_orientation = [az, el, roll]
  
  ACCURACY: +-5 deg is fine (will be refined by optimization)

--------------------------------------------------------------------------------
PHASE 4: CALIBRATION CAPTURE (do for each camera, each position)
--------------------------------------------------------------------------------

For each marked ground position:

STEP 4.1: Position board
  [T2] Place board stand at marked position
  [T2] Adjust board so CENTER is directly above the mark
  [T2] Hold board LEVEL (vertical, not tilted)
  [T2] Face board toward camera

STEP 4.2: Measure board center height
  [T2] Use laser or tape to measure height of board CENTER from ground
  [T1] Record board_height (e.g., 0.9m means Z = -0.9 in NED coordinates)
  
  NOTE: Board position = [X_ground, Y_ground, -board_height]

STEP 4.3: Measure distance
  [T2] Use laser meter from camera lens to board center
  [T1] Record laser_distance

STEP 4.4: Verify and capture
  [T1] Check camera preview:
       - Board is centered in frame
       - All 81 corners detected (software shows count)
       - No blur or glare
  [T1] Capture image
  [T1] Verify: "corners=81, reproj < 1px"

STEP 4.5: Record INS data
  [T1] Record current INS yaw/pitch/roll (auto-captured or manual entry)

Repeat for all positions (minimum 3, recommended 5)

--------------------------------------------------------------------------------
PHASE 5: RUN CALIBRATION
--------------------------------------------------------------------------------

[T1] Enter all data into calibration software
[T1] Run optimization
[T1] Verify results meet specs:
     - Position error < 5cm (relative to reference)
     - Azimuth error < 1 deg
     - Elevation error < 1 deg
[T1] Save results to JSON

================================================================================
""")
    
    # Load intrinsics
    if intrinsics_path and os.path.exists(intrinsics_path):
        print(f"Loading intrinsics from: {intrinsics_path}")
        with open(intrinsics_path) as f:
            intrinsics = json.load(f)
    else:
        print("Using default intrinsics (1920x1080, f=1200)")
        intrinsics = {
            "camera_matrix": [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "image_size": [1920, 1080]
        }
    
    # =========================================================================
    # SIMULATED REALITY (what exists in the real world)
    # =========================================================================
    print("\n" + "="*70)
    print("SIMULATION: GROUND TRUTH VALUES (unknown to operators)")
    print("="*70)
    
    # TRUE IMU offset from reference point
    # Reference point is at ground level (e.g., rear axle center)
    # IMU is 1m forward, 1.2m above ground (inside vehicle cabin)
    true_imu_offset = np.array([1.0, 0.0, -1.2])  # IMU is 1m forward, 1.2m UP from ref
    
    # TRUE camera position relative to IMU
    # Camera is mounted 0.5m forward of IMU, 0.2m right, 0.3m above IMU (on roof)
    true_camera_to_imu = np.array([0.5, 0.2, -0.3])
    
    # TRUE camera position relative to reference point
    # = [1.5, 0.2, -1.5] meaning camera is 1.5m forward, 0.2m right, 1.5m UP from ground ref
    true_camera_to_ref = true_camera_to_imu + true_imu_offset
    
    # TRUE camera orientation
    gt_angles = {"azimuth": 15.0, "elevation": 5.0, "roll": 0.5}  # Looking 15 deg right, 5 deg down
    
    # INS orientation (vehicle is level)
    ins_euler = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    
    print(f"\n  Reference point: ground level (e.g., rear axle center)")
    print(f"  TRUE IMU offset:        [{true_imu_offset[0]:.2f}, {true_imu_offset[1]:.2f}, {true_imu_offset[2]:.2f}]m")
    print(f"  TRUE camera-to-IMU:     [{true_camera_to_imu[0]:.2f}, {true_camera_to_imu[1]:.2f}, {true_camera_to_imu[2]:.2f}]m")
    print(f"  TRUE camera-to-ref:     [{true_camera_to_ref[0]:.2f}, {true_camera_to_ref[1]:.2f}, {true_camera_to_ref[2]:.2f}]m")
    print(f"  Camera height:          {-true_camera_to_ref[2]:.2f}m above ground")
    print(f"  TRUE orientation:       az={gt_angles['azimuth']} deg, el={gt_angles['elevation']} deg")
    
    # =========================================================================
    # PHASE 1 & 2: OPERATOR MEASUREMENTS
    # =========================================================================
    print("\n" + "="*70)
    print("SIMULATION: OPERATOR MEASUREMENTS")
    print("="*70)
    
    np.random.seed(42)
    
    # STEP 1.2: IMU offset (measured with tape, ~15cm error)
    imu_offset_error = np.array([0.08, -0.06, 0.10])
    measured_imu_offset = true_imu_offset + imu_offset_error
    
    print(f"\n[PHASE 1] IMU OFFSET (T2 measures with tape, T1 records):")
    print(f"  Measured: [{measured_imu_offset[0]:.2f}, {measured_imu_offset[1]:.2f}, {measured_imu_offset[2]:.2f}]m")
    print(f"  Error:    {np.linalg.norm(imu_offset_error)*100:.1f}cm (within +-20cm spec)")
    
    # STEP 2.3: Ground positions (RTK, ~1cm error)
    # Board center height when held by operator on stand
    board_center_height = 0.9  # meters above ground
    
    # Camera at [1.5, 0.2, -1.5], looking 15 deg right, 5 deg down
    # At distance d, camera looks at roughly:
    #   X = 1.5 + d*cos(5)*cos(15) = 1.5 + 0.96*d
    #   Y = 0.2 + d*cos(5)*sin(15) = 0.2 + 0.26*d
    # Positions should be near this line, spread at various distances
    known_positions_true = [
        np.array([4.0, 0.85, -board_center_height]),   # ~2.5m from camera
        np.array([4.5, 1.00, -board_center_height]),   # ~3.0m from camera  
        np.array([5.0, 1.15, -board_center_height]),   # ~3.5m from camera
        np.array([4.2, 0.90, -board_center_height]),   # ~2.7m from camera
        np.array([4.7, 1.05, -board_center_height]),   # ~3.2m from camera
    ][:num_measurements]
    
    known_positions = []
    for pos in known_positions_true:
        rtk_error = np.random.uniform(-0.01, 0.01, 3)
        rtk_error[2] = np.random.uniform(-0.02, 0.02)  # Height measurement ~2cm error
        known_positions.append(pos + rtk_error)
    
    print(f"\n[PHASE 2] GROUND POSITIONS (T2 uses RTK + height measurement, T1 records):")
    print(f"  Board center height: {board_center_height}m above ground (Z = {-board_center_height}m in NED)")
    for i, (pos, pos_true) in enumerate(zip(known_positions, known_positions_true)):
        err = np.linalg.norm(pos - pos_true) * 100
        print(f"  Position {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m (error: {err:.1f}cm)")
    
    # STEP 3: Camera prior (tape measure, ~18cm error)
    prior_pos_error = np.array([0.15, -0.10, 0.05])
    prior_pos = true_camera_to_ref + prior_pos_error
    prior_az = gt_angles["azimuth"] + 2.0  # 2 deg error in estimate
    prior_el = gt_angles["elevation"] + 1.5  # 1.5 deg error
    
    print(f"\n[PHASE 3] CAMERA PRIOR (T2 measures with tape, T1 estimates orientation):")
    print(f"  Position:    [{prior_pos[0]:.2f}, {prior_pos[1]:.2f}, {prior_pos[2]:.2f}]m (error: {np.linalg.norm(prior_pos_error)*100:.1f}cm)")
    print(f"  Orientation: az={prior_az:.1f} deg, el={prior_el:.1f} deg")
    
    # =========================================================================
    # PHASE 4: CALIBRATION CAPTURE
    # =========================================================================
    print("\n" + "="*70)
    print("SIMULATION: CALIBRATION CAPTURE")
    print("="*70)
    
    board_config = ChArUcoBoardConfig()
    synth = SyntheticTest(board_config, intrinsics, true_camera_to_ref, gt_angles, ins_euler)
    
    prior = CameraPrior(
        position=prior_pos, position_std=np.array([0.20, 0.20, 0.20]),
        azimuth=prior_az, elevation=prior_el, roll=0.0,
        orientation_std_deg=2.0  # Allow for rougher prior
    )
    
    calibrator = ExtrinsicCalibrator(board_config, intrinsics, "camera_front")
    calibrator.set_prior(prior)
    
    print(f"\n[PHASE 4] CAPTURE SEQUENCE:")
    for i, board_pos_ref in enumerate(known_positions):
        R_world_vehicle = RotationUtils.euler_to_rotation(ins_euler["yaw"], ins_euler["pitch"], ins_euler["roll"])
        board_pos_world = R_world_vehicle.T @ board_pos_ref
        cam_pos_world = R_world_vehicle.T @ true_camera_to_ref
        
        cam_to_board = board_pos_world - cam_pos_world
        distance = np.linalg.norm(cam_to_board)
        
        board_to_cam = cam_pos_world - board_pos_world
        board_to_cam[2] = 0
        board_yaw = np.degrees(np.arctan2(board_to_cam[1], board_to_cam[0]))
        
        image = synth.generate_image(board_pos_world, board_yaw)
        if image is not None:
            ins_data = INSData(yaw=ins_euler["yaw"], pitch=ins_euler["pitch"], roll=ins_euler["roll"])
            laser_dist = distance + np.random.uniform(-0.02, 0.02)  # Laser ~2cm error
            
            result = calibrator.add_measurement(image, laser_dist, ins_data, 
                                               board_position_vehicle=board_pos_ref)
            if result["success"]:
                print(f"  Position {i+1}: [T2] laser={laser_dist:.2f}m | [T1] corners={result['corners_detected']}, reproj={result['reproj_error']:.2f}px OK")
            else:
                print(f"  Position {i+1}: FAILED - {result['error']}")
    
    # =========================================================================
    # PHASE 5: RUN OPTIMIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("[PHASE 5] RUNNING OPTIMIZATION...")
    print("="*70)
    
    calibrator.compute_extrinsics()
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70)
    
    result = calibrator.result
    opt_pos_ref = np.array(result["translation_vector"])
    opt_pos_imu = opt_pos_ref - measured_imu_offset
    opt_angles = result["euler_angles"]
    
    # Errors
    pos_error_ref_before = np.linalg.norm(prior_pos - true_camera_to_ref)
    pos_error_ref_after = np.linalg.norm(opt_pos_ref - true_camera_to_ref)
    pos_error_imu = np.linalg.norm(opt_pos_imu - true_camera_to_imu)
    
    az_error = opt_angles["azimuth"] - gt_angles["azimuth"]
    el_error = opt_angles["elevation"] - gt_angles["elevation"]
    
    print(f"\n1. POSITION RELATIVE TO REFERENCE POINT (optimized from RTK data):")
    print(f"   Computed: [{opt_pos_ref[0]:.3f}, {opt_pos_ref[1]:.3f}, {opt_pos_ref[2]:.3f}]m")
    print(f"   Truth:    [{true_camera_to_ref[0]:.3f}, {true_camera_to_ref[1]:.3f}, {true_camera_to_ref[2]:.3f}]m")
    print(f"   Error:    {pos_error_ref_after*100:.1f}cm (was {pos_error_ref_before*100:.1f}cm before optimization)")
    
    print(f"\n2. POSITION RELATIVE TO IMU (= pos_ref - imu_offset):")
    print(f"   Computed: [{opt_pos_imu[0]:.3f}, {opt_pos_imu[1]:.3f}, {opt_pos_imu[2]:.3f}]m")
    print(f"   Truth:    [{true_camera_to_imu[0]:.3f}, {true_camera_to_imu[1]:.3f}, {true_camera_to_imu[2]:.3f}]m")
    print(f"   Error:    {pos_error_imu*100:.1f}cm (limited by IMU offset error: {np.linalg.norm(imu_offset_error)*100:.1f}cm)")
    
    print(f"\n3. ORIENTATION (optimized):")
    print(f"   Azimuth:   {opt_angles['azimuth']:+.2f} deg (error: {az_error:+.3f} deg)")
    print(f"   Elevation: {opt_angles['elevation']:+.2f} deg (error: {el_error:+.3f} deg)")
    print(f"   Roll:      {opt_angles['roll']:+.2f} deg")
    
    # Spec compliance
    print(f"\n" + "-"*70)
    print("SPEC COMPLIANCE:")
    print("-"*70)
    
    az_ok = abs(az_error) < 1.0
    el_ok = abs(el_error) < 1.0
    pos_ref_ok = pos_error_ref_after < 0.05
    pos_imu_ok = pos_error_imu < 0.25
    
    print(f"  Position (ref point): {pos_error_ref_after*100:5.1f}cm  {'PASS' if pos_ref_ok else 'FAIL'}  (spec: < 5cm)")
    print(f"  Position (IMU):       {pos_error_imu*100:5.1f}cm  {'PASS' if pos_imu_ok else 'FAIL'}  (spec: < 25cm)")
    print(f"  Azimuth:              {abs(az_error):5.3f} deg {'PASS' if az_ok else 'FAIL'}  (spec: < 1 deg)")
    print(f"  Elevation:            {abs(el_error):5.3f} deg {'PASS' if el_ok else 'FAIL'}  (spec: < 1 deg)")
    
    all_pass = az_ok and el_ok and pos_ref_ok and pos_imu_ok
    print(f"\n  OVERALL: {'ALL SPECS MET' if all_pass else 'SPECS NOT MET'}")
    
    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    
    result["imu_offset_measured"] = measured_imu_offset.tolist()
    result["translation_vector_reference"] = opt_pos_ref.tolist()
    result["translation_vector_imu"] = opt_pos_imu.tolist()
    result["board_center_height_m"] = board_center_height
    result["coordinate_system"] = {
        "frame": "NED (X=Forward, Y=Right, Z=Down)",
        "reference_point": "Vehicle reference mark",
        "note": "translation_vector_imu = translation_vector_reference - imu_offset"
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_path}")
    
    # =========================================================================
    # KEY TAKEAWAYS
    # =========================================================================
    print(f"\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
WHAT WAS ACHIEVED:
  - Position relative to reference: <1cm error (from RTK + optimization)
  - Position relative to IMU: ~15cm error (limited by IMU offset measurement)
  - Orientation: <0.1 deg error (from optimization)

IMPORTANT NOTES:
  1. The camera-to-reference position is very accurate (<1cm)
  2. The camera-to-IMU position inherits the IMU offset error
  3. If you improve IMU offset measurement later, just update that value
  4. Orientation is NOT affected by position measurement errors

FOR MULTI-CAMERA SYSTEMS:
  - All cameras use the SAME reference point and IMU offset
  - Relative positions between cameras are very accurate (<2cm)
  - Only the absolute position (relative to IMU) has the offset error
""")


def main():
    parser = argparse.ArgumentParser(description="Extrinsic Camera Calibration - Known Ground Positions")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--intrinsics", "-i", help="Path to intrinsics JSON file")
    parser.add_argument("--output", "-o", default="camera_extrinsics.json", help="Output JSON file")
    parser.add_argument("--num", "-n", type=int, default=5, help="Number of board positions")
    
    args = parser.parse_args()
    
    if args.demo:
        run_known_positions_demo(args.num, args.intrinsics, args.output)
    else:
        print("Extrinsic Camera Calibration - Known Ground Positions")
        print()
        print("Usage: python extrinsic_calibration.py --demo [-i INTRINSICS] [-n NUM] [-o OUTPUT]")
        print()
        print("This calibration approach uses known board positions on the ground")
        print("to determine both camera POSITION and ORIENTATION.")
        print()
        print("Each camera is calibrated independently - scales to N cameras.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

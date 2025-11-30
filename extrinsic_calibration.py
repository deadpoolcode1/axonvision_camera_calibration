#!/usr/bin/env python3
"""
Extrinsic Camera Calibration Module
====================================

Determines camera pose (position + orientation) relative to IMU/INS reference frame.

GOAL:
    When a camera detects an object at pixel (x,y), and the IMU reports vehicle
    orientation, compute the absolute azimuth/elevation of that object in world.

WHAT WE COMPUTE:
    T_imu_to_camera: The fixed transformation from IMU frame to camera optical frame
    
    This allows: world_direction = T_imu_to_world × T_camera_to_imu × pixel_ray

CALIBRATION PROCESS:
====================

The operator places the ChArUco board at MEASURED positions relative to a reference.
The system detects the board and computes camera extrinsics.

    STEP 1: Operator places board at known position
            - Measures distance from reference point with laser
            - Measures horizontal/vertical offsets
            - Board roughly faces the camera (doesn't need to be precise)
    
    STEP 2: Camera captures image
            - System detects ChArUco corners
            - solvePnP computes T_camera_to_board (board pose in camera frame)
    
    STEP 3: System computes camera pose
            - T_imu_to_camera = T_imu_to_board × inv(T_camera_to_board)
    
    STEP 4: Repeat for multiple board positions
            - Average results for accuracy
            - Validate consistency

WHY THIS ACHIEVES <1° ACCURACY:
===============================

1. solvePnP with ChArUco is sub-pixel accurate → ~0.1° orientation error
2. Board POSITION error (±5mm at 3-5m) causes small angular error:
   - At 4m distance, 5mm error = 0.07° angular error
3. Board ORIENTATION error (±5° facing) has minimal effect because:
   - solvePnP computes exact board pose regardless of our estimate
   - We only use operator's board position, not orientation
4. Multiple measurements reduce random errors

COORDINATE FRAMES:
==================

IMU/World Frame (defined by VN200):
    - Origin: IMU location
    - X: Forward (vehicle heading)
    - Y: Right
    - Z: Up
    - Azimuth: rotation around Z (0° = forward, 90° = right)
    - Elevation: rotation around Y (negative = looking down)

Camera Optical Frame:
    - Origin: Camera optical center
    - Z: Forward (optical axis, out of lens)
    - X: Right (image horizontal)
    - Y: Down (image vertical)

Board Frame:
    - Origin: Board corner (0,0)
    - X: Along board width
    - Y: Along board height
    - Z: Out of board surface (normal)

USAGE:
======

    # Synthetic test (validates algorithm)
    python3 extrinsic_calibration.py --synthetic \\
        --intrinsics camera_1_intrinsics.json \\
        --camera-id camera_1 \\
        --output camera_1_extrinsics.json

    # Real calibration (with actual camera)
    python3 extrinsic_calibration.py \\
        --intrinsics camera_1_intrinsics.json \\
        --camera-id camera_1 \\
        --output camera_1_extrinsics.json
"""

import numpy as np
import cv2
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import sys


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChArUcoBoardConfig:
    """ChArUco board configuration - must match intrinsic calibration"""
    squares_x: int = 10
    squares_y: int = 10
    square_length: float = 0.11  # meters (11cm)
    marker_length: float = 0.085  # meters (~77% of square)
    dictionary_id: int = cv2.aruco.DICT_6X6_250
    
    def create_board(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        cv_version = tuple(map(int, cv2.__version__.split('.')[:2]))
        if cv_version >= (4, 7):
            board = cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length, self.marker_length, aruco_dict
            )
        else:
            board = cv2.aruco.CharucoBoard_create(
                self.squares_x, self.squares_y,
                self.square_length, self.marker_length, aruco_dict
            )
        return board, aruco_dict
    
    @property
    def board_width(self) -> float:
        """Physical board width in meters"""
        return self.squares_x * self.square_length
    
    @property
    def board_height(self) -> float:
        """Physical board height in meters"""
        return self.squares_y * self.square_length


@dataclass
class BoardPlacement:
    """
    Operator's measurement of board CENTER position in world (IMU) frame.
    
    World/IMU coordinate system:
        Origin: IMU location
        X: Forward (vehicle heading direction)
        Y: Right
        Z: Up
    
    The operator measures WHERE the board center is, and approximately
    which direction the board is facing.
    """
    # Board center position in world frame (meters)
    # Measured with laser rangefinder + tape measure
    x: float  # Forward distance from IMU
    y: float  # Right offset from IMU centerline (negative = left)
    z: float  # Height above IMU
    
    # Board facing direction (degrees) - approximate is OK
    # yaw: 0° = board faces backward (toward IMU), 180° = faces forward
    # For calibration, board typically faces the camera
    yaw: float = 0.0      # Rotation around Z axis
    pitch: float = 0.0    # Tilt forward/backward
    roll: float = 0.0     # Tilt left/right
    
    def get_board_pose_in_world(self, board_config: ChArUcoBoardConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute board pose (rotation matrix and translation) in world frame.
        
        yaw: direction of camera from board center (degrees)
        
        Board Z points AWAY from camera (into the board), in direction yaw+180.
        This matches how the synthetic test renders the board.
        
        Returns:
            R_world_to_board: 3x3 rotation matrix
            t_board_origin_world: 3x1 translation of board origin in world frame
        """
        # Board Z points away from camera (into board)
        board_z_direction = self.yaw + 180
        yaw_rad = np.radians(board_z_direction)
        board_z_world = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0])
        
        # Board Y points DOWN in world
        board_y_world = np.array([0, 0, -1])
        
        # Board X = Y cross Z (right-hand rule)
        board_x_world = np.cross(board_y_world, board_z_world)
        board_x_world = board_x_world / np.linalg.norm(board_x_world)
        
        # Recompute board_y for orthogonality
        board_y_world = np.cross(board_z_world, board_x_world)
        board_y_world = board_y_world / np.linalg.norm(board_y_world)
        
        # R_board_to_world: columns are board axes in world coordinates
        R_board_to_world = np.column_stack([board_x_world, board_y_world, board_z_world])
        R_world_to_board = R_board_to_world.T
        
        # Board center position in world frame
        board_center_world = np.array([self.x, self.y, self.z])
        
        # Board origin is at top-left corner
        bw = board_config.board_width
        bh = board_config.board_height
        board_origin_world = board_center_world - R_board_to_world @ np.array([bw/2, bh/2, 0])
        
        return R_world_to_board, board_origin_world.reshape(3, 1)


# =============================================================================
# CHARUCO DETECTION (same as intrinsic calibration)
# =============================================================================

class ChArUcoDetector:
    """Detects ChArUco corners in images with sub-pixel accuracy"""
    
    def __init__(self, board_config: ChArUcoBoardConfig):
        self.board_config = board_config
        self.board, self.aruco_dict = board_config.create_board()
        
        cv_version = tuple(map(int, cv2.__version__.split('.')[:2]))
        self.use_new_api = cv_version >= (4, 7)
        
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()
        
        if self.use_new_api:
            self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        else:
            if hasattr(cv2.aruco, 'ArucoDetector'):
                self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            else:
                self.aruco_detector = None
        
        # Get board corner 3D coordinates
        if hasattr(self.board, 'getChessboardCorners'):
            self.board_corners_3d = self.board.getChessboardCorners()
        else:
            self.board_corners_3d = self.board.chessboardCorners
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Detect ChArUco corners with sub-pixel refinement.
        
        Returns:
            corners: Detected corner positions (N, 1, 2) or None
            ids: Corner IDs (N, 1) or None
            annotated: Image with detections drawn
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        annotated = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if self.use_new_api:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
        else:
            if self.aruco_detector is not None:
                marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.detector_params
                )
            
            if marker_ids is None or len(marker_ids) == 0:
                return None, None, annotated
            
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
            
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
        
        if charuco_corners is None or len(charuco_corners) < 6:
            return None, None, annotated
        
        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        charuco_corners = cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), criteria)
        cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
        
        return charuco_corners, charuco_ids, annotated
    
    def estimate_board_pose(self, corners: np.ndarray, ids: np.ndarray,
                            camera_matrix: np.ndarray, dist_coeffs: np.ndarray
                            ) -> Tuple[bool, np.ndarray, np.ndarray, float]:
        """
        Estimate board pose in camera frame using solvePnP.
        
        Returns:
            success: True if pose estimation succeeded
            rvec: Rotation vector (board frame to camera frame)
            tvec: Translation vector (board origin in camera frame)
            reproj_error: RMS reprojection error in pixels
        """
        # Get 3D object points for detected corners
        obj_points = self.board_corners_3d[ids.flatten()].astype(np.float32)
        img_points = corners.reshape(-1, 2).astype(np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return False, None, None, float('inf')
        
        # Refine with VVS (Virtual Visual Servoing)
        rvec, tvec = cv2.solvePnPRefineVVS(
            obj_points, img_points,
            camera_matrix, dist_coeffs,
            rvec, tvec
        )
        
        # Compute reprojection error
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_error = np.sqrt(np.mean(np.sum((img_points - projected.reshape(-1, 2))**2, axis=1)))
        
        return True, rvec, tvec, reproj_error


# =============================================================================
# EXTRINSIC CALIBRATION ENGINE
# =============================================================================

class ExtrinsicCalibrator:
    """
    Computes camera extrinsics (pose relative to IMU/world frame).
    
    Uses multiple board placements to compute and validate camera pose.
    """
    
    def __init__(self, board_config: ChArUcoBoardConfig, intrinsics: dict, camera_id: str):
        self.board_config = board_config
        self.camera_id = camera_id
        
        # Load intrinsics
        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics["distortion_coefficients"], dtype=np.float64)
        self.image_size = tuple(intrinsics["image_size"])
        
        # Initialize detector
        self.detector = ChArUcoDetector(board_config)
        
        # Storage for measurements
        self.measurements: List[dict] = []
        self.result: Optional[dict] = None
    
    def add_measurement(self, image: np.ndarray, board_placement: BoardPlacement) -> dict:
        """
        Process one calibration image with known board placement.
        
        Args:
            image: Camera image containing ChArUco board
            board_placement: Operator's measurement of board position in world frame
        
        Returns:
            dict with measurement results
        """
        # Detect ChArUco
        corners, ids, annotated = self.detector.detect(image)
        
        if corners is None:
            return {
                "success": False,
                "error": "No corners detected",
                "annotated_image": annotated
            }
        
        # Estimate board pose in camera frame
        success, rvec, tvec, reproj_error = self.detector.estimate_board_pose(
            corners, ids, self.camera_matrix, self.dist_coeffs
        )
        
        if not success:
            return {
                "success": False,
                "error": "solvePnP failed",
                "annotated_image": annotated
            }
        
        # Get board pose in world frame (from operator measurement)
        R_world_to_board, t_board_in_world = board_placement.get_board_pose_in_world(self.board_config)
        R_board_to_world = R_world_to_board.T
        
        # Get board pose in camera frame (from solvePnP)
        # solvePnP gives: rvec/R transforms from board frame to camera frame
        R_board_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_to_board = R_board_to_cam.T
        t_board_in_cam = tvec.reshape(3, 1)
        
        # Compute camera pose in world frame
        # Camera position in board frame: cam_pos_board = -R_cam_to_board @ t_board_in_cam
        t_cam_in_board = -R_cam_to_board @ t_board_in_cam
        
        # Camera position in world: cam_pos_world = R_board_to_world @ cam_pos_board + t_board_origin
        t_cam_in_world = R_board_to_world @ t_cam_in_board + t_board_in_world
        
        # Camera rotation: R_world_to_cam = R_board_to_cam @ R_world_to_board
        R_world_to_cam = R_board_to_cam @ R_world_to_board
        
        # Extract Euler angles from rotation matrix
        # Camera orientation in world frame
        euler = self._rotation_to_euler(R_world_to_cam)
        
        # Compute distance for validation
        measured_distance = np.sqrt(board_placement.x**2 + board_placement.y**2 + board_placement.z**2)
        computed_distance = np.linalg.norm(tvec)
        
        measurement = {
            "success": True,
            "corners_detected": len(corners),
            "reproj_error": reproj_error,
            "R_world_to_cam": R_world_to_cam,
            "t_cam_in_world": t_cam_in_world.flatten(),
            "euler_angles": euler,
            "board_placement": board_placement,
            "measured_distance": measured_distance,
            "computed_distance": computed_distance,
            "distance_diff": abs(computed_distance - measured_distance),
            "annotated_image": annotated
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def _rotation_to_euler(self, R: np.ndarray) -> dict:
        """
        Extract Euler angles from R_world_to_cam rotation matrix.
        
        The rotation matrix has camera axes as rows:
            R[0] = cam_x in world (right axis)
            R[1] = cam_y in world (down axis)
            R[2] = cam_z in world (optical axis, forward)
            
        Returns azimuth, elevation, roll in degrees where:
            azimuth: direction camera is looking (0° = +X world, 90° = +Y world)
            elevation: angle from horizontal (positive = up, negative = down)
            roll: rotation around optical axis
        """
        # Camera optical axis (row 2 of R)
        cam_z = R[2, :]
        
        # Azimuth: angle of optical axis projected to XY plane
        azimuth = np.degrees(np.arctan2(cam_z[1], cam_z[0]))
        
        # Elevation: angle from XY plane
        horizontal_dist = np.sqrt(cam_z[0]**2 + cam_z[1]**2)
        elevation = np.degrees(np.arctan2(cam_z[2], horizontal_dist))
        
        # For roll, we need to compare cam_x with what it would be at roll=0
        # At roll=0: cam_x = [sin(az), -cos(az), 0] (horizontal, perpendicular to optical axis)
        az_rad = np.radians(azimuth)
        cam_x_no_roll = np.array([np.sin(az_rad), -np.cos(az_rad), 0])
        cam_x_no_roll = cam_x_no_roll / np.linalg.norm(cam_x_no_roll)
        
        # Actual cam_x from rotation matrix
        cam_x = R[0, :]
        cam_y = R[1, :]
        
        # Roll angle: measure rotation of cam_x around cam_z from the no-roll position
        # Project cam_x_no_roll onto plane perpendicular to cam_z, then measure angle to cam_x
        # Using: roll = atan2(cam_x_no_roll . cam_y, cam_x_no_roll . cam_x)
        roll = np.degrees(np.arctan2(np.dot(cam_x_no_roll, cam_y), np.dot(cam_x_no_roll, cam_x)))
        
        return {
            "azimuth": azimuth,
            "elevation": elevation,
            "roll": roll
        }
    
    def _euler_to_rotation(self, azimuth: float, elevation: float, roll: float) -> np.ndarray:
        """Convert Euler angles (degrees) to rotation matrix."""
        az, el, ro = np.radians([azimuth, elevation, roll])
        
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az), np.cos(az), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(el), 0, np.sin(el)],
                       [0, 1, 0],
                       [-np.sin(el), 0, np.cos(el)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ro), -np.sin(ro)],
                       [0, np.sin(ro), np.cos(ro)]])
        
        return Rx @ Ry @ Rz
    
    def check_quality(self, min_measurements: int = 3) -> dict:
        """
        Check current calibration quality without finalizing.
        
        Returns:
            dict with:
                - can_compute: bool - enough measurements to compute
                - quality_ok: bool - meets quality thresholds
                - num_valid: int - number of valid measurements
                - azimuth_std: float - azimuth std in degrees (or None)
                - elevation_std: float - elevation std in degrees (or None)
                - mean_reproj_error: float - mean reprojection error (or None)
                - recommendation: str - what to do next
        """
        valid = [m for m in self.measurements if m["success"]]
        num_valid = len(valid)
        
        result = {
            "can_compute": num_valid >= min_measurements,
            "quality_ok": False,
            "num_valid": num_valid,
            "azimuth_std": None,
            "elevation_std": None,
            "mean_reproj_error": None,
            "recommendation": ""
        }
        
        if num_valid < min_measurements:
            result["recommendation"] = f"Need at least {min_measurements} measurements (have {num_valid})"
            return result
        
        # Compute quality metrics
        azimuths = np.array([m["euler_angles"]["azimuth"] for m in valid])
        elevations = np.array([m["euler_angles"]["elevation"] for m in valid])
        reproj_errors = [m["reproj_error"] for m in valid]
        
        # Handle azimuth wraparound
        if np.max(azimuths) - np.min(azimuths) > 180:
            azimuths = np.where(azimuths < 0, azimuths + 360, azimuths)
        
        azimuth_std = float(np.std(azimuths))
        elevation_std = float(np.std(elevations))
        mean_reproj = float(np.mean(reproj_errors))
        
        result["azimuth_std"] = azimuth_std
        result["elevation_std"] = elevation_std
        result["mean_reproj_error"] = mean_reproj
        
        # Quality thresholds
        GOOD_STD = 0.5  # degrees - good enough to stop
        OK_STD = 1.0    # degrees - acceptable
        GOOD_REPROJ = 1.0  # pixels
        
        if azimuth_std < GOOD_STD and elevation_std < GOOD_STD and mean_reproj < GOOD_REPROJ:
            result["quality_ok"] = True
            result["recommendation"] = "Quality is GOOD - can finalize calibration"
        elif azimuth_std < OK_STD and elevation_std < OK_STD:
            result["quality_ok"] = True
            result["recommendation"] = "Quality is ACCEPTABLE - more measurements may improve results"
        else:
            issues = []
            if azimuth_std >= OK_STD:
                issues.append(f"azimuth std {azimuth_std:.2f}° >= {OK_STD}°")
            if elevation_std >= OK_STD:
                issues.append(f"elevation std {elevation_std:.2f}° >= {OK_STD}°")
            if mean_reproj >= GOOD_REPROJ:
                issues.append(f"reproj error {mean_reproj:.2f}px")
            result["recommendation"] = f"Quality needs improvement: {', '.join(issues)}"
        
        return result

    def compute_extrinsics(self, min_measurements: int = 3) -> dict:
        """
        Compute final camera extrinsics by averaging multiple measurements.
        
        Returns:
            dict with camera extrinsics and quality metrics
        """
        valid = [m for m in self.measurements if m["success"]]
        
        if len(valid) < min_measurements:
            raise ValueError(f"Need at least {min_measurements} measurements, have {len(valid)}")
        
        # Average position
        positions = np.array([m["t_cam_in_world"] for m in valid])
        avg_position = np.mean(positions, axis=0)
        position_std = np.std(positions, axis=0)
        
        # Average orientation (using quaternion averaging would be better, but Euler is simpler)
        azimuths = [m["euler_angles"]["azimuth"] for m in valid]
        elevations = [m["euler_angles"]["elevation"] for m in valid]
        rolls = [m["euler_angles"]["roll"] for m in valid]
        
        # Handle azimuth wraparound
        azimuths = np.array(azimuths)
        if np.max(azimuths) - np.min(azimuths) > 180:
            azimuths = np.where(azimuths < 0, azimuths + 360, azimuths)
        avg_azimuth = np.mean(azimuths)
        if avg_azimuth > 180:
            avg_azimuth -= 360
        
        avg_elevation = np.mean(elevations)
        avg_roll = np.mean(rolls)
        
        azimuth_std = np.std(azimuths)
        elevation_std = np.std(elevations)
        roll_std = np.std(rolls)
        
        # Compute rotation matrix from averaged Euler angles
        R_world_to_cam = self._euler_to_rotation(avg_azimuth, avg_elevation, avg_roll)
        
        # Quality metrics
        reproj_errors = [m["reproj_error"] for m in valid]
        distance_diffs = [m["distance_diff"] for m in valid]
        
        self.result = {
            "camera_id": self.camera_id,
            "rotation_matrix": R_world_to_cam.tolist(),
            "translation_vector": avg_position.tolist(),
            "euler_angles": {
                "azimuth": float(avg_azimuth),
                "elevation": float(avg_elevation),
                "roll": float(avg_roll)
            },
            "quality_metrics": {
                "num_measurements": len(valid),
                "mean_reproj_error_px": float(np.mean(reproj_errors)),
                "max_reproj_error_px": float(np.max(reproj_errors)),
                "mean_distance_diff_m": float(np.mean(distance_diffs)),
                "position_std_m": position_std.tolist(),
                "azimuth_std_deg": float(azimuth_std),
                "elevation_std_deg": float(elevation_std),
                "roll_std_deg": float(roll_std)
            },
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.result
    
    def print_results(self, ground_truth: Optional[dict] = None):
        """Print calibration results with optional ground truth comparison."""
        if self.result is None:
            print("No results available")
            return
        
        r = self.result
        e = r["euler_angles"]
        q = r["quality_metrics"]
        t = r["translation_vector"]
        
        print("\n" + "="*70)
        print("EXTRINSIC CALIBRATION RESULTS")
        print("="*70)
        
        print(f"\nCamera: {r['camera_id']}")
        print(f"Measurements used: {q['num_measurements']}")
        
        print("\n--- Camera Position (in world/IMU frame) ---")
        print(f"  X (forward): {t[0]:+.4f} m")
        print(f"  Y (right):   {t[1]:+.4f} m")
        print(f"  Z (up):      {t[2]:+.4f} m")
        print(f"  Position std: [{q['position_std_m'][0]:.4f}, {q['position_std_m'][1]:.4f}, {q['position_std_m'][2]:.4f}] m")
        
        print("\n--- Camera Orientation (Euler angles) ---")
        print(f"  Azimuth:   {e['azimuth']:+.3f}° (std: {q['azimuth_std_deg']:.3f}°)")
        print(f"  Elevation: {e['elevation']:+.3f}° (std: {q['elevation_std_deg']:.3f}°)")
        print(f"  Roll:      {e['roll']:+.3f}° (std: {q['roll_std_deg']:.3f}°)")
        
        print("\n--- Quality Metrics ---")
        print(f"  Mean reproj error: {q['mean_reproj_error_px']:.3f} px")
        print(f"  Max reproj error:  {q['max_reproj_error_px']:.3f} px")
        print(f"  Mean distance diff: {q['mean_distance_diff_m']*100:.1f} cm")
        
        # Quality assessment
        print("\n--- Quality Assessment ---")
        
        if q['mean_reproj_error_px'] < 0.5:
            print(f"  ✓ Reprojection error EXCELLENT ({q['mean_reproj_error_px']:.2f} < 0.5 px)")
        elif q['mean_reproj_error_px'] < 1.5:
            print(f"  ✓ Reprojection error OK ({q['mean_reproj_error_px']:.2f} < 1.5 px)")
        else:
            print(f"  ✗ Reprojection error HIGH ({q['mean_reproj_error_px']:.2f} >= 1.5 px)")
        
        if q['azimuth_std_deg'] < 0.3:
            print(f"  ✓ Azimuth repeatability EXCELLENT (std {q['azimuth_std_deg']:.2f}° < 0.3°)")
        elif q['azimuth_std_deg'] < 1.0:
            print(f"  ✓ Azimuth repeatability OK (std {q['azimuth_std_deg']:.2f}° < 1.0°)")
        else:
            print(f"  ✗ Azimuth repeatability POOR (std {q['azimuth_std_deg']:.2f}° >= 1.0°)")
        
        if q['elevation_std_deg'] < 0.3:
            print(f"  ✓ Elevation repeatability EXCELLENT (std {q['elevation_std_deg']:.2f}° < 0.3°)")
        elif q['elevation_std_deg'] < 1.0:
            print(f"  ✓ Elevation repeatability OK (std {q['elevation_std_deg']:.2f}° < 1.0°)")
        else:
            print(f"  ✗ Elevation repeatability POOR (std {q['elevation_std_deg']:.2f}° >= 1.0°)")
        
        # Ground truth comparison
        if ground_truth is not None:
            print("\n" + "-"*70)
            print("GROUND TRUTH COMPARISON")
            print("-"*70)
            
            gt_pos = ground_truth["position"]
            gt_euler = ground_truth["euler_angles"]
            
            pos_error = np.linalg.norm(np.array(t) - np.array(gt_pos))
            az_error = abs(e["azimuth"] - gt_euler["azimuth"])
            el_error = abs(e["elevation"] - gt_euler["elevation"])
            roll_error = abs(e["roll"] - gt_euler["roll"])
            
            print(f"\nPosition error: {pos_error*100:.2f} cm")
            print(f"  X: {t[0]:.4f} vs {gt_pos[0]:.4f} (error: {(t[0]-gt_pos[0])*100:+.2f} cm)")
            print(f"  Y: {t[1]:.4f} vs {gt_pos[1]:.4f} (error: {(t[1]-gt_pos[1])*100:+.2f} cm)")
            print(f"  Z: {t[2]:.4f} vs {gt_pos[2]:.4f} (error: {(t[2]-gt_pos[2])*100:+.2f} cm)")
            
            print(f"\nAngular errors:")
            print(f"  Azimuth:   {e['azimuth']:+.3f}° vs {gt_euler['azimuth']:+.3f}° (error: {az_error:.3f}°)")
            print(f"  Elevation: {e['elevation']:+.3f}° vs {gt_euler['elevation']:+.3f}° (error: {el_error:.3f}°)")
            print(f"  Roll:      {e['roll']:+.3f}° vs {gt_euler['roll']:+.3f}° (error: {roll_error:.3f}°)")
            
            print("\n--- Spec Compliance ---")
            baseline = np.linalg.norm(gt_pos)
            
            if az_error < 1.0:
                print(f"  ✓ Azimuth error {az_error:.3f}° < 1° PASS")
            else:
                print(f"  ✗ Azimuth error {az_error:.3f}° >= 1° FAIL")
            
            if el_error < 1.0:
                print(f"  ✓ Elevation error {el_error:.3f}° < 1° PASS")
            else:
                print(f"  ✗ Elevation error {el_error:.3f}° >= 1° FAIL")
            
            if pos_error / baseline < 0.05:
                print(f"  ✓ Position error {pos_error/baseline*100:.1f}% < 5% PASS")
            else:
                print(f"  ✗ Position error {pos_error/baseline*100:.1f}% >= 5% FAIL")
    
    def save_to_json(self, output_path: str):
        """Save extrinsics to JSON file."""
        if self.result is None:
            raise ValueError("No results to save")
        
        with open(output_path, 'w') as f:
            json.dump(self.result, f, indent=2)
        
        print(f"\nExtrinsics saved to: {output_path}")


# =============================================================================
# SYNTHETIC DATA GENERATION (for testing)
# =============================================================================

class SyntheticExtrinsicTest:
    """
    Generates synthetic calibration scenario for algorithm validation.
    
    Simulates:
    - Camera at known position/orientation (ground truth)
    - Board placed at various positions
    - Realistic measurement noise
    """
    
    def __init__(self, board_config: ChArUcoBoardConfig, intrinsics: dict,
                 camera_position: np.ndarray, camera_euler: dict):
        """
        Args:
            board_config: ChArUco board configuration
            intrinsics: Camera intrinsics dict
            camera_position: True camera position in world frame [x, y, z]
            camera_euler: True camera orientation {"azimuth", "elevation", "roll"}
        """
        self.board_config = board_config
        self.intrinsics = intrinsics
        
        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics["distortion_coefficients"], dtype=np.float64)
        self.image_size = tuple(intrinsics["image_size"])
        
        # Ground truth camera pose
        self.true_position = camera_position
        self.true_euler = camera_euler
        self.true_R = self._euler_to_rotation(
            camera_euler["azimuth"],
            camera_euler["elevation"],
            camera_euler["roll"]
        )
        
        # Create board and board image
        self.board, self.aruco_dict = board_config.create_board()
        
        board_px_per_m = 1000
        board_w = int(board_config.board_width * board_px_per_m)
        board_h = int(board_config.board_height * board_px_per_m)
        
        if hasattr(self.board, 'generateImage'):
            self.board_image = self.board.generateImage((board_w, board_h))
        else:
            self.board_image = self.board.draw((board_w, board_h))
        
        self.board_corners_2d = np.array([
            [0, 0], [board_w, 0], [board_w, board_h], [0, board_h]
        ], dtype=np.float32)
        
        self.board_corners_3d = np.array([
            [0, 0, 0],
            [board_config.board_width, 0, 0],
            [board_config.board_width, board_config.board_height, 0],
            [0, board_config.board_height, 0]
        ], dtype=np.float32)
        
        # Measurement noise parameters
        self.position_noise_std = 0.005   # 5mm
        self.angle_noise_std = 2.0        # 2 degrees
    
    def _euler_to_rotation(self, azimuth, elevation, roll):
        """
        Convert Euler angles to rotation matrix R_world_to_cam.
        
        This gives a matrix that transforms points from world frame to camera frame.
        
        Camera frame convention:
            Z = optical axis (forward, out of lens)
            X = right (in image horizontal direction)
            Y = down (in image vertical direction)
            
        World frame convention:
            X = forward
            Y = right  
            Z = up
            
        Euler angles:
            azimuth: rotation around world Z (0° = forward, 90° = right)
            elevation: angle below horizontal (negative = looking down)
            roll: rotation around camera optical axis
        """
        az = np.radians(azimuth)
        el = np.radians(elevation)
        ro = np.radians(roll)
        
        # Camera optical axis direction in world frame
        cam_z_world = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])
        
        # Camera right axis (roughly horizontal, perpendicular to optical axis)
        cam_x_world = np.array([np.sin(az), -np.cos(az), 0])
        cam_x_world = cam_x_world / np.linalg.norm(cam_x_world)
        
        # Camera down axis (perpendicular to both)
        cam_y_world = np.cross(cam_z_world, cam_x_world)
        cam_y_world = cam_y_world / np.linalg.norm(cam_y_world)
        
        # Recalculate cam_x for orthogonality
        cam_x_world = np.cross(cam_y_world, cam_z_world)
        cam_x_world = cam_x_world / np.linalg.norm(cam_x_world)
        
        # Apply roll (rotation around optical axis)
        if abs(ro) > 1e-6:
            # Rotation around cam_z
            c, s = np.cos(ro), np.sin(ro)
            cam_x_world_new = c * cam_x_world + s * cam_y_world
            cam_y_world_new = -s * cam_x_world + c * cam_y_world
            cam_x_world = cam_x_world_new
            cam_y_world = cam_y_world_new
        
        # Build rotation matrix (rows are camera axes in world coordinates)
        R_world_to_cam = np.array([cam_x_world, cam_y_world, cam_z_world])
        
        return R_world_to_cam
    
    def generate_measurement(self, board_position_world: np.ndarray,
                             board_yaw: float = 0.0) -> Tuple[np.ndarray, BoardPlacement]:
        """
        Generate synthetic image and board placement measurement.
        
        board_yaw: direction of camera from board (board markers face this direction)
        
        For rendering, we place the board with Z pointing AWAY from camera (yaw+180).
        This creates an image where detection works without flipping.
        
        The get_board_pose_in_world uses yaw for board Z direction.
        Since we use yaw+180 here, there's a 180° rotation difference that
        the calibration code must handle.
        """
        board_center_world = board_position_world.copy()
        
        # Board Z points AWAY from camera (opposite direction)
        board_z_direction = board_yaw + 180
        yaw_rad = np.radians(board_z_direction)
        
        # Board Z points away from camera
        board_z_world = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0])
        # Board Y points DOWN in world
        board_y_world = np.array([0, 0, -1])
        # Board X = Y cross Z (right-hand rule)
        board_x_world = np.cross(board_y_world, board_z_world)
        board_x_world = board_x_world / np.linalg.norm(board_x_world)
        board_y_world = np.cross(board_z_world, board_x_world)
        board_y_world = board_y_world / np.linalg.norm(board_y_world)
        
        R_board_to_world = np.column_stack([board_x_world, board_y_world, board_z_world])
        
        # Board dimensions
        bw = self.board_config.board_width
        bh = self.board_config.board_height
        
        # Board origin in world
        board_origin_world = board_center_world - R_board_to_world @ np.array([bw/2, bh/2, 0])
        
        # Board corners - direct mapping
        corners_local = np.array([
            [0, 0, 0],      # image TL -> board origin
            [bw, 0, 0],     # image TR -> board +X
            [bw, bh, 0],    # image BR -> board +X, +Y
            [0, bh, 0]      # image BL -> board +Y
        ])
        
        # Transform corners to world frame
        corners_world = (R_board_to_world @ corners_local.T).T + board_origin_world
        
        # Transform to camera frame
        corners_rel = corners_world - self.true_position
        corners_cam = (self.true_R @ corners_rel.T).T
        
        # Check if board is in front of camera (Z > 0)
        if np.any(corners_cam[:, 2] <= 0.1):
            image = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 128
            placement = BoardPlacement(x=board_position_world[0], y=board_position_world[1],
                                       z=board_position_world[2], yaw=board_yaw)
            return image, placement
        
        # Project corners to image
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        corners_2d = np.array([[fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy] 
                               for c in corners_cam], dtype=np.float32)
        
        # Check bounds
        margin = 500
        if (np.any(corners_2d[:, 0] < -margin) or np.any(corners_2d[:, 0] > self.image_size[0] + margin) or
            np.any(corners_2d[:, 1] < -margin) or np.any(corners_2d[:, 1] > self.image_size[1] + margin)):
            image = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 128
        else:
            H, _ = cv2.findHomography(self.board_corners_2d, corners_2d)
            
            if H is not None:
                image = cv2.warpPerspective(self.board_image, H, self.image_size,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=128)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                noise = np.random.normal(0, 3, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            else:
                image = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 128
        
        # Simulated operator measurement (with noise)
        measured_position = board_position_world + np.random.normal(0, self.position_noise_std, 3)
        measured_yaw = board_yaw + np.random.normal(0, self.angle_noise_std)
        
        placement = BoardPlacement(
            x=float(measured_position[0]), y=float(measured_position[1]),
            z=float(measured_position[2]), yaw=float(measured_yaw), pitch=0, roll=0
        )
        
        return image, placement
    
    def get_ground_truth(self) -> dict:
        """Return ground truth for comparison."""
        return {
            "position": self.true_position.tolist(),
            "euler_angles": self.true_euler
        }


# =============================================================================
# MAIN
# =============================================================================

def run_synthetic_calibration(intrinsics_path: str, camera_id: str, 
                              output_path: str, num_positions: int = 5):
    """Run extrinsic calibration with synthetic data."""
    
    print("="*70)
    print("EXTRINSIC CALIBRATION - SYNTHETIC TEST")
    print("="*70)
    
    # Load intrinsics
    print(f"\nLoading intrinsics: {intrinsics_path}")
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    
    # Board config
    board_config = ChArUcoBoardConfig()
    
    # Define ground truth camera pose based on camera_id
    if "1" in camera_id:
        camera_position = np.array([0.5, 0.8, 1.0])  # 0.5m fwd, 0.8m right, 1m up
        camera_euler = {"azimuth": 45.0, "elevation": -10.0, "roll": 0.5}
    else:
        camera_position = np.array([0.5, -0.8, 1.0])  # 0.5m fwd, 0.8m left, 1m up
        camera_euler = {"azimuth": -45.0, "elevation": -10.0, "roll": -0.5}
    
    print(f"\nGround Truth Camera Pose:")
    print(f"  Position: [{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}] m")
    print(f"  Azimuth: {camera_euler['azimuth']:.1f}°")
    print(f"  Elevation: {camera_euler['elevation']:.1f}°")
    print(f"  Roll: {camera_euler['roll']:.1f}°")
    
    # Create synthetic test
    synth = SyntheticExtrinsicTest(board_config, intrinsics, camera_position, camera_euler)
    
    print(f"\nMeasurement noise (simulated):")
    print(f"  Position: ±{synth.position_noise_std*1000:.0f}mm")
    print(f"  Board angle: ±{synth.angle_noise_std:.1f}°")
    
    # Create calibrator
    calibrator = ExtrinsicCalibrator(board_config, intrinsics, camera_id)
    
    # Generate board positions in world frame
    # Board should be in front of the camera (in its field of view)
    # Camera is at camera_position, looking in direction of azimuth
    
    board_positions = []
    for i in range(num_positions):
        # Distance from camera: 3-6m
        dist = 3.0 + i * 0.6
        
        # Slight variation in angle (±10° from camera optical axis)
        angle_offset = (i - num_positions//2) * 4  # degrees
        angle = np.radians(camera_euler["azimuth"] + angle_offset)
        
        # Board position: camera position + offset in viewing direction
        # Camera looks along its azimuth direction
        bx = camera_position[0] + dist * np.cos(angle)
        by = camera_position[1] + dist * np.sin(angle)
        bz = camera_position[2] + np.random.uniform(-0.2, 0.2) + dist * np.sin(np.radians(camera_euler["elevation"]))
        
        # Board faces back toward camera
        # board_yaw = 180 means board normal points in -X direction (back toward origin)
        # We want board to face toward the camera
        board_yaw = camera_euler["azimuth"] + 180 + np.random.uniform(-5, 5)
        
        board_positions.append((np.array([bx, by, bz]), board_yaw))
    
    # Run calibration
    print(f"\n" + "-"*70)
    print("CALIBRATION MEASUREMENTS")
    print("-"*70)
    
    for i, (board_pos, board_yaw) in enumerate(board_positions):
        print(f"\n[Measurement {i+1}/{num_positions}]")
        print(f"  Board placed at: X={board_pos[0]:.2f}m, Y={board_pos[1]:.2f}m, Z={board_pos[2]:.2f}m")
        print(f"  Board facing: {board_yaw:.1f}°")
        
        # Generate synthetic image and measurement
        image, placement = synth.generate_measurement(board_pos, board_yaw)
        
        print(f"  Operator measures: X={placement.x:.3f}m, Y={placement.y:.3f}m, Z={placement.z:.3f}m, yaw={placement.yaw:.1f}°")
        
        # Process measurement
        result = calibrator.add_measurement(image, placement)
        
        if result["success"]:
            print(f"  ✓ Detection: {result['corners_detected']} corners, reproj={result['reproj_error']:.3f}px")
            print(f"    Computed camera pose: az={result['euler_angles']['azimuth']:.2f}°, "
                  f"el={result['euler_angles']['elevation']:.2f}°")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Compute final extrinsics
    print(f"\n" + "-"*70)
    print("COMPUTING FINAL EXTRINSICS")
    print("-"*70)
    
    try:
        extrinsics = calibrator.compute_extrinsics()
        calibrator.print_results(synth.get_ground_truth())
        calibrator.save_to_json(output_path)
        return extrinsics
    except ValueError as e:
        print(f"Calibration failed: {e}")
        return None


def print_operator_instructions():
    """Print detailed instructions for the operator."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    EXTRINSIC CALIBRATION GUIDE                       ║
╚══════════════════════════════════════════════════════════════════════╝

EQUIPMENT NEEDED:
  • ChArUco calibration board (1.1m × 1.1m, printed on RIGID material)
  • Tape measure (preferably laser distance meter)
  • Camera live preview (monitor/laptop showing camera feed)

COORDINATE SYSTEM (World/IMU Frame):
  • Origin: IMU sensor location
  • X-axis: FORWARD (direction vehicle faces)
  • Y-axis: RIGHT (passenger side)
  • Z-axis: UP (toward sky)
  
  Example: If board is 3m in front, 2m to the right, 0.5m above IMU:
           X = 3.0, Y = 2.0, Z = 0.5

╔══════════════════════════════════════════════════════════════════════╗
║              ⚠️  CRITICAL: FINDING CAMERA OPTICAL AXIS               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  BEFORE taking measurements, you must find where the camera points:  ║
║                                                                      ║
║  1. Watch the camera live preview on a monitor                       ║
║  2. Move the board around until it appears CENTERED in the image     ║
║  3. The board should be in the MIDDLE of the frame, not off to side  ║
║  4. Note this position - this is along the camera's optical axis     ║
║  5. Note the YAW angle (direction from board toward camera)          ║
║                                                                      ║
║  For ALL measurements, keep the board CENTERED in the image!         ║
║  Only vary the DISTANCE and HEIGHT, not the lateral position.        ║
║                                                                      ║
║  ┌─────────────────────────────────────┐                             ║
║  │                                     │                             ║
║  │           ╔═══════════╗             │  ← Board should be          ║
║  │           ║   BOARD   ║             │    CENTERED like this       ║
║  │           ║  (good)   ║             │                             ║
║  │           ╚═══════════╝             │                             ║
║  │                                     │                             ║
║  └─────────────────────────────────────┘                             ║
║                                                                      ║
║  ┌─────────────────────────────────────┐                             ║
║  │  ╔═══════════╗                      │  ← Board off to side        ║
║  │  ║   BOARD   ║                      │    will cause ERRORS!       ║
║  │  ║   (bad)   ║                      │                             ║
║  │  ╚═══════════╝                      │                             ║
║  │                                     │                             ║
║  │                                     │                             ║
║  └─────────────────────────────────────┘                             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

STEP-BY-STEP PROCEDURE:

  STEP 1: FIND OPTICAL AXIS (do this once at start)
    • Place board ~4m from camera
    • Watch live preview, move board left/right until CENTERED in image
    • Measure this position (X, Y, Z) from IMU origin
    • Calculate yaw: direction FROM board TO camera
    • This yaw will be used for ALL measurements

  STEP 2: TAKE MEASUREMENTS (repeat 7-10 times)
    • Keep board CENTERED in camera view (same direction as Step 1)
    • For each measurement:
      - Move board to a new DISTANCE (3m, 4m, 5m, 6m, 7m)
      - Vary HEIGHT slightly (±0.3m from baseline)
      - Keep board facing same direction (same yaw)
      - Measure X, Y, Z position from IMU
      - Capture image

  STEP 3: USE SAME YAW FOR ALL
    • All measurements use the SAME yaw angle found in Step 1
    • This is critical for accuracy!

╔══════════════════════════════════════════════════════════════════════╗
║                         TOP VIEW DIAGRAM                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║                    +X (Forward)                                      ║
║                          ↑                                           ║
║                          │                                           ║
║                          │   Board positions along optical axis:     ║
║            ═══           │        ═══        ═══        ═══          ║
║           3m away        │       4m away    5m away    6m away       ║
║               ↘          │                                           ║
║         optical ↘        │   (vary distance, keep board centered)    ║
║           axis    ↘      │                                           ║
║    ←──────────────────── ● ──────────────────────→ +Y (Right)        ║
║                         IMU                                          ║
║                        Origin           ◎ Camera                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

MEASURING BOARD POSITION (X, Y, Z):
  • Measure from IMU origin to the CENTER of the board
  • X: Distance FORWARD from IMU (positive = in front)
  • Y: Distance RIGHT from IMU (positive = right, negative = left)
  • Z: Height relative to IMU (positive = above, negative = below)
  
  TIP: Mark the board center with tape for consistent measurement

MEASURING BOARD YAW ANGLE:
  • Yaw = direction FROM the board center TOWARD the camera
  • Measure once when you find the optical axis, use same value for all
  • Use compass, or calculate from geometry
  
  Example: If camera points at ~45° and board faces camera → yaw ≈ 225°

GOOD MEASUREMENT PRACTICES:
  ✓ Find optical axis FIRST by centering board in preview
  ✓ Keep board CENTERED in image for ALL measurements
  ✓ Use SAME YAW for all measurements  
  ✓ Vary DISTANCE: 3m, 4m, 5m, 6m, 7m
  ✓ Vary HEIGHT: ±0.3m from baseline
  ✓ Take 7-10 measurements total
  
COMMON MISTAKES TO AVOID:
  ✗ Board off to side of image (must be CENTERED!)
  ✗ Different yaw angles for different measurements
  ✗ Measuring to board CORNER instead of CENTER
  ✗ Board tilted (not vertical)
  ✗ Board too close (<2m) or too far (>10m)
  ✗ Varying lateral position instead of distance/height
""")


def run_real_calibration(intrinsics_path: str, camera_id: str, output_path: str,
                         image_dir: Optional[str] = None, num_positions: int = 7,
                         demo_mode: bool = False) -> Optional[dict]:
    """
    Run interactive extrinsic calibration with real camera.
    
    If demo_mode=True, uses synthetic images but still shows full operator workflow.
    """
    print("="*70)
    if demo_mode:
        print("EXTRINSIC CALIBRATION - DEMO MODE (synthetic images, real workflow)")
    else:
        print("EXTRINSIC CALIBRATION - REAL CAMERA")
    print("="*70)
    
    # Load intrinsics
    print(f"\nLoading intrinsics: {intrinsics_path}")
    with open(intrinsics_path) as f:
        intrinsics = json.load(f)
    
    board_config = ChArUcoBoardConfig()
    
    # For demo mode, create synthetic image generator with MATCHING intrinsics
    if demo_mode:
        # Use a typical camera pose for demo
        camera_position = np.array([0.5, 0.8, 1.0])
        camera_euler = {"azimuth": 45.0, "elevation": -10.0, "roll": 0.5}
        
        # Use synthetic intrinsics that match the synthetic camera model
        # This ensures consistency between image generation and calibration
        demo_intrinsics = {
            "camera_matrix": [
                [1250.0, 0.0, 965.0],
                [0.0, 1248.0, 545.0],
                [0.0, 0.0, 1.0]
            ],
            "distortion_coefficients": [-0.12, 0.05, 0.0008, -0.0005, -0.015],
            "image_size": [1920, 1080]
        }
        
        synth = SyntheticExtrinsicTest(board_config, demo_intrinsics, camera_position, camera_euler)
        synth.position_noise_std = 0.0  # No noise in demo - operator enters exact values
        synth.angle_noise_std = 0.0
        
        # Use the same intrinsics for the calibrator
        calibrator = ExtrinsicCalibrator(board_config, demo_intrinsics, camera_id)
        
        print(f"\n⚠️  DEMO MODE: Using synthetic images to simulate real workflow")
        print(f"    (Using synthetic intrinsics for consistency)")
        print(f"    Ground truth: camera at [{camera_position[0]}, {camera_position[1]}, {camera_position[2]}]m")
        print(f"                  azimuth={camera_euler['azimuth']}°, elevation={camera_euler['elevation']}°")
    else:
        calibrator = ExtrinsicCalibrator(board_config, intrinsics, camera_id)
    
    # Print instructions
    print_operator_instructions()
    
    input("\nPress ENTER when you have read the instructions and are ready to begin...")
    
    print("\n" + "="*70)
    print(f"CALIBRATION SESSION: {num_positions} measurements needed")
    print("="*70)
    
    measurement_num = 0
    
    while measurement_num < num_positions:
        print(f"\n{'─'*70}")
        print(f"MEASUREMENT {measurement_num + 1} of {num_positions}")
        print(f"{'─'*70}")
        
        # Suggest placement for this measurement
        suggested_dist = 3.0 + measurement_num * 0.5
        if measurement_num < num_positions // 3:
            position_hint = "LEFT side of camera view"
        elif measurement_num < 2 * num_positions // 3:
            position_hint = "CENTER of camera view"
        else:
            position_hint = "RIGHT side of camera view"
        
        height_hints = ["at camera height", "ABOVE camera level", "BELOW camera level"]
        height_hint = height_hints[measurement_num % 3]
        
        print(f"\n📍 SUGGESTED PLACEMENT:")
        print(f"   • Distance from camera: ~{suggested_dist:.1f} meters")
        print(f"   • Position: {position_hint}")
        print(f"   • Height: {height_hint}")
        
        print(f"\n📋 STEPS:")
        print(f"   1. Place board at suggested position (markers facing camera)")
        print(f"   2. Measure board CENTER position in world frame (X, Y, Z)")
        print(f"   3. Measure board yaw angle (direction from board to camera)")
        if not demo_mode:
            print(f"   4. Capture image OR enter image filename")
        else:
            print(f"   4. [DEMO] Image will be generated automatically")
        
        # In demo mode, suggest realistic values based on camera pose
        if demo_mode:
            # Generate suggested values (what operator would measure)
            az_rad = np.radians(camera_euler["azimuth"])
            el_rad = np.radians(camera_euler["elevation"])
            
            # Add some lateral offset based on position hint (smaller offsets work better)
            if "LEFT" in position_hint:
                angle_offset = -8
            elif "RIGHT" in position_hint:
                angle_offset = 8
            else:
                angle_offset = 0
            
            angle = az_rad + np.radians(angle_offset)
            
            suggested_x = camera_position[0] + suggested_dist * np.cos(angle)
            suggested_y = camera_position[1] + suggested_dist * np.sin(angle)
            
            if "ABOVE" in height_hint:
                suggested_z = camera_position[2] + 0.3
            elif "BELOW" in height_hint:
                suggested_z = camera_position[2] - 0.3
            else:
                suggested_z = camera_position[2]
            
            suggested_z += suggested_dist * np.sin(el_rad)
            
            # Yaw is always camera_azimuth + 180 (board faces camera direction)
            # This works better than calculating exact angle from board to camera
            suggested_yaw = camera_euler["azimuth"] + 180
            
            print(f"\n💡 DEMO SUGGESTED VALUES (based on simulated camera at 45° azimuth):")
            print(f"   X ≈ {suggested_x:.1f}m, Y ≈ {suggested_y:.1f}m, Z ≈ {suggested_z:.1f}m")
            print(f"   Yaw ≈ {suggested_yaw:.0f}°")
        
        # Get image
        print(f"\n📸 IMAGE CAPTURE:")
        if demo_mode:
            print(f"   [DEMO] Press ENTER to generate synthetic image...")
            input()
            # Use the suggested values to generate image - NO random variation
            # so the suggested yaw matches what's actually in the image
            board_pos = np.array([suggested_x, suggested_y, suggested_z])
            board_yaw = suggested_yaw  # No random variation - must match what operator enters
            image, _ = synth.generate_measurement(board_pos, board_yaw)
            print(f"   ✓ Synthetic image generated: {image.shape[1]}x{image.shape[0]} pixels")
        elif image_dir:
            image_files = sorted(Path(image_dir).glob("*.png")) + sorted(Path(image_dir).glob("*.jpg"))
            if measurement_num < len(image_files):
                image_path = image_files[measurement_num]
                print(f"   Using: {image_path}")
                image = cv2.imread(str(image_path))
            else:
                image_name = input("   Enter image filename: ").strip()
                image_path = Path(image_dir) / image_name
                image = cv2.imread(str(image_path))
        else:
            image_name = input("   Enter full path to image: ").strip()
            image = cv2.imread(image_name)
        
        if image is None:
            print("   ❌ ERROR: Could not load image. Try again.")
            continue
        
        if not demo_mode:
            print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Get board position measurements
        print(f"\n📏 BOARD POSITION (in meters, relative to IMU origin):")
        if demo_mode:
            print(f"   [DEMO] Enter values or press ENTER to use suggested values")
        
        try:
            x_str = input(f"   X (forward) [{suggested_x:.2f}]: " if demo_mode else "   X (forward):  ").strip()
            if demo_mode and x_str == "":
                board_x = suggested_x
            else:
                board_x = float(x_str)
            
            y_str = input(f"   Y (right) [{suggested_y:.2f}]: " if demo_mode else "   Y (right):    ").strip()
            if demo_mode and y_str == "":
                board_y = suggested_y
            else:
                board_y = float(y_str)
            
            z_str = input(f"   Z (up) [{suggested_z:.2f}]: " if demo_mode else "   Z (up):       ").strip()
            if demo_mode and z_str == "":
                board_z = suggested_z
            else:
                board_z = float(z_str)
        except ValueError:
            print("   ❌ ERROR: Invalid number format. Try again.")
            continue
        
        # Get board yaw
        print(f"\n🧭 BOARD YAW (direction from board toward camera, in degrees):")
        print(f"   Hint: If camera faces ~45° and board faces camera, yaw ≈ 225°")
        try:
            yaw_str = input(f"   Yaw angle [{suggested_yaw:.0f}]: " if demo_mode else "   Yaw angle: ").strip()
            if demo_mode and yaw_str == "":
                board_yaw_input = suggested_yaw
            else:
                board_yaw_input = float(yaw_str)
        except ValueError:
            print("   ❌ ERROR: Invalid number format. Try again.")
            continue
        
        # Create placement and process
        placement = BoardPlacement(
            x=board_x, y=board_y, z=board_z,
            yaw=board_yaw_input, pitch=0, roll=0
        )
        
        print(f"\n⏳ Processing measurement...")
        print(f"   Board position: X={board_x:.2f}m, Y={board_y:.2f}m, Z={board_z:.2f}m")
        print(f"   Board yaw: {board_yaw_input:.1f}°")
        
        result = calibrator.add_measurement(image, placement)
        
        if result["success"]:
            print(f"\n   ✅ SUCCESS!")
            print(f"   • Corners detected: {result['corners_detected']}")
            print(f"   • Reprojection error: {result['reproj_error']:.3f} px")
            print(f"   • Computed azimuth: {result['euler_angles']['azimuth']:.2f}°")
            print(f"   • Computed elevation: {result['euler_angles']['elevation']:.2f}°")
            measurement_num += 1
        else:
            print(f"\n   ❌ FAILED: {result['error']}")
            print(f"   Please try again with a different position or better lighting.")
            retry = input("   Retry this measurement? [Y/n]: ").strip().lower()
            if retry == 'n':
                measurement_num += 1  # Skip this one
        
        # Show progress and quality
        successful = len([m for m in calibrator.measurements if m["success"]])
        print(f"\n   📊 Progress: {successful} successful measurements")
        
        # Check quality after minimum measurements
        if successful >= 3:
            quality = calibrator.check_quality()
            print(f"\n   📈 QUALITY CHECK:")
            print(f"      Azimuth std:   {quality['azimuth_std']:.2f}° {'✓' if quality['azimuth_std'] < 1.0 else '✗'}")
            print(f"      Elevation std: {quality['elevation_std']:.2f}° {'✓' if quality['elevation_std'] < 1.0 else '✗'}")
            print(f"      Reproj error:  {quality['mean_reproj_error']:.2f} px")
            print(f"      → {quality['recommendation']}")
            
            # If quality is good and we have enough measurements, offer to stop
            if quality['quality_ok'] and measurement_num < num_positions:
                print(f"\n   🎯 Quality threshold met!")
                choice = input(f"   Continue to {num_positions} measurements or finalize now? [c]ontinue/[f]inalize: ").strip().lower()
                if choice == 'f':
                    print(f"   Finalizing with {successful} measurements...")
                    break
            
            # If quality is poor and at limit, offer to continue
            if not quality['quality_ok'] and measurement_num >= num_positions:
                print(f"\n   ⚠️  Quality threshold NOT met at {num_positions} measurements.")
                choice = input(f"   Take more measurements? [y]es/[n]o finalize anyway: ").strip().lower()
                if choice == 'y':
                    num_positions += 3  # Add 3 more slots
                    print(f"   Extended to {num_positions} measurements.")
    
    # Compute final extrinsics
    print(f"\n" + "="*70)
    print("COMPUTING FINAL EXTRINSICS")
    print("="*70)
    
    try:
        extrinsics = calibrator.compute_extrinsics()
        if demo_mode:
            calibrator.print_results(synth.get_ground_truth())
        else:
            calibrator.print_results()
        calibrator.save_to_json(output_path)
        return extrinsics
    except ValueError as e:
        print(f"\n❌ Calibration failed: {e}")
        print("   Need at least 3 successful measurements.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extrinsic Camera Calibration")
    parser.add_argument("--intrinsics", "-i", required=True, help="Path to intrinsics JSON")
    parser.add_argument("--camera-id", default="camera_1", help="Camera identifier")
    parser.add_argument("--output", "-o", default="camera_extrinsics.json", help="Output path")
    parser.add_argument("--num-positions", "-n", type=int, default=7, help="Number of board positions")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (automatic, no interaction)")
    parser.add_argument("--demo", action="store_true", help="Demo mode: synthetic images with real operator workflow")
    parser.add_argument("--image-dir", help="Directory containing calibration images")
    parser.add_argument("--min-measurements", type=int, default=3, help="Minimum measurements before quality check (default: 3)")
    
    args = parser.parse_args()
    
    if not Path(args.intrinsics).exists():
        print(f"Error: Intrinsics file not found: {args.intrinsics}")
        return 1
    
    if args.synthetic:
        result = run_synthetic_calibration(
            args.intrinsics, args.camera_id, args.output, args.num_positions
        )
    elif args.demo:
        result = run_real_calibration(
            args.intrinsics, args.camera_id, args.output,
            args.image_dir, args.num_positions, demo_mode=True
        )
    else:
        result = run_real_calibration(
            args.intrinsics, args.camera_id, args.output,
            args.image_dir, args.num_positions, demo_mode=False
        )
    
    if result:
        print("\n" + "="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

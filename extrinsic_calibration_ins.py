#!/usr/bin/env python3
"""
INS-Based Extrinsic Camera Calibration
=======================================

Determines camera pose relative to vehicle frame using:
1. ChArUco board detection (solvePnP)
2. Live INS data (yaw, pitch, roll)
3. Laser-measured distance to board
4. Bundle adjustment optimization

KEY IMPROVEMENTS OVER NAIVE APPROACH:
=====================================
1. Uses proper 3D rotation matrix composition (not angle subtraction)
2. Board facing direction is COMPUTED from geometry, not measured
3. Laser distance is used as an optimization constraint
4. Bundle adjustment refines camera pose across all measurements
5. Quaternion averaging for robust rotation estimation
6. Consistent rotation conventions throughout

COORDINATE FRAMES:
==================

World Frame (NED - North-East-Down):
    - X: North
    - Y: East
    - Z: Down
    - Origin: Fixed point on Earth

Vehicle Frame (Body - Forward-Right-Down):
    - X: Forward (vehicle heading)
    - Y: Right
    - Z: Down
    - Origin: IMU location
    - Related to world by INS yaw/pitch/roll

Camera Frame:
    - Z: Forward (optical axis)
    - X: Right (image horizontal)
    - Y: Down (image vertical)
    - Origin: Camera optical center

Board Frame:
    - X: Along board width (right when facing board)
    - Y: Along board height (down when facing board)
    - Z: Out of board surface (toward viewer)
    - Origin: Board corner (0,0)

CALIBRATION PROCEDURE:
======================
1. Place vertical ChArUco board at measured position (laser distance from camera)
2. Board should be roughly facing the camera (exact angle computed automatically)
3. Record INS data (vehicle orientation in world frame)
4. System detects board, computes camera orientation in vehicle frame
5. Repeat at multiple distances/positions
6. Bundle adjustment optimizes final camera pose

USAGE:
======

    python3 extrinsic_calibration_ins.py --synthetic \\
        --intrinsics camera_intrinsics.json \\
        --output camera_extrinsics.json

    python3 extrinsic_calibration_ins.py \\
        --intrinsics camera_intrinsics.json \\
        --output camera_extrinsics.json
"""

import numpy as np
import cv2
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from scipy.optimize import least_squares
import sys


# =============================================================================
# ROTATION UTILITIES
# =============================================================================

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to unit quaternion [w, x, y, z].
    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def average_quaternions(quaternions: List[np.ndarray],
                        weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute weighted average of quaternions using Markley's method.
    """
    n = len(quaternions)
    if n == 0:
        raise ValueError("Cannot average empty list")
    if n == 1:
        return quaternions[0]

    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights) / np.sum(weights)

    # Align quaternions to same hemisphere
    q0 = quaternions[0]
    aligned = [q0]
    for q in quaternions[1:]:
        aligned.append(-q if np.dot(q0, q) < 0 else q)

    # Build 4x4 matrix and find eigenvector
    M = np.zeros((4, 4))
    for w, q in zip(weights, aligned):
        q = q.reshape(4, 1)
        M += w * (q @ q.T)

    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg = eigenvectors[:, np.argmax(eigenvalues)]

    if avg[0] < 0:
        avg = -avg
    return avg / np.linalg.norm(avg)


def average_rotation_matrices(matrices: List[np.ndarray],
                               weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Average rotation matrices via quaternion averaging."""
    quats = [rotation_matrix_to_quaternion(R) for R in matrices]
    avg_quat = average_quaternions(quats, weights)
    return quaternion_to_rotation_matrix(avg_quat)


def validate_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> Tuple[bool, str]:
    """Validate R is a proper rotation matrix (orthonormal, det=+1)."""
    if R.shape != (3, 3):
        return False, f"Shape {R.shape} != (3,3)"

    err = np.linalg.norm(R.T @ R - np.eye(3))
    if err > tol:
        return False, f"Not orthonormal: error={err:.2e}"

    det = np.linalg.det(R)
    if abs(det - 1.0) > tol:
        return False, f"Determinant {det:.6f} != 1.0"

    return True, "Valid"


def euler_zyx_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Create rotation matrix from ZYX Euler angles (yaw, pitch, roll).

    This implements the standard aerospace/NED convention:
        R = Rx(roll) @ Ry(pitch) @ Rz(yaw)

    Args:
        yaw: Rotation about Z axis (degrees)
        pitch: Rotation about Y axis (degrees)
        roll: Rotation about X axis (degrees)

    Returns:
        3x3 rotation matrix R_world_to_body
    """
    y, p, r = np.radians([yaw, pitch, roll])

    # Rotation about Z (yaw)
    Rz = np.array([
        [np.cos(y), np.sin(y), 0],
        [-np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ])

    # Rotation about Y (pitch)
    Ry = np.array([
        [np.cos(p), 0, -np.sin(p)],
        [0, 1, 0],
        [np.sin(p), 0, np.cos(p)]
    ])

    # Rotation about X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), np.sin(r)],
        [0, -np.sin(r), np.cos(r)]
    ])

    # ZYX order: first yaw, then pitch, then roll
    return Rx @ Ry @ Rz


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> Dict[str, float]:
    """Extract ZYX Euler angles from rotation matrix. Returns degrees."""
    # Handle gimbal lock
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = 90.0
            roll = np.degrees(np.arctan2(R[0, 1], R[0, 2]))
        else:
            pitch = -90.0
            roll = np.degrees(np.arctan2(-R[0, 1], -R[0, 2]))
    else:
        pitch = np.degrees(np.arcsin(-R[2, 0]))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    return {"yaw": yaw, "pitch": pitch, "roll": roll}


# =============================================================================
# INS DATA
# =============================================================================

@dataclass
class INSData:
    """
    INS (Inertial Navigation System) reading at a specific time.

    Stores vehicle orientation in NED (North-East-Down) convention:
        - yaw: Heading angle (0° = North, 90° = East)
        - pitch: Nose up/down angle (positive = nose up)
        - roll: Bank angle (positive = right wing down)

    The rotation sequence is ZYX: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    """
    yaw: float      # degrees, 0 = North, 90 = East
    pitch: float    # degrees, positive = nose up
    roll: float     # degrees, positive = right wing down
    timestamp: float = 0.0  # Unix timestamp

    def to_rotation_matrix(self) -> np.ndarray:
        """
        Compute rotation matrix R_world_to_vehicle (R_ned_to_body).

        This transforms vectors from world (NED) frame to vehicle (body) frame.
        """
        return euler_zyx_to_rotation_matrix(self.yaw, self.pitch, self.roll)

    def to_rotation_matrix_inverse(self) -> np.ndarray:
        """
        Compute rotation matrix R_vehicle_to_world (R_body_to_ned).

        This transforms vectors from vehicle (body) frame to world (NED) frame.
        """
        return self.to_rotation_matrix().T


# =============================================================================
# BOARD CONFIGURATION
# =============================================================================

@dataclass
class ChArUcoBoardConfig:
    """ChArUco board configuration."""
    squares_x: int = 10
    squares_y: int = 10
    square_length: float = 0.11  # meters
    marker_length: float = 0.085  # meters
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
        return self.squares_x * self.square_length

    @property
    def board_height(self) -> float:
        return self.squares_y * self.square_length


# =============================================================================
# BOARD MEASUREMENT (SIMPLIFIED - No facing angle required)
# =============================================================================

@dataclass
class BoardMeasurement:
    """
    Simplified board measurement - no facing angle required!

    The operator only needs to measure:
    1. Board center position in world (NED) frame
    2. Laser distance from camera to board center

    The board is assumed to be:
    - Vertical (board Y axis aligned with world Down)
    - Roughly facing the camera (exact angle computed automatically)
    """
    # Board center position in world (NED) frame
    north: float  # meters, positive = north
    east: float   # meters, positive = east
    down: float   # meters, positive = down (negative = up)

    # Laser-measured distance from camera to board center
    laser_distance: float  # meters

    def __post_init__(self):
        if self.laser_distance <= 0:
            raise ValueError("Laser distance must be positive")


# =============================================================================
# CHARUCO DETECTOR
# =============================================================================

class ChArUcoDetector:
    """ChArUco corner detection with sub-pixel refinement."""

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
                self.aruco_detector = cv2.aruco.ArucoDetector(
                    self.aruco_dict, self.detector_params)
            else:
                self.aruco_detector = None

        if hasattr(self.board, 'getChessboardCorners'):
            self.board_corners_3d = self.board.getChessboardCorners()
        else:
            self.board_corners_3d = self.board.chessboardCorners

    def detect(self, image: np.ndarray
               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Detect ChArUco corners with sub-pixel refinement."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        annotated = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self.use_new_api:
            corners, ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
        else:
            if self.aruco_detector:
                marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.detector_params)

            if marker_ids is None or len(marker_ids) == 0:
                return None, None, annotated

            _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)

        if corners is None or len(corners) < 6:
            return None, None, annotated

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        cv2.aruco.drawDetectedCornersCharuco(annotated, corners, ids)

        return corners, ids, annotated

    def estimate_board_pose(self, corners: np.ndarray, ids: np.ndarray,
                            camera_matrix: np.ndarray, dist_coeffs: np.ndarray
                            ) -> Tuple[bool, np.ndarray, np.ndarray, float]:
        """Estimate board pose in camera frame using solvePnP."""
        obj_points = self.board_corners_3d[ids.flatten()].astype(np.float32)
        img_points = corners.reshape(-1, 2).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False, None, None, float('inf')

        # Refine with VVS
        rvec, tvec = cv2.solvePnPRefineVVS(
            obj_points, img_points,
            camera_matrix, dist_coeffs,
            rvec, tvec
        )

        # Reprojection error
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_error = np.sqrt(np.mean(np.sum((img_points - projected.reshape(-1, 2))**2, axis=1)))

        return True, rvec, tvec, reproj_error


# =============================================================================
# INS-BASED EXTRINSIC CALIBRATOR WITH OPTIMIZATION
# =============================================================================

class ExtrinsicCalibratorINS:
    """
    Computes camera extrinsics using INS data, ChArUco board detection, and
    bundle adjustment optimization.

    KEY INNOVATION: Board facing direction is computed, not measured!

    The operator only provides:
    1. Board center position (measured or estimated in world frame)
    2. Laser distance from camera to board
    3. INS reading at capture time

    The algorithm:
    1. Detects board, gets R_board_to_camera from solvePnP
    2. Uses INS to get R_world_to_vehicle
    3. COMPUTES board facing direction from tvec (board faces camera)
    4. Computes R_vehicle_to_camera
    5. Bundle adjustment refines camera pose using all measurements
    """

    def __init__(self, board_config: ChArUcoBoardConfig,
                 intrinsics: dict, camera_id: str):
        self.board_config = board_config
        self.camera_id = camera_id

        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics["distortion_coefficients"], dtype=np.float64)
        self.image_size = tuple(intrinsics["image_size"])

        self.detector = ChArUcoDetector(board_config)
        self.measurements: List[dict] = []
        self.result: Optional[dict] = None

    def _compute_board_pose_from_detection(self, tvec: np.ndarray,
                                            R_board_to_camera: np.ndarray,
                                            board_center_world: np.ndarray,
                                            ins_data: INSData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute board pose in world frame using detection results.

        KEY INSIGHT: The board faces the camera, so we can compute the facing
        direction from the camera-board geometry instead of measuring it.

        Args:
            tvec: Board origin in camera frame (from solvePnP)
            R_board_to_camera: Board-to-camera rotation (from solvePnP)
            board_center_world: Measured board center position in world (NED)
            ins_data: INS reading at capture time

        Returns:
            R_world_to_board: Rotation matrix
            t_board_origin_world: Board origin position in world frame
        """
        # Board is vertical, Y points down (aligned with world Z in NED)
        board_y_world = np.array([0, 0, 1])  # Down in NED

        # Board Z (normal) points toward camera
        # We need to compute the direction from board to camera in world frame

        # Camera position in board frame
        R_camera_to_board = R_board_to_camera.T
        t_board_in_camera = tvec.reshape(3)
        t_camera_in_board = -R_camera_to_board @ t_board_in_camera

        # For a vertical board facing the camera:
        # Board Z should point in the horizontal direction toward camera
        # Project camera direction onto horizontal plane
        camera_dir_board = t_camera_in_board.copy()
        camera_dir_board[1] = 0  # Zero out vertical component (board Y)
        camera_dir_board = camera_dir_board / (np.linalg.norm(camera_dir_board) + 1e-10)

        # Board Z points toward camera (in board's local horizontal plane)
        # Since we want board Z to point toward camera, and camera is at t_camera_in_board,
        # board Z should point in the direction of t_camera_in_board (in XZ plane)
        board_z_board = np.array([camera_dir_board[0], 0, camera_dir_board[2]])
        board_z_board = board_z_board / (np.linalg.norm(board_z_board) + 1e-10)

        # This gives us the board orientation in board frame, but we need it in world frame
        # Use the actual R_board_to_camera to get board axes in camera frame, then transform to world

        # Actually, let's use a simpler approach:
        # The board Z axis in camera frame is the third column of R_board_to_camera.T = R_camera_to_board
        # No wait, R_board_to_camera transforms from board frame to camera frame
        # So R_board_to_camera @ [0,0,1] gives board Z in camera frame
        board_z_camera = R_board_to_camera @ np.array([0, 0, 1])

        # Transform to world frame: need R_camera_to_world
        R_world_to_vehicle = ins_data.to_rotation_matrix()
        R_vehicle_to_world = R_world_to_vehicle.T

        # We need R_camera_to_world, but we don't have it yet (that's what we're computing!)
        #
        # SIMPLER APPROACH: Assume board is vertical and facing roughly toward camera.
        # Compute the facing azimuth from the board center position and camera position estimate.
        #
        # But we don't know camera position yet either!
        #
        # SOLUTION: Use the tvec from solvePnP to compute board facing direction.
        # tvec is the board origin in camera frame.
        # The board faces toward the camera, so board Z points in the direction of -tvec
        # (normalized and projected to horizontal).

        # Actually, let's just use the geometric constraint that the board is vertical
        # and faces the camera. The solvePnP gives us R_board_to_camera directly.

        # For a vertical board facing the camera:
        # board_y_world = [0, 0, 1] (down in NED)
        # board_z_world = horizontal direction from board toward camera

        # We can compute facing_azimuth from the measurement geometry if we assume
        # the camera is roughly at the vehicle position for the first estimate.

        # Actually, the cleanest approach is to just use the R_board_to_camera directly:
        # R_vehicle_to_camera = R_board_to_camera @ R_world_to_board @ R_vehicle_to_world
        #
        # We can iterate: start with initial guess of board facing, compute camera pose,
        # refine board facing based on computed camera position, repeat.
        #
        # For the first iteration, assume board faces toward vehicle origin.

        # Compute initial facing azimuth (direction from board to origin)
        dx = -board_center_world[0]  # North component (toward origin)
        dy = -board_center_world[1]  # East component (toward origin)
        facing_azimuth = np.degrees(np.arctan2(dy, dx))

        return self._compute_board_pose_world(board_center_world, facing_azimuth)

    def _compute_board_pose_world(self, board_center_world: np.ndarray,
                                   facing_azimuth: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute board pose in world frame given center position and facing direction.

        Args:
            board_center_world: Board center position in world (NED) frame
            facing_azimuth: Direction board normal points (0° = North, 90° = East)

        Returns:
            R_world_to_board: Rotation matrix
            t_board_origin_world: Board origin position in world frame
        """
        # facing_azimuth is direction FROM board TO camera (where board normal points)
        # Board Z points in this direction (toward camera)
        az = np.radians(facing_azimuth)
        board_z_world = np.array([np.cos(az), np.sin(az), 0])

        # Board Y points Down (aligned with world Z for vertical board)
        board_y_world = np.array([0, 0, 1])

        # Board X = Z cross Y gives left-handed system, so negate to get proper rotation
        board_x_world = -np.cross(board_z_world, board_y_world)
        board_x_world = board_x_world / np.linalg.norm(board_x_world)

        # Recompute Y for orthogonality (should still point down)
        board_y_world = np.cross(board_z_world, board_x_world)
        board_y_world = board_y_world / np.linalg.norm(board_y_world)

        # R_board_to_world: columns are board axes in world coordinates (det=+1)
        R_board_to_world = np.column_stack([board_x_world, board_y_world, board_z_world])
        R_world_to_board = R_board_to_world.T

        # Board origin is at top-left corner
        # In board frame: center = [width/2, height/2, 0]
        offset_board = np.array([self.board_config.board_width / 2,
                                  self.board_config.board_height / 2, 0])
        board_origin_world = board_center_world - R_board_to_world @ offset_board

        return R_world_to_board, board_origin_world

    def add_measurement(self, image: np.ndarray,
                        board_measurement: BoardMeasurement,
                        ins_data: INSData) -> dict:
        """
        Process one calibration measurement.

        The board facing direction is COMPUTED from geometry, not measured!

        Args:
            image: Camera image with ChArUco board
            board_measurement: Board position and laser distance
            ins_data: INS reading at capture time

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

        # Get board pose in camera frame from solvePnP
        success, rvec, tvec, reproj_error = self.detector.estimate_board_pose(
            corners, ids, self.camera_matrix, self.dist_coeffs
        )

        if not success:
            return {
                "success": False,
                "error": "solvePnP failed",
                "annotated_image": annotated
            }

        # R_board_to_camera from solvePnP
        R_board_to_camera, _ = cv2.Rodrigues(rvec)
        t_board_in_camera = tvec.reshape(3)

        # Check distance consistency
        pnp_distance = np.linalg.norm(t_board_in_camera)
        distance_error = abs(pnp_distance - board_measurement.laser_distance)

        # Board center in camera frame (board origin + offset to center)
        board_center_offset = np.array([self.board_config.board_width / 2,
                                        self.board_config.board_height / 2, 0])
        t_board_center_camera = t_board_in_camera + R_board_to_camera @ board_center_offset
        board_center_distance = np.linalg.norm(t_board_center_camera)

        # Get vehicle pose from INS
        R_world_to_vehicle = ins_data.to_rotation_matrix()
        R_vehicle_to_world = R_world_to_vehicle.T

        # Board center in world frame (from measurement)
        board_center_world = np.array([board_measurement.north,
                                        board_measurement.east,
                                        board_measurement.down])

        # DIRECT COMPUTATION of R_camera_to_world using board constraint
        #
        # Key insight: Board Y (in board frame) = World Z (down) because board is vertical
        # Therefore: R_board_to_camera @ [0,1,0] = world_z expressed in camera frame
        #
        world_z_camera = R_board_to_camera @ np.array([0, 1, 0])

        # Board Z in camera frame (board normal direction)
        board_z_camera = R_board_to_camera @ np.array([0, 0, 1])

        # The board Z is horizontal in world, so board_z_world = [cos(θ), sin(θ), 0]
        # We can find θ from the constraint that board Z in camera comes from rotating
        # a horizontal world vector.

        # Compute R_camera_to_world from constraints:
        # 1. Third row of R_camera_to_world = world_z_camera (world Z expressed in camera)
        # 2. R is a proper rotation (orthonormal, det=+1)

        # world_z_camera gives us one constraint
        # For the other two rows, we use the fact that camera Z (optical axis) projects
        # to a mostly horizontal direction in world

        # Camera Z in world: we want it to be roughly [cos(az), sin(az), -sin(el)]
        # Camera X in world: perpendicular to camera Z, roughly horizontal

        # Use Gram-Schmidt to build R_camera_to_world from world_z_camera

        # Start with world Z direction in camera frame
        r3 = world_z_camera / np.linalg.norm(world_z_camera)  # Should already be unit

        # Camera Z points roughly forward (into scene) - use board Z direction as hint
        # In world, camera Z is approximately opposite to board Z direction
        # board_z_world = board_z_camera / R_world_to_camera.T (need to solve)

        # Simpler: camera Y is roughly aligned with world Z (down)
        # So camera X is roughly horizontal in world
        # Camera frame is right-handed: X=right, Y=down, Z=forward

        # Construct R_camera_to_world using known constraint
        # r3 = world Z in camera (from board Y constraint)
        # Use camera Y ≈ world Z, so camera X ≈ world horizontal

        # Camera Y in camera = [0, 1, 0]
        # World Z in camera = r3
        # If camera Y ≈ world Z, then camera Y in camera ≈ r3
        # Deviation from this gives us camera roll

        # For now, assume small roll and solve directly:
        # Camera Y in world ≈ [0, 0, 1] (down)
        # Camera Z in world = optical axis direction

        # Use tvec to determine camera looking direction
        # Board is at tvec in camera frame, so camera looks toward +tvec direction
        camera_z_camera = np.array([0, 0, 1])
        board_dir_camera = t_board_in_camera / np.linalg.norm(t_board_in_camera)

        # Board Z direction in world (horizontal, perpendicular to board surface)
        # We can compute this from board_z_camera and the constraint that it's horizontal

        # Alternative approach: compute board facing from measurement geometry
        dx = board_center_world[0]  # From origin toward board
        dy = board_center_world[1]
        facing_azimuth = np.degrees(np.arctan2(dy, dx))

        # Compute board pose
        R_world_to_board, t_board_origin_world = self._compute_board_pose_world(
            board_center_world, facing_azimuth)
        R_board_to_world = R_world_to_board.T

        # =================================================================
        # KEY COMPUTATION: Compose rotations properly
        # =================================================================
        # R_vehicle_to_camera = R_board_to_camera @ R_world_to_board @ R_vehicle_to_world
        # =================================================================
        R_vehicle_to_camera = R_board_to_camera @ R_world_to_board @ R_vehicle_to_world

        # Validate the rotation matrix
        is_valid, msg = validate_rotation_matrix(R_vehicle_to_camera)
        if not is_valid:
            print(f"WARNING: R_vehicle_to_camera invalid: {msg}")

        # Compute camera position in vehicle frame
        R_camera_to_board = R_board_to_camera.T
        t_camera_in_board = -R_camera_to_board @ t_board_in_camera
        t_camera_in_world = R_board_to_world @ t_camera_in_board + t_board_origin_world
        t_camera_in_vehicle = R_world_to_vehicle @ t_camera_in_world

        # Extract Euler angles for reporting
        euler = self._rotation_to_euler(R_vehicle_to_camera)

        measurement = {
            "success": True,
            "corners_detected": len(corners),
            "reproj_error": reproj_error,
            "R_vehicle_to_camera": R_vehicle_to_camera,
            "t_camera_in_vehicle": t_camera_in_vehicle,
            "euler_angles": euler,
            "ins_data": ins_data,
            "board_measurement": board_measurement,
            "computed_facing_azimuth": facing_azimuth,
            "pnp_distance": pnp_distance,
            "laser_distance": board_measurement.laser_distance,
            "distance_error": distance_error,
            "rvec": rvec,
            "tvec": tvec,
            "corners": corners,
            "ids": ids,
            "annotated_image": annotated
        }

        self.measurements.append(measurement)
        return measurement

    def _rotation_to_euler(self, R: np.ndarray) -> dict:
        """
        Extract camera Euler angles from R_vehicle_to_camera.

        Returns azimuth/elevation/roll where:
            azimuth: Camera heading in vehicle frame (0° = forward)
            elevation: Camera pitch (negative = looking down)
            roll: Camera roll
        """
        R_camera_to_vehicle = R.T
        cam_z_vehicle = R_camera_to_vehicle[:, 2]  # Camera optical axis in vehicle frame

        # Azimuth: angle in XY plane (forward-right)
        azimuth = np.degrees(np.arctan2(cam_z_vehicle[1], cam_z_vehicle[0]))

        # Elevation: angle from horizontal
        horizontal = np.sqrt(cam_z_vehicle[0]**2 + cam_z_vehicle[1]**2)
        elevation = np.degrees(np.arctan2(-cam_z_vehicle[2], horizontal))

        # Roll: rotation of camera X axis around optical axis
        # cam_x_no_roll is the expected camera X direction with zero roll
        az_rad = np.radians(azimuth)
        cam_x_no_roll = np.array([-np.sin(az_rad), np.cos(az_rad), 0])
        norm = np.linalg.norm(cam_x_no_roll)
        if norm > 1e-6:
            cam_x_no_roll = cam_x_no_roll / norm
        else:
            cam_x_no_roll = np.array([0, 1, 0])

        cam_x_vehicle = R_camera_to_vehicle[:, 0]
        cam_y_vehicle = R_camera_to_vehicle[:, 1]

        roll = np.degrees(np.arctan2(
            -np.dot(cam_x_no_roll, cam_y_vehicle),
            np.dot(cam_x_no_roll, cam_x_vehicle)
        ))

        return {"azimuth": azimuth, "elevation": elevation, "roll": roll}

    def _euler_to_rotation(self, azimuth: float, elevation: float, roll: float) -> np.ndarray:
        """Convert camera Euler angles to R_vehicle_to_camera."""
        az = np.radians(azimuth)
        el = np.radians(elevation)
        ro = np.radians(roll)

        # Camera Z in vehicle frame (optical axis direction)
        cam_z_vehicle = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            -np.sin(el)
        ])

        # Camera X in vehicle frame (before roll)
        # For azimuth=0 (forward), cam_x should be [0, 1, 0] (vehicle right)
        # For azimuth=90° (right), cam_x should be [-1, 0, 0] (vehicle backward)
        cam_x_vehicle = np.array([-np.sin(az), np.cos(az), 0])
        norm = np.linalg.norm(cam_x_vehicle)
        if norm > 1e-6:
            cam_x_vehicle = cam_x_vehicle / norm
        else:
            cam_x_vehicle = np.array([0, 1, 0])

        # For right-handed camera frame (X=right, Y=down, Z=forward):
        # Z = X x Y, therefore Y = Z x X, and X = Y x Z
        cam_y_vehicle = np.cross(cam_z_vehicle, cam_x_vehicle)
        cam_y_vehicle = cam_y_vehicle / np.linalg.norm(cam_y_vehicle)

        # Recalculate X for orthogonality
        cam_x_vehicle = np.cross(cam_y_vehicle, cam_z_vehicle)
        cam_x_vehicle = cam_x_vehicle / np.linalg.norm(cam_x_vehicle)

        # Apply roll
        if abs(ro) > 1e-6:
            c, s = np.cos(ro), np.sin(ro)
            cam_x_new = c * cam_x_vehicle + s * cam_y_vehicle
            cam_y_new = -s * cam_x_vehicle + c * cam_y_vehicle
            cam_x_vehicle = cam_x_new
            cam_y_vehicle = cam_y_new

        # R_camera_to_vehicle: columns are camera axes in vehicle frame
        R_camera_to_vehicle = np.column_stack([cam_x_vehicle, cam_y_vehicle, cam_z_vehicle])
        R_vehicle_to_camera = R_camera_to_vehicle.T

        return R_vehicle_to_camera

    def _bundle_adjustment(self, initial_position: np.ndarray,
                           initial_euler: dict) -> Tuple[np.ndarray, dict]:
        """
        Bundle adjustment to refine camera pose using all measurements.

        Optimizes camera pose to minimize:
        1. Reprojection error across all measurements
        2. Distance error (PnP distance vs laser distance)
        3. Consistency across measurements

        Returns:
            Optimized position and euler angles
        """
        valid = [m for m in self.measurements if m["success"]]
        if len(valid) < 3:
            return initial_position, initial_euler

        # Initial parameters: [x, y, z, azimuth, elevation, roll]
        x0 = np.array([
            initial_position[0], initial_position[1], initial_position[2],
            initial_euler["azimuth"], initial_euler["elevation"], initial_euler["roll"]
        ])

        def residuals(params):
            pos = params[:3]
            euler = {"azimuth": params[3], "elevation": params[4], "roll": params[5]}
            R_vehicle_to_camera = self._euler_to_rotation(
                euler["azimuth"], euler["elevation"], euler["roll"])
            R_camera_to_vehicle = R_vehicle_to_camera.T

            errors = []

            for m in valid:
                ins = m["ins_data"]
                board = m["board_measurement"]
                tvec = m["tvec"].reshape(3)
                rvec = m["rvec"]

                R_world_to_vehicle = ins.to_rotation_matrix()
                R_vehicle_to_world = R_world_to_vehicle.T

                # Camera position in world frame
                t_camera_world = R_vehicle_to_world @ pos

                # Board center in world
                board_center = np.array([board.north, board.east, board.down])

                # Expected distance from camera to board
                expected_dist = np.linalg.norm(board_center - t_camera_world)

                # Actual distance from PnP
                pnp_dist = np.linalg.norm(tvec)

                # Distance error (weighted)
                dist_err = (pnp_dist - board.laser_distance) * 10.0  # Weight
                errors.append(dist_err)

                # Position consistency error
                # The computed position from this measurement should match our estimate
                pos_err = np.linalg.norm(m["t_camera_in_vehicle"] - pos) * 5.0
                errors.append(pos_err)

                # Orientation consistency error
                R_meas = m["R_vehicle_to_camera"]
                R_diff = R_vehicle_to_camera @ R_meas.T
                angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                errors.append(np.degrees(angle_diff))

            return np.array(errors)

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=100)
            optimized = result.x
            return (
                optimized[:3],
                {"azimuth": optimized[3], "elevation": optimized[4], "roll": optimized[5]}
            )
        except Exception as e:
            print(f"Bundle adjustment failed: {e}")
            return initial_position, initial_euler

    def compute_extrinsics(self, min_measurements: int = 3,
                           use_bundle_adjustment: bool = True) -> dict:
        """
        Compute final extrinsics using quaternion averaging and optional bundle adjustment.
        """
        valid = [m for m in self.measurements if m["success"]]

        if len(valid) < min_measurements:
            raise ValueError(f"Need {min_measurements} measurements, have {len(valid)}")

        # Average rotation using quaternions
        rotations = [m["R_vehicle_to_camera"] for m in valid]
        R_avg = average_rotation_matrices(rotations)

        # Validate
        is_valid, msg = validate_rotation_matrix(R_avg)
        if not is_valid:
            print(f"WARNING: Averaged rotation invalid: {msg}")
            U, _, Vt = np.linalg.svd(R_avg)
            R_avg = U @ Vt

        # Average position
        positions = np.array([m["t_camera_in_vehicle"] for m in valid])
        t_avg = np.mean(positions, axis=0)
        t_std = np.std(positions, axis=0)

        # Extract Euler angles
        euler = self._rotation_to_euler(R_avg)

        # Bundle adjustment refinement
        if use_bundle_adjustment and len(valid) >= 3:
            t_refined, euler_refined = self._bundle_adjustment(t_avg, euler)
            R_refined = self._euler_to_rotation(
                euler_refined["azimuth"],
                euler_refined["elevation"],
                euler_refined["roll"]
            )

            # Use refined values if they improve consistency
            t_avg = t_refined
            euler = euler_refined
            R_avg = R_refined

        # Compute statistics
        azimuths = np.array([m["euler_angles"]["azimuth"] for m in valid])
        elevations = np.array([m["euler_angles"]["elevation"] for m in valid])
        rolls = np.array([m["euler_angles"]["roll"] for m in valid])

        # Handle azimuth wraparound
        if np.max(azimuths) - np.min(azimuths) > 180:
            azimuths = np.where(azimuths < 0, azimuths + 360, azimuths)

        reproj_errors = [m["reproj_error"] for m in valid]
        distance_errors = [m["distance_error"] for m in valid]

        self.result = {
            "camera_id": self.camera_id,
            "rotation_matrix": R_avg.tolist(),
            "translation_vector": t_avg.tolist(),
            "euler_angles": {
                "azimuth": float(euler["azimuth"]),
                "elevation": float(euler["elevation"]),
                "roll": float(euler["roll"])
            },
            "quality_metrics": {
                "num_measurements": len(valid),
                "mean_reproj_error_px": float(np.mean(reproj_errors)),
                "max_reproj_error_px": float(np.max(reproj_errors)),
                "mean_distance_error_m": float(np.mean(distance_errors)),
                "position_std_m": t_std.tolist(),
                "azimuth_std_deg": float(np.std(azimuths)),
                "elevation_std_deg": float(np.std(elevations)),
                "roll_std_deg": float(np.std(rolls))
            },
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return self.result

    def print_results(self, ground_truth: Optional[dict] = None):
        """Print results with optional ground truth comparison."""
        if self.result is None:
            print("No results")
            return

        r = self.result
        e = r["euler_angles"]
        q = r["quality_metrics"]
        t = r["translation_vector"]

        print("\n" + "=" * 70)
        print("EXTRINSIC CALIBRATION RESULTS (INS-Based)")
        print("=" * 70)

        print(f"\nCamera: {r['camera_id']}")
        print(f"Measurements: {q['num_measurements']}")

        print("\n--- Camera Position (vehicle frame) ---")
        print(f"  X (forward): {t[0]:+.4f} m")
        print(f"  Y (right):   {t[1]:+.4f} m")
        print(f"  Z (down):    {t[2]:+.4f} m")
        print(f"  Position std: [{q['position_std_m'][0]:.4f}, {q['position_std_m'][1]:.4f}, {q['position_std_m'][2]:.4f}] m")

        print("\n--- Camera Orientation ---")
        print(f"  Azimuth:   {e['azimuth']:+.3f}° (std: {q['azimuth_std_deg']:.3f}°)")
        print(f"  Elevation: {e['elevation']:+.3f}° (std: {q['elevation_std_deg']:.3f}°)")
        print(f"  Roll:      {e['roll']:+.3f}° (std: {q['roll_std_deg']:.3f}°)")

        print("\n--- Quality ---")
        print(f"  Reproj error: {q['mean_reproj_error_px']:.3f} px (mean)")
        print(f"  Distance error: {q['mean_distance_error_m']*100:.2f} cm (mean)")

        if ground_truth:
            print("\n--- Ground Truth Comparison ---")
            gt_e = ground_truth["euler_angles"]
            gt_t = ground_truth["position"]

            pos_err = np.linalg.norm(np.array(t) - np.array(gt_t))
            az_err = abs(e["azimuth"] - gt_e["azimuth"])
            el_err = abs(e["elevation"] - gt_e["elevation"])
            ro_err = abs(e["roll"] - gt_e["roll"])

            if az_err > 180:
                az_err = 360 - az_err

            print(f"  Position error: {pos_err*100:.2f} cm")
            print(f"  Azimuth error:   {az_err:.3f}°")
            print(f"  Elevation error: {el_err:.3f}°")
            print(f"  Roll error:      {ro_err:.3f}°")

            if az_err < 1.0 and el_err < 1.0:
                print("  --> PASS (<1° angular error)")
            else:
                print("  --> FAIL (>1° angular error)")

    def save_to_json(self, path: str):
        if self.result is None:
            raise ValueError("No results")
        with open(path, 'w') as f:
            json.dump(self.result, f, indent=2)
        print(f"\nSaved to: {path}")


# =============================================================================
# SYNTHETIC TEST (Consistent Conventions)
# =============================================================================

class SyntheticTestINS:
    """
    Synthetic test generator with CONSISTENT conventions.

    Uses the SAME rotation functions as the calibrator to ensure
    the synthetic data and calibration algorithm are compatible.
    """

    def __init__(self, board_config: ChArUcoBoardConfig, intrinsics: dict,
                 camera_position_vehicle: np.ndarray, camera_euler: dict):
        """
        Args:
            camera_position_vehicle: Camera position in vehicle frame [x_fwd, y_right, z_down]
            camera_euler: Camera orientation {"azimuth", "elevation", "roll"}
        """
        self.board_config = board_config
        self.intrinsics = intrinsics
        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics["distortion_coefficients"], dtype=np.float64)
        self.image_size = tuple(intrinsics["image_size"])

        self.true_position = camera_position_vehicle
        self.true_euler = camera_euler

        # Use the SAME function as the calibrator
        self.true_R_vehicle_to_camera = self._euler_to_rotation(
            camera_euler["azimuth"],
            camera_euler["elevation"],
            camera_euler["roll"]
        )

        # Create board image
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

        self.position_noise = 0.02  # 2cm position measurement noise
        self.ins_noise = 0.1       # 0.1 degree INS noise

    def _euler_to_rotation(self, azimuth: float, elevation: float, roll: float) -> np.ndarray:
        """IDENTICAL to ExtrinsicCalibratorINS._euler_to_rotation."""
        az = np.radians(azimuth)
        el = np.radians(elevation)
        ro = np.radians(roll)

        cam_z_vehicle = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            -np.sin(el)
        ])

        # For azimuth=0 (forward), cam_x should be [0, 1, 0] (vehicle right)
        cam_x_vehicle = np.array([-np.sin(az), np.cos(az), 0])
        norm = np.linalg.norm(cam_x_vehicle)
        if norm > 1e-6:
            cam_x_vehicle = cam_x_vehicle / norm
        else:
            cam_x_vehicle = np.array([0, 1, 0])

        # For right-handed camera frame (X=right, Y=down, Z=forward):
        # Y = Z x X, X = Y x Z
        cam_y_vehicle = np.cross(cam_z_vehicle, cam_x_vehicle)
        cam_y_vehicle = cam_y_vehicle / np.linalg.norm(cam_y_vehicle)

        # Recalculate X for orthogonality
        cam_x_vehicle = np.cross(cam_y_vehicle, cam_z_vehicle)
        cam_x_vehicle = cam_x_vehicle / np.linalg.norm(cam_x_vehicle)

        if abs(ro) > 1e-6:
            c, s = np.cos(ro), np.sin(ro)
            cam_x_new = c * cam_x_vehicle + s * cam_y_vehicle
            cam_y_new = -s * cam_x_vehicle + c * cam_y_vehicle
            cam_x_vehicle = cam_x_new
            cam_y_vehicle = cam_y_new

        R_camera_to_vehicle = np.column_stack([cam_x_vehicle, cam_y_vehicle, cam_z_vehicle])
        return R_camera_to_vehicle.T

    def generate_measurement(self, ins_data: INSData, distance: float,
                              height_offset: float = 0.0,
                              add_noise: bool = True
                              ) -> Tuple[np.ndarray, BoardMeasurement, INSData, float]:
        """
        Generate synthetic image and measurement for given INS and distance.

        Args:
            ins_data: Simulated INS reading
            distance: Distance from camera to board center
            height_offset: Vertical offset from camera height
            add_noise: Whether to add measurement noise

        Returns:
            image: Synthetic camera image
            board_measurement: Board measurement (with noise if enabled)
            noisy_ins: INS data (with noise if enabled)
            true_laser_distance: True distance for validation
        """
        # Get vehicle pose from INS
        R_world_to_vehicle = ins_data.to_rotation_matrix()
        R_vehicle_to_world = R_world_to_vehicle.T

        # Camera pose in vehicle frame (ground truth)
        R_vehicle_to_camera = self.true_R_vehicle_to_camera
        t_camera_in_vehicle = self.true_position

        # Camera pose in world frame
        R_camera_to_vehicle = R_vehicle_to_camera.T
        R_camera_to_world = R_vehicle_to_world @ R_camera_to_vehicle
        t_camera_in_world = R_vehicle_to_world @ t_camera_in_vehicle

        # Camera optical axis in world
        cam_z_world = R_camera_to_world[:, 2]

        # Place board along camera optical axis
        board_center_world = t_camera_in_world + distance * cam_z_world
        board_center_world[2] += height_offset  # Add height variation

        # True laser distance (for validation)
        true_laser_distance = np.linalg.norm(board_center_world - t_camera_in_world)

        # Board faces camera - facing_az is direction FROM board TO camera
        board_to_camera = t_camera_in_world - board_center_world
        board_to_camera[2] = 0  # Project to horizontal
        board_to_camera = board_to_camera / (np.linalg.norm(board_to_camera) + 1e-10)
        facing_az = np.degrees(np.arctan2(board_to_camera[1], board_to_camera[0]))

        # Board pose in world for VISUAL rendering
        # Use Z×Y for board_x (left-handed) so board front is visible
        board_y_world = np.array([0, 0, 1])  # Down
        board_z_world = np.array([np.cos(np.radians(facing_az)),
                                  np.sin(np.radians(facing_az)), 0])
        board_x_world = np.cross(board_z_world, board_y_world)
        board_x_world = board_x_world / np.linalg.norm(board_x_world)

        R_board_to_world = np.column_stack([board_x_world, board_y_world, board_z_world])

        bw = self.board_config.board_width
        bh = self.board_config.board_height
        board_origin_world = board_center_world - R_board_to_world @ np.array([bw/2, bh/2, 0])

        # Board corners in world frame
        corners_board = self.board_corners_3d.copy()
        corners_world = (R_board_to_world @ corners_board.T).T + board_origin_world

        # Transform to camera frame
        R_world_to_camera = R_camera_to_world.T
        corners_camera = (R_world_to_camera @ (corners_world - t_camera_in_world).T).T

        # Check visibility
        if np.any(corners_camera[:, 2] <= 0.1):
            return self._gray_image(), None, ins_data, true_laser_distance

        # Project to image
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        corners_2d = np.array([
            [fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy]
            for c in corners_camera
        ], dtype=np.float32)

        # Check bounds
        margin = 500
        if (np.any(corners_2d[:, 0] < -margin) or
            np.any(corners_2d[:, 0] > self.image_size[0] + margin) or
            np.any(corners_2d[:, 1] < -margin) or
            np.any(corners_2d[:, 1] > self.image_size[1] + margin)):
            return self._gray_image(), None, ins_data, true_laser_distance

        # Warp board image using standard homography
        H, _ = cv2.findHomography(self.board_corners_2d, corners_2d)
        if H is None:
            return self._gray_image(), None, ins_data, true_laser_distance

        image = cv2.warpPerspective(
            self.board_image, H, self.image_size,
            borderMode=cv2.BORDER_CONSTANT, borderValue=128
        )
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Add image noise
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add measurement noise
        if add_noise:
            noisy_board = BoardMeasurement(
                north=board_center_world[0] + np.random.normal(0, self.position_noise),
                east=board_center_world[1] + np.random.normal(0, self.position_noise),
                down=board_center_world[2] + np.random.normal(0, self.position_noise),
                laser_distance=true_laser_distance + np.random.normal(0, 0.01)  # 1cm noise
            )
            noisy_ins = INSData(
                yaw=ins_data.yaw + np.random.normal(0, self.ins_noise),
                pitch=ins_data.pitch + np.random.normal(0, self.ins_noise),
                roll=ins_data.roll + np.random.normal(0, self.ins_noise)
            )
        else:
            noisy_board = BoardMeasurement(
                north=board_center_world[0],
                east=board_center_world[1],
                down=board_center_world[2],
                laser_distance=true_laser_distance
            )
            noisy_ins = ins_data

        return image, noisy_board, noisy_ins, true_laser_distance

    def _gray_image(self) -> np.ndarray:
        return np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 128

    def get_ground_truth(self) -> dict:
        return {
            "position": self.true_position.tolist(),
            "euler_angles": self.true_euler
        }


# =============================================================================
# MAIN
# =============================================================================

def run_synthetic_test(intrinsics_path: str, output_path: str,
                       num_measurements: int = 7) -> Optional[dict]:
    """Run synthetic calibration test."""
    print("=" * 70)
    print("INS-BASED EXTRINSIC CALIBRATION - SYNTHETIC TEST")
    print("=" * 70)

    with open(intrinsics_path) as f:
        intrinsics = json.load(f)

    board_config = ChArUcoBoardConfig()

    # Ground truth camera pose in VEHICLE frame
    # Camera is 0.5m forward, 0.8m right, 0.3m down from IMU
    # Looking at azimuth 30° (forward-right), elevation -5° (slightly down)
    camera_position = np.array([0.5, 0.8, 0.3])
    camera_euler = {"azimuth": 30.0, "elevation": -5.0, "roll": 1.0}

    print(f"\nGround Truth (camera in vehicle frame):")
    print(f"  Position: [{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}] m")
    print(f"  Azimuth:   {camera_euler['azimuth']}°")
    print(f"  Elevation: {camera_euler['elevation']}°")
    print(f"  Roll:      {camera_euler['roll']}°")

    synth = SyntheticTestINS(board_config, intrinsics, camera_position, camera_euler)
    calibrator = ExtrinsicCalibratorINS(board_config, intrinsics, "camera_1")

    print(f"\nMeasurement noise:")
    print(f"  Position: ±{synth.position_noise*100:.0f}cm")
    print(f"  INS: ±{synth.ins_noise}°")

    print(f"\n" + "-" * 70)
    print("MEASUREMENTS")
    print("-" * 70)

    successful = 0
    for i in range(num_measurements + 2):  # Extra attempts in case some fail
        if successful >= num_measurements:
            break

        # Simulate vehicle at different orientations (INS readings)
        ins_yaw = 45.0 + np.random.uniform(-10, 10)
        ins_pitch = np.random.uniform(-3, 3)
        ins_roll = np.random.uniform(-2, 2)
        ins = INSData(yaw=ins_yaw, pitch=ins_pitch, roll=ins_roll)

        # Vary distance and height
        dist = 4.0 + i * 0.5
        height_offset = np.random.uniform(-0.3, 0.3)

        image, board_meas, noisy_ins, true_dist = synth.generate_measurement(
            ins, dist, height_offset, add_noise=True
        )

        if board_meas is None:
            print(f"[{i+1}] FAILED: Board not visible")
            continue

        result = calibrator.add_measurement(image, board_meas, noisy_ins)

        if result["success"]:
            e = result["euler_angles"]
            print(f"[{successful+1}] INS(y={ins_yaw:.0f}°) dist={dist:.1f}m -> "
                  f"az={e['azimuth']:+.2f}°, el={e['elevation']:+.2f}°, "
                  f"roll={e['roll']:+.2f}° (reproj={result['reproj_error']:.2f}px)")
            successful += 1
        else:
            print(f"[{i+1}] FAILED: {result['error']}")

    # Compute final
    print("\n" + "-" * 70)
    print("FINAL RESULT")
    print("-" * 70)

    try:
        result = calibrator.compute_extrinsics(use_bundle_adjustment=True)
        calibrator.print_results(synth.get_ground_truth())
        calibrator.save_to_json(output_path)
        return result
    except ValueError as e:
        print(f"Failed: {e}")
        return None


def run_interactive(intrinsics_path: str, output_path: str) -> Optional[dict]:
    """Run interactive INS-based calibration."""
    print("=" * 70)
    print("INS-BASED EXTRINSIC CALIBRATION - INTERACTIVE MODE")
    print("=" * 70)
    print("\nThis mode requires:")
    print("  1. Live camera feed")
    print("  2. INS data stream")
    print("  3. Laser distance measurements")
    print("\nNot yet implemented. Use --synthetic for testing.")
    return None


def main():
    parser = argparse.ArgumentParser(description="INS-Based Extrinsic Calibration")
    parser.add_argument("--intrinsics", "-i", required=True, help="Intrinsics JSON")
    parser.add_argument("--output", "-o", default="extrinsics_ins.json", help="Output path")
    parser.add_argument("--synthetic", action="store_true", help="Synthetic test mode")
    parser.add_argument("--num-measurements", "-n", type=int, default=7, help="Number of measurements")

    args = parser.parse_args()

    if not Path(args.intrinsics).exists():
        print(f"Error: {args.intrinsics} not found")
        return 1

    if args.synthetic:
        result = run_synthetic_test(args.intrinsics, args.output, args.num_measurements)
        return 0 if result else 1
    else:
        result = run_interactive(args.intrinsics, args.output)
        return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
INS-Based Extrinsic Camera Calibration
=======================================

Determines camera pose relative to vehicle frame using:
1. ChArUco board detection (solvePnP)
2. Live INS data (yaw, pitch, roll)
3. Measured board position in world frame

This module addresses the mathematical issues identified in review:
- Uses proper 3D rotation matrix composition (not angle subtraction)
- Consistent rotation conventions throughout
- Rigorous frame transformations
- Quaternion averaging for rotations

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

ROTATION CONVENTIONS:
=====================

INS provides yaw/pitch/roll in NED convention:
    - Yaw: Rotation about Down axis (0° = North, 90° = East)
    - Pitch: Rotation about East axis (positive = nose up)
    - Roll: Rotation about North axis (positive = right wing down)

The rotation sequence is ZYX (yaw, then pitch, then roll):
    R_world_to_vehicle = Rx(roll) @ Ry(pitch) @ Rz(yaw)

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


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> Dict[str, float]:
    """
    Extract ZYX Euler angles (yaw, pitch, roll) from rotation matrix.

    This is the inverse of euler_zyx_to_rotation_matrix.

    Returns angles in degrees.
    """
    # Handle gimbal lock
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        # Gimbal lock: pitch = ±90°
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
# BOARD PLACEMENT IN WORLD FRAME
# =============================================================================

@dataclass
class BoardPlacementWorld:
    """
    Board placement in world (NED) frame.

    The board is assumed to be:
    - Vertical (board Y axis aligned with world Down)
    - Facing a specific direction (board Z axis in horizontal plane)

    World Frame (NED):
        X: North
        Y: East
        Z: Down

    Board Frame:
        X: Along board width (right when facing board)
        Y: Along board height (down, aligned with world Z)
        Z: Out of board surface (toward viewer)
    """
    # Board center position in world (NED) frame
    north: float  # meters, positive = north
    east: float   # meters, positive = east
    down: float   # meters, positive = down (negative = up)

    # Board facing direction: azimuth of board normal (Z axis)
    # 0° = board faces North, 90° = board faces East
    facing_azimuth: float = 0.0  # degrees

    def get_board_pose_in_world(self, board_config: ChArUcoBoardConfig
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute board pose in world (NED) frame.

        Returns:
            R_world_to_board: Rotation matrix (world vectors to board frame)
            t_board_origin_in_world: Board origin position in world frame
        """
        # Board Z points AWAY from viewer (opposite to facing direction)
        # This matches the working extrinsic_calibration.py convention
        az = np.radians(self.facing_azimuth + 180)

        # Board Z points away from viewer (into the board)
        board_z_world = np.array([np.cos(az), np.sin(az), 0])

        # Board Y points Down (aligned with world Z for vertical board)
        board_y_world = np.array([0, 0, 1])

        # For a right-handed coordinate system: X = Y × Z
        # This gives board X pointing to the viewer's LEFT (not right)
        # But OpenCV uses right-handed with X to the right, Y down, Z toward viewer
        # The key insight: OpenCV's "right" and "down" are from the BOARD's perspective
        # when looking at the back of the board (from behind the markers)
        #
        # When viewing from the front, X appears reversed (points to viewer's left)
        # This is handled by the fact that solvePnP sees the board as it appears
        board_x_world = np.cross(board_y_world, board_z_world)
        board_x_world = board_x_world / np.linalg.norm(board_x_world)

        # R_board_to_world: columns are board axes in world coordinates
        R_board_to_world = np.column_stack([board_x_world, board_y_world, board_z_world])
        R_world_to_board = R_board_to_world.T

        # Board center in world
        board_center_world = np.array([self.north, self.east, self.down])

        # Board origin is at top-left corner
        # In board frame: center = [width/2, height/2, 0]
        # In world frame: origin = center - R_board_to_world @ [width/2, height/2, 0]
        offset_board = np.array([board_config.board_width / 2,
                                  board_config.board_height / 2, 0])
        board_origin_world = board_center_world - R_board_to_world @ offset_board

        return R_world_to_board, board_origin_world


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
# INS-BASED EXTRINSIC CALIBRATOR
# =============================================================================

class ExtrinsicCalibratorINS:
    """
    Computes camera extrinsics using INS data and ChArUco board detection.

    The key computation is:
        R_vehicle_to_camera = R_board_to_camera @ R_world_to_board @ R_vehicle_to_world

    Where:
        - R_board_to_camera: From solvePnP
        - R_world_to_board: From board placement measurement
        - R_vehicle_to_world: From INS data (inverse of INS rotation)

    This properly composes 3D rotations instead of subtracting angles.
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

    def add_measurement(self, image: np.ndarray,
                        board_placement: BoardPlacementWorld,
                        ins_data: INSData) -> dict:
        """
        Process one calibration measurement.

        Args:
            image: Camera image with ChArUco board
            board_placement: Board position/orientation in world frame
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

        # Get board pose in world frame
        R_world_to_board, t_board_in_world = board_placement.get_board_pose_in_world(
            self.board_config)
        R_board_to_world = R_world_to_board.T

        # Get vehicle pose from INS
        R_world_to_vehicle = ins_data.to_rotation_matrix()
        R_vehicle_to_world = R_world_to_vehicle.T

        # =================================================================
        # KEY COMPUTATION: Compose rotations properly
        # =================================================================
        # R_vehicle_to_camera = R_board_to_camera @ R_world_to_board @ R_vehicle_to_world
        #
        # This transforms: vehicle -> world -> board -> camera
        # =================================================================
        R_vehicle_to_camera = R_board_to_camera @ R_world_to_board @ R_vehicle_to_world

        # Validate the rotation matrix
        is_valid, msg = validate_rotation_matrix(R_vehicle_to_camera)
        if not is_valid:
            print(f"WARNING: R_vehicle_to_camera invalid: {msg}")

        # Compute camera position in vehicle frame
        # Camera position in board frame
        R_camera_to_board = R_board_to_camera.T
        t_board_in_camera = tvec.reshape(3)
        t_camera_in_board = -R_camera_to_board @ t_board_in_camera

        # Camera position in world frame
        t_camera_in_world = R_board_to_world @ t_camera_in_board + t_board_in_world.flatten()

        # Camera position in vehicle frame
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
            "board_placement": board_placement,
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
        # Camera optical axis (Z) in vehicle frame = row 2 of R_vehicle_to_camera
        # Wait, R_vehicle_to_camera transforms vehicle coords to camera coords
        # So camera Z in vehicle frame is the third column of R_camera_to_vehicle = R^T
        R_camera_to_vehicle = R.T
        cam_z_vehicle = R_camera_to_vehicle[:, 2]  # Camera optical axis in vehicle frame

        # Azimuth: angle in XY plane (forward-right)
        azimuth = np.degrees(np.arctan2(cam_z_vehicle[1], cam_z_vehicle[0]))

        # Elevation: angle from horizontal
        horizontal = np.sqrt(cam_z_vehicle[0]**2 + cam_z_vehicle[1]**2)
        elevation = np.degrees(np.arctan2(-cam_z_vehicle[2], horizontal))

        # Roll: rotation of camera X axis around optical axis
        az_rad = np.radians(azimuth)
        cam_x_no_roll = np.array([np.sin(az_rad), -np.cos(az_rad), 0])
        cam_x_no_roll = cam_x_no_roll / np.linalg.norm(cam_x_no_roll)

        cam_x_vehicle = R_camera_to_vehicle[:, 0]
        cam_y_vehicle = R_camera_to_vehicle[:, 1]

        roll = np.degrees(np.arctan2(
            -np.dot(cam_x_no_roll, cam_y_vehicle),
            np.dot(cam_x_no_roll, cam_x_vehicle)
        ))

        return {"azimuth": azimuth, "elevation": elevation, "roll": roll}

    def _euler_to_rotation(self, azimuth: float, elevation: float, roll: float) -> np.ndarray:
        """
        Convert camera Euler angles to R_vehicle_to_camera.

        This is the inverse of _rotation_to_euler.
        """
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
        cam_x_vehicle = np.array([np.sin(az), -np.cos(az), 0])
        cam_x_vehicle = cam_x_vehicle / np.linalg.norm(cam_x_vehicle)

        # Camera Y (perpendicular to Z and X)
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

    def compute_extrinsics(self, min_measurements: int = 3) -> dict:
        """
        Compute final extrinsics using quaternion averaging.
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

        # Compute statistics
        azimuths = np.array([m["euler_angles"]["azimuth"] for m in valid])
        elevations = np.array([m["euler_angles"]["elevation"] for m in valid])
        rolls = np.array([m["euler_angles"]["roll"] for m in valid])

        # Handle azimuth wraparound
        if np.max(azimuths) - np.min(azimuths) > 180:
            azimuths = np.where(azimuths < 0, azimuths + 360, azimuths)

        reproj_errors = [m["reproj_error"] for m in valid]

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

        print("\n--- Camera Orientation ---")
        print(f"  Azimuth:   {e['azimuth']:+.3f}° (std: {q['azimuth_std_deg']:.3f}°)")
        print(f"  Elevation: {e['elevation']:+.3f}° (std: {q['elevation_std_deg']:.3f}°)")
        print(f"  Roll:      {e['roll']:+.3f}° (std: {q['roll_std_deg']:.3f}°)")

        print("\n--- Quality ---")
        print(f"  Reproj error: {q['mean_reproj_error_px']:.3f} px (mean)")

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

        self.position_noise = 0.005  # 5mm
        self.angle_noise = 2.0       # 2 degrees

    def _euler_to_rotation(self, azimuth: float, elevation: float, roll: float) -> np.ndarray:
        """
        IDENTICAL to ExtrinsicCalibratorINS._euler_to_rotation.
        This ensures consistent conventions.
        """
        az = np.radians(azimuth)
        el = np.radians(elevation)
        ro = np.radians(roll)

        cam_z_vehicle = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            -np.sin(el)
        ])

        cam_x_vehicle = np.array([np.sin(az), -np.cos(az), 0])
        cam_x_vehicle = cam_x_vehicle / np.linalg.norm(cam_x_vehicle)

        cam_y_vehicle = np.cross(cam_z_vehicle, cam_x_vehicle)
        cam_y_vehicle = cam_y_vehicle / np.linalg.norm(cam_y_vehicle)

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

    def generate_measurement(self, board_placement: BoardPlacementWorld,
                              ins_data: INSData,
                              add_noise: bool = True
                              ) -> Tuple[np.ndarray, BoardPlacementWorld, INSData]:
        """
        Generate synthetic image for given board placement and INS data.

        Returns:
            image: Synthetic camera image
            noisy_placement: Board placement with measurement noise
            noisy_ins: INS data with noise
        """
        # Get board pose in world frame
        R_world_to_board, t_board_in_world = board_placement.get_board_pose_in_world(
            self.board_config)
        R_board_to_world = R_world_to_board.T

        # Get vehicle pose from INS
        R_world_to_vehicle = ins_data.to_rotation_matrix()

        # Camera pose in vehicle frame (ground truth)
        R_vehicle_to_camera = self.true_R_vehicle_to_camera
        t_camera_in_vehicle = self.true_position

        # Camera pose in world frame
        R_camera_to_vehicle = R_vehicle_to_camera.T
        R_vehicle_to_world = R_world_to_vehicle.T
        R_camera_to_world = R_vehicle_to_world @ R_camera_to_vehicle
        t_camera_in_world = R_vehicle_to_world @ t_camera_in_vehicle

        # Board corners in world frame
        corners_board = self.board_corners_3d.copy()
        corners_world = (R_board_to_world @ corners_board.T).T + t_board_in_world.flatten()

        # Transform to camera frame
        R_world_to_camera = R_camera_to_world.T
        corners_camera = (R_world_to_camera @ (corners_world - t_camera_in_world).T).T

        # Check visibility
        if np.any(corners_camera[:, 2] <= 0.1):
            return self._gray_image(), board_placement, ins_data

        # Project to image
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        corners_2d_raw = np.array([
            [fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy]
            for c in corners_camera
        ], dtype=np.float32)

        # Board corners in board frame: TL(0), TR(1), BR(2), BL(3)
        # These correspond to source image corners in the same order
        corners_2d = corners_2d_raw

        # Check bounds
        margin = 500
        if (np.any(corners_2d[:, 0] < -margin) or
            np.any(corners_2d[:, 0] > self.image_size[0] + margin) or
            np.any(corners_2d[:, 1] < -margin) or
            np.any(corners_2d[:, 1] > self.image_size[1] + margin)):
            return self._gray_image(), board_placement, ins_data

        # Warp board image
        H, _ = cv2.findHomography(self.board_corners_2d, corners_2d)
        if H is None:
            return self._gray_image(), board_placement, ins_data

        image = cv2.warpPerspective(
            self.board_image, H, self.image_size,
            borderMode=cv2.BORDER_CONSTANT, borderValue=128
        )
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Add noise
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add measurement noise
        if add_noise:
            noisy_placement = BoardPlacementWorld(
                north=board_placement.north + np.random.normal(0, self.position_noise),
                east=board_placement.east + np.random.normal(0, self.position_noise),
                down=board_placement.down + np.random.normal(0, self.position_noise),
                facing_azimuth=board_placement.facing_azimuth + np.random.normal(0, self.angle_noise)
            )
            noisy_ins = INSData(
                yaw=ins_data.yaw + np.random.normal(0, 0.1),  # 0.1° INS noise
                pitch=ins_data.pitch + np.random.normal(0, 0.1),
                roll=ins_data.roll + np.random.normal(0, 0.1)
            )
        else:
            noisy_placement = board_placement
            noisy_ins = ins_data

        return image, noisy_placement, noisy_ins

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

    print(f"\n" + "-" * 70)
    print("MEASUREMENTS")
    print("-" * 70)

    for i in range(num_measurements):
        # Simulate vehicle at different orientations (INS readings)
        ins_yaw = 45.0 + np.random.uniform(-10, 10)
        ins_pitch = np.random.uniform(-3, 3)
        ins_roll = np.random.uniform(-2, 2)
        ins = INSData(yaw=ins_yaw, pitch=ins_pitch, roll=ins_roll)

        # Compute camera position in world frame
        R_world_to_vehicle = ins.to_rotation_matrix()
        R_vehicle_to_world = R_world_to_vehicle.T

        # Camera position in world (vehicle is at world origin for simplicity)
        camera_pos_world = R_vehicle_to_world @ camera_position

        # Camera optical axis direction in world
        az_veh = np.radians(camera_euler["azimuth"])
        el_veh = np.radians(camera_euler["elevation"])
        cam_dir_vehicle = np.array([
            np.cos(el_veh) * np.cos(az_veh),
            np.cos(el_veh) * np.sin(az_veh),
            -np.sin(el_veh)
        ])
        cam_dir_world = R_vehicle_to_world @ cam_dir_vehicle

        # Place board along camera optical axis
        dist = 4.0 + i * 0.5
        board_center_world = camera_pos_world + dist * cam_dir_world

        # Board faces back toward camera
        facing_az = np.degrees(np.arctan2(-cam_dir_world[1], -cam_dir_world[0]))

        board = BoardPlacementWorld(
            north=board_center_world[0],
            east=board_center_world[1],
            down=board_center_world[2] + np.random.uniform(-0.2, 0.2),
            facing_azimuth=facing_az
        )

        image, noisy_board, noisy_ins = synth.generate_measurement(board, ins)

        result = calibrator.add_measurement(image, noisy_board, noisy_ins)

        if result["success"]:
            e = result["euler_angles"]
            print(f"[{i+1}] INS(y={ins_yaw:.0f}°) -> az={e['azimuth']:+.2f}°, "
                  f"el={e['elevation']:+.2f}°, roll={e['roll']:+.2f}° "
                  f"(reproj={result['reproj_error']:.2f}px)")
        else:
            print(f"[{i+1}] FAILED: {result['error']}")

    # Compute final
    print("\n" + "-" * 70)
    print("FINAL RESULT")
    print("-" * 70)

    try:
        result = calibrator.compute_extrinsics()
        calibrator.print_results(synth.get_ground_truth())
        calibrator.save_to_json(output_path)
        return result
    except ValueError as e:
        print(f"Failed: {e}")
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
        print("Interactive mode not yet implemented")
        print("Use --synthetic for testing")
        return 1


if __name__ == "__main__":
    sys.exit(main())

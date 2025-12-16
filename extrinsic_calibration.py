#!/usr/bin/env python3
"""
Extrinsic Camera Calibration with INS Integration - Full Bundle Adjustment
===========================================================================

This calibration system uses dual-RTK measurements and bundle adjustment
to determine the IMU-to-camera transform (6-DOF).

METHOD: DUAL-RTK + BUNDLE ADJUSTMENT
- RTK #1: Vehicle/IMU antenna position (static)
- RTK #2: Board antenna position (moved with board)
- ChArUco board for image-based measurements
- IMU attitude for orientation prior
- Full reprojection residuals on all detected Charuco corners

COORDINATE SYSTEMS:
- World Frame (W): NED, defined by RTK/INS (North-East-Down)
- IMU Frame (I): NED convention (X=Forward, Y=Right, Z=Down)
- Camera Frame (C): X=Right, Y=Down, Z=Forward (optical axis)
- Board Frame (B): X=Right, Y=Down, Z=Out (normal toward camera)

OPTIMIZATION VARIABLES:
- T_IC: IMU→camera transform (6-DOF) - the result we want
- T_WI: World→IMU transform (6-DOF)
- T_WB_k: World→board transform for each capture k (6-DOF)

RESIDUALS:
1. Charuco reprojection: pixel errors between detected and projected corners
2. Board RTK position: predicted vs measured board antenna position
3. Vehicle RTK position: predicted vs measured vehicle antenna position
4. IMU attitude prior: alignment with INS-measured attitude
5. Board levelness prior: soft constraint for roughly vertical board
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
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
    squares_x: int = 8
    squares_y: int = 8
    square_size: float = 0.11      # 11cm squares
    marker_size: float = 0.085     # 8.5cm markers
    dictionary_id: int = cv2.aruco.DICT_6X6_250
    rtk_antenna_offset_board: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    @property
    def board_width(self) -> float:
        return self.squares_x * self.square_size
    
    @property
    def board_height(self) -> float:
        return self.squares_y * self.square_size
    
    @property
    def board_center_offset(self) -> np.ndarray:
        return np.array([self.board_width / 2, self.board_height / 2, 0])
    
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
    """Configuration for the bundle adjustment optimization."""
    corner_detection_std_px: float = 0.5
    rtk_position_std: float = 0.02
    imu_attitude_std_deg: float = 0.5
    board_level_std_deg: float = 5.0
    camera_translation_prior_std: float = 0.10
    camera_rotation_prior_std_deg: float = 5.0
    
    max_iterations: int = 500
    ftol: float = 1e-12
    
    weight_reprojection: float = 1.0
    weight_rtk_board: float = 1.0
    weight_rtk_vehicle: float = 10.0  # Reduced from 50
    weight_imu_attitude: float = 10.0  # Reduced from 20
    weight_board_level: float = 0.1
    weight_camera_prior: float = 2.0  # Increased from 0.5


# =============================================================================
# ROTATION UTILITIES
# =============================================================================

class RotationUtils:
    """Rotation utilities using scipy.spatial.transform.Rotation."""
    
    @staticmethod
    def euler_to_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        
        Returns R_world_to_body: transforms world coordinates to body coordinates.
        Convention: ZYX intrinsic (yaw about Z, then pitch about Y, then roll about X)
        """
        r = Rotation.from_euler('ZYX', [yaw_deg, pitch_deg, roll_deg], degrees=True)
        return r.as_matrix()
    
    @staticmethod
    def rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (yaw, pitch, roll in degrees)."""
        r = Rotation.from_matrix(R)
        angles = r.as_euler('ZYX', degrees=True)
        return float(angles[0]), float(angles[1]), float(angles[2])
    
    @staticmethod
    def rotation_to_camera_angles(R_camera_to_imu: np.ndarray) -> Dict[str, float]:
        """Convert camera-to-IMU rotation to intuitive angles."""
        optical_axis = R_camera_to_imu @ np.array([0, 0, 1])
        azimuth = np.degrees(np.arctan2(optical_axis[1], optical_axis[0]))
        horizontal = np.sqrt(optical_axis[0]**2 + optical_axis[1]**2)
        elevation = np.degrees(np.arctan2(optical_axis[2], horizontal))
        
        camera_y = R_camera_to_imu @ np.array([0, 1, 0])
        camera_y_perp = camera_y - np.dot(camera_y, optical_axis) * optical_axis
        norm = np.linalg.norm(camera_y_perp)
        
        if norm > 0.01:
            camera_y_perp /= norm
            imu_down = np.array([0, 0, 1])
            expected = imu_down - np.dot(imu_down, optical_axis) * optical_axis
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
        """Convert camera angles to R_camera_to_imu."""
        az = np.radians(azimuth)
        el = np.radians(elevation)
        ro = np.radians(roll)
        
        cam_z = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        cam_z /= np.linalg.norm(cam_z)
        
        cam_x = np.array([-np.sin(az), np.cos(az), 0])
        if np.linalg.norm(cam_x) < 0.01:
            cam_x = np.array([1, 0, 0])
        cam_x /= np.linalg.norm(cam_x)
        
        cam_y = np.cross(cam_z, cam_x)
        cam_y /= np.linalg.norm(cam_y)
        cam_x = np.cross(cam_y, cam_z)
        cam_x /= np.linalg.norm(cam_x)
        
        if abs(ro) > 1e-6:
            c, s = np.cos(ro), np.sin(ro)
            cam_x, cam_y = c * cam_x + s * cam_y, -s * cam_x + c * cam_y
        
        return np.column_stack([cam_x, cam_y, cam_z])
    
    @staticmethod
    def rotation_log(R: np.ndarray) -> np.ndarray:
        """SO(3) logarithm map."""
        return Rotation.from_matrix(R).as_rotvec()


# =============================================================================
# SE(3) TRANSFORM UTILITIES
# =============================================================================

class SE3:
    """
    SE(3) rigid body transformation utilities.
    
    Convention: T_AB = [R_AB, t_AB] represents pose of frame B in frame A.
    - To transform point p_B to frame A: p_A = R_AB @ p_B + t_AB
    - Composition: T_AC = T_AB @ T_BC
    """
    
    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    @staticmethod
    def to_Rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return T[:3, :3].copy(), T[:3, 3].copy()
    
    @staticmethod
    def inverse(T: np.ndarray) -> np.ndarray:
        R, t = SE3.to_Rt(T)
        return SE3.from_Rt(R.T, -R.T @ t)
    
    @staticmethod
    def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        return T1 @ T2
    
    @staticmethod
    def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
        R, t = SE3.to_Rt(T)
        return R @ p + t
    
    @staticmethod
    def from_params(params: np.ndarray) -> np.ndarray:
        t = params[:3]
        rotvec = params[3:6]
        R = Rotation.from_rotvec(rotvec).as_matrix()
        return SE3.from_Rt(R, t)
    
    @staticmethod
    def to_params(T: np.ndarray) -> np.ndarray:
        R, t = SE3.to_Rt(T)
        rotvec = Rotation.from_matrix(R).as_rotvec()
        return np.concatenate([t, rotvec])


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class INSData:
    """INS orientation at capture time (NED convention)."""
    yaw: float
    pitch: float
    roll: float
    timestamp: float = 0.0
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Returns R_world_to_imu: transforms world coords to IMU coords."""
        return RotationUtils.euler_to_rotation(self.yaw, self.pitch, self.roll)


@dataclass
class RTKMeasurement:
    """RTK position measurement."""
    position_world: np.ndarray
    std: float = 0.02


@dataclass
class Measurement:
    """A single calibration measurement."""
    corners_2d: np.ndarray
    corner_ids: np.ndarray
    corners_3d_board: np.ndarray
    image_shape: Tuple[int, int]
    R_board_in_camera: np.ndarray
    t_board_in_camera: np.ndarray
    reproj_error_pnp: float
    ins_data: INSData
    board_rtk: Optional[RTKMeasurement] = None
    vehicle_rtk: Optional[RTKMeasurement] = None
    laser_distance: Optional[float] = None


@dataclass
class CameraPrior:
    """Prior estimate of camera pose relative to IMU."""
    position: np.ndarray
    position_std: np.ndarray
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    roll: Optional[float] = None
    orientation_std_deg: float = 5.0


@dataclass
class IMUConfig:
    """IMU-related configuration."""
    rtk_antenna_offset_imu: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or len(corners) < 6:
            return None
        return corners.reshape(-1, 2), ids.flatten()
    
    def estimate_pose(self, corners_2d: np.ndarray, ids: np.ndarray,
                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray
                      ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        obj_points = self.corners_3d[ids]
        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners_2d.reshape(-1, 1, 2),
            camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_err = np.sqrt(np.mean((corners_2d - proj.reshape(-1, 2))**2))
        return R, t, reproj_err


# =============================================================================
# BUNDLE ADJUSTMENT CALIBRATOR
# =============================================================================

class ExtrinsicCalibrator:
    """
    Bundle Adjustment-based extrinsic calibrator.
    
    Directly estimates the IMU→camera transform (T_IC) using:
    1. Charuco reprojection residuals
    2. Board RTK position residuals  
    3. Vehicle RTK position residuals
    4. IMU attitude prior residuals
    5. Board levelness soft prior
    """
    
    def __init__(self, board_config: ChArUcoBoardConfig, intrinsics: dict,
                 camera_id: str, config: CalibrationConfig = None,
                 imu_config: IMUConfig = None):
        self.board_config = board_config
        self.camera_id = camera_id
        self.config = config or CalibrationConfig()
        self.imu_config = imu_config or IMUConfig()
        
        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics["distortion_coefficients"], dtype=np.float64)
        
        self.detector = ChArUcoDetector(board_config)
        self.measurements: List[Measurement] = []
        self.prior: Optional[CameraPrior] = None
        self.result: Optional[dict] = None
    
    def set_prior(self, prior: CameraPrior):
        self.prior = prior
    
    def add_measurement(self, image: np.ndarray, ins_data: INSData,
                        board_rtk: RTKMeasurement = None,
                        vehicle_rtk: RTKMeasurement = None,
                        laser_distance: float = None) -> dict:
        detection = self.detector.detect(image)
        if detection is None:
            return {"success": False, "error": "No corners detected"}
        
        corners_2d, ids = detection
        corners_3d = self.detector.corners_3d[ids]
        
        pose = self.detector.estimate_pose(corners_2d, ids, self.camera_matrix, self.dist_coeffs)
        if pose is None:
            return {"success": False, "error": "PnP failed"}
        
        R_board_camera, t_board_camera, reproj_err = pose
        
        meas = Measurement(
            corners_2d=corners_2d,
            corner_ids=ids,
            corners_3d_board=corners_3d,
            image_shape=image.shape[:2],
            R_board_in_camera=R_board_camera,
            t_board_in_camera=t_board_camera,
            reproj_error_pnp=reproj_err,
            ins_data=ins_data,
            board_rtk=board_rtk,
            vehicle_rtk=vehicle_rtk,
            laser_distance=laser_distance
        )
        
        self.measurements.append(meas)
        
        return {
            "success": True,
            "corners_detected": len(corners_2d),
            "reproj_error": reproj_err,
            "pnp_distance": np.linalg.norm(t_board_camera)
        }
    
    def _project_points(self, points_camera: np.ndarray) -> np.ndarray:
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        z = np.maximum(points_camera[:, 2], 1e-6)
        u = fx * points_camera[:, 0] / z + cx
        v = fy * points_camera[:, 1] / z + cy
        return np.column_stack([u, v])
    
    def _residual_function(self, params: np.ndarray) -> np.ndarray:
        """
        Compute residual vector for bundle adjustment.
        
        Parameters:
        - params[0:6]: T_IC (IMU→camera) as [tx, ty, tz, rx, ry, rz]
        - params[6:12]: T_WI (world→IMU pose)
        - params[12:]: T_WB_k for each measurement
        """
        n_meas = len(self.measurements)
        
        T_IC = SE3.from_params(params[0:6])
        T_WI = SE3.from_params(params[6:12])
        
        # Camera pose in world: T_WC = T_WI @ T_IC
        T_WC = SE3.compose(T_WI, T_IC)
        T_CW = SE3.inverse(T_WC)
        
        R_IC, t_IC = SE3.to_Rt(T_IC)
        R_WI, t_WI = SE3.to_Rt(T_WI)
        
        residuals = []
        
        for i, meas in enumerate(self.measurements):
            idx = 12 + i * 6
            T_WB = SE3.from_params(params[idx:idx+6])
            R_WB, t_WB = SE3.to_Rt(T_WB)
            
            # Board in camera: T_CB = T_CW @ T_WB
            T_CB = SE3.compose(T_CW, T_WB)
            R_CB, t_CB = SE3.to_Rt(T_CB)
            
            # 1. CHARUCO REPROJECTION
            corners_camera = (R_CB @ meas.corners_3d_board.T).T + t_CB
            projected = self._project_points(corners_camera)
            reproj_error = (meas.corners_2d - projected).flatten()
            weight = self.config.weight_reprojection / self.config.corner_detection_std_px
            residuals.extend(weight * reproj_error)
            
            # 2. BOARD RTK POSITION
            if meas.board_rtk is not None:
                p_Rb_B = self.board_config.rtk_antenna_offset_board
                p_Rb_W_pred = SE3.transform_point(T_WB, p_Rb_B)
                p_Rb_W_meas = meas.board_rtk.position_world
                rtk_residual = (p_Rb_W_pred - p_Rb_W_meas) / meas.board_rtk.std
                residuals.extend(self.config.weight_rtk_board * rtk_residual)
            
            # 3. VEHICLE RTK POSITION
            if meas.vehicle_rtk is not None:
                p_Rv_I = self.imu_config.rtk_antenna_offset_imu
                p_Rv_W_pred = SE3.transform_point(T_WI, p_Rv_I)
                p_Rv_W_meas = meas.vehicle_rtk.position_world
                rtk_residual = (p_Rv_W_pred - p_Rv_W_meas) / meas.vehicle_rtk.std
                residuals.extend(self.config.weight_rtk_vehicle * rtk_residual)
            
            # 4. IMU ATTITUDE PRIOR
            # INS gives R_world_to_imu. Our T_WI has R_WI which should be R_imu_to_world.
            R_world_to_imu_meas = meas.ins_data.to_rotation_matrix()
            R_WI_expected = R_world_to_imu_meas.T  # R_imu_to_world
            R_att_error = R_WI_expected.T @ R_WI
            att_error = RotationUtils.rotation_log(R_att_error)
            att_weight = self.config.weight_imu_attitude / np.radians(self.config.imu_attitude_std_deg)
            residuals.extend(att_weight * att_error)
            
            # 5. BOARD LEVELNESS SOFT PRIOR
            board_z_world = R_WB @ np.array([0, 0, 1])
            level_residual = board_z_world[2]
            level_weight = self.config.weight_board_level / np.radians(self.config.board_level_std_deg)
            residuals.append(level_weight * level_residual)
        
        # 6. CAMERA POSE PRIOR
        if self.prior is not None:
            for j in range(3):
                prior_residual = (t_IC[j] - self.prior.position[j]) / self.prior.position_std[j]
                residuals.append(self.config.weight_camera_prior * prior_residual)
            
            if self.prior.azimuth is not None or self.prior.elevation is not None:
                angles = RotationUtils.rotation_to_camera_angles(R_IC)
                if self.prior.azimuth is not None:
                    residuals.append(self.config.weight_camera_prior * 
                                   (angles['azimuth'] - self.prior.azimuth) / self.prior.orientation_std_deg)
                if self.prior.elevation is not None:
                    residuals.append(self.config.weight_camera_prior * 
                                   (angles['elevation'] - self.prior.elevation) / self.prior.orientation_std_deg)
                if self.prior.roll is not None:
                    residuals.append(self.config.weight_camera_prior * 
                                   (angles['roll'] - self.prior.roll) / self.prior.orientation_std_deg)
        
        return np.array(residuals)
    
    def _compute_initial_estimate(self) -> np.ndarray:
        """Compute initial parameter estimate."""
        n_meas = len(self.measurements)
        
        # T_WI from INS + vehicle RTK (average across measurements for robustness)
        R_world_to_imu_sum = np.zeros((3, 3))
        t_WI_sum = np.zeros(3)
        count = 0
        
        for meas in self.measurements:
            R_world_to_imu = meas.ins_data.to_rotation_matrix()
            R_WI = R_world_to_imu.T
            R_world_to_imu_sum += R_WI
            
            if meas.vehicle_rtk is not None:
                t_WI = meas.vehicle_rtk.position_world - R_WI @ self.imu_config.rtk_antenna_offset_imu
                t_WI_sum += t_WI
                count += 1
        
        # Average rotation (approximate - better would be Procrustes)
        U, _, Vt = np.linalg.svd(R_world_to_imu_sum)
        R_WI_init = U @ Vt
        if np.linalg.det(R_WI_init) < 0:
            R_WI_init = U @ np.diag([1, 1, -1]) @ Vt
        
        t_WI_init = t_WI_sum / count if count > 0 else np.zeros(3)
        T_WI_init = SE3.from_Rt(R_WI_init, t_WI_init)
        
        # Estimate T_IC from multiple measurements
        # For each measurement: T_WC = T_WB @ inv(T_CB), then T_IC = inv(T_WI) @ T_WC
        T_IC_estimates = []
        
        for meas in self.measurements:
            if meas.board_rtk is None:
                continue
            
            # T_CB from PnP
            R_CB = meas.R_board_in_camera
            t_CB = meas.t_board_in_camera
            T_CB = SE3.from_Rt(R_CB, t_CB)
            T_BC = SE3.inverse(T_CB)
            
            # T_WB from RTK (assume board is approximately level and facing camera)
            t_WB = meas.board_rtk.position_world  # Simplified: RTK at board center
            # Estimate board orientation from camera direction
            R_IW_init = R_WI_init.T
            camera_dir_world = R_WI_init @ (RotationUtils.camera_angles_to_rotation(
                self.prior.azimuth if self.prior and self.prior.azimuth else 0,
                self.prior.elevation if self.prior and self.prior.elevation else 0,
                0
            ) @ np.array([0, 0, 1]))
            
            # Board faces opposite to camera
            board_z = -camera_dir_world
            board_z[2] = 0  # Keep horizontal
            board_z /= np.linalg.norm(board_z) if np.linalg.norm(board_z) > 0.01 else 1
            board_x = np.cross(np.array([0, 0, 1]), board_z)
            if np.linalg.norm(board_x) < 0.01:
                board_x = np.array([1, 0, 0])
            board_x /= np.linalg.norm(board_x)
            board_y = np.cross(board_z, board_x)
            R_WB = np.column_stack([board_x, board_y, board_z])
            
            T_WB = SE3.from_Rt(R_WB, t_WB)
            
            # T_WC = T_WB @ T_BC
            T_WC = SE3.compose(T_WB, T_BC)
            
            # T_IC = inv(T_WI) @ T_WC
            T_IW = SE3.inverse(T_WI_init)
            T_IC_est = SE3.compose(T_IW, T_WC)
            T_IC_estimates.append(T_IC_est)
        
        # Use first estimate or prior
        if T_IC_estimates:
            T_IC_init = T_IC_estimates[0]
        elif self.prior is not None:
            t_IC_init = self.prior.position.copy()
            az = self.prior.azimuth if self.prior.azimuth is not None else 0.0
            el = self.prior.elevation if self.prior.elevation is not None else 0.0
            ro = self.prior.roll if self.prior.roll is not None else 0.0
            R_IC_init = RotationUtils.camera_angles_to_rotation(az, el, ro)
            T_IC_init = SE3.from_Rt(R_IC_init, t_IC_init)
        else:
            T_IC_init = np.eye(4)
        
        # Build parameter vector
        params = np.zeros(12 + 6 * n_meas)
        params[0:6] = SE3.to_params(T_IC_init)
        params[6:12] = SE3.to_params(T_WI_init)
        
        # T_WB from PnP
        T_WC_init = SE3.compose(T_WI_init, T_IC_init)
        
        for i, meas in enumerate(self.measurements):
            R_CB = meas.R_board_in_camera
            t_CB = meas.t_board_in_camera
            T_CB = SE3.from_Rt(R_CB, t_CB)
            T_WB_init = SE3.compose(T_WC_init, T_CB)
            
            idx = 12 + i * 6
            params[idx:idx+6] = SE3.to_params(T_WB_init)
        
        return params
    
    def compute_extrinsics(self) -> dict:
        """Compute IMU-to-camera extrinsics via bundle adjustment."""
        if len(self.measurements) < 3:
            raise ValueError("Need at least 3 measurements")
        
        x0 = self._compute_initial_estimate()
        
        try:
            result = least_squares(
                self._residual_function, x0, method='trf',
                max_nfev=self.config.max_iterations * len(x0),
                ftol=self.config.ftol, x_scale='jac', verbose=0
            )
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            result = type('obj', (object,), {'x': x0, 'success': False, 'cost': float('inf'), 
                                              'nfev': 0, 'message': str(e)})()
        
        return self._extract_results(result)
    
    def _extract_results(self, result) -> dict:
        n_meas = len(self.measurements)
        
        T_IC = SE3.from_params(result.x[0:6])
        T_WI = SE3.from_params(result.x[6:12])
        
        R_IC, t_IC = SE3.to_Rt(T_IC)
        R_WI, t_WI = SE3.to_Rt(T_WI)
        
        camera_angles = RotationUtils.rotation_to_camera_angles(R_IC)
        imu_yaw, imu_pitch, imu_roll = RotationUtils.rotation_to_euler(R_WI.T)
        
        # Compute reprojection errors
        reproj_errors = []
        T_WC = SE3.compose(T_WI, T_IC)
        T_CW = SE3.inverse(T_WC)
        
        for i, meas in enumerate(self.measurements):
            idx = 12 + i * 6
            T_WB = SE3.from_params(result.x[idx:idx+6])
            T_CB = SE3.compose(T_CW, T_WB)
            R_CB, t_CB = SE3.to_Rt(T_CB)
            corners_camera = (R_CB @ meas.corners_3d_board.T).T + t_CB
            projected = self._project_points(corners_camera)
            err = np.sqrt(np.mean((meas.corners_2d - projected)**2))
            reproj_errors.append(err)
        
        self.result = {
            "camera_id": self.camera_id,
            "imu_to_camera_transform": {
                "rotation_matrix": R_IC.tolist(),
                "translation_vector": t_IC.tolist(),
                "euler_angles": camera_angles,
                "transform_matrix_4x4": T_IC.tolist()
            },
            "imu_world_pose": {
                "rotation_matrix": R_WI.tolist(),
                "translation_vector": t_WI.tolist(),
                "euler_angles_ypr_deg": [imu_yaw, imu_pitch, imu_roll]
            },
            "coordinate_system": {
                "frame": "NED",
                "origin": "IMU center",
                "note": "T_IC transforms points from IMU frame to camera frame"
            },
            "quality_metrics": {
                "num_measurements": n_meas,
                "optimization_converged": result.success,
                "final_cost": float(result.cost),
                "mean_reproj_error_px": float(np.mean(reproj_errors)),
                "max_reproj_error_px": float(np.max(reproj_errors)),
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return self.result
    
    def save_to_json(self, path: str):
        if self.result is None:
            raise ValueError("No results to save")
        with open(path, 'w') as f:
            json.dump(self.result, f, indent=2)
        print(f"Saved to: {path}")


# =============================================================================
# SYNTHETIC TEST
# =============================================================================

class SyntheticTest:
    """Generates synthetic test data."""
    
    def __init__(self, board_config: ChArUcoBoardConfig, intrinsics: dict,
                 camera_position_imu: np.ndarray, camera_angles: dict, 
                 imu_position_world: np.ndarray, ins_euler: dict):
        self.board_config = board_config
        self.camera_position_imu = camera_position_imu
        self.camera_angles = camera_angles
        self.imu_position_world = imu_position_world
        self.ins_euler = ins_euler
        
        self.camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        self.image_size = tuple(intrinsics["image_size"])
        
        board, _ = board_config.create_board()
        px_per_m = 3000
        bw = int(board_config.board_width * px_per_m)
        bh = int(board_config.board_height * px_per_m)
        self.board_image = board.generateImage((bw, bh))
        
        # Compute camera pose in world
        R_world_to_imu = RotationUtils.euler_to_rotation(ins_euler["yaw"], ins_euler["pitch"], ins_euler["roll"])
        R_WI = R_world_to_imu.T  # IMU-to-world for pose
        R_IC = RotationUtils.camera_angles_to_rotation(
            camera_angles["azimuth"], camera_angles["elevation"], camera_angles["roll"]
        )
        
        self.T_WI = SE3.from_Rt(R_WI, imu_position_world)
        self.T_IC = SE3.from_Rt(R_IC, camera_position_imu)
        self.T_WC = SE3.compose(self.T_WI, self.T_IC)
        
        R_WC, t_WC = SE3.to_Rt(self.T_WC)
        self.camera_pos_world = t_WC
    
    def generate_measurement(self, board_center_world: np.ndarray, board_yaw: float,
                             board_pitch: float = 0.0, board_roll: float = 0.0,
                             rtk_noise_std: float = 0.02) -> Tuple[Optional[np.ndarray], dict]:
        """Generate a synthetic measurement.
        
        board_center_world: Position of board center in world frame
        board_yaw: Board faces this direction (degrees, 0=North, 90=East)
        board_pitch/roll: Small tilts from vertical (degrees)
        
        Board frame (ChArUco convention):
        - Origin at top-left corner
        - X points right (toward TR)
        - Y points down (toward BL)
        - Z points out of board (toward camera)
        
        World frame: NED (X=North, Y=East, Z=Down)
        """
        bw, bh = self.board_config.board_width, self.board_config.board_height
        
        # Construct R_WB for board facing toward camera
        # For detection to work, in camera frame we need:
        #   board X → camera right (+X)
        #   board Y → camera down (+Y)  
        #   board Z → camera forward (+Z)
        yaw_rad = np.radians(board_yaw)
        
        # Board Z points OPPOSITE to yaw direction (away from camera, into the board)
        # This way when transformed to camera frame, it points +Z (forward)
        board_z = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0])
        
        # Board Y points UP in world (will be down in camera Y due to NED)
        board_y = np.array([0, 0, -1])
        
        # Board X = Y × Z
        board_x = np.cross(board_y, board_z)
        board_x /= np.linalg.norm(board_x)
        
        R_WB = np.column_stack([board_x, board_y, board_z])
        
        # Apply small pitch/roll perturbations if any
        if abs(board_pitch) > 0.01 or abs(board_roll) > 0.01:
            R_perturb = RotationUtils.euler_to_rotation(0, board_pitch, board_roll)
            R_WB = R_WB @ R_perturb.T
        
        # Board origin is at top-left, compute from center
        center_in_board = np.array([bw/2, bh/2, 0])
        board_origin_world = board_center_world - R_WB @ center_in_board
        
        T_WB = SE3.from_Rt(R_WB, board_origin_world)
        
        image = self._render_board(T_WB)
        if image is None:
            return None, {}
        
        # RTK antenna position
        p_Rb_B = self.board_config.rtk_antenna_offset_board
        p_Rb_W = SE3.transform_point(T_WB, p_Rb_B) + np.random.randn(3) * rtk_noise_std
        
        # Vehicle RTK
        p_Rv_I = np.array([0.0, 0.0, 0.0])
        p_Rv_W = SE3.transform_point(self.T_WI, p_Rv_I) + np.random.randn(3) * rtk_noise_std
        
        ins_data = INSData(yaw=self.ins_euler["yaw"], pitch=self.ins_euler["pitch"], roll=self.ins_euler["roll"])
        
        data = {
            "board_rtk": RTKMeasurement(position_world=p_Rb_W, std=rtk_noise_std),
            "vehicle_rtk": RTKMeasurement(position_world=p_Rv_W, std=rtk_noise_std),
            "ins_data": ins_data,
            "T_WB_true": T_WB,
        }
        
        return image, data
    
    def _render_board(self, T_WB: np.ndarray) -> Optional[np.ndarray]:
        """Render the board at the given world pose using OpenCV projection."""
        bw, bh = self.board_config.board_width, self.board_config.board_height
        
        # Compute board pose in camera frame: T_CB = T_CW @ T_WB
        T_CW = SE3.inverse(self.T_WC)
        T_CB = SE3.compose(T_CW, T_WB)
        R_CB, t_CB = SE3.to_Rt(T_CB)
        
        # Check if board is in front of camera
        if t_CB[2] <= 0.5:
            return None
        
        # Convert to OpenCV rvec/tvec
        rvec, _ = cv2.Rodrigues(R_CB)
        tvec = t_CB.reshape(3, 1)
        
        # Project board corners
        obj_corners = np.array([
            [0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]
        ], dtype=np.float64)
        
        K = self.camera_matrix
        img_corners, _ = cv2.projectPoints(obj_corners, rvec, tvec, K, np.zeros(5))
        img_corners = img_corners.reshape(-1, 2).astype(np.float32)
        
        w, h = self.image_size
        
        # Check if all corners are in image
        if not all(0 <= c[0] < w and 0 <= c[1] < h for c in img_corners):
            return None
        
        # Warp board image
        bw_px, bh_px = self.board_image.shape[1], self.board_image.shape[0]
        src_corners = np.array([
            [0, 0], [bw_px, 0], [bw_px, bh_px], [0, bh_px]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(src_corners, img_corners)
        if H is None:
            return None
        
        warped_gray = cv2.warpPerspective(self.board_image, H, (w, h), 
                                          flags=cv2.INTER_LINEAR, borderValue=128)
        
        # Create BGR output
        image = np.full((h, w, 3), 128, dtype=np.uint8)
        mask = warped_gray != 128
        for c in range(3):
            image[:,:,c][mask] = warped_gray[mask]
        
        return image


# =============================================================================
# DEMO
# =============================================================================

def run_demo(num_measurements: int = 5, intrinsics_path: str = None, 
             output_path: str = "camera_extrinsics.json"):
    """Demo showing IMU-to-camera calibration with dual RTK and bundle adjustment.
    
    Simulates a real calibration workflow:
    - Technician 1: Measures approximate camera mounting angles with protractor/inclinometer
    - Technician 2: Moves ChArUco board to various positions, records RTK coordinates
    - System: Captures images and runs bundle adjustment
    """
    print("\n" + "="*70)
    print("EXTRINSIC CALIBRATION - Dual RTK + Bundle Adjustment")
    print("="*70)
    
    # Load intrinsics
    if intrinsics_path and os.path.exists(intrinsics_path):
        print(f"\nLoading camera intrinsics from: {intrinsics_path}")
        with open(intrinsics_path) as f:
            intrinsics = json.load(f)
    else:
        print("\nUsing default intrinsics (1920x1080, f=1200)")
        intrinsics = {
            "camera_matrix": [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "image_size": [1920, 1080]
        }
    
    # Ground truth (unknown to calibration - this is what we're trying to find)
    true_camera_to_imu = np.array([0.5, 0.2, -0.3])
    true_camera_angles = {"azimuth": 15.0, "elevation": 5.0, "roll": 0.5}
    imu_position_world = np.array([0.0, 0.0, -1.5])
    ins_euler = {"yaw": 45.0, "pitch": 0.0, "roll": 0.0}
    
    print("\n" + "="*70)
    print("PHASE 1: VEHICLE SETUP")
    print("="*70)
    print("\nVehicle positioned and INS initialized.")
    print(f"  INS reports vehicle heading: {ins_euler['yaw']:.1f}° (NED yaw)")
    print(f"  INS reports pitch: {ins_euler['pitch']:.1f}°, roll: {ins_euler['roll']:.1f}°")
    print(f"  Vehicle RTK antenna height: {abs(imu_position_world[2]):.2f}m above ground")
    
    np.random.seed(42)
    
    board_config = ChArUcoBoardConfig()
    synth = SyntheticTest(board_config, intrinsics, true_camera_to_imu, true_camera_angles,
                          imu_position_world, ins_euler)
    
    # =========================================================================
    # PHASE 2: TECHNICIAN 1 - Camera mounting measurements
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: TECHNICIAN 1 - CAMERA MOUNTING MEASUREMENTS")
    print("="*70)
    
    # Technician 1 measures approximate camera position and angles
    # These have some measurement error (simulated)
    measured_position = true_camera_to_imu + np.array([0.08, -0.05, 0.03])
    measured_azimuth = true_camera_angles["azimuth"] + 2.0
    measured_elevation = true_camera_angles["elevation"] + 1.0
    
    print("\nTechnician 1 measures camera mounting with tape measure and inclinometer:")
    print(f"  Camera position relative to IMU:")
    print(f"    Forward (X): {measured_position[0]*100:.1f} cm")
    print(f"    Right (Y):   {measured_position[1]*100:.1f} cm")
    print(f"    Down (Z):    {measured_position[2]*100:.1f} cm")
    print(f"  Camera pointing angles:")
    print(f"    Azimuth:     {measured_azimuth:.1f}° (0°=forward, positive=right)")
    print(f"    Elevation:   {measured_elevation:.1f}° (positive=down)")
    
    prior = CameraPrior(
        position=measured_position,
        position_std=np.array([0.15, 0.15, 0.15]),  # 15cm uncertainty
        azimuth=measured_azimuth,
        elevation=measured_elevation,
        roll=0.0,
        orientation_std_deg=5.0  # 5° uncertainty
    )
    
    print(f"\n  Measurement uncertainty: ±15cm position, ±5° orientation")
    
    # =========================================================================
    # PHASE 3: TECHNICIAN 2 - Board positioning and data collection
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: TECHNICIAN 2 - BOARD POSITIONING & DATA COLLECTION")
    print("="*70)
    
    print(f"\nChArUco board specifications:")
    print(f"  Size: {board_config.squares_x}x{board_config.squares_y} squares")
    print(f"  Square size: {board_config.square_size*100:.0f}cm")
    print(f"  Total dimensions: {board_config.board_width:.2f}m x {board_config.board_height:.2f}m")
    
    calibrator = ExtrinsicCalibrator(board_config, intrinsics, "camera_front")
    calibrator.set_prior(prior)
    
    # Camera direction for board placement
    R_WC, _ = SE3.to_Rt(synth.T_WC)
    camera_z_world = R_WC @ np.array([0, 0, 1])
    camera_x_world = R_WC @ np.array([1, 0, 0])
    
    print(f"\nTechnician 2 moves board to {num_measurements} positions:")
    print("-" * 60)
    
    for i in range(num_measurements):
        # Place board at varying distances, lateral positions, AND heights
        dist = 2.5 + i * 0.4
        lateral = (i - num_measurements//2) * 0.25
        height_offset = (i - num_measurements//2) * 0.10  # Moderate height variation ±30cm max
        
        # Board center position
        board_center = synth.camera_pos_world + dist * camera_z_world + lateral * camera_x_world
        board_center[2] = synth.camera_pos_world[2] + height_offset  # Vary height
        
        # Board yaw
        to_camera = synth.camera_pos_world - board_center
        to_camera[2] = 0
        board_yaw = np.degrees(np.arctan2(to_camera[1], to_camera[0]))
        
        # Small random tilt
        board_pitch = np.random.uniform(-3, 3)
        board_roll = np.random.uniform(-3, 3)
        
        image, data = synth.generate_measurement(board_center, board_yaw, board_pitch, board_roll, 0.015)
        
        print(f"\n  Position {i+1}:")
        if data.get("board_rtk"):
            rtk_pos = data["board_rtk"].position_world
            print(f"    Board RTK reading: N={rtk_pos[0]:.3f}m, E={rtk_pos[1]:.3f}m, D={rtk_pos[2]:.3f}m")
        if data.get("vehicle_rtk"):
            veh_rtk = data["vehicle_rtk"].position_world
            print(f"    Vehicle RTK reading: N={veh_rtk[0]:.3f}m, E={veh_rtk[1]:.3f}m, D={veh_rtk[2]:.3f}m")
        print(f"    Board distance: ~{dist:.1f}m, lateral offset: {lateral:+.1f}m, height offset: {height_offset:+.2f}m")
        
        if image is not None:
            result = calibrator.add_measurement(
                image, data["ins_data"],
                board_rtk=data["board_rtk"],
                vehicle_rtk=data["vehicle_rtk"]
            )
            if result["success"]:
                print(f"    Image captured: {result['corners_detected']} corners detected")
                print(f"    Initial PnP reprojection error: {result['reproj_error']:.2f} pixels")
            else:
                print(f"    FAILED: {result['error']}")
        else:
            print(f"    Board not visible in camera FOV")
    
    # =========================================================================
    # PHASE 4: BUNDLE ADJUSTMENT OPTIMIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 4: BUNDLE ADJUSTMENT OPTIMIZATION")
    print("="*70)
    
    print("\nRunning bundle adjustment...")
    print("  Optimizing: camera-to-IMU transform (6-DOF)")
    print("  Constraints: reprojection, RTK positions, INS attitude, priors")
    
    result = calibrator.compute_extrinsics()
    
    q = result["quality_metrics"]
    print(f"\n  Optimization {'converged' if q['optimization_converged'] else 'FAILED'}:")
    print(f"    Iterations: completed")
    print(f"    Final cost: {q['final_cost']:.2f}")
    print(f"    Mean reprojection error: {q['mean_reproj_error_px']:.3f} pixels")
    
    # =========================================================================
    # PHASE 5: RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 5: CALIBRATION RESULTS")
    print("="*70)
    
    t_result = np.array(result["imu_to_camera_transform"]["translation_vector"])
    angles_result = result["imu_to_camera_transform"]["euler_angles"]
    
    print("\n  COMPUTED CAMERA-TO-IMU TRANSFORM:")
    print(f"    Position (in IMU frame):")
    print(f"      Forward (X): {t_result[0]*100:+.2f} cm")
    print(f"      Right (Y):   {t_result[1]*100:+.2f} cm")
    print(f"      Down (Z):    {t_result[2]*100:+.2f} cm")
    print(f"    Orientation:")
    print(f"      Azimuth:     {angles_result['azimuth']:+.2f}°")
    print(f"      Elevation:   {angles_result['elevation']:+.2f}°")
    print(f"      Roll:        {angles_result['roll']:+.2f}°")
    
    # Compare with ground truth
    pos_error = np.linalg.norm(t_result - true_camera_to_imu)
    az_error = angles_result["azimuth"] - true_camera_angles["azimuth"]
    el_error = angles_result["elevation"] - true_camera_angles["elevation"]
    
    print("\n  COMPARISON WITH GROUND TRUTH:")
    print(f"    Position error: {pos_error*100:.2f} cm")
    print(f"    Azimuth error:  {az_error:+.2f}°")
    print(f"    Elevation error: {el_error:+.2f}°")
    
    print("\n  IMPROVEMENT OVER MANUAL MEASUREMENT:")
    manual_pos_error = np.linalg.norm(measured_position - true_camera_to_imu)
    manual_az_error = measured_azimuth - true_camera_angles["azimuth"]
    manual_el_error = measured_elevation - true_camera_angles["elevation"]
    print(f"    Position: {manual_pos_error*100:.1f}cm → {pos_error*100:.1f}cm")
    print(f"    Azimuth:  {abs(manual_az_error):.1f}° → {abs(az_error):.1f}°")
    print(f"    Elevation: {abs(manual_el_error):.1f}° → {abs(el_error):.1f}°")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved to: {output_path}")
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extrinsic Camera Calibration - Dual RTK + Bundle Adjustment")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--intrinsics", "-i", help="Path to intrinsics JSON file")
    parser.add_argument("--output", "-o", default="camera_extrinsics.json", help="Output JSON file")
    parser.add_argument("--num", "-n", type=int, default=5, help="Number of board positions")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo(args.num, args.intrinsics, args.output)
    else:
        print("Extrinsic Camera Calibration - Dual RTK + Bundle Adjustment")
        print("\nUsage: python extrinsic_calibration.py --demo [-i INTRINSICS] [-n NUM] [-o OUTPUT]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Intrinsic Camera Calibration Module

Uses real OpenCV ChArUco detection pipeline.
Supports both real camera input and synthetic image generation for testing.
"""

import os

# Set environment variables to avoid Qt threading issues with OpenCV
# CRITICAL: This must be done BEFORE cv2 is imported because Qt initializes at import time
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')  # Use X11 backend instead of Wayland
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_QT', '0')  # Disable Qt priority for video

import numpy as np
import cv2
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ChArUcoBoardConfig:
    """ChArUco board configuration"""
    squares_x: int = 10
    squares_y: int = 10
    square_length: float = 0.11  # meters
    marker_length: float = 0.085  # meters (typically ~77% of square)
    dictionary_id: int = cv2.aruco.DICT_6X6_250
    
    def create_board(self):
        """Create ChArUco board object"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        
        # Check OpenCV version for API compatibility
        cv_version = tuple(map(int, cv2.__version__.split('.')[:2]))
        
        if cv_version >= (4, 7):
            # OpenCV 4.7+ API
            board = cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                aruco_dict
            )
        else:
            # Older OpenCV API
            board = cv2.aruco.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length,
                self.marker_length,
                aruco_dict
            )
        return board, aruco_dict


@dataclass 
class CameraConfig:
    """Camera configuration"""
    image_width: int = 1920
    image_height: int = 1080
    # Ground truth intrinsics (for synthetic image generation)
    fx: float = 1200.0
    fy: float = 1200.0
    cx: float = 960.0
    cy: float = 540.0
    # Distortion coefficients (k1, k2, p1, p2, k3)
    k1: float = -0.15
    k2: float = 0.08
    p1: float = 0.001
    p2: float = -0.001
    k3: float = -0.02
    
    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)


class SyntheticImageGenerator:
    """
    Generates synthetic images of ChArUco board for calibration testing.
    Renders actual ChArUco board image and warps it to simulate camera view.
    """
    
    def __init__(self, board_config: ChArUcoBoardConfig, camera_config: CameraConfig):
        self.board_config = board_config
        self.camera_config = camera_config
        self.board, self.aruco_dict = board_config.create_board()
        
        # Generate high-res board image
        board_pixels_per_meter = 1000  # 1mm resolution
        board_width_px = int(board_config.squares_x * board_config.square_length * board_pixels_per_meter)
        board_height_px = int(board_config.squares_y * board_config.square_length * board_pixels_per_meter)
        
        # Handle different OpenCV versions
        if hasattr(self.board, 'generateImage'):
            # OpenCV 4.7+
            self.board_image = self.board.generateImage((board_width_px, board_height_px))
        else:
            # OpenCV 4.6 and earlier - use draw method
            self.board_image = self.board.draw((board_width_px, board_height_px))
        
        # Board corner coordinates in 3D (meters), board lies on Z=0 plane
        self.board_corners_3d = np.array([
            [0, 0, 0],
            [board_config.squares_x * board_config.square_length, 0, 0],
            [board_config.squares_x * board_config.square_length, 
             board_config.squares_y * board_config.square_length, 0],
            [0, board_config.squares_y * board_config.square_length, 0]
        ], dtype=np.float32)
        
        # Corresponding corners in board image pixels
        self.board_corners_2d = np.array([
            [0, 0],
            [board_width_px, 0],
            [board_width_px, board_height_px],
            [0, board_height_px]
        ], dtype=np.float32)
        
    def generate_board_pose(self, distance: float, angle_x: float, angle_y: float,
                            offset_x: float, offset_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate rotation and translation for board pose.
        
        Args:
            distance: Distance from camera to board center (meters)
            angle_x: Board tilt around X axis (degrees) 
            angle_y: Board tilt around Y axis (degrees)
            offset_x: Horizontal offset in image (normalized -1 to 1)
            offset_y: Vertical offset in image (normalized -1 to 1)
        
        Returns:
            rvec, tvec: Rotation and translation vectors
        """
        rx = np.radians(angle_x)
        ry = np.radians(angle_y)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        R = Ry @ Rx
        rvec, _ = cv2.Rodrigues(R)
        
        # Board center offset
        board_center_x = (self.board_config.squares_x / 2) * self.board_config.square_length
        board_center_y = (self.board_config.squares_y / 2) * self.board_config.square_length
        
        # Calculate translation
        fov_x = 2 * np.arctan(self.camera_config.image_width / (2 * self.camera_config.fx))
        fov_y = 2 * np.arctan(self.camera_config.image_height / (2 * self.camera_config.fy))
        
        tx = offset_x * distance * np.tan(fov_x / 2) - board_center_x
        ty = offset_y * distance * np.tan(fov_y / 2) - board_center_y
        tz = distance
        
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)
        
        return rvec, tvec
    
    def render_image(self, rvec: np.ndarray, tvec: np.ndarray, 
                     add_noise: bool = True) -> np.ndarray:
        """
        Render synthetic camera image of the ChArUco board at given pose.
        
        Uses perspective warp to simulate camera view with distortion.
        """
        # Project 3D board corners to 2D image points
        image_corners, _ = cv2.projectPoints(
            self.board_corners_3d,
            rvec, tvec,
            self.camera_config.camera_matrix,
            self.camera_config.dist_coeffs
        )
        image_corners = image_corners.reshape(-1, 2).astype(np.float32)
        
        # Compute homography from board image to camera image
        H, _ = cv2.findHomography(self.board_corners_2d, image_corners)
        
        # Warp board image to camera view
        output_size = (self.camera_config.image_width, self.camera_config.image_height)
        warped = cv2.warpPerspective(
            self.board_image, H, output_size,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=128  # Gray background
        )
        
        # Convert to 3-channel for realistic image
        if len(warped.shape) == 2:
            warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        
        # Add realistic noise
        if add_noise:
            noise = np.random.normal(0, 3, warped.shape).astype(np.int16)
            warped = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return warped
    
    def generate_diverse_poses(self, num_images: int) -> List[Tuple]:
        """Generate diverse board poses for good calibration coverage"""
        poses = []
        
        # Coverage regions
        offsets = [
            (0.0, 0.0),     # center
            (-0.4, -0.3),   # top-left
            (0.4, -0.3),    # top-right
            (-0.4, 0.3),    # bottom-left
            (0.4, 0.3),     # bottom-right
            (-0.5, 0.0),    # left
            (0.5, 0.0),     # right
            (0.0, -0.4),    # top
            (0.0, 0.4),     # bottom
        ]
        
        distances = [3.0, 4.0, 5.0, 6.0, 7.0]
        
        angles = [
            (0, 0),
            (20, 0), (-20, 0),
            (0, 20), (0, -20),
            (15, 15), (-15, 15),
            (25, 10), (-25, -10),
            (30, 0), (0, 30),
        ]
        
        for i in range(num_images):
            distance = distances[i % len(distances)]
            angle_x, angle_y = angles[i % len(angles)]
            offset_x, offset_y = offsets[i % len(offsets)]
            
            # Add randomness
            distance += np.random.uniform(-0.3, 0.3)
            angle_x += np.random.uniform(-3, 3)
            angle_y += np.random.uniform(-3, 3)
            offset_x += np.random.uniform(-0.05, 0.05)
            offset_y += np.random.uniform(-0.05, 0.05)
            
            poses.append((distance, angle_x, angle_y, offset_x, offset_y))
        
        return poses


class ImageSource:
    """Abstract base for image sources (real camera or synthetic)"""
    
    def get_image(self) -> Optional[np.ndarray]:
        raise NotImplementedError
    
    def release(self):
        pass


class SyntheticImageSource(ImageSource):
    """Generates synthetic calibration images"""
    
    def __init__(self, board_config: ChArUcoBoardConfig, camera_config: CameraConfig,
                 num_images: int = 25):
        self.generator = SyntheticImageGenerator(board_config, camera_config)
        self.poses = self.generator.generate_diverse_poses(num_images)
        self.current_index = 0
        self.camera_config = camera_config  # Store ground truth for validation
        
    def get_image(self) -> Optional[np.ndarray]:
        if self.current_index >= len(self.poses):
            return None
        
        pose = self.poses[self.current_index]
        rvec, tvec = self.generator.generate_board_pose(*pose)
        image = self.generator.render_image(rvec, tvec)
        
        self.current_index += 1
        return image
    
    def get_pose_info(self) -> str:
        """Get info about current pose for logging"""
        if self.current_index == 0 or self.current_index > len(self.poses):
            return ""
        pose = self.poses[self.current_index - 1]
        return f"distance={pose[0]:.1f}m, angles=({pose[1]:.0f}°,{pose[2]:.0f}°), offset=({pose[3]:.2f},{pose[4]:.2f})"


class RealCameraSource(ImageSource):
    """Captures from real camera via OpenCV VideoCapture"""

    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")

    def get_image(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


class NetworkCameraSource(ImageSource):
    """Captures from network camera via H265/RTP multicast stream using GStreamer"""

    def __init__(self, ip: str = "10.100.102.222", api_port: int = 5000,
                 multicast_host: str = "239.255.0.1", stream_port: int = 5010,
                 width: int = 1920, height: int = 1080, timeout: float = 5.0):
        self.ip = ip
        self.api_port = api_port
        self.multicast_host = multicast_host
        self.stream_port = stream_port
        self.width = width
        self.height = height
        self.timeout = timeout
        self.cap: Optional[cv2.VideoCapture] = None
        self._stream_started = False

        # Try to import requests for API calls
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise RuntimeError("'requests' module required. Install with: pip install requests")

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:{self.api_port}"

    def _start_stream(self) -> bool:
        """Start camera streaming via API"""
        try:
            payload = {
                "host": self.multicast_host,
                "port": self.stream_port
            }
            response = self.requests.post(
                f"{self.base_url}/api/stream/start",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            self._stream_started = response.status_code == 200
            return self._stream_started
        except Exception as e:
            print(f"Failed to start stream: {e}")
            return False

    def _stop_stream(self):
        """Stop camera streaming via API"""
        try:
            self.requests.post(
                f"{self.base_url}/api/stream/stop",
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
        except Exception:
            pass
        self._stream_started = False

    def connect(self) -> bool:
        """Connect to the network camera"""
        import time

        # Start the camera stream via API
        print(f"Connecting to camera at {self.ip}...")
        if not self._start_stream():
            print("Failed to start camera stream via API")
            return False

        print("Stream started, connecting via GStreamer...")
        time.sleep(1.0)  # Give stream time to start

        # GStreamer pipeline for OpenCV
        pipeline = (
            f"udpsrc address={self.multicast_host} port={self.stream_port} "
            f'caps="application/x-rtp,media=video,encoding-name=H265,payload=96" ! '
            f"rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=1"
        )

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print("Failed to open GStreamer pipeline")
            self._stop_stream()
            return False

        print("Connected to network camera")
        return True

    def get_image(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._stream_started:
            self._stop_stream()


def get_opencv_version() -> Tuple[int, int]:
    """Parse OpenCV version safely, returning (major, minor) tuple."""
    try:
        version_str = cv2.__version__
        # Handle versions like "4.8.1" or "4.8.1-dev" or "4.8.1.78"
        parts = version_str.split('.')
        major = int(parts[0]) if parts else 4
        minor = int(parts[1].split('-')[0]) if len(parts) > 1 else 0
        return (major, minor)
    except (ValueError, IndexError, AttributeError):
        # Default to older API if version parsing fails
        return (4, 0)


class ChArUcoDetector:
    """Detects ChArUco corners in images using OpenCV"""

    def __init__(self, board_config: ChArUcoBoardConfig):
        self.board_config = board_config
        self.board, self.aruco_dict = board_config.create_board()

        # Check OpenCV version for API compatibility
        cv_version = get_opencv_version()
        self.use_new_api = cv_version >= (4, 7)

        # Create detector parameters - handle different OpenCV versions
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()

        if self.use_new_api:
            # OpenCV 4.7+ uses CharucoDetector
            try:
                self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
            except Exception as e:
                # Fall back to old API if CharucoDetector fails
                print(f"Warning: CharucoDetector init failed ({e}), using legacy API")
                self.use_new_api = False
                self._init_legacy_detector()
        else:
            self._init_legacy_detector()

    def _init_legacy_detector(self):
        """Initialize legacy ArUco detector for older OpenCV versions."""
        # Check if ArucoDetector exists (4.5-4.6) or use detectMarkers directly
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        else:
            self.aruco_detector = None  # Will use cv2.aruco.detectMarkers directly
        
    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Detect ChArUco corners in image.

        Returns:
            charuco_corners: Detected corner positions (N, 1, 2) or None
            charuco_ids: Corner IDs (N, 1) or None
            annotated_image: Image with detections drawn
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create annotated image
        annotated = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Initialize variables
        marker_corners = None
        marker_ids = None
        charuco_corners = None
        charuco_ids = None

        try:
            if self.use_new_api:
                # OpenCV 4.7+ API
                charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            else:
                # Older OpenCV API
                if self.aruco_detector is not None:
                    marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
                else:
                    # Very old OpenCV (< 4.5) - use function directly
                    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict, parameters=self.detector_params
                    )

                if marker_ids is None or len(marker_ids) == 0:
                    return None, None, annotated

                # Interpolate ChArUco corners
                num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )

            # Draw detected markers if available
            if marker_corners is not None and marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)

            # Check if we have valid charuco corners
            if charuco_corners is None or charuco_ids is None:
                return None, None, annotated

            # Handle case where corners is empty array
            if not isinstance(charuco_corners, np.ndarray) or charuco_corners.size == 0:
                return None, None, annotated

            if len(charuco_corners) < 4:
                return None, None, annotated

            # Ensure corners are in correct format for cornerSubPix (float32)
            if charuco_corners.dtype != np.float32:
                charuco_corners = charuco_corners.astype(np.float32)

            # Refine corner positions to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            charuco_corners = cv2.cornerSubPix(
                gray, charuco_corners, (5, 5), (-1, -1), criteria
            )

            # Draw ChArUco corners
            cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)

        except Exception as e:
            # If detection fails for any reason, return safely
            import sys
            print(f"Warning: ChArUco detection error: {e}", file=sys.stderr)
            return None, None, annotated

        return charuco_corners, charuco_ids, annotated


class IntrinsicCalibrator:
    """Performs intrinsic calibration from ChArUco detections"""
    
    def __init__(self, board_config: ChArUcoBoardConfig, camera_id: str = "camera_1"):
        self.board_config = board_config
        self.board, _ = board_config.create_board()
        self.camera_id = camera_id
        
        # Collected calibration data
        self.all_charuco_corners: List[np.ndarray] = []
        self.all_charuco_ids: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None
        
        # Results
        self.calibration_result: Optional[dict] = None
        
    def add_detection(self, corners: np.ndarray, ids: np.ndarray, 
                      image_size: Tuple[int, int]) -> bool:
        """
        Add a detection to the calibration dataset.
        
        Returns:
            True if detection was accepted, False if rejected
        """
        if corners is None or ids is None:
            return False
        
        if len(corners) < 6:  # Need minimum corners for good calibration
            return False
        
        if self.image_size is None:
            self.image_size = image_size
        elif self.image_size != image_size:
            print(f"Warning: Image size mismatch. Expected {self.image_size}, got {image_size}")
            return False
        
        self.all_charuco_corners.append(corners)
        self.all_charuco_ids.append(ids)
        return True
    
    def get_num_images(self) -> int:
        return len(self.all_charuco_corners)
    
    def calibrate(self, min_images: int = 10) -> Optional[dict]:
        """
        Run camera calibration using collected ChArUco detections.
        
        Returns:
            Dictionary with calibration results, or None if failed
        """
        if len(self.all_charuco_corners) < min_images:
            print(f"Error: Need at least {min_images} images, have {len(self.all_charuco_corners)}")
            return None
        
        if self.image_size is None:
            print("Error: No image size set")
            return None
        
        print(f"\nRunning OpenCV ChArUco calibration with {len(self.all_charuco_corners)} images...")
        
        # Build object points and image points arrays
        all_obj_points = []
        all_img_points = []
        
        # Get board corners - handle different OpenCV versions
        if hasattr(self.board, 'getChessboardCorners'):
            board_corners = self.board.getChessboardCorners()
        else:
            board_corners = self.board.chessboardCorners
        
        for i in range(len(self.all_charuco_corners)):
            ids = self.all_charuco_ids[i].flatten()
            obj_pts = board_corners[ids]
            img_pts = self.all_charuco_corners[i].reshape(-1, 2)
            
            all_obj_points.append(obj_pts.astype(np.float32))
            all_img_points.append(img_pts.astype(np.float32))
        
        # Calibration flags
        flags = (
            cv2.CALIB_RATIONAL_MODEL +
            cv2.CALIB_FIX_K4 + 
            cv2.CALIB_FIX_K5 + 
            cv2.CALIB_FIX_K6
        )
        
        # Run calibration using standard calibrateCamera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points,
            all_img_points,
            self.image_size,
            None, None,
            flags=flags
        )
        
        if ret is None or ret > 10.0:  # Sanity check
            print(f"Calibration failed or very poor quality (RMS={ret})")
            return None
        
        # Calculate per-image reprojection errors
        per_image_errors = self._calculate_per_image_errors(
            camera_matrix, dist_coeffs, rvecs, tvecs
        )
        
        self.calibration_result = {
            "camera_id": self.camera_id,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.flatten().tolist()[:5],  # k1,k2,p1,p2,k3
            "image_size": list(self.image_size),
            "rms_error": float(ret),
            "per_image_errors": per_image_errors,
            "num_images": len(self.all_charuco_corners),
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.calibration_result
    
    def _calculate_per_image_errors(self, camera_matrix, dist_coeffs, rvecs, tvecs) -> List[float]:
        """Calculate reprojection error for each calibration image"""
        errors = []
        
        # Get board corners - handle different OpenCV versions
        if hasattr(self.board, 'getChessboardCorners'):
            board_corners = self.board.getChessboardCorners()
        else:
            board_corners = self.board.chessboardCorners
        
        for i in range(len(self.all_charuco_corners)):
            # Get object points for detected corners
            ids = self.all_charuco_ids[i].flatten()
            obj_points = board_corners[ids]
            
            # Project to image
            projected, _ = cv2.projectPoints(
                obj_points, rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )
            
            # Calculate error
            detected = self.all_charuco_corners[i].reshape(-1, 2)
            projected = projected.reshape(-1, 2)
            diff = detected - projected
            error = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            errors.append(float(error))
        
        return errors
    
    def print_results(self, ground_truth: Optional[CameraConfig] = None):
        """Print calibration results"""
        if self.calibration_result is None:
            print("No calibration results available")
            return
            
        r = self.calibration_result
        cm = np.array(r["camera_matrix"])
        dc = np.array(r["distortion_coefficients"])
        
        print("\n" + "="*60)
        print("INTRINSIC CALIBRATION RESULTS")
        print("="*60)
        print(f"Camera ID: {r['camera_id']}")
        print(f"Image size: {r['image_size'][0]} x {r['image_size'][1]}")
        print(f"Number of images: {r['num_images']}")
        print(f"RMS Reprojection Error: {r['rms_error']:.4f} pixels")
        
        print("\nCamera Matrix:")
        print(f"  fx = {cm[0,0]:.2f}")
        print(f"  fy = {cm[1,1]:.2f}")
        print(f"  cx = {cm[0,2]:.2f}")
        print(f"  cy = {cm[1,2]:.2f}")
        
        print("\nDistortion Coefficients:")
        print(f"  k1 = {dc[0]:.6f}")
        print(f"  k2 = {dc[1]:.6f}")
        print(f"  p1 = {dc[2]:.6f}")
        print(f"  p2 = {dc[3]:.6f}")
        print(f"  k3 = {dc[4]:.6f}")
        
        # Quality assessment
        print("\nQuality Assessment:")
        if r['rms_error'] < 0.5:
            print(f"  ✓ GOOD - RMS error {r['rms_error']:.3f} < 0.5 pixels")
        elif r['rms_error'] < 1.0:
            print(f"  ~ ACCEPTABLE - RMS error {r['rms_error']:.3f} < 1.0 pixels")
        else:
            print(f"  ✗ POOR - RMS error {r['rms_error']:.3f} >= 1.0 pixels")
        
        # Ground truth comparison
        if ground_truth is not None:
            print("\n" + "-"*60)
            print("GROUND TRUTH COMPARISON (synthetic data)")
            print("-"*60)
            gt_cm = ground_truth.camera_matrix
            gt_dc = ground_truth.dist_coeffs
            
            print("\nCamera Matrix Errors:")
            print(f"  fx: {cm[0,0]:.2f} vs {gt_cm[0,0]:.2f} (error: {cm[0,0]-gt_cm[0,0]:+.2f})")
            print(f"  fy: {cm[1,1]:.2f} vs {gt_cm[1,1]:.2f} (error: {cm[1,1]-gt_cm[1,1]:+.2f})")
            print(f"  cx: {cm[0,2]:.2f} vs {gt_cm[0,2]:.2f} (error: {cm[0,2]-gt_cm[0,2]:+.2f})")
            print(f"  cy: {cm[1,2]:.2f} vs {gt_cm[1,2]:.2f} (error: {cm[1,2]-gt_cm[1,2]:+.2f})")
            
            print("\nDistortion Coefficient Errors:")
            for j, name in enumerate(['k1', 'k2', 'p1', 'p2', 'k3']):
                print(f"  {name}: {dc[j]:.6f} vs {gt_dc[j]:.6f} (error: {dc[j]-gt_dc[j]:+.6f})")
    
    def save_to_json(self, output_path: str):
        """Save calibration results to JSON file"""
        if self.calibration_result is None:
            raise ValueError("No calibration results to save")
        
        output = {
            "camera_id": self.calibration_result["camera_id"],
            "camera_matrix": self.calibration_result["camera_matrix"],
            "distortion_coefficients": self.calibration_result["distortion_coefficients"],
            "image_size": self.calibration_result["image_size"],
            "rms_error": round(self.calibration_result["rms_error"], 4),
            "calibration_date": self.calibration_result["calibration_date"]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nCalibration saved to: {output_path}")


def run_calibration(image_source: ImageSource, 
                    board_config: ChArUcoBoardConfig,
                    camera_id: str,
                    output_path: str,
                    num_images: int = 25,
                    save_images: bool = False,
                    output_dir: Optional[str] = None,
                    ground_truth: Optional[CameraConfig] = None) -> Optional[dict]:
    """
    Run the full intrinsic calibration pipeline.
    
    This is the main calibration loop - same code path for real or synthetic images.
    """
    print("="*60)
    print("INTRINSIC CALIBRATION")
    print("="*60)
    print(f"Camera ID: {camera_id}")
    print(f"Target images: {num_images}")
    print(f"Board: {board_config.squares_x}x{board_config.squares_y} ChArUco, "
          f"{board_config.square_length*100:.0f}cm squares")
    
    # Initialize detector and calibrator
    detector = ChArUcoDetector(board_config)
    calibrator = IntrinsicCalibrator(board_config, camera_id)
    
    if save_images and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nCapturing calibration images...")
    print("-"*60)
    
    image_count = 0
    accepted_count = 0
    
    while accepted_count < num_images:
        # Get image from source
        image = image_source.get_image()
        if image is None:
            print("No more images available from source")
            break
        
        image_count += 1
        image_size = (image.shape[1], image.shape[0])
        
        # Detect ChArUco corners
        corners, ids, annotated = detector.detect(image)
        
        # Get pose info if available (for synthetic source)
        pose_info = ""
        if hasattr(image_source, 'get_pose_info'):
            pose_info = image_source.get_pose_info()
        
        # Try to add detection
        if corners is not None and calibrator.add_detection(corners, ids, image_size):
            accepted_count += 1
            status = "✓ ACCEPTED"
            corner_count = len(corners)
            
            # Save image if requested
            if save_images and output_dir:
                cv2.imwrite(f"{output_dir}/calib_{accepted_count:02d}.png", annotated)
        else:
            status = "✗ REJECTED"
            corner_count = len(corners) if corners is not None else 0
        
        print(f"  Image {image_count:2d}: {status} ({corner_count:2d} corners) {pose_info}")
    
    print("-"*60)
    print(f"Collected {accepted_count} valid images")
    
    # Run calibration
    result = calibrator.calibrate(min_images=10)
    
    if result is None:
        print("Calibration failed!")
        return None
    
    # Print results
    calibrator.print_results(ground_truth=ground_truth)
    
    # Save to JSON
    calibrator.save_to_json(output_path)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Intrinsic Camera Calibration using ChArUco board"
    )
    parser.add_argument(
        "--camera-id", 
        default="camera_1",
        help="Camera identifier (default: camera_1)"
    )
    parser.add_argument(
        "--output", "-o",
        default="camera_intrinsics.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=25,
        help="Number of calibration images to capture"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic images instead of real camera"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index for real camera mode"
    )
    parser.add_argument(
        "--network-camera",
        action="store_true",
        help="Use network camera (H265/RTP stream) instead of local camera"
    )
    parser.add_argument(
        "--camera-ip",
        default="10.100.102.222",
        help="Network camera IP address (default: 10.100.102.222)"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=5000,
        help="Network camera API port (default: 5000)"
    )
    parser.add_argument(
        "--multicast",
        default="239.255.0.1",
        help="Multicast address for stream (default: 239.255.0.1)"
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=5010,
        help="Stream port (default: 5010)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save captured calibration images"
    )
    parser.add_argument(
        "--image-dir",
        default="calibration_images",
        help="Directory to save calibration images"
    )
    # Board configuration
    parser.add_argument("--board-squares-x", type=int, default=10)
    parser.add_argument("--board-squares-y", type=int, default=10)
    parser.add_argument("--board-square-size", type=float, default=0.11,
                        help="Square size in meters")
    
    args = parser.parse_args()
    
    # Board configuration
    board_config = ChArUcoBoardConfig(
        squares_x=args.board_squares_x,
        squares_y=args.board_squares_y,
        square_length=args.board_square_size,
        marker_length=args.board_square_size * 0.77,
    )
    
    # Camera configuration (ground truth for synthetic mode)
    camera_config = CameraConfig(
        image_width=1920,
        image_height=1080,
        fx=1250.0,
        fy=1248.0,
        cx=965.0,
        cy=545.0,
        k1=-0.12,
        k2=0.05,
        p1=0.0008,
        p2=-0.0005,
        k3=-0.015
    )
    
    # Create image source
    if args.synthetic:
        print("Mode: SYNTHETIC (fake camera images)")
        image_source = SyntheticImageSource(board_config, camera_config, args.num_images + 5)
        ground_truth = camera_config
    elif args.network_camera:
        print("Mode: NETWORK CAMERA (H265/RTP stream)")
        print(f"  Camera IP: {args.camera_ip}:{args.api_port}")
        print(f"  Stream: {args.multicast}:{args.stream_port}")
        try:
            image_source = NetworkCameraSource(
                ip=args.camera_ip,
                api_port=args.api_port,
                multicast_host=args.multicast,
                stream_port=args.stream_port
            )
            if not image_source.connect():
                print("Error: Failed to connect to network camera")
                print("Run 'python3 camera_streaming.py test' to diagnose connectivity")
                return 1
            ground_truth = None
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1
    else:
        print("Mode: REAL CAMERA (local device)")
        try:
            image_source = RealCameraSource(args.camera_index)
            ground_truth = None
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Use --synthetic flag for testing without a real camera")
            print("Use --network-camera flag for network camera streaming")
            return 1
    
    try:
        result = run_calibration(
            image_source=image_source,
            board_config=board_config,
            camera_id=args.camera_id,
            output_path=args.output,
            num_images=args.num_images,
            save_images=args.save_images,
            output_dir=args.image_dir,
            ground_truth=ground_truth
        )
        
        if result:
            print("\n" + "="*60)
            print("CALIBRATION COMPLETE")
            print("="*60)
            return 0
        else:
            return 1
            
    finally:
        image_source.release()


if __name__ == "__main__":
    exit(main())

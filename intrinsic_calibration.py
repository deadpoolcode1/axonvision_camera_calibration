#!/usr/bin/env python3
"""
Intrinsic Camera Calibration Module

Uses real OpenCV ChArUco detection pipeline.
Supports both real camera input and synthetic image generation for testing.
"""

import os

# Set environment variables to avoid Qt/GTK threading issues with OpenCV
# CRITICAL: This must be done BEFORE cv2 is imported because Qt/GTK initialize at import time
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11 backend instead of Wayland
os.environ['OPENCV_VIDEOIO_PRIORITY_QT'] = '0'  # Disable Qt priority for video
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''  # Prevent Qt plugin conflicts
os.environ['GDK_BACKEND'] = 'x11'  # Force GTK to use X11

import numpy as np
import cv2
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    import io
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    inch = 72  # Fallback: 1 inch = 72 points (standard PDF unit)
    RLImage = None  # Fallback for type annotation


# Supported ArUco dictionaries with their names
ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


@dataclass
class ChArUcoBoardConfig:
    """ChArUco board configuration"""
    squares_x: int = 8
    squares_y: int = 8
    square_length: float = 0.11  # meters
    marker_length: float = 0.075  # meters (75mm marker size)
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
                 bitrate: int = 4000000, width: int = 1920, height: int = 1080,
                 timeout: float = 5.0):
        self.ip = ip
        self.api_port = api_port
        self.multicast_host = multicast_host
        self.stream_port = stream_port
        self.bitrate = bitrate
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
                "port": self.stream_port,
                "bitrate": self.bitrate
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

        # First, try to stop any existing stream (in case previous session left it running)
        print(f"Connecting to camera at {self.ip}...")
        self._stop_stream()
        time.sleep(0.5)  # Brief pause after stopping

        # Start the camera stream via API
        if not self._start_stream():
            print("Failed to start camera stream via API")
            return False

        print("Stream started, connecting via GStreamer...")
        time.sleep(1.5)  # Give stream time to start (increased for reliability)

        # GStreamer pipeline for OpenCV
        pipeline = (
            f"udpsrc multicast-group={self.multicast_host} port={self.stream_port} ! "
            f"application/x-rtp,payload=96 ! "
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


class DirectoryImageSource(ImageSource):
    """Loads images from a directory for debug/replay mode"""

    def __init__(self, directory: str, extensions: List[str] = None, prefer_raw: bool = True):
        """
        Load images from directory.

        Args:
            directory: Path to directory containing calibration images
            extensions: List of image extensions to load (default: png, jpg, jpeg)
            prefer_raw: If True, prefer raw_*.png over annotated_*.png files
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise RuntimeError(f"Directory not found: {directory}")

        if extensions is None:
            extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']

        # Find all image files sorted by name
        self.image_files: List[Path] = []
        for ext in extensions:
            self.image_files.extend(self.directory.glob(f"*.{ext}"))
            self.image_files.extend(self.directory.glob(f"*.{ext.upper()}"))

        # Remove duplicates and sort
        self.image_files = sorted(set(self.image_files), key=lambda x: x.name)

        # Filter to prefer raw images over annotated when both exist
        if prefer_raw:
            raw_files = [f for f in self.image_files if f.name.startswith('raw_')]
            if raw_files:
                # Use only raw files if they exist (standard save format)
                self.image_files = raw_files
                print(f"Using {len(self.image_files)} raw images (ignoring annotated versions)")
            else:
                # Filter out annotated files if raw don't exist but annotated do
                non_annotated = [f for f in self.image_files if not f.name.startswith('annotated_')]
                if non_annotated and len(non_annotated) < len(self.image_files):
                    self.image_files = non_annotated

        self.current_index = 0

        if not self.image_files:
            raise RuntimeError(f"No image files found in: {directory}")

        print(f"Found {len(self.image_files)} images in {directory}")

    def get_image(self) -> Optional[np.ndarray]:
        if self.current_index >= len(self.image_files):
            return None

        image_path = self.image_files[self.current_index]
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: Failed to load {image_path}")
            self.current_index += 1
            return self.get_image()  # Try next image

        self.current_index += 1
        return image

    def get_current_filename(self) -> str:
        """Get filename of last returned image"""
        if self.current_index == 0 or self.current_index > len(self.image_files):
            return ""
        return self.image_files[self.current_index - 1].name

    def get_total_images(self) -> int:
        return len(self.image_files)


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
        # Force using function-based API (cv2.aruco.detectMarkers) instead of
        # class-based ArucoDetector which has bugs in some OpenCV 4.6 builds
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
                    # Use function-based API without explicit parameters
                    # (passing DetectorParameters crashes on OpenCV 4.6 with certain builds)
                    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict
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


class CalibrationReportGenerator:
    """Generates PDF reports for camera calibration results"""

    def __init__(self, output_path: str):
        if not HAS_REPORTLAB:
            raise RuntimeError("reportlab is required for PDF generation. Install with: pip install reportlab")
        self.output_path = output_path
        self.images_data: List[Dict[str, Any]] = []
        self.styles = getSampleStyleSheet()

        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12
        )

    def add_image(self, image: np.ndarray, image_name: str, num_corners: int,
                  reprojection_error: Optional[float] = None,
                  corners: Optional[np.ndarray] = None,
                  ids: Optional[np.ndarray] = None):
        """Add an image with detection info for the report"""
        self.images_data.append({
            'image': image.copy(),
            'name': image_name,
            'num_corners': num_corners,
            'reprojection_error': reprojection_error,
            'corners': corners.copy() if corners is not None else None,
            'ids': ids.copy() if ids is not None else None
        })

    def _image_to_reportlab(self, cv_image: np.ndarray, max_width: float = 5*inch,
                            max_height: float = 3.5*inch) -> RLImage:
        """Convert OpenCV image to ReportLab Image object"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Resize if needed to fit page
        h, w = rgb_image.shape[:2]
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode to PNG bytes
        success, buffer = cv2.imencode('.png', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode image")

        # Create ReportLab Image
        img_buffer = io.BytesIO(buffer.tobytes())
        return RLImage(img_buffer, width=rgb_image.shape[1], height=rgb_image.shape[0])

    def generate_report(self, calibration_result: dict, board_config: 'ChArUcoBoardConfig'):
        """Generate the PDF report"""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        story = []

        # Title
        story.append(Paragraph("Camera Calibration Report", self.title_style))
        story.append(Spacer(1, 12))

        # Summary section
        story.append(Paragraph("Calibration Summary", self.heading_style))

        # Summary table
        cm = np.array(calibration_result["camera_matrix"])
        dc = np.array(calibration_result["distortion_coefficients"])

        summary_data = [
            ["Camera ID", calibration_result["camera_id"]],
            ["Calibration Date", calibration_result["calibration_date"]],
            ["Image Size", f"{calibration_result['image_size'][0]} x {calibration_result['image_size'][1]}"],
            ["Number of Images", str(calibration_result["num_images"])],
            ["RMS Error", f"{calibration_result['rms_error']:.4f} pixels"],
            ["Board Configuration", f"{board_config.squares_x}x{board_config.squares_y} squares"],
            ["Square Size", f"{board_config.square_length*100:.1f} cm"],
            ["Marker Size", f"{board_config.marker_length*100:.1f} cm"],
        ]

        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Camera Matrix section
        story.append(Paragraph("Camera Intrinsics", self.heading_style))
        intrinsics_data = [
            ["Parameter", "Value"],
            ["Focal Length X (fx)", f"{cm[0,0]:.2f} pixels"],
            ["Focal Length Y (fy)", f"{cm[1,1]:.2f} pixels"],
            ["Principal Point X (cx)", f"{cm[0,2]:.2f} pixels"],
            ["Principal Point Y (cy)", f"{cm[1,2]:.2f} pixels"],
            ["Distortion k1", f"{dc[0]:.6f}"],
            ["Distortion k2", f"{dc[1]:.6f}"],
            ["Distortion p1", f"{dc[2]:.6f}"],
            ["Distortion p2", f"{dc[3]:.6f}"],
            ["Distortion k3", f"{dc[4]:.6f}"],
        ]

        intrinsics_table = Table(intrinsics_data, colWidths=[2.5*inch, 3*inch])
        intrinsics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(intrinsics_table)
        story.append(Spacer(1, 20))

        # Per-image errors table
        story.append(Paragraph("Per-Image Reprojection Errors", self.heading_style))
        errors_data = [["Image #", "Corners Detected", "Reprojection Error (px)"]]

        for i, img_data in enumerate(self.images_data):
            error_str = f"{img_data['reprojection_error']:.4f}" if img_data['reprojection_error'] else "N/A"
            errors_data.append([
                str(i + 1),
                str(img_data['num_corners']),
                error_str
            ])

        errors_table = Table(errors_data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
        errors_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(errors_table)
        story.append(PageBreak())

        # Individual image pages
        story.append(Paragraph("Calibration Images with Detections", self.heading_style))
        story.append(Spacer(1, 12))

        for i, img_data in enumerate(self.images_data):
            # Image title
            error_str = f"{img_data['reprojection_error']:.4f} px" if img_data['reprojection_error'] else "N/A"
            img_title = f"Image {i+1}: {img_data['name']} - {img_data['num_corners']} corners, Error: {error_str}"
            story.append(Paragraph(img_title, self.styles['Normal']))
            story.append(Spacer(1, 6))

            # Add image
            rl_image = self._image_to_reportlab(img_data['image'])
            story.append(rl_image)
            story.append(Spacer(1, 20))

            # Page break every 2 images (except last)
            if (i + 1) % 2 == 0 and i < len(self.images_data) - 1:
                story.append(PageBreak())

        # Build PDF
        doc.build(story)
        print(f"\nPDF report saved to: {self.output_path}")


def run_calibration(image_source: ImageSource,
                    board_config: ChArUcoBoardConfig,
                    camera_id: str,
                    output_path: str,
                    num_images: int = 25,
                    save_images: bool = False,
                    output_dir: Optional[str] = None,
                    ground_truth: Optional[CameraConfig] = None,
                    generate_pdf: bool = False,
                    pdf_path: Optional[str] = None) -> Optional[dict]:
    """
    Run the full intrinsic calibration pipeline.

    This is the main calibration loop - same code path for real or synthetic images.

    Args:
        image_source: Source of images (camera, directory, or synthetic)
        board_config: ChArUco board configuration
        camera_id: Identifier for the camera
        output_path: Path for JSON calibration output
        num_images: Target number of calibration images
        save_images: Whether to save captured images for later replay
        output_dir: Directory to save images
        ground_truth: Ground truth camera config (for synthetic validation)
        generate_pdf: Whether to generate a PDF report
        pdf_path: Path for PDF report output
    """
    print("="*60)
    print("INTRINSIC CALIBRATION")
    print("="*60)
    print(f"Camera ID: {camera_id}")
    print(f"Target images: {num_images}")
    print(f"Board: {board_config.squares_x}x{board_config.squares_y} ChArUco, "
          f"{board_config.square_length*100:.1f}cm squares, "
          f"{board_config.marker_length*100:.1f}cm markers")

    # Get dictionary name for display
    dict_name = "UNKNOWN"
    for name, did in ARUCO_DICTIONARIES.items():
        if did == board_config.dictionary_id:
            dict_name = name
            break
    print(f"Dictionary: {dict_name}")

    # Initialize detector and calibrator
    detector = ChArUcoDetector(board_config)
    calibrator = IntrinsicCalibrator(board_config, camera_id)

    # Setup image saving
    if save_images and output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Save board configuration metadata for replay
        metadata = {
            "board_config": {
                "squares_x": board_config.squares_x,
                "squares_y": board_config.squares_y,
                "square_length": board_config.square_length,
                "marker_length": board_config.marker_length,
                "dictionary_id": board_config.dictionary_id,
                "dictionary_name": dict_name
            },
            "camera_id": camera_id,
            "capture_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "images": []
        }

    # Setup PDF report generator
    report_generator = None
    if generate_pdf:
        if not HAS_REPORTLAB:
            print("Warning: reportlab not installed, skipping PDF generation")
            print("  Install with: pip install reportlab")
        else:
            if pdf_path is None:
                pdf_path = output_path.replace('.json', '_report.pdf')
            report_generator = CalibrationReportGenerator(pdf_path)

    print(f"\nCapturing calibration images...")
    print("-"*60)

    image_count = 0
    accepted_count = 0
    accepted_images = []  # Store images and their data for PDF

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

        # Get pose/filename info if available
        pose_info = ""
        image_name = f"image_{image_count:02d}"
        if hasattr(image_source, 'get_pose_info'):
            pose_info = image_source.get_pose_info()
        if hasattr(image_source, 'get_current_filename'):
            image_name = image_source.get_current_filename() or image_name

        # Try to add detection
        if corners is not None and calibrator.add_detection(corners, ids, image_size):
            accepted_count += 1
            status = "✓ ACCEPTED"
            corner_count = len(corners)

            # Store for later processing (PDF report, etc.)
            accepted_images.append({
                'image': image.copy(),
                'annotated': annotated.copy(),
                'corners': corners.copy(),
                'ids': ids.copy(),
                'name': image_name
            })

            # Save image if requested
            if save_images and output_dir:
                raw_path = f"{output_dir}/raw_{accepted_count:02d}.png"
                annotated_path = f"{output_dir}/annotated_{accepted_count:02d}.png"
                cv2.imwrite(raw_path, image)
                cv2.imwrite(annotated_path, annotated)
                metadata["images"].append({
                    "index": accepted_count,
                    "raw_file": f"raw_{accepted_count:02d}.png",
                    "annotated_file": f"annotated_{accepted_count:02d}.png",
                    "num_corners": corner_count
                })
        else:
            status = "✗ REJECTED"
            corner_count = len(corners) if corners is not None else 0

        print(f"  Image {image_count:2d}: {status} ({corner_count:2d} corners) {pose_info}")

    print("-"*60)
    print(f"Collected {accepted_count} valid images")

    # Save metadata if saving images
    if save_images and output_dir:
        metadata_path = Path(output_dir) / "calibration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    # Run calibration
    result = calibrator.calibrate(min_images=10)

    if result is None:
        print("Calibration failed!")
        return None

    # Add per-image data to report and update with reprojection errors
    if report_generator and result:
        per_image_errors = result.get('per_image_errors', [])
        for i, img_data in enumerate(accepted_images):
            error = per_image_errors[i] if i < len(per_image_errors) else None
            report_generator.add_image(
                image=img_data['annotated'],
                image_name=img_data['name'],
                num_corners=len(img_data['corners']),
                reprojection_error=error,
                corners=img_data['corners'],
                ids=img_data['ids']
            )

    # Print results
    calibrator.print_results(ground_truth=ground_truth)

    # Save to JSON
    calibrator.save_to_json(output_path)

    # Generate PDF report
    if report_generator and result:
        report_generator.generate_report(result, board_config)

    return result


def main():
    # Build dictionary choices string for help text
    dict_choices = list(ARUCO_DICTIONARIES.keys())

    parser = argparse.ArgumentParser(
        description="Intrinsic Camera Calibration using ChArUco board",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic images (no camera needed)
  python intrinsic_calibration.py --synthetic --generate-pdf

  # Run from saved images directory (debug mode)
  python intrinsic_calibration.py --from-directory ./calibration_images

  # Save images during calibration for later replay
  python intrinsic_calibration.py --save-images --image-dir ./my_images

  # Custom board configuration (sizes in cm)
  python intrinsic_calibration.py --squares-x 8 --squares-y 8 --square-size 11 --marker-size 7.5 --dictionary DICT_6X6_250
"""
    )

    # Output options
    parser.add_argument(
        "--camera-id",
        default="camera_1",
        help="Camera identifier (default: camera_1). Output will be saved to camera_intrinsic/camera_intrinsics_<camera_id>.json"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path (default: camera_intrinsic/camera_intrinsics_<camera_id>.json)"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=25,
        help="Number of calibration images to capture (default: 25)"
    )

    # Image source options (mutually exclusive)
    source_group = parser.add_argument_group('Image Source Options')
    source_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic images instead of real camera"
    )
    source_group.add_argument(
        "--from-directory",
        type=str,
        metavar="DIR",
        help="Load images from directory (debug/replay mode)"
    )
    source_group.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index for real camera mode (default: 0)"
    )
    source_group.add_argument(
        "--network-camera",
        action="store_true",
        help="Use network camera (H265/RTP stream) instead of local camera"
    )

    # Network camera options
    network_group = parser.add_argument_group('Network Camera Options')
    network_group.add_argument(
        "--camera-ip",
        default="10.100.102.222",
        help="Network camera IP address (default: 10.100.102.222)"
    )
    network_group.add_argument(
        "--api-port",
        type=int,
        default=5000,
        help="Network camera API port (default: 5000)"
    )
    network_group.add_argument(
        "--multicast",
        default="239.255.0.1",
        help="Multicast address for stream (default: 239.255.0.1)"
    )
    network_group.add_argument(
        "--stream-port",
        type=int,
        default=5010,
        help="Stream port (default: 5010)"
    )

    # Image saving options
    save_group = parser.add_argument_group('Image Saving Options')
    save_group.add_argument(
        "--save-images",
        action="store_true",
        help="Save captured calibration images for later replay"
    )
    save_group.add_argument(
        "--image-dir",
        default="calibration_images",
        help="Directory to save calibration images (default: calibration_images)"
    )

    # PDF report options
    report_group = parser.add_argument_group('Report Options')
    report_group.add_argument(
        "--generate-pdf",
        action="store_true",
        help="Generate PDF report with images and reprojection errors"
    )
    report_group.add_argument(
        "--pdf-output",
        type=str,
        metavar="PATH",
        help="Custom path for PDF report (default: <output>_report.pdf)"
    )

    # Board configuration options
    board_group = parser.add_argument_group('ChArUco Board Configuration')
    board_group.add_argument(
        "--squares-x",
        type=int,
        default=8,
        help="Number of squares in X direction (default: 8)"
    )
    board_group.add_argument(
        "--squares-y",
        type=int,
        default=8,
        help="Number of squares in Y direction (default: 8)"
    )
    board_group.add_argument(
        "--square-size",
        type=float,
        default=11.0,
        help="Square size in centimeters (default: 11cm)"
    )
    board_group.add_argument(
        "--marker-size",
        type=float,
        default=7.5,
        help="ArUco marker size in centimeters (default: 7.5cm)"
    )
    board_group.add_argument(
        "--dictionary",
        type=str,
        default="DICT_6X6_250",
        choices=dict_choices,
        help="ArUco dictionary (default: DICT_6X6_250)"
    )

    # Legacy argument names (for backwards compatibility)
    parser.add_argument("--board-squares-x", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--board-squares-y", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--board-square-size", type=float, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle legacy argument names
    squares_x = args.board_squares_x if args.board_squares_x is not None else args.squares_x
    squares_y = args.board_squares_y if args.board_squares_y is not None else args.squares_y
    square_size_meters = (args.board_square_size if args.board_square_size is not None
                          else args.square_size / 100.0)  # Convert cm to meters
    marker_size_meters = args.marker_size / 100.0  # Convert cm to meters

    # Validate marker size vs square size
    if marker_size_meters >= square_size_meters:
        print(f"Error: Marker size ({args.marker_size}cm) must be smaller than square size ({args.square_size}cm)")
        return 1

    # Get dictionary ID
    dictionary_id = ARUCO_DICTIONARIES.get(args.dictionary)
    if dictionary_id is None:
        print(f"Error: Unknown dictionary '{args.dictionary}'")
        print(f"Available dictionaries: {', '.join(dict_choices)}")
        return 1

    # Board configuration
    board_config = ChArUcoBoardConfig(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_size_meters,
        marker_length=marker_size_meters,
        dictionary_id=dictionary_id,
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

    # Create image source based on mode
    if args.from_directory:
        print(f"Mode: DIRECTORY (debug/replay from {args.from_directory})")
        try:
            # Try to load metadata if available
            metadata_path = Path(args.from_directory) / "calibration_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    saved_config = metadata.get("board_config", {})
                    print(f"  Found metadata - using saved board configuration:")
                    print(f"    Squares: {saved_config.get('squares_x', '?')}x{saved_config.get('squares_y', '?')}")
                    print(f"    Dictionary: {saved_config.get('dictionary_name', 'unknown')}")

                    # Update board config from metadata if not explicitly overridden
                    if args.board_squares_x is None and args.squares_x == 8:
                        board_config.squares_x = saved_config.get('squares_x', board_config.squares_x)
                    if args.board_squares_y is None and args.squares_y == 8:
                        board_config.squares_y = saved_config.get('squares_y', board_config.squares_y)
                    if 'square_length' in saved_config:
                        board_config.square_length = saved_config['square_length']
                    if 'marker_length' in saved_config:
                        board_config.marker_length = saved_config['marker_length']
                    if 'dictionary_id' in saved_config:
                        board_config.dictionary_id = saved_config['dictionary_id']

                    # Recreate board with updated config
                    board_config.board, board_config.aruco_dict = None, None

            image_source = DirectoryImageSource(args.from_directory)
            ground_truth = None
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1
    elif args.synthetic:
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
            print("Use --from-directory flag to load from saved images")
            print("Use --network-camera flag for network camera streaming")
            return 1

    # Set default output path based on camera_id if not specified
    output_path = args.output
    if output_path is None:
        # Create camera_intrinsic directory if it doesn't exist
        intrinsic_dir = Path("camera_intrinsic")
        intrinsic_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(intrinsic_dir / f"camera_intrinsics_{args.camera_id}.json")
        print(f"Output path: {output_path}")

    try:
        result = run_calibration(
            image_source=image_source,
            board_config=board_config,
            camera_id=args.camera_id,
            output_path=output_path,
            num_images=args.num_images,
            save_images=args.save_images,
            output_dir=args.image_dir,
            ground_truth=ground_truth,
            generate_pdf=args.generate_pdf,
            pdf_path=args.pdf_output
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

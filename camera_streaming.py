#!/usr/bin/env python3
"""
AxonVision Camera Streaming Module

Provides network camera streaming support for real-time calibration.
Supports H265/RTP multicast streaming via GStreamer.

Camera API:
- Base URL: http://<camera_ip>:5000/
- Start stream: POST /api/stream/start {"host": "239.255.0.1", "port": 5010}
- Stop stream: POST /api/stream/stop

Usage:
    python3 camera_streaming.py --help
    python3 camera_streaming.py test          # Test connectivity
    python3 camera_streaming.py preview       # Start camera preview
    python3 camera_streaming.py calibrate     # Run calibration with live camera
"""

import subprocess
import sys
import time
import socket
import json
import argparse
import signal
import os
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
from enum import Enum
from contextlib import contextmanager

# Check for required modules
try:
    import requests
except ImportError:
    print("ERROR: 'requests' module not found. Install with: pip install requests")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: 'numpy' module not found. Install with: pip install 'numpy<2'")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: 'opencv-contrib-python' not found. Install with: pip install opencv-contrib-python==4.8.1.78")
    sys.exit(1)


class StatusColor:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    colors = {
        "ok": StatusColor.GREEN,
        "warn": StatusColor.YELLOW,
        "error": StatusColor.RED,
        "info": StatusColor.BLUE,
        "header": StatusColor.CYAN + StatusColor.BOLD
    }
    color = colors.get(status, StatusColor.RESET)
    symbol = {"ok": "[OK]", "warn": "[WARN]", "error": "[FAIL]", "info": "[*]", "header": "==>"}.get(status, "[*]")
    print(f"{color}{symbol}{StatusColor.RESET} {message}")


def print_header(title: str):
    """Print a section header"""
    print(f"\n{StatusColor.CYAN}{StatusColor.BOLD}{'='*60}{StatusColor.RESET}")
    print(f"{StatusColor.CYAN}{StatusColor.BOLD}  {title}{StatusColor.RESET}")
    print(f"{StatusColor.CYAN}{StatusColor.BOLD}{'='*60}{StatusColor.RESET}\n")


@dataclass
class CameraConfig:
    """Network camera configuration"""
    ip: str = "10.100.102.222"
    api_port: int = 5000
    multicast_host: str = "239.255.0.1"
    stream_port: int = 5010
    width: int = 1920
    height: int = 1080
    timeout: float = 5.0

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:{self.api_port}"

    @property
    def stream_start_url(self) -> str:
        return f"{self.base_url}/api/stream/start"

    @property
    def stream_stop_url(self) -> str:
        return f"{self.base_url}/api/stream/stop"


class ConnectionTestResult(Enum):
    """Result of connection test"""
    SUCCESS = "success"
    NETWORK_UNREACHABLE = "network_unreachable"
    HOST_DOWN = "host_down"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test"""
    test_name: str
    passed: bool
    message: str
    details: Optional[str] = None
    latency_ms: Optional[float] = None


class CameraConnectivity:
    """Handles camera connectivity tests and diagnostics"""

    def __init__(self, config: CameraConfig):
        self.config = config

    def ping_test(self, count: int = 3) -> DiagnosticResult:
        """Test network connectivity using ping"""
        print_status(f"Pinging {self.config.ip}...", "info")

        try:
            # Use system ping command
            result = subprocess.run(
                ["ping", "-c", str(count), "-W", "2", self.config.ip],
                capture_output=True,
                text=True,
                timeout=count * 3 + 5
            )

            if result.returncode == 0:
                # Parse latency from ping output
                output = result.stdout
                latency = None
                for line in output.split('\n'):
                    if 'avg' in line.lower() or 'average' in line.lower():
                        # Extract average latency: "rtt min/avg/max/mdev = x/y/z/w ms"
                        try:
                            parts = line.split('=')[1].strip().split('/')
                            latency = float(parts[1])
                        except (IndexError, ValueError):
                            pass

                return DiagnosticResult(
                    test_name="Ping Test",
                    passed=True,
                    message=f"Camera reachable at {self.config.ip}",
                    details=output,
                    latency_ms=latency
                )
            else:
                return DiagnosticResult(
                    test_name="Ping Test",
                    passed=False,
                    message=f"Camera unreachable at {self.config.ip}",
                    details=result.stderr or result.stdout
                )

        except subprocess.TimeoutExpired:
            return DiagnosticResult(
                test_name="Ping Test",
                passed=False,
                message="Ping timed out"
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="Ping Test",
                passed=False,
                message=f"Ping failed: {str(e)}"
            )

    def port_check(self) -> DiagnosticResult:
        """Check if API port is open"""
        print_status(f"Checking port {self.config.api_port}...", "info")

        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            result = sock.connect_ex((self.config.ip, self.config.api_port))
            latency = (time.time() - start) * 1000
            sock.close()

            if result == 0:
                return DiagnosticResult(
                    test_name="Port Check",
                    passed=True,
                    message=f"Port {self.config.api_port} is open",
                    latency_ms=latency
                )
            else:
                return DiagnosticResult(
                    test_name="Port Check",
                    passed=False,
                    message=f"Port {self.config.api_port} is closed or filtered"
                )

        except socket.timeout:
            return DiagnosticResult(
                test_name="Port Check",
                passed=False,
                message="Connection timed out"
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="Port Check",
                passed=False,
                message=f"Port check failed: {str(e)}"
            )

    def api_health_check(self) -> DiagnosticResult:
        """Check if camera API is responding"""
        print_status(f"Testing API at {self.config.base_url}...", "info")

        try:
            start = time.time()
            response = requests.get(
                self.config.base_url,
                timeout=self.config.timeout
            )
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                return DiagnosticResult(
                    test_name="API Health",
                    passed=True,
                    message="Camera API is responding",
                    details=f"Status: {response.status_code}",
                    latency_ms=latency
                )
            else:
                return DiagnosticResult(
                    test_name="API Health",
                    passed=True,  # API responded, just not 200
                    message=f"API responded with status {response.status_code}",
                    latency_ms=latency
                )

        except requests.exceptions.ConnectionError as e:
            return DiagnosticResult(
                test_name="API Health",
                passed=False,
                message="Cannot connect to camera API",
                details=str(e)
            )
        except requests.exceptions.Timeout:
            return DiagnosticResult(
                test_name="API Health",
                passed=False,
                message="API request timed out"
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="API Health",
                passed=False,
                message=f"API check failed: {str(e)}"
            )

    def multicast_check(self) -> DiagnosticResult:
        """Check if system supports multicast"""
        print_status("Checking multicast support...", "info")

        try:
            # Check if the multicast address is in a valid range (224.0.0.0 - 239.255.255.255)
            ip_parts = [int(p) for p in self.config.multicast_host.split('.')]
            if not (224 <= ip_parts[0] <= 239):
                return DiagnosticResult(
                    test_name="Multicast Check",
                    passed=False,
                    message=f"Invalid multicast address: {self.config.multicast_host}"
                )

            # Try to create a multicast socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Try to join multicast group
            try:
                mreq = socket.inet_aton(self.config.multicast_host) + socket.inet_aton('0.0.0.0')
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                sock.close()

                return DiagnosticResult(
                    test_name="Multicast Check",
                    passed=True,
                    message=f"Multicast supported for {self.config.multicast_host}"
                )
            except Exception as e:
                sock.close()
                return DiagnosticResult(
                    test_name="Multicast Check",
                    passed=False,
                    message=f"Cannot join multicast group: {str(e)}"
                )

        except Exception as e:
            return DiagnosticResult(
                test_name="Multicast Check",
                passed=False,
                message=f"Multicast check failed: {str(e)}"
            )

    def gstreamer_check(self) -> DiagnosticResult:
        """Check if GStreamer is installed with required plugins"""
        print_status("Checking GStreamer installation...", "info")

        try:
            # Check gst-launch-1.0
            result = subprocess.run(
                ["gst-launch-1.0", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return DiagnosticResult(
                    test_name="GStreamer Check",
                    passed=False,
                    message="GStreamer not found",
                    details="Install with: sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav"
                )

            version = result.stdout.split('\n')[0] if result.stdout else "Unknown"

            # Check for required plugins
            required_plugins = ['udpsrc', 'rtph265depay', 'h265parse', 'avdec_h265', 'videoconvert', 'autovideosink']
            missing = []

            for plugin in required_plugins:
                check = subprocess.run(
                    ["gst-inspect-1.0", plugin],
                    capture_output=True,
                    timeout=5
                )
                if check.returncode != 0:
                    missing.append(plugin)

            if missing:
                return DiagnosticResult(
                    test_name="GStreamer Check",
                    passed=False,
                    message=f"Missing GStreamer plugins: {', '.join(missing)}",
                    details="Install: sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav"
                )

            return DiagnosticResult(
                test_name="GStreamer Check",
                passed=True,
                message=f"GStreamer ready: {version}",
                details="All required plugins available"
            )

        except FileNotFoundError:
            return DiagnosticResult(
                test_name="GStreamer Check",
                passed=False,
                message="GStreamer not installed",
                details="Install with: sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav"
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="GStreamer Check",
                passed=False,
                message=f"GStreamer check failed: {str(e)}"
            )

    def run_full_diagnostics(self) -> List[DiagnosticResult]:
        """Run all diagnostic tests"""
        print_header("Camera Connectivity Diagnostics")

        results = []

        # Run tests in order
        tests = [
            self.ping_test,
            self.port_check,
            self.api_health_check,
            self.multicast_check,
            self.gstreamer_check
        ]

        for test in tests:
            result = test()
            results.append(result)

            status = "ok" if result.passed else "error"
            print_status(result.message, status)
            if result.latency_ms:
                print(f"      Latency: {result.latency_ms:.1f}ms")
            if result.details and not result.passed:
                print(f"      Details: {result.details}")
            print()

        # Summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)

        print_header("Diagnostics Summary")
        print(f"  Tests Passed: {passed}/{total}")

        if passed == total:
            print_status("All tests passed! Camera is ready.", "ok")
        else:
            print_status("Some tests failed. Please check the issues above.", "warn")

        return results


class CameraStreamController:
    """Controls camera streaming via API"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.stream_active = False
        self._gstreamer_process: Optional[subprocess.Popen] = None

    def start_stream(self) -> Tuple[bool, str]:
        """Start camera streaming via API"""
        print_status("Starting camera stream...", "info")

        try:
            payload = {
                "host": self.config.multicast_host,
                "port": self.config.stream_port
            }

            response = requests.post(
                self.config.stream_start_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                self.stream_active = True
                return True, "Stream started successfully"
            else:
                return False, f"Failed to start stream: HTTP {response.status_code}"

        except requests.exceptions.RequestException as e:
            return False, f"Failed to start stream: {str(e)}"

    def stop_stream(self) -> Tuple[bool, str]:
        """Stop camera streaming via API"""
        print_status("Stopping camera stream...", "info")

        try:
            response = requests.post(
                self.config.stream_stop_url,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout
            )

            self.stream_active = False

            if response.status_code == 200:
                return True, "Stream stopped successfully"
            else:
                return True, f"Stream stop request sent (HTTP {response.status_code})"

        except requests.exceptions.RequestException as e:
            self.stream_active = False
            return False, f"Error stopping stream: {str(e)}"

    def get_gstreamer_pipeline(self, output: str = "autovideosink") -> str:
        """Get GStreamer pipeline command"""
        return (
            f"gst-launch-1.0 udpsrc address={self.config.multicast_host} "
            f"port={self.config.stream_port} "
            f'caps="application/x-rtp,media=video,encoding-name=H265,payload=96" ! '
            f"rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! {output}"
        )

    def get_gstreamer_opencv_pipeline(self) -> str:
        """Get GStreamer pipeline for OpenCV capture"""
        return (
            f"udpsrc address={self.config.multicast_host} port={self.config.stream_port} "
            f'caps="application/x-rtp,media=video,encoding-name=H265,payload=96" ! '
            f"rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=1"
        )

    def start_preview(self, window_name: str = "Camera Preview") -> subprocess.Popen:
        """Start GStreamer preview window"""
        pipeline = self.get_gstreamer_pipeline(f'autovideosink name="{window_name}"')

        print_status(f"Starting preview window...", "info")
        print(f"  Pipeline: {pipeline}\n")

        self._gstreamer_process = subprocess.Popen(
            pipeline,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return self._gstreamer_process

    def stop_preview(self):
        """Stop GStreamer preview"""
        if self._gstreamer_process:
            self._gstreamer_process.terminate()
            try:
                self._gstreamer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gstreamer_process.kill()
            self._gstreamer_process = None
            print_status("Preview stopped", "info")


class NetworkCameraSource:
    """Camera source using network streaming for OpenCV integration"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.controller = CameraStreamController(config)
        self.cap: Optional[cv2.VideoCapture] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to the camera stream"""
        # Start the camera stream via API
        success, message = self.controller.start_stream()
        if not success:
            print_status(message, "error")
            return False

        print_status(message, "ok")

        # Give the stream a moment to start
        time.sleep(1.0)

        # Connect using GStreamer pipeline
        pipeline = self.controller.get_gstreamer_opencv_pipeline()
        print_status(f"Connecting to stream via GStreamer...", "info")

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print_status("Failed to open GStreamer pipeline with OpenCV", "error")
            self.controller.stop_stream()
            return False

        self._connected = True
        print_status("Connected to camera stream", "ok")
        return True

    def get_image(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if not self._connected or self.cap is None:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Release the camera"""
        if self.cap:
            self.cap.release()
            self.cap = None

        if self._connected:
            self.controller.stop_stream()
            self._connected = False

    @contextmanager
    def stream_context(self):
        """Context manager for camera streaming"""
        try:
            if self.connect():
                yield self
            else:
                yield None
        finally:
            self.release()


class CameraPreviewApp:
    """Interactive camera preview with calibration integration"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.controller = CameraStreamController(config)
        self.running = False
        self._capture_callback: Optional[Callable[[np.ndarray], None]] = None

    def set_capture_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for captured frames"""
        self._capture_callback = callback

    def run_preview_opencv(self, window_name: str = "Camera Preview"):
        """Run interactive preview using OpenCV window"""
        print_header("Camera Preview (OpenCV)")
        print("Controls:")
        print("  [SPACE] - Capture frame")
        print("  [S]     - Save snapshot")
        print("  [Q/ESC] - Quit")
        print()

        source = NetworkCameraSource(self.config)

        if not source.connect():
            print_status("Failed to connect to camera", "error")
            return

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        self.running = True
        frame_count = 0
        snapshot_count = 0

        try:
            while self.running:
                frame = source.get_image()

                if frame is None:
                    print_status("Lost connection to camera", "warn")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # Add overlay text
                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"Frame: {frame_count} | Press 'Q' to quit, 'S' to save",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(window_name, overlay)

                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                    self.running = False

                elif key == ord(' '):  # Space - capture
                    if self._capture_callback:
                        self._capture_callback(frame.copy())
                        print_status(f"Frame captured (#{frame_count})", "ok")

                elif key in [ord('s'), ord('S')]:  # S - save snapshot
                    snapshot_count += 1
                    filename = f"snapshot_{snapshot_count:04d}.png"
                    cv2.imwrite(filename, frame)
                    print_status(f"Saved: {filename}", "ok")

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            source.release()
            cv2.destroyAllWindows()

        print_status("Preview ended", "info")

    def run_preview_gstreamer(self):
        """Run preview using native GStreamer (better performance)"""
        print_header("Camera Preview (GStreamer)")
        print("This opens a native GStreamer window.")
        print("Press Ctrl+C in terminal to stop.\n")

        # Start camera stream
        success, message = self.controller.start_stream()
        if not success:
            print_status(message, "error")
            return

        print_status(message, "ok")
        time.sleep(0.5)

        # Start GStreamer preview
        process = self.controller.start_preview()

        try:
            # Wait for process or interrupt
            while process.poll() is None:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n")
            print_status("Stopping preview...", "info")

        finally:
            self.controller.stop_preview()
            self.controller.stop_stream()

        print_status("Preview ended", "info")


def run_connectivity_test(config: CameraConfig):
    """Run full connectivity test"""
    connectivity = CameraConnectivity(config)
    results = connectivity.run_full_diagnostics()

    # Return exit code based on critical tests
    critical_passed = all(r.passed for r in results[:3])  # ping, port, api
    return 0 if critical_passed else 1


def run_preview(config: CameraConfig, mode: str = "gstreamer"):
    """Run camera preview"""
    app = CameraPreviewApp(config)

    if mode == "opencv":
        app.run_preview_opencv()
    else:
        app.run_preview_gstreamer()


def run_quick_capture(config: CameraConfig, output_file: str = "capture.png"):
    """Quickly capture a single frame"""
    print_header("Quick Capture")

    source = NetworkCameraSource(config)

    with source.stream_context() as cam:
        if cam is None:
            print_status("Failed to connect to camera", "error")
            return 1

        # Warm up - skip first few frames
        print_status("Warming up camera...", "info")
        for _ in range(10):
            cam.get_image()
            time.sleep(0.1)

        # Capture frame
        frame = cam.get_image()

        if frame is not None:
            cv2.imwrite(output_file, frame)
            print_status(f"Captured: {output_file} ({frame.shape[1]}x{frame.shape[0]})", "ok")
            return 0
        else:
            print_status("Failed to capture frame", "error")
            return 1


def run_calibration_mode(config: CameraConfig):
    """Run intrinsic calibration with live camera"""
    print_header("Live Camera Calibration")

    # Check if intrinsic_calibration module exists
    try:
        import intrinsic_calibration as ic
    except ImportError:
        print_status("intrinsic_calibration.py not found in current directory", "error")
        return 1

    # Run connectivity test first
    print_status("Running connectivity check...", "info")
    connectivity = CameraConnectivity(config)

    ping_result = connectivity.ping_test()
    if not ping_result.passed:
        print_status(f"Camera not reachable: {ping_result.message}", "error")
        return 1

    api_result = connectivity.api_health_check()
    if not api_result.passed:
        print_status(f"Camera API not available: {api_result.message}", "error")
        return 1

    print_status("Camera connectivity OK", "ok")
    print()

    # Create camera source
    source = NetworkCameraSource(config)

    if not source.connect():
        print_status("Failed to connect to camera stream", "error")
        return 1

    print()
    print_status("Starting calibration...", "info")
    print("Follow the on-screen instructions to capture calibration images.")
    print()

    # Use the intrinsic calibration module
    try:
        board_config = ic.ChArUcoBoardConfig()
        detector = ic.ChArUcoDetector(board_config)
        calibrator = ic.IntrinsicCalibrator(board_config)

        # Run interactive calibration
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 1280, 720)

        captured_images = []
        target_images = 25

        print(f"Capture {target_images} images of the ChArUco board from different angles.")
        print("Press SPACE to capture, Q to finish early.\n")

        while len(captured_images) < target_images:
            frame = source.get_image()

            if frame is None:
                time.sleep(0.1)
                continue

            # Try to detect ChArUco board
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = detector.detect_markers(gray)

            display = frame.copy()

            if corners is not None and ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                charuco_corners, charuco_ids = detector.detect_charuco(gray, corners, ids)

                if charuco_corners is not None and len(charuco_corners) > 10:
                    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                    status_text = f"Detected {len(charuco_corners)} corners - Press SPACE to capture"
                    color = (0, 255, 0)
                else:
                    status_text = "Move board closer or adjust angle"
                    color = (0, 165, 255)
            else:
                status_text = "No board detected - Show ChArUco board to camera"
                color = (0, 0, 255)

            # Overlay status
            cv2.putText(display, f"Captured: {len(captured_images)}/{target_images}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, status_text,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and charuco_corners is not None and len(charuco_corners) > 10:
                captured_images.append(frame.copy())
                print_status(f"Captured image {len(captured_images)}/{target_images}", "ok")
                time.sleep(0.3)  # Debounce

            elif key in [ord('q'), ord('Q'), 27]:
                if len(captured_images) >= 5:
                    break
                else:
                    print_status("Need at least 5 images for calibration", "warn")

        cv2.destroyAllWindows()
        source.release()

        if len(captured_images) < 5:
            print_status("Not enough images captured", "error")
            return 1

        # Run calibration
        print()
        print_status(f"Running calibration with {len(captured_images)} images...", "info")

        for img in captured_images:
            calibrator.add_image(img)

        result = calibrator.calibrate()

        if result is not None:
            # Save results
            output_file = "camera_intrinsics.json"
            calibrator.save_calibration(output_file)
            print()
            print_status(f"Calibration successful! Saved to {output_file}", "ok")
            print(f"  RMS Error: {result['rms_error']:.4f} pixels")
            return 0
        else:
            print_status("Calibration failed", "error")
            return 1

    except Exception as e:
        source.release()
        cv2.destroyAllWindows()
        print_status(f"Calibration error: {str(e)}", "error")
        return 1


def interactive_menu(config: CameraConfig):
    """Run interactive CLI menu"""
    print_header("AxonVision Camera Control")
    print(f"Camera: {config.ip}:{config.api_port}")
    print(f"Stream: {config.multicast_host}:{config.stream_port}")
    print()

    while True:
        print("\n" + "=" * 40)
        print("  Main Menu")
        print("=" * 40)
        print("  1. Test Camera Connectivity")
        print("  2. Preview Camera (GStreamer)")
        print("  3. Preview Camera (OpenCV)")
        print("  4. Quick Capture")
        print("  5. Run Calibration")
        print("  6. Start Stream Only")
        print("  7. Stop Stream Only")
        print("  0. Exit")
        print()

        try:
            choice = input("Select option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == '1':
            run_connectivity_test(config)
        elif choice == '2':
            run_preview(config, "gstreamer")
        elif choice == '3':
            run_preview(config, "opencv")
        elif choice == '4':
            filename = input("Output filename [capture.png]: ").strip() or "capture.png"
            run_quick_capture(config, filename)
        elif choice == '5':
            run_calibration_mode(config)
        elif choice == '6':
            controller = CameraStreamController(config)
            success, msg = controller.start_stream()
            print_status(msg, "ok" if success else "error")
            if success:
                print(f"\nTo view stream, run:")
                print(f"  {controller.get_gstreamer_pipeline()}")
        elif choice == '7':
            controller = CameraStreamController(config)
            success, msg = controller.stop_stream()
            print_status(msg, "ok" if success else "error")
        elif choice == '0':
            break
        else:
            print_status("Invalid option", "warn")

    print_status("Goodbye!", "info")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AxonVision Camera Streaming Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_streaming.py                    # Interactive menu
  python3 camera_streaming.py test               # Run connectivity tests
  python3 camera_streaming.py preview            # Start GStreamer preview
  python3 camera_streaming.py preview --opencv   # Start OpenCV preview
  python3 camera_streaming.py capture            # Capture single frame
  python3 camera_streaming.py calibrate          # Run calibration mode
  python3 camera_streaming.py start              # Start stream only
  python3 camera_streaming.py stop               # Stop stream only
        """
    )

    parser.add_argument(
        'command',
        nargs='?',
        choices=['test', 'preview', 'capture', 'calibrate', 'start', 'stop', 'menu'],
        default='menu',
        help='Command to run (default: menu)'
    )

    parser.add_argument(
        '--ip',
        default='10.100.102.222',
        help='Camera IP address (default: 10.100.102.222)'
    )

    parser.add_argument(
        '--api-port',
        type=int,
        default=5000,
        help='Camera API port (default: 5000)'
    )

    parser.add_argument(
        '--multicast',
        default='239.255.0.1',
        help='Multicast address (default: 239.255.0.1)'
    )

    parser.add_argument(
        '--stream-port',
        type=int,
        default=5010,
        help='Stream port (default: 5010)'
    )

    parser.add_argument(
        '--opencv',
        action='store_true',
        help='Use OpenCV for preview instead of GStreamer'
    )

    parser.add_argument(
        '-o', '--output',
        default='capture.png',
        help='Output filename for capture command (default: capture.png)'
    )

    args = parser.parse_args()

    # Create config
    config = CameraConfig(
        ip=args.ip,
        api_port=args.api_port,
        multicast_host=args.multicast,
        stream_port=args.stream_port
    )

    # Route to command
    if args.command == 'test':
        return run_connectivity_test(config)
    elif args.command == 'preview':
        mode = 'opencv' if args.opencv else 'gstreamer'
        run_preview(config, mode)
        return 0
    elif args.command == 'capture':
        return run_quick_capture(config, args.output)
    elif args.command == 'calibrate':
        return run_calibration_mode(config)
    elif args.command == 'start':
        controller = CameraStreamController(config)
        success, msg = controller.start_stream()
        print_status(msg, "ok" if success else "error")
        if success:
            print(f"\nTo view stream, run:")
            print(f"  {controller.get_gstreamer_pipeline()}")
        return 0 if success else 1
    elif args.command == 'stop':
        controller = CameraStreamController(config)
        success, msg = controller.stop_stream()
        print_status(msg, "ok" if success else "error")
        return 0 if success else 1
    else:  # menu
        interactive_menu(config)
        return 0


if __name__ == '__main__':
    sys.exit(main())

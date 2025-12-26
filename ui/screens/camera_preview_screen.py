"""
Camera Preview Screen (Screen 3)

Hardware verification and intrinsic calibration:
- Shows all cameras with live preview and ping status
- Read-only camera configuration table
- Intrinsic calibration for each camera
- Real-time LLA (Lat/Long/Alt) and Yaw/Pitch/Roll data display
- Block progression if any camera missing intrinsic calibration
"""

from pathlib import Path
import subprocess
import platform
import random
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QGridLayout, QSizePolicy, QMessageBox,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject
from PySide6.QtGui import QPixmap, QImage

import numpy as np
import cv2

from ..styles import COLORS
from ..data_models import PlatformConfiguration, MOUNTING_POSITIONS

# Module-level list to keep references to orphaned threads AND their workers
# This prevents QThread/QObject from being garbage collected while still running
# Each entry is a tuple of (thread, worker) to keep both alive
_orphaned_threads: list = []


def _on_orphaned_thread_finished(orphan_tuple):
    """Remove finished orphaned thread from tracking list."""
    if orphan_tuple in _orphaned_threads:
        _orphaned_threads.remove(orphan_tuple)
    thread, worker = orphan_tuple
    if worker:
        worker.deleteLater()
    thread.deleteLater()


# Import camera streaming from intrinsic_calibration module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from intrinsic_calibration import NetworkCameraSource
except ImportError:
    NetworkCameraSource = None

from ..dialogs import IntrinsicCalibrationDialog


class CameraConnectionWorker(QObject):
    """Worker for threaded camera ping and connection operations."""

    # Signals to update UI from thread
    ping_result = Signal(int, bool)  # camera_number, ping_status
    connection_result = Signal(int, bool, str)  # camera_number, success, error_message
    all_complete = Signal()  # Emitted when all cameras have been processed

    def __init__(self, camera_ips: dict, parent=None):
        """
        Initialize worker with camera IPs.

        Args:
            camera_ips: Dict mapping camera_number to ip_address
        """
        super().__init__(parent)
        self.camera_ips = camera_ips
        self._is_running = True

    def stop(self):
        """Stop the worker."""
        self._is_running = False

    def run_ping_tests(self):
        """Run ping tests for all cameras in background."""
        for camera_number, ip_address in self.camera_ips.items():
            if not self._is_running:
                break
            ping_ok = self._ping_device(ip_address)
            self.ping_result.emit(camera_number, ping_ok)

        self.all_complete.emit()

    def _ping_device(self, ip_address: str) -> bool:
        """Ping a device to check if it's reachable."""
        if not ip_address:
            return False

        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            timeout_param = '-w' if platform.system().lower() == 'windows' else '-W'

            result = subprocess.run(
                ['ping', param, '1', timeout_param, '2', ip_address],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return False


class CameraStreamWorker(QObject):
    """Worker for connecting a single camera stream in background."""

    connection_ready = Signal(bool, str)  # success, error_message

    def __init__(self, ip_address: str, parent=None):
        super().__init__(parent)
        self.ip_address = ip_address
        self.camera_source = None
        self._is_running = True

    def stop(self):
        """Stop the worker."""
        self._is_running = False

    def connect_camera(self):
        """Connect to camera in background thread."""
        if not self.ip_address or not self._is_running:
            self.connection_ready.emit(False, "No IP address")
            return

        try:
            if NetworkCameraSource is None:
                self.connection_ready.emit(False, "NetworkCameraSource not available")
                return

            self.camera_source = NetworkCameraSource(
                ip=self.ip_address,
                api_port=5000,
                multicast_host="239.255.0.1",
                stream_port=5010,
                timeout=10.0
            )

            if self.camera_source.connect():
                self.connection_ready.emit(True, "")
            else:
                error_msg = self.camera_source.last_error or "Connection failed"
                self.connection_ready.emit(False, error_msg)
        except Exception as e:
            self.connection_ready.emit(False, str(e))

    def get_camera_source(self):
        """Get the connected camera source."""
        return self.camera_source


class SensorDataWidget(QFrame):
    """Widget displaying real-time LLA and YPR sensor data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_data)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['white']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QLabel("Real-Time Sensor Data")
        header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['primary']};")
        layout.addWidget(header)

        # LLA Section
        lla_frame = QFrame()
        lla_frame.setStyleSheet("border: none;")
        lla_layout = QGridLayout(lla_frame)
        lla_layout.setSpacing(8)

        lla_label = QLabel("Position (LLA)")
        lla_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_dark']};")
        lla_layout.addWidget(lla_label, 0, 0, 1, 2)

        lla_layout.addWidget(QLabel("Latitude:"), 1, 0)
        self.lat_value = QLabel("---.------°")
        self.lat_value.setStyleSheet(f"font-family: monospace; color: {COLORS['primary']};")
        lla_layout.addWidget(self.lat_value, 1, 1)

        lla_layout.addWidget(QLabel("Longitude:"), 2, 0)
        self.lon_value = QLabel("---.------°")
        self.lon_value.setStyleSheet(f"font-family: monospace; color: {COLORS['primary']};")
        lla_layout.addWidget(self.lon_value, 2, 1)

        lla_layout.addWidget(QLabel("Altitude:"), 3, 0)
        self.alt_value = QLabel("----.-- m")
        self.alt_value.setStyleSheet(f"font-family: monospace; color: {COLORS['primary']};")
        lla_layout.addWidget(self.alt_value, 3, 1)

        layout.addWidget(lla_frame)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background-color: {COLORS['border']};")
        layout.addWidget(sep)

        # YPR Section
        ypr_frame = QFrame()
        ypr_frame.setStyleSheet("border: none;")
        ypr_layout = QGridLayout(ypr_frame)
        ypr_layout.setSpacing(8)

        ypr_label = QLabel("Orientation (YPR)")
        ypr_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_dark']};")
        ypr_layout.addWidget(ypr_label, 0, 0, 1, 2)

        ypr_layout.addWidget(QLabel("Yaw:"), 1, 0)
        self.yaw_value = QLabel("---.--°")
        self.yaw_value.setStyleSheet(f"font-family: monospace; color: {COLORS['primary']};")
        ypr_layout.addWidget(self.yaw_value, 1, 1)

        ypr_layout.addWidget(QLabel("Pitch:"), 2, 0)
        self.pitch_value = QLabel("---.--°")
        self.pitch_value.setStyleSheet(f"font-family: monospace; color: {COLORS['primary']};")
        ypr_layout.addWidget(self.pitch_value, 2, 1)

        ypr_layout.addWidget(QLabel("Roll:"), 3, 0)
        self.roll_value = QLabel("---.--°")
        self.roll_value.setStyleSheet(f"font-family: monospace; color: {COLORS['primary']};")
        ypr_layout.addWidget(self.roll_value, 3, 1)

        layout.addWidget(ypr_frame)

        # Status
        self.status_label = QLabel("⚪ Waiting for data...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def start_updates(self):
        """Start updating sensor data."""
        self.timer.start(100)  # 10 Hz update
        self.status_label.setText("🟢 Receiving data")
        self.status_label.setStyleSheet(f"color: {COLORS['success']};")

    def stop_updates(self):
        """Stop updating sensor data."""
        self.timer.stop()
        self.status_label.setText("⚪ Stopped")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")

    def _update_data(self):
        """Update sensor values with simulated data (replace with real INS data)."""
        # Simulated data - in production, this would read from INS
        lat = 37.7749 + random.uniform(-0.0001, 0.0001)
        lon = -122.4194 + random.uniform(-0.0001, 0.0001)
        alt = 10.5 + random.uniform(-0.1, 0.1)

        yaw = random.uniform(0, 360)
        pitch = random.uniform(-5, 5)
        roll = random.uniform(-3, 3)

        self.lat_value.setText(f"{lat:011.6f}°")
        self.lon_value.setText(f"{lon:012.6f}°")
        self.alt_value.setText(f"{alt:07.2f} m")

        self.yaw_value.setText(f"{yaw:06.2f}°")
        self.pitch_value.setText(f"{pitch:+06.2f}°")
        self.roll_value.setText(f"{roll:+06.2f}°")


class CameraPreviewCard(QFrame):
    """A card widget displaying a single camera preview with status."""

    calibrate_requested = Signal(int)  # camera_number

    def __init__(self, camera_number: int, camera_id: str, ip_address: str,
                 mounting_position: str, has_intrinsic: bool, parent=None):
        super().__init__(parent)
        self.camera_number = camera_number
        self.camera_id = camera_id
        self.ip_address = ip_address
        self.mounting_position = mounting_position
        self.has_intrinsic = has_intrinsic
        self.ping_status = None

        self.camera_source: Optional[NetworkCameraSource] = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self._is_streaming = False

        # Threading for camera connection
        self._stream_thread: Optional[QThread] = None
        self._stream_worker: Optional[CameraStreamWorker] = None
        self._finishing_threads: list = []  # Keep refs to threads until they finish

        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("card")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame#card {{
                background-color: {COLORS['white']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header with camera info and status
        header_layout = QHBoxLayout()

        camera_label = QLabel(f"Camera {self.camera_number}")
        camera_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            color: {COLORS['primary']};
        """)
        header_layout.addWidget(camera_label)

        header_layout.addStretch()

        # Ping status indicator
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self._update_status_display()
        header_layout.addWidget(self.status_label)

        layout.addLayout(header_layout)

        # Camera ID, IP, and Position info
        info_label = QLabel(f"ID: {self.camera_id}  |  IP: {self.ip_address}")
        info_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        layout.addWidget(info_label)

        pos_label = QLabel(f"Position: {self.mounting_position}")
        pos_label.setStyleSheet(f"color: {COLORS['text_dark']}; font-size: 13px; font-weight: bold;")
        layout.addWidget(pos_label)

        # Preview area
        self.video_label = QLabel()
        self.video_label.setFixedSize(280, 180)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"""
            background-color: #1a1a1a;
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            color: {COLORS['text_muted']};
            font-size: 12px;
        """)
        self.video_label.setText("Connecting...")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Intrinsic calibration status and button
        intrinsic_layout = QHBoxLayout()

        self.intrinsic_status = QLabel()
        self._update_intrinsic_display()
        intrinsic_layout.addWidget(self.intrinsic_status)

        intrinsic_layout.addStretch()

        self.calibrate_btn = QPushButton("Calibrate Intrinsic")
        self.calibrate_btn.setObjectName("calibrate_button")
        self.calibrate_btn.setToolTip("Run intrinsic calibration for this camera")
        self.calibrate_btn.clicked.connect(lambda: self.calibrate_requested.emit(self.camera_number))
        intrinsic_layout.addWidget(self.calibrate_btn)

        layout.addLayout(intrinsic_layout)

    def _update_status_display(self):
        """Update the ping status display."""
        if self.ping_status is None:
            self.status_label.setText("Testing...")
            self.status_label.setStyleSheet(f"""
                color: {COLORS['text_muted']};
                font-size: 12px;
                padding: 4px 8px;
            """)
        elif self.ping_status:
            self.status_label.setText("✓ Connected")
            self.status_label.setStyleSheet(f"""
                color: {COLORS['success']};
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                background-color: #E8F5E9;
                border-radius: 4px;
            """)
        else:
            self.status_label.setText("✗ Not Detected")
            self.status_label.setStyleSheet(f"""
                color: {COLORS['danger']};
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                background-color: #FFEBEE;
                border-radius: 4px;
            """)

    def _update_intrinsic_display(self):
        """Update the intrinsic calibration status display."""
        if self.has_intrinsic:
            self.intrinsic_status.setText("✓ Intrinsic Calibrated")
            self.intrinsic_status.setStyleSheet(f"""
                color: {COLORS['success']};
                font-weight: bold;
                font-size: 13px;
            """)
        else:
            self.intrinsic_status.setText("✗ Needs Calibration")
            self.intrinsic_status.setStyleSheet(f"""
                color: {COLORS['danger']};
                font-weight: bold;
                font-size: 13px;
            """)

    def set_ping_status(self, status: bool):
        """Set the ping status and update display."""
        self.ping_status = status
        self._update_status_display()

    def set_intrinsic_status(self, has_intrinsic: bool):
        """Set the intrinsic calibration status."""
        self.has_intrinsic = has_intrinsic
        self._update_intrinsic_display()

    def start_streaming(self):
        """Start camera streaming in a background thread."""
        if self._is_streaming or NetworkCameraSource is None:
            return
        if not self.ip_address:
            self.video_label.setText("No IP address")
            return

        # Clean up any existing thread
        self._cleanup_stream_thread()

        self.video_label.setText("Connecting...")

        # Create worker and thread for background connection
        self._stream_thread = QThread()
        self._stream_worker = CameraStreamWorker(self.ip_address)
        self._stream_worker.moveToThread(self._stream_thread)

        # Connect signals
        self._stream_thread.started.connect(self._stream_worker.connect_camera)
        self._stream_worker.connection_ready.connect(self._on_stream_connected)

        # Start the thread
        self._stream_thread.start()

    def _on_stream_connected(self, success: bool, error_msg: str):
        """Handle camera connection result from worker thread."""
        if success and self._stream_worker:
            self.camera_source = self._stream_worker.get_camera_source()
            self._is_streaming = True
            self.timer.start(33)  # ~30 FPS
        else:
            self.video_label.setText(f"Error:\n{error_msg[:30]}...")
            self.video_label.setStyleSheet(f"""
                background-color: #1a1a1a;
                border: 1px solid {COLORS['danger']};
                border-radius: 4px;
                color: {COLORS['danger']};
                font-size: 10px;
            """)

        # Clean up thread after connection attempt
        self._cleanup_stream_thread()

    def _cleanup_stream_thread(self, wait_for_finish: bool = False):
        """Clean up the stream worker thread.

        Args:
            wait_for_finish: If True, block until thread finishes (for widget deletion).
                           If False, allow thread to finish asynchronously.
        """
        worker = self._stream_worker
        thread = self._stream_thread

        if worker:
            worker.stop()

        if thread:
            thread.quit()

            if wait_for_finish:
                # Block until thread finishes - needed before widget deletion
                # Wait up to 12 seconds (network timeout can be 5s for stop + 5s for start)
                if not thread.wait(12000):
                    # Thread didn't stop gracefully, force terminate
                    thread.terminate()
                    thread.wait(2000)

                # Check if thread is STILL running after terminate
                # If so, move to module-level orphaned list to prevent GC crash
                # IMPORTANT: Also keep worker alive - it's still being used by the thread!
                if thread.isRunning():
                    orphan = (thread, worker)
                    _orphaned_threads.append(orphan)
                    thread.finished.connect(lambda o=orphan: _on_orphaned_thread_finished(o))
            else:
                # Non-blocking cleanup - thread will finish on its own
                # Keep reference until thread finishes to prevent GC crash
                self._finishing_threads.append(thread)
                thread.finished.connect(lambda t=thread: self._on_thread_finished(t))

            self._stream_thread = None

        # Always clear the instance worker reference
        # (orphaned list keeps its own reference if needed)
        self._stream_worker = None

    def _on_thread_finished(self, thread: QThread):
        """Remove finished thread from tracking list."""
        if thread in self._finishing_threads:
            self._finishing_threads.remove(thread)
        thread.deleteLater()

    def stop_streaming(self, wait_for_finish: bool = False):
        """Stop camera streaming.

        Args:
            wait_for_finish: If True, block until background thread finishes.
                           Use True when widget is about to be deleted.
        """
        self.timer.stop()
        self._is_streaming = False
        self._cleanup_stream_thread(wait_for_finish=wait_for_finish)
        if self.camera_source:
            self.camera_source.release()
            self.camera_source = None

    def _update_frame(self):
        """Update video frame from camera."""
        if not self.camera_source:
            return
        frame = self.camera_source.get_image()
        if frame is not None:
            self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        """Convert OpenCV frame to QPixmap and display."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        target_w, target_h = 280, 180
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = rgb_frame.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))


class CameraPreviewScreen(QWidget):
    """
    Camera Preview Screen

    Step 2 of 6 in the calibration workflow (Hardware Verification).
    Displays camera previews, read-only config table, intrinsic calibration,
    and real-time sensor data.
    """

    cancel_requested = Signal()
    next_requested = Signal(PlatformConfiguration)

    def __init__(self, config: PlatformConfiguration = None, parent=None):
        super().__init__(parent)
        self.config = config or PlatformConfiguration()
        self.base_path = "."
        self.camera_cards = []
        self.ping_failed_cameras = []

        # Threading for camera verification
        self._ping_thread: Optional[QThread] = None
        self._ping_worker: Optional[CameraConnectionWorker] = None
        self._ping_results: dict = {}  # camera_number -> ping_status

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(15)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)

        screen_label = QLabel("Screen 3: Hardware Verification")
        screen_label.setObjectName("screen_indicator")
        header_layout.addWidget(screen_label)

        title = QLabel("Camera Verification & Intrinsic Calibration")
        title.setObjectName("title")
        header_layout.addWidget(title)

        step_label = QLabel("Step 2 of 6")
        step_label.setObjectName("step_indicator")
        header_layout.addWidget(step_label)

        main_layout.addLayout(header_layout)

        # Status message area
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['table_header']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        status_layout = QHBoxLayout(self.status_frame)

        self.status_icon = QLabel()
        self.status_icon.setFixedWidth(30)
        status_layout.addWidget(self.status_icon)

        self.status_message = QLabel("Testing camera connectivity...")
        self.status_message.setStyleSheet("font-size: 14px;")
        status_layout.addWidget(self.status_message, 1)

        main_layout.addWidget(self.status_frame)

        # Main content with camera table and sensor data side panel
        content_layout = QHBoxLayout()

        # Left side - Camera table (read-only)
        left_panel = QFrame()
        left_panel.setObjectName("card")
        left_layout = QVBoxLayout(left_panel)

        table_header = QLabel("Camera Configuration (Read-Only)")
        table_header.setObjectName("section_header")
        left_layout.addWidget(table_header)

        self.config_table = QTableWidget()
        self._setup_config_table()
        left_layout.addWidget(self.config_table)

        content_layout.addWidget(left_panel, 2)

        # Right side - Sensor data
        self.sensor_widget = SensorDataWidget()
        self.sensor_widget.setFixedWidth(250)
        content_layout.addWidget(self.sensor_widget)

        main_layout.addLayout(content_layout)

        # Camera previews section
        preview_frame = QFrame()
        preview_frame.setObjectName("card")
        preview_layout = QVBoxLayout(preview_frame)

        preview_header_row = QHBoxLayout()
        preview_header = QLabel("Camera Previews & Intrinsic Calibration")
        preview_header.setObjectName("section_header")
        preview_header_row.addWidget(preview_header)
        preview_header_row.addStretch()

        self.refresh_btn = QPushButton("↻ Refresh")
        self.refresh_btn.setObjectName("refresh_button")
        self.refresh_btn.setToolTip("Re-test camera connectivity")
        self.refresh_btn.clicked.connect(self.verify_cameras)
        preview_header_row.addWidget(self.refresh_btn)

        preview_layout.addLayout(preview_header_row)

        instructions = QLabel(
            "All cameras must have intrinsic calibration completed before proceeding. "
            "Click 'Calibrate Intrinsic' on cameras marked as needing calibration."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
        preview_layout.addWidget(instructions)

        # Scrollable camera grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setMinimumHeight(300)

        self.cameras_container = QWidget()
        self.cameras_layout = QGridLayout(self.cameras_container)
        self.cameras_layout.setSpacing(15)
        self.cameras_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll_area.setWidget(self.cameras_container)
        preview_layout.addWidget(scroll_area, 1)

        main_layout.addWidget(preview_frame, 1)

        # Bottom navigation bar
        nav_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("< Back to Configuration")
        self.cancel_btn.setObjectName("cancel_button")
        self.cancel_btn.setToolTip("Return to platform configuration")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.cancel_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.cancel_btn)

        nav_layout.addStretch()

        self.next_btn = QPushButton("Next: Extrinsic Calibration >")
        self.next_btn.setObjectName("nav_button")
        self.next_btn.setToolTip("Proceed to extrinsic calibration (requires all intrinsics)")
        self.next_btn.clicked.connect(self._on_next_clicked)
        self.next_btn.setEnabled(False)
        self.next_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.next_btn)

        main_layout.addLayout(nav_layout)

    def _setup_config_table(self):
        """Setup the read-only configuration table."""
        columns = ['#', 'Camera ID', 'Type', 'Model', 'Position', 'IP Address', 'Intrinsic']
        self.config_table.setColumnCount(len(columns))
        self.config_table.setHorizontalHeaderLabels(columns)

        header = self.config_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Fixed)

        self.config_table.setColumnWidth(0, 40)
        self.config_table.setColumnWidth(2, 110)
        self.config_table.setColumnWidth(3, 90)
        self.config_table.setColumnWidth(6, 80)

        self.config_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.config_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.config_table.verticalHeader().setVisible(False)
        self.config_table.setAlternatingRowColors(True)

    def _populate_config_table(self):
        """Populate the configuration table with camera data."""
        self.config_table.setRowCount(0)

        for camera in self.config.cameras:
            row = self.config_table.rowCount()
            self.config_table.insertRow(row)

            # Camera number
            num_item = QTableWidgetItem(str(camera.camera_number))
            num_item.setTextAlignment(Qt.AlignCenter)
            self.config_table.setItem(row, 0, num_item)

            # Camera ID
            self.config_table.setItem(row, 1, QTableWidgetItem(camera.camera_id))

            # Type
            self.config_table.setItem(row, 2, QTableWidgetItem(camera.camera_type))

            # Model
            self.config_table.setItem(row, 3, QTableWidgetItem(camera.camera_model))

            # Position
            self.config_table.setItem(row, 4, QTableWidgetItem(camera.mounting_position))

            # IP Address
            self.config_table.setItem(row, 5, QTableWidgetItem(camera.ip_address))

            # Intrinsic status
            has_intrinsic = camera.has_intrinsic_calibration(self.base_path)
            status_item = QTableWidgetItem("✓" if has_intrinsic else "✗")
            status_item.setTextAlignment(Qt.AlignCenter)
            if has_intrinsic:
                status_item.setForeground(Qt.darkGreen)
            else:
                status_item.setForeground(Qt.red)
            self.config_table.setItem(row, 6, status_item)

            self.config_table.setRowHeight(row, 35)

    def set_config(self, config: PlatformConfiguration):
        """Set a new configuration and update UI."""
        self.config = config
        self._populate_config_table()
        self._rebuild_camera_grid()

    def set_base_path(self, path: str):
        """Set the base path."""
        self.base_path = path

    def _rebuild_camera_grid(self):
        """Rebuild the camera preview grid from configuration."""
        # Clear existing cards
        # Wait for threads to finish before deleting to avoid crash
        for card in self.camera_cards:
            card.stop_streaming(wait_for_finish=True)
            card.deleteLater()
        self.camera_cards.clear()

        # Clear layout
        while self.cameras_layout.count():
            item = self.cameras_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create camera cards
        row = 0
        col = 0
        max_cols = 2

        for camera in self.config.cameras:
            has_intrinsic = camera.has_intrinsic_calibration(self.base_path)
            card = CameraPreviewCard(
                camera_number=camera.camera_number,
                camera_id=camera.camera_id,
                ip_address=camera.ip_address,
                mounting_position=camera.mounting_position,
                has_intrinsic=has_intrinsic,
                parent=self
            )
            card.calibrate_requested.connect(self._on_calibrate_requested)
            self.camera_cards.append(card)
            self.cameras_layout.addWidget(card, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        self.cameras_layout.setRowStretch(row + 1, 1)
        self.cameras_layout.setColumnStretch(max_cols, 1)

    def verify_cameras(self):
        """Verify all cameras are reachable and check intrinsic status (threaded)."""
        self.ping_failed_cameras.clear()
        self._ping_results.clear()
        self.next_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)  # Disable refresh during testing

        self._update_status("testing", "Testing camera connectivity...")

        # Clean up any existing ping thread
        self._cleanup_ping_thread()

        # Build camera IPs dict
        camera_ips = {card.camera_number: card.ip_address for card in self.camera_cards}

        # Create worker and thread for background ping tests
        self._ping_thread = QThread()
        self._ping_worker = CameraConnectionWorker(camera_ips)
        self._ping_worker.moveToThread(self._ping_thread)

        # Connect signals
        self._ping_thread.started.connect(self._ping_worker.run_ping_tests)
        self._ping_worker.ping_result.connect(self._on_ping_result)
        self._ping_worker.all_complete.connect(self._on_all_pings_complete)

        # Start the thread
        self._ping_thread.start()

    def _on_ping_result(self, camera_number: int, ping_ok: bool):
        """Handle individual ping result from worker thread."""
        self._ping_results[camera_number] = ping_ok

        # Update the card UI
        card = next((c for c in self.camera_cards if c.camera_number == camera_number), None)
        if card:
            card.set_ping_status(ping_ok)
            if not ping_ok:
                self.ping_failed_cameras.append(card.camera_id)

    def _on_all_pings_complete(self):
        """Handle completion of all ping tests."""
        # Clean up thread
        self._cleanup_ping_thread()

        # Re-enable refresh button
        self.refresh_btn.setEnabled(True)

        # Check intrinsic calibration status (this is fast, no threading needed)
        all_calibrated = True
        for card in self.camera_cards:
            camera = next((c for c in self.config.cameras if c.camera_number == card.camera_number), None)
            if camera:
                has_intrinsic = camera.has_intrinsic_calibration(self.base_path)
                card.set_intrinsic_status(has_intrinsic)
                if not has_intrinsic:
                    all_calibrated = False

        # Update table intrinsic status
        self._populate_config_table()

        # Check connection results
        all_connected = len(self.ping_failed_cameras) == 0

        # Update status message
        if not all_connected:
            failed_list = ", ".join(self.ping_failed_cameras)
            self._update_status(
                "error",
                f"Camera(s) not detected: {failed_list}. Check connections."
            )
            self._show_camera_failure_dialog()
        elif not all_calibrated:
            self._update_status(
                "warning",
                "Some cameras need intrinsic calibration. Complete calibration to proceed."
            )
        else:
            self._update_status("success", "All cameras connected and calibrated!")
            self.next_btn.setEnabled(True)

        # Start video streaming for connected cameras (each starts its own thread)
        self.start_all_streams()

    def _cleanup_ping_thread(self):
        """Clean up the ping worker thread."""
        if self._ping_worker:
            self._ping_worker.stop()

        if self._ping_thread:
            self._ping_thread.quit()
            # Wait up to 3 seconds for ping tests to complete
            if not self._ping_thread.wait(3000):
                # Thread didn't stop gracefully, force terminate
                self._ping_thread.terminate()
                self._ping_thread.wait(1000)
            self._ping_thread = None
        self._ping_worker = None

    def _ping_device(self, ip_address: str) -> bool:
        """Ping a device to check if it's reachable."""
        if not ip_address:
            return False

        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            timeout_param = '-w' if platform.system().lower() == 'windows' else '-W'

            result = subprocess.run(
                ['ping', param, '1', timeout_param, '2', ip_address],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return False

    def _update_status(self, status_type: str, message: str):
        """Update the status message area."""
        if status_type == "testing":
            self.status_frame.setStyleSheet(f"""
                QFrame {{ background-color: {COLORS['table_header']}; border-radius: 6px; padding: 10px; }}
            """)
            self.status_icon.setText("⏳")
            self.status_icon.setStyleSheet("font-size: 20px;")
        elif status_type == "success":
            self.status_frame.setStyleSheet(f"""
                QFrame {{ background-color: #E8F5E9; border-radius: 6px; padding: 10px; }}
            """)
            self.status_icon.setText("✓")
            self.status_icon.setStyleSheet(f"font-size: 20px; color: {COLORS['success']};")
        elif status_type == "warning":
            self.status_frame.setStyleSheet(f"""
                QFrame {{ background-color: #FFF8E1; border-radius: 6px; padding: 10px; }}
            """)
            self.status_icon.setText("⚠")
            self.status_icon.setStyleSheet(f"font-size: 20px; color: {COLORS['warning']};")
        elif status_type == "error":
            self.status_frame.setStyleSheet(f"""
                QFrame {{ background-color: #FFEBEE; border-radius: 6px; padding: 10px; }}
            """)
            self.status_icon.setText("✗")
            self.status_icon.setStyleSheet(f"font-size: 20px; color: {COLORS['danger']};")

        self.status_message.setText(message)

    def _show_camera_failure_dialog(self):
        """Show a dialog when cameras fail ping test."""
        failed_list = "\n".join(f"  - {cam_id}" for cam_id in self.ping_failed_cameras)
        QMessageBox.warning(
            self,
            "Camera Connection Failed",
            f"The following camera(s) could not be detected:\n\n"
            f"{failed_list}\n\n"
            "Please check:\n"
            "  1. Camera is powered on\n"
            "  2. Network cable is connected\n"
            "  3. IP address is correct\n\n"
            "Go back to configuration if you need to change settings."
        )

    def _on_calibrate_requested(self, camera_number: int):
        """Handle intrinsic calibration request for a camera."""
        camera = next((c for c in self.config.cameras if c.camera_number == camera_number), None)
        if not camera:
            return

        # Check if already calibrated
        intrinsic_path = Path(self.base_path) / f"camera_intrinsic/camera_intrinsics_{camera.camera_id}.json"
        if intrinsic_path.exists():
            reply = QMessageBox.question(
                self,
                "Existing Calibration Found",
                f"Camera '{camera.camera_id}' already has an intrinsic calibration file.\n\n"
                "Do you want to perform a new calibration?\n"
                "(This will overwrite the existing calibration)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Stop streaming for this camera during calibration
        card = next((c for c in self.camera_cards if c.camera_number == camera_number), None)
        if card:
            card.stop_streaming()

        # Launch calibration dialog
        dialog = IntrinsicCalibrationDialog(
            camera_id=camera.camera_id,
            ip_address=camera.ip_address,
            base_path=self.base_path,
            parent=self
        )

        dialog.calibration_completed.connect(
            lambda success, msg: self._on_calibration_completed(camera_number, success, msg)
        )

        dialog.exec()

    def _on_calibration_completed(self, camera_number: int, success: bool, message: str):
        """Handle calibration completion."""
        if success:
            # Update intrinsic status
            camera = next((c for c in self.config.cameras if c.camera_number == camera_number), None)
            if camera:
                has_intrinsic = camera.has_intrinsic_calibration(self.base_path)
                card = next((c for c in self.camera_cards if c.camera_number == camera_number), None)
                if card:
                    card.set_intrinsic_status(has_intrinsic)

            # Update table
            self._populate_config_table()

            # Check if all cameras now have intrinsics
            all_calibrated = all(
                c.has_intrinsic_calibration(self.base_path) for c in self.config.cameras
            )
            if all_calibrated and not self.ping_failed_cameras:
                self._update_status("success", "All cameras connected and calibrated!")
                self.next_btn.setEnabled(True)
            else:
                self._update_status(
                    "warning",
                    "Some cameras still need intrinsic calibration."
                )

        # Restart streaming
        card = next((c for c in self.camera_cards if c.camera_number == camera_number), None)
        if card:
            card.start_streaming()

    def _on_cancel_clicked(self):
        """Handle Cancel/Back button click."""
        self.stop_all_streams()
        self.sensor_widget.stop_updates()
        self.cancel_requested.emit()

    def _on_next_clicked(self):
        """Handle Next button click."""
        # Verify all intrinsics are present
        missing_intrinsics = [
            c.camera_id for c in self.config.cameras
            if not c.has_intrinsic_calibration(self.base_path)
        ]

        if missing_intrinsics:
            QMessageBox.warning(
                self,
                "Intrinsic Calibration Required",
                f"The following cameras are missing intrinsic calibration:\n\n"
                + "\n".join(f"  - {cam_id}" for cam_id in missing_intrinsics) +
                "\n\nPlease complete intrinsic calibration for all cameras before proceeding."
            )
            return

        self.stop_all_streams()
        self.sensor_widget.stop_updates()

        QMessageBox.information(
            self,
            "Next Step: Extrinsic Calibration",
            f"Platform: {self.config.platform_type} - {self.config.platform_id}\n"
            f"Cameras verified: {len(self.config.cameras)}\n\n"
            "All cameras have intrinsic calibration.\n"
            "Proceeding to extrinsic calibration."
        )

        self.next_requested.emit(self.config)

    def start_all_streams(self):
        """Start video streaming for all camera cards that passed ping."""
        for card in self.camera_cards:
            if card.ping_status:
                card.start_streaming()

    def stop_all_streams(self):
        """Stop video streaming for all camera cards."""
        for card in self.camera_cards:
            card.stop_streaming()

    def showEvent(self, event):
        """Start sensor updates and verify cameras when screen is shown."""
        super().showEvent(event)
        self.sensor_widget.start_updates()

    def hideEvent(self, event):
        """Stop all streams and sensor updates when screen is hidden."""
        self._cleanup_ping_thread()
        self.stop_all_streams()
        self.sensor_widget.stop_updates()
        super().hideEvent(event)

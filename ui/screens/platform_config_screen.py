"""
Platform Configuration Screen (Screen 2)

Configure platform information and camera setup:
- Platform type and ID
- Camera configuration table (type, model, mounting position, IP)
- Camera preview with live feed
- Validation for camera limits and positions
"""

from pathlib import Path
import subprocess
import platform
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSizePolicy, QMessageBox,
    QScrollArea, QGridLayout, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject
from PySide6.QtGui import QPixmap, QImage

import numpy as np
import cv2

from ..styles import COLORS
from ..data_models import (
    PlatformConfiguration, CameraDefinition,
    MOUNTING_POSITIONS, CAMERA_TYPES, CAMERA_MODELS, PLATFORM_TYPES,
    CAMERA_ROLES, VALID_3_1_POSITIONS, MAX_CAMERAS, MAX_AI_CENTRAL_CAMERAS
)

# Import camera streaming from intrinsic_calibration module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from intrinsic_calibration import NetworkCameraSource
except ImportError:
    NetworkCameraSource = None


class CameraStreamWorker(QObject):
    """Worker for connecting a single camera stream in background thread."""

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
                timeout=5.0
            )

            if self.camera_source.connect():
                self.connection_ready.emit(True, "")
            else:
                error_msg = getattr(self.camera_source, 'last_error', None) or "Connection failed"
                self.connection_ready.emit(False, error_msg)
        except Exception as e:
            self.connection_ready.emit(False, str(e))

    def get_camera_source(self):
        """Get the connected camera source."""
        return self.camera_source


class CameraPreviewWidget(QFrame):
    """Widget showing a single camera preview."""

    def __init__(self, camera_id: str, ip_address: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.ip_address = ip_address
        self._pending_ip = None  # Pending IP for debounced restart
        self.camera_source: Optional['NetworkCameraSource'] = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self._is_streaming = False
        self._is_connecting = False  # Track if connection is in progress
        self._stream_thread: Optional[QThread] = None
        self._stream_worker: Optional[CameraStreamWorker] = None
        # Debounce timer to avoid restarting on every keystroke
        self._restart_debounce = QTimer()
        self._restart_debounce.setSingleShot(True)
        self._restart_debounce.timeout.connect(self._debounced_restart)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['white']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        self.setFixedSize(200, 140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Camera ID label
        self.id_label = QLabel(self.camera_id or "Camera")
        self.id_label.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLORS['primary']};")
        self.id_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.id_label)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(188, 105)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"""
            background-color: #1a1a1a;
            border-radius: 4px;
            color: {COLORS['text_muted']};
            font-size: 10px;
        """)
        self.video_label.setText("No preview")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

    def update_camera_info(self, camera_id: str, ip_address: str):
        """Update camera information and restart stream if needed."""
        old_ip = self.ip_address
        self.camera_id = camera_id
        self.ip_address = ip_address
        self.id_label.setText(camera_id or "Camera")
        # Restart if IP changed and was streaming OR connecting
        # Use debounce to avoid restarting on every keystroke
        if old_ip != ip_address and (self._is_streaming or self._is_connecting):
            self._pending_ip = ip_address
            self._restart_debounce.start(500)  # 500ms debounce

    def _debounced_restart(self):
        """Restart streaming after debounce delay."""
        if self._pending_ip and self._pending_ip == self.ip_address:
            self.stop_streaming()
            self.start_streaming()
        self._pending_ip = None

    def start_streaming(self):
        """Start camera streaming in a background thread."""
        if self._is_streaming or self._is_connecting or NetworkCameraSource is None:
            return
        if not self.ip_address:
            self.video_label.setText("No IP")
            return

        self._is_connecting = True
        self.video_label.setText("Connecting...")

        # Clean up any previous thread
        self._cleanup_stream_thread()

        # Create worker and thread for non-blocking connection
        self._stream_thread = QThread()
        self._stream_worker = CameraStreamWorker(self.ip_address)
        self._stream_worker.moveToThread(self._stream_thread)

        # Connect signals
        self._stream_thread.started.connect(self._stream_worker.connect_camera)
        self._stream_worker.connection_ready.connect(self._on_stream_connected)

        # Start thread - connection happens in background
        self._stream_thread.start()

    def _on_stream_connected(self, success: bool, error_msg: str):
        """Handle camera connection result from worker thread."""
        self._is_connecting = False

        if success and self._stream_worker:
            self.camera_source = self._stream_worker.get_camera_source()
            self._is_streaming = True
            self.timer.start(100)  # 10 FPS for preview
        else:
            error_display = error_msg[:20] + "..." if len(error_msg) > 20 else error_msg
            self.video_label.setText(f"Failed: {error_display}" if error_msg else "Connection failed")

        # Clean up thread after connection attempt
        self._cleanup_stream_thread()

    def _cleanup_stream_thread(self):
        """Clean up the stream worker thread."""
        if self._stream_worker:
            # Disconnect signals to prevent stale callbacks from old connections
            try:
                self._stream_worker.connection_ready.disconnect(self._on_stream_connected)
            except (RuntimeError, TypeError):
                pass  # Signal might not be connected or already disconnected
            self._stream_worker.stop()
            self._stream_worker = None
        if self._stream_thread:
            self._stream_thread.quit()
            self._stream_thread.wait(100)  # Brief wait - don't block UI long
            self._stream_thread = None

    def stop_streaming(self):
        """Stop camera streaming."""
        self.timer.stop()
        self._restart_debounce.stop()  # Cancel any pending restart
        self._pending_ip = None
        self._is_streaming = False
        self._is_connecting = False
        self._cleanup_stream_thread()
        if self.camera_source:
            self.camera_source.release()
            self.camera_source = None

    def _update_frame(self):
        """Update video frame."""
        if not self.camera_source:
            return
        frame = self.camera_source.get_image()
        if frame is not None:
            self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        """Display frame on label."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        scale = min(188/w, 105/h)
        new_w, new_h = int(w*scale), int(h*scale)
        rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = rgb_frame.shape[:2]
        q_image = QImage(rgb_frame.data, w, h, 3*w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))


class CameraTableWidget(QTableWidget):
    """Custom table widget for camera configuration."""

    camera_removed = Signal(int)
    camera_data_changed = Signal()  # Emitted when any camera data changes

    COLUMNS = ['#', 'Camera ID', 'Type', 'Role', 'Camera Model', 'Mounting Position', 'IP Address', 'Action']
    COLUMN_WIDTHS = [40, 100, 90, 80, 110, 160, 140, 80]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_path = "."
        self._setup_table()

    def _setup_table(self):
        """Setup table structure."""
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)

        header = self.horizontalHeader()
        for i, width in enumerate(self.COLUMN_WIDTHS):
            self.setColumnWidth(i, width)

        header.setSectionResizeMode(0, QHeaderView.Fixed)    # #
        header.setSectionResizeMode(1, QHeaderView.Fixed)    # Camera ID
        header.setSectionResizeMode(2, QHeaderView.Fixed)    # Type
        header.setSectionResizeMode(3, QHeaderView.Fixed)    # Role
        header.setSectionResizeMode(4, QHeaderView.Fixed)    # Camera Model
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # Mounting Position
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # IP Address
        header.setSectionResizeMode(7, QHeaderView.Fixed)    # Action

        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(True)
        self.setAlternatingRowColors(True)

    def add_camera_row(self, camera: CameraDefinition, base_path: str = ".") -> int:
        """Add a new camera row to the table."""
        self.base_path = base_path
        row = self.rowCount()
        self.insertRow(row)

        # Camera number (read-only)
        num_item = QTableWidgetItem(str(camera.camera_number))
        num_item.setTextAlignment(Qt.AlignCenter)
        num_item.setFlags(num_item.flags() & ~Qt.ItemIsEditable)
        self.setItem(row, 0, num_item)

        # Camera ID input
        camera_id_edit = QLineEdit(camera.camera_id)
        camera_id_edit.setPlaceholderText("cam_1")
        camera_id_edit.setToolTip("Unique identifier for this camera")
        camera_id_edit.textChanged.connect(lambda: self.camera_data_changed.emit())
        self.setCellWidget(row, 1, camera_id_edit)

        # Type dropdown
        type_combo = QComboBox()
        type_combo.addItems(CAMERA_TYPES)
        type_combo.setCurrentText(camera.camera_type)
        type_combo.setToolTip("Select camera type: AI Central (1 max), 1:1, or 3:1")
        type_combo.currentTextChanged.connect(lambda text, r=row: self._on_type_changed(r, text))
        self.setCellWidget(row, 2, type_combo)

        # Role dropdown (Manager/Worker for 3:1 only)
        role_combo = QComboBox()
        role_combo.addItem("")  # Empty option
        role_combo.addItems(CAMERA_ROLES)
        role_combo.setCurrentText(camera.camera_role if camera.camera_type == "3:1" else "")
        role_combo.setToolTip("Select role (Manager/Worker) - only for 3:1 cameras")
        role_combo.setEnabled(camera.camera_type == "3:1")
        role_combo.currentTextChanged.connect(lambda: self.camera_data_changed.emit())
        self.setCellWidget(row, 3, role_combo)

        # Model dropdown
        model_combo = QComboBox()
        model_combo.addItems(CAMERA_MODELS)
        model_combo.setCurrentText(camera.camera_model)
        model_combo.setToolTip("Select camera hardware model")
        model_combo.currentTextChanged.connect(lambda: self.camera_data_changed.emit())
        self.setCellWidget(row, 4, model_combo)

        # Mounting Position dropdown
        position_combo = QComboBox()
        position_combo.addItems(MOUNTING_POSITIONS)
        position_combo.setCurrentText(camera.mounting_position)
        position_combo.setToolTip("Select mounting position. 3:1 cameras can only be Front/Rear Center")
        position_combo.currentTextChanged.connect(lambda text, r=row: self._on_position_changed(r, text))
        self.setCellWidget(row, 5, position_combo)

        # IP Address input
        ip_edit = QLineEdit(camera.ip_address)
        ip_edit.setPlaceholderText("192.168.1.xxx")
        ip_edit.setToolTip("Camera IP address for network connection")
        ip_edit.textChanged.connect(lambda: self.camera_data_changed.emit())
        self.setCellWidget(row, 6, ip_edit)

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("remove_button")
        remove_btn.setToolTip("Remove this camera from the configuration")
        remove_btn.clicked.connect(lambda checked, r=row: self._on_remove_clicked(r))
        remove_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        remove_container = QWidget()
        remove_layout = QHBoxLayout(remove_container)
        remove_layout.setContentsMargins(2, 2, 2, 2)
        remove_layout.addWidget(remove_btn, alignment=Qt.AlignCenter)
        self.setCellWidget(row, 7, remove_container)

        self.setRowHeight(row, 45)
        return row

    def _on_type_changed(self, row: int, text: str):
        """Handle camera type change."""
        role_combo = self.cellWidget(row, 3)
        position_combo = self.cellWidget(row, 5)

        if role_combo:
            if text == "3:1":
                role_combo.setEnabled(True)
            else:
                role_combo.setEnabled(False)
                role_combo.setCurrentText("")

        # Validate position for 3:1
        if text == "3:1" and position_combo:
            current_pos = position_combo.currentText()
            if current_pos not in VALID_3_1_POSITIONS and current_pos != "N/A":
                position_combo.setCurrentText("N/A")

        self.camera_data_changed.emit()

    def _on_position_changed(self, row: int, text: str):
        """Handle mounting position change."""
        self.camera_data_changed.emit()

    def _on_remove_clicked(self, row: int):
        """Handle remove button click."""
        self.camera_removed.emit(row)

    def get_camera_data(self, row: int) -> dict:
        """Get camera data from a specific row."""
        if row >= self.rowCount():
            return {}

        return {
            'camera_number': int(self.item(row, 0).text()),
            'camera_id': self.cellWidget(row, 1).text(),
            'camera_type': self.cellWidget(row, 2).currentText(),
            'camera_role': self.cellWidget(row, 3).currentText(),
            'camera_model': self.cellWidget(row, 4).currentText(),
            'mounting_position': self.cellWidget(row, 5).currentText(),
            'ip_address': self.cellWidget(row, 6).text(),
        }

    def get_all_cameras(self) -> list:
        """Get data for all cameras in the table."""
        cameras = []
        for row in range(self.rowCount()):
            cameras.append(self.get_camera_data(row))
        return cameras

    def clear_all(self):
        """Clear all rows from the table."""
        self.setRowCount(0)

    def highlight_duplicate_positions(self, duplicate_rows: list):
        """Highlight rows with duplicate positions in red."""
        for row in range(self.rowCount()):
            position_combo = self.cellWidget(row, 5)
            if position_combo:
                if row in duplicate_rows:
                    position_combo.setStyleSheet(f"border: 2px solid {COLORS['danger']}; background-color: #FFEBEE;")
                else:
                    position_combo.setStyleSheet("")


class PlatformConfigScreen(QWidget):
    """
    Platform Configuration Screen

    Step 1 of 6 in the calibration workflow.
    Configures platform information and camera setup with live preview.
    """

    cancel_requested = Signal()
    next_requested = Signal(PlatformConfiguration)

    def __init__(self, config: PlatformConfiguration = None, parent=None):
        super().__init__(parent)
        self.config = config or PlatformConfiguration()
        self.base_path = "."
        self.preview_widgets = []
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(15)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)

        screen_label = QLabel("Screen 2: Platform Setup")
        screen_label.setObjectName("screen_indicator")
        header_layout.addWidget(screen_label)

        title = QLabel("Platform Configuration")
        title.setObjectName("title")
        header_layout.addWidget(title)

        step_label = QLabel("Step 1 of 6")
        step_label.setObjectName("step_indicator")
        header_layout.addWidget(step_label)

        main_layout.addLayout(header_layout)

        # Content area with splitter for table and preview
        content_splitter = QSplitter(Qt.Horizontal)

        # Left side - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # Platform Information section
        platform_frame = QFrame()
        platform_frame.setObjectName("card")
        platform_layout = QVBoxLayout(platform_frame)
        platform_layout.setSpacing(10)

        platform_header = QLabel("Platform Information")
        platform_header.setObjectName("section_header")
        platform_layout.addWidget(platform_header)

        # Platform Type
        type_row = QHBoxLayout()
        type_label = QLabel("Platform Type:")
        type_label.setToolTip("Select the type of platform/vehicle")
        self.platform_type_combo = QComboBox()
        self.platform_type_combo.addItems(PLATFORM_TYPES)
        self.platform_type_combo.setFixedWidth(180)
        type_row.addWidget(type_label)
        type_row.addWidget(self.platform_type_combo)
        type_row.addStretch()
        platform_layout.addLayout(type_row)

        # Platform ID
        id_row = QHBoxLayout()
        id_label = QLabel("Platform ID:")
        id_label.setToolTip("Unique identifier for this platform (e.g., PQ4459-001)")
        self.platform_id_edit = QLineEdit()
        self.platform_id_edit.setPlaceholderText("e.g., PQ4459-001")
        self.platform_id_edit.setFixedWidth(180)
        id_row.addWidget(id_label)
        id_row.addWidget(self.platform_id_edit)
        id_row.addStretch()
        platform_layout.addLayout(id_row)

        left_layout.addWidget(platform_frame)

        # Camera Configuration section
        camera_frame = QFrame()
        camera_frame.setObjectName("card")
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setSpacing(10)

        # Header row with title and buttons
        camera_header_row = QHBoxLayout()
        camera_header = QLabel("Camera Configuration")
        camera_header.setObjectName("section_header")
        camera_header_row.addWidget(camera_header)
        camera_header_row.addStretch()

        # Camera count label
        self.camera_count_label = QLabel("0/6 cameras")
        self.camera_count_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.camera_count_label.setToolTip(f"Maximum {MAX_CAMERAS} cameras allowed")
        camera_header_row.addWidget(self.camera_count_label)

        # Refresh button
        self.refresh_btn = QPushButton("↻ Refresh Preview")
        self.refresh_btn.setObjectName("refresh_button")
        self.refresh_btn.setToolTip("Refresh camera preview streams")
        self.refresh_btn.clicked.connect(self._on_refresh_preview)
        camera_header_row.addWidget(self.refresh_btn)

        camera_layout.addLayout(camera_header_row)

        # Validation message area
        self.validation_label = QLabel("")
        self.validation_label.setObjectName("error_label")
        self.validation_label.setWordWrap(True)
        self.validation_label.hide()
        camera_layout.addWidget(self.validation_label)

        # Camera table
        self.camera_table = CameraTableWidget()
        self.camera_table.camera_removed.connect(self._on_camera_removed)
        self.camera_table.camera_data_changed.connect(self._on_camera_data_changed)
        self.camera_table.setMinimumHeight(180)
        camera_layout.addWidget(self.camera_table)

        # Add Camera button
        add_btn_layout = QHBoxLayout()
        self.add_camera_btn = QPushButton("+ Add Camera")
        self.add_camera_btn.setObjectName("add_button")
        self.add_camera_btn.setToolTip(f"Add a new camera (max {MAX_CAMERAS})")
        self.add_camera_btn.clicked.connect(self._on_add_camera)
        add_btn_layout.addWidget(self.add_camera_btn)
        add_btn_layout.addStretch()
        camera_layout.addLayout(add_btn_layout)

        left_layout.addWidget(camera_frame, 1)

        content_splitter.addWidget(left_widget)

        # Right side - Camera Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)

        preview_frame = QFrame()
        preview_frame.setObjectName("card")
        preview_layout = QVBoxLayout(preview_frame)

        preview_header = QLabel("Camera Preview")
        preview_header.setObjectName("section_header")
        preview_layout.addWidget(preview_header)

        preview_hint = QLabel("Live video feeds from configured cameras")
        preview_hint.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
        preview_layout.addWidget(preview_hint)

        # Scrollable preview area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        self.preview_container = QWidget()
        self.preview_grid = QGridLayout(self.preview_container)
        self.preview_grid.setSpacing(10)
        self.preview_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll_area.setWidget(self.preview_container)
        preview_layout.addWidget(scroll_area, 1)

        right_layout.addWidget(preview_frame)
        content_splitter.addWidget(right_widget)

        # Set initial sizes
        content_splitter.setSizes([600, 450])

        main_layout.addWidget(content_splitter, 1)

        # Bottom navigation bar
        nav_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancel_button")
        self.cancel_btn.setToolTip("Return to welcome screen")
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        nav_layout.addWidget(self.cancel_btn)

        nav_layout.addStretch()

        self.next_btn = QPushButton("Next: Verify Hardware >")
        self.next_btn.setObjectName("nav_button")
        self.next_btn.setToolTip("Proceed to hardware verification")
        self.next_btn.clicked.connect(self._on_next_clicked)
        nav_layout.addWidget(self.next_btn)

        main_layout.addLayout(nav_layout)

    def _load_config(self):
        """Load configuration into UI."""
        self.platform_type_combo.setCurrentText(self.config.platform_type)
        self.platform_id_edit.setText(self.config.platform_id)

        self.camera_table.clear_all()
        for camera in self.config.cameras:
            self.camera_table.add_camera_row(camera, self.base_path)

        # Add default cameras if none exist
        if not self.config.cameras:
            for _ in range(4):
                self._on_add_camera()

        self._update_camera_count()
        self._rebuild_preview_grid()

    def _on_add_camera(self):
        """Add a new camera to the configuration."""
        if len(self.config.cameras) >= MAX_CAMERAS:
            QMessageBox.warning(
                self,
                "Camera Limit Reached",
                f"Maximum of {MAX_CAMERAS} cameras allowed."
            )
            return

        camera = self.config.add_camera()
        self.camera_table.add_camera_row(camera, self.base_path)
        self._update_camera_count()
        # Add only the new preview widget, don't rebuild all existing ones
        self._add_single_preview(camera)

    def _add_single_preview(self, camera):
        """Add a single camera preview widget without affecting existing previews."""
        preview = CameraPreviewWidget(camera.camera_id, camera.ip_address)
        self.preview_widgets.append(preview)

        # Calculate grid position (2 columns)
        index = len(self.preview_widgets) - 1
        row = index // 2
        col = index % 2
        self.preview_grid.addWidget(preview, row, col)

        # Start streaming for this preview if screen is visible
        if self.isVisible():
            preview.start_streaming()

    def _on_camera_removed(self, row: int):
        """Remove a camera from the configuration."""
        if self.camera_table.rowCount() <= 1:
            QMessageBox.warning(
                self,
                "Cannot Remove",
                "At least one camera must be configured."
            )
            return

        self.config.remove_camera(row)
        self._reload_camera_table()
        self._update_camera_count()
        self._rebuild_preview_grid()

    def _on_camera_data_changed(self):
        """Handle any camera data change - validate and update preview."""
        self._validate_configuration()
        self._update_preview_info()

    def _update_camera_count(self):
        """Update the camera count label."""
        count = len(self.config.cameras)
        self.camera_count_label.setText(f"{count}/{MAX_CAMERAS} cameras")

        if count >= MAX_CAMERAS:
            self.add_camera_btn.setEnabled(False)
            self.camera_count_label.setStyleSheet(f"color: {COLORS['warning']}; font-weight: bold;")
        else:
            self.add_camera_btn.setEnabled(True)
            self.camera_count_label.setStyleSheet(f"color: {COLORS['text_muted']};")

    def _validate_configuration(self) -> tuple:
        """Validate current configuration. Returns (is_valid, error_messages)."""
        errors = []
        duplicate_rows = []
        cameras = self.camera_table.get_all_cameras()

        # Check AI Central count
        ai_central_count = sum(1 for cam in cameras if cam['camera_type'] == 'AI Central')
        if ai_central_count > MAX_AI_CENTRAL_CAMERAS:
            errors.append(f"Only {MAX_AI_CENTRAL_CAMERAS} AI Central camera allowed (found {ai_central_count})")

        # Check 3:1 positions
        for i, cam in enumerate(cameras):
            if cam['camera_type'] == '3:1':
                pos = cam['mounting_position']
                if pos not in VALID_3_1_POSITIONS and pos != "N/A":
                    errors.append(f"Camera {cam['camera_number']}: 3:1 cameras can only be at Front Center or Rear Center")

        # Check for N/A positions
        na_cameras = [cam['camera_number'] for cam in cameras if cam['mounting_position'] == 'N/A']
        if na_cameras:
            errors.append(f"Camera(s) {', '.join(map(str, na_cameras))}: Position must be selected (currently N/A)")

        # Check for duplicate positions (excluding N/A)
        positions = {}
        for i, cam in enumerate(cameras):
            pos = cam['mounting_position']
            if pos != "N/A":
                if pos in positions:
                    duplicate_rows.append(i)
                    duplicate_rows.append(positions[pos])
                    errors.append(f"Duplicate position '{pos}' found")
                else:
                    positions[pos] = i

        # Check 3:1 cameras have role selected
        for cam in cameras:
            if cam['camera_type'] == '3:1' and not cam['camera_role']:
                errors.append(f"Camera {cam['camera_number']}: 3:1 cameras must have a role (Manager/Worker)")

        # Highlight duplicate positions
        self.camera_table.highlight_duplicate_positions(list(set(duplicate_rows)))

        # Update validation label
        if errors:
            self.validation_label.setText("⚠ " + "; ".join(errors))
            self.validation_label.show()
            return False, errors
        else:
            self.validation_label.hide()
            return True, []

    def _rebuild_preview_grid(self):
        """Rebuild the camera preview grid."""
        # Stop and clear existing previews
        for widget in self.preview_widgets:
            widget.stop_streaming()
            widget.deleteLater()
        self.preview_widgets.clear()

        # Clear layout
        while self.preview_grid.count():
            item = self.preview_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create preview widgets for each camera
        row, col = 0, 0
        for camera in self.config.cameras:
            preview = CameraPreviewWidget(camera.camera_id, camera.ip_address)
            self.preview_widgets.append(preview)
            self.preview_grid.addWidget(preview, row, col)

            col += 1
            if col >= 2:
                col = 0
                row += 1

        # Note: Camera streaming is started in showEvent() when screen is visible
        # Do NOT start cameras here during construction

    def _update_preview_info(self):
        """Update preview widgets with current camera info."""
        cameras = self.camera_table.get_all_cameras()
        for i, preview in enumerate(self.preview_widgets):
            if i < len(cameras):
                preview.update_camera_info(cameras[i]['camera_id'], cameras[i]['ip_address'])

    def _start_all_previews(self):
        """Start streaming for all preview widgets."""
        for preview in self.preview_widgets:
            preview.start_streaming()

    def _stop_all_previews(self):
        """Stop streaming for all preview widgets."""
        for preview in self.preview_widgets:
            preview.stop_streaming()

    def _on_refresh_preview(self):
        """Refresh all camera previews."""
        self._stop_all_previews()
        self._update_config_from_ui()
        self._rebuild_preview_grid()
        # Restart streaming after refresh (when user explicitly clicks refresh)
        QTimer.singleShot(500, self._start_all_previews)

    def _reload_camera_table(self):
        """Reload the camera table from config."""
        self.camera_table.clear_all()
        for camera in self.config.cameras:
            self.camera_table.add_camera_row(camera, self.base_path)

    def _on_next_clicked(self):
        """Handle Next button click."""
        # Validate platform ID
        if not self.platform_id_edit.text().strip():
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please enter a Platform ID."
            )
            self.platform_id_edit.setFocus()
            return

        # Update config from UI
        self._update_config_from_ui()

        # Validate configuration
        is_valid, errors = self._validate_configuration()
        if not is_valid:
            QMessageBox.warning(
                self,
                "Configuration Error",
                "Please fix the following issues:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return

        # Stop previews before navigating
        self._stop_all_previews()

        # Emit signal with updated config
        self.next_requested.emit(self.config)

    def _update_config_from_ui(self):
        """Update the configuration from current UI state."""
        self.config.platform_type = self.platform_type_combo.currentText()
        self.config.platform_id = self.platform_id_edit.text().strip()

        # Update camera data
        camera_data_list = self.camera_table.get_all_cameras()
        for i, data in enumerate(camera_data_list):
            if i < len(self.config.cameras):
                cam = self.config.cameras[i]
                cam.camera_id = data['camera_id']
                cam.camera_type = data['camera_type']
                cam.camera_role = data['camera_role']
                cam.camera_model = data['camera_model']
                cam.mounting_position = data['mounting_position']
                cam.ip_address = data['ip_address']

    def set_config(self, config: PlatformConfiguration):
        """Set a new configuration and update UI."""
        self.config = config
        self._load_config()

    def set_base_path(self, path: str):
        """Set the base path for intrinsic file detection."""
        self.base_path = path
        self._reload_camera_table()

    def showEvent(self, event):
        """Start previews when screen is shown."""
        super().showEvent(event)
        QTimer.singleShot(500, self._start_all_previews)

    def hideEvent(self, event):
        """Stop previews when screen is hidden."""
        self._stop_all_previews()
        super().hideEvent(event)

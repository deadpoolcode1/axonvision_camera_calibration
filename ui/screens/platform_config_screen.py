"""
Platform Configuration Screen (Screen 2)

Configure platform information and camera setup:
- Platform type and ID
- Camera configuration table (type, model, mounting position, IP)
- Auto-detection of intrinsic calibration files
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from ..styles import COLORS
from ..data_models import (
    PlatformConfiguration, CameraDefinition,
    MOUNTING_POSITIONS, CAMERA_TYPES, CAMERA_MODELS, PLATFORM_TYPES
)


class CameraTableWidget(QTableWidget):
    """Custom table widget for camera configuration."""

    camera_removed = Signal(int)  # Emits row index when camera is removed

    COLUMNS = ['Camera', 'Type', 'Camera Model', 'Mounting Position', 'IP Address', 'Intrinsic', 'Action']
    COLUMN_WIDTHS = [60, 80, 120, 150, 140, 80, 80]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_table()

    def _setup_table(self):
        """Setup table structure."""
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)

        # Set column widths
        header = self.horizontalHeader()
        for i, width in enumerate(self.COLUMN_WIDTHS):
            self.setColumnWidth(i, width)

        # Make last columns stretch
        header.setSectionResizeMode(3, QHeaderView.Stretch)  # Mounting Position
        header.setSectionResizeMode(4, QHeaderView.Stretch)  # IP Address

        # Table settings
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(True)
        self.setAlternatingRowColors(True)

    def add_camera_row(self, camera: CameraDefinition, base_path: str = ".") -> int:
        """Add a new camera row to the table."""
        row = self.rowCount()
        self.insertRow(row)

        # Camera number (read-only label)
        num_item = QTableWidgetItem(str(camera.camera_number))
        num_item.setTextAlignment(Qt.AlignCenter)
        num_item.setFlags(num_item.flags() & ~Qt.ItemIsEditable)
        self.setItem(row, 0, num_item)

        # Type dropdown
        type_combo = QComboBox()
        type_combo.addItems(CAMERA_TYPES)
        type_combo.setCurrentText(camera.camera_type)
        type_combo.currentTextChanged.connect(lambda text, r=row: self._on_type_changed(r, text))
        self.setCellWidget(row, 1, type_combo)

        # Model dropdown
        model_combo = QComboBox()
        model_combo.addItems(CAMERA_MODELS)
        model_combo.setCurrentText(camera.camera_model)
        model_combo.currentTextChanged.connect(lambda text, r=row: self._on_model_changed(r, text))
        self.setCellWidget(row, 2, model_combo)

        # Mounting Position dropdown
        position_combo = QComboBox()
        position_combo.addItems(MOUNTING_POSITIONS)
        position_combo.setCurrentText(camera.mounting_position)
        position_combo.currentTextChanged.connect(lambda text, r=row: self._on_position_changed(r, text))
        self.setCellWidget(row, 3, position_combo)

        # IP Address input
        ip_edit = QLineEdit(camera.ip_address)
        ip_edit.setPlaceholderText("192.168.1.xxx")
        ip_edit.textChanged.connect(lambda text, r=row: self._on_ip_changed(r, text))
        self.setCellWidget(row, 4, ip_edit)

        # Intrinsic status indicator
        has_intrinsic = camera.has_intrinsic_calibration(base_path)
        intrinsic_label = QLabel()
        if has_intrinsic:
            intrinsic_label.setText("\u2713")
            intrinsic_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold; font-size: 16px;")
            intrinsic_label.setToolTip(f"Found: {camera.intrinsic_file_path}")
        else:
            intrinsic_label.setText("\u2717")
            intrinsic_label.setStyleSheet(f"color: {COLORS['danger']}; font-weight: bold; font-size: 16px;")
            intrinsic_label.setToolTip(f"Not found: {camera.intrinsic_file_path}")
        intrinsic_label.setAlignment(Qt.AlignCenter)

        # Wrap in container for proper alignment
        intrinsic_container = QWidget()
        intrinsic_layout = QHBoxLayout(intrinsic_container)
        intrinsic_layout.setContentsMargins(0, 0, 0, 0)
        intrinsic_layout.addWidget(intrinsic_label, alignment=Qt.AlignCenter)
        self.setCellWidget(row, 5, intrinsic_container)

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("remove_button")
        remove_btn.clicked.connect(lambda checked, r=row: self._on_remove_clicked(r))
        self.setCellWidget(row, 6, remove_btn)

        # Adjust row height
        self.setRowHeight(row, 45)

        return row

    def _on_type_changed(self, row: int, text: str):
        """Handle camera type change."""
        # Store data in item for later retrieval
        pass

    def _on_model_changed(self, row: int, text: str):
        """Handle camera model change."""
        pass

    def _on_position_changed(self, row: int, text: str):
        """Handle mounting position change."""
        pass

    def _on_ip_changed(self, row: int, text: str):
        """Handle IP address change."""
        pass

    def _on_remove_clicked(self, row: int):
        """Handle remove button click."""
        self.camera_removed.emit(row)

    def get_camera_data(self, row: int) -> dict:
        """Get camera data from a specific row."""
        if row >= self.rowCount():
            return {}

        return {
            'camera_number': int(self.item(row, 0).text()),
            'camera_type': self.cellWidget(row, 1).currentText(),
            'camera_model': self.cellWidget(row, 2).currentText(),
            'mounting_position': self.cellWidget(row, 3).currentText(),
            'ip_address': self.cellWidget(row, 4).text(),
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

    def update_intrinsic_status(self, row: int, has_intrinsic: bool, file_path: str):
        """Update the intrinsic calibration status for a row."""
        container = self.cellWidget(row, 5)
        if container:
            label = container.findChild(QLabel)
            if label:
                if has_intrinsic:
                    label.setText("\u2713")
                    label.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold; font-size: 16px;")
                    label.setToolTip(f"Found: {file_path}")
                else:
                    label.setText("\u2717")
                    label.setStyleSheet(f"color: {COLORS['danger']}; font-weight: bold; font-size: 16px;")
                    label.setToolTip(f"Not found: {file_path}")


class PlatformConfigScreen(QWidget):
    """
    Platform Configuration Screen

    Step 1 of 6 in the calibration workflow.
    Configures platform information and camera setup.
    """

    # Signals
    cancel_requested = Signal()
    next_requested = Signal(PlatformConfiguration)

    def __init__(self, config: PlatformConfiguration = None, parent=None):
        super().__init__(parent)
        self.config = config or PlatformConfiguration()
        self.base_path = "."  # Will be set from main window
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(20)

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

        # Content area (scrollable)
        content_frame = QFrame()
        content_frame.setObjectName("card")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setSpacing(20)

        # Platform Information section
        platform_section = QVBoxLayout()
        platform_section.setSpacing(10)

        platform_header = QLabel("Platform Information")
        platform_header.setObjectName("section_header")
        platform_section.addWidget(platform_header)

        # Platform Type
        type_layout = QVBoxLayout()
        type_layout.setSpacing(4)
        type_label = QLabel("Platform Type:")
        self.platform_type_combo = QComboBox()
        self.platform_type_combo.addItems(PLATFORM_TYPES)
        self.platform_type_combo.setFixedWidth(200)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.platform_type_combo)
        platform_section.addLayout(type_layout)

        # Platform ID
        id_layout = QVBoxLayout()
        id_layout.setSpacing(4)
        id_label = QLabel("Platform ID:")
        self.platform_id_edit = QLineEdit()
        self.platform_id_edit.setPlaceholderText("e.g., PQ4459-001")
        self.platform_id_edit.setFixedWidth(200)
        id_layout.addWidget(id_label)
        id_layout.addWidget(self.platform_id_edit)
        platform_section.addLayout(id_layout)

        content_layout.addLayout(platform_section)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {COLORS['border']};")
        separator.setFixedHeight(1)
        content_layout.addWidget(separator)

        # Camera Configuration section
        camera_section = QVBoxLayout()
        camera_section.setSpacing(10)

        camera_header = QLabel("Camera Configuration")
        camera_header.setObjectName("section_header")
        camera_section.addWidget(camera_header)

        # Camera table
        self.camera_table = CameraTableWidget()
        self.camera_table.camera_removed.connect(self._on_camera_removed)
        self.camera_table.setMinimumHeight(200)
        camera_section.addWidget(self.camera_table)

        # Add Camera button
        add_btn_layout = QHBoxLayout()
        self.add_camera_btn = QPushButton("+ Add Camera")
        self.add_camera_btn.setObjectName("add_button")
        self.add_camera_btn.clicked.connect(self._on_add_camera)
        add_btn_layout.addWidget(self.add_camera_btn)
        add_btn_layout.addStretch()
        camera_section.addLayout(add_btn_layout)

        content_layout.addLayout(camera_section)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setStyleSheet(f"background-color: {COLORS['border']};")
        separator2.setFixedHeight(1)
        content_layout.addWidget(separator2)

        # Calibration Options section (informational only)
        options_section = QVBoxLayout()
        options_section.setSpacing(10)

        options_header = QLabel("Calibration Options")
        options_header.setObjectName("section_header")
        options_section.addWidget(options_header)

        info_label = QLabel(
            "Intrinsic calibration files are automatically detected at:\n"
            "camera_intrinsic/camera_intrinsics_<camera_id>.json\n\n"
            "Run intrinsic calibration for cameras missing calibration files."
        )
        info_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
        info_label.setWordWrap(True)
        options_section.addWidget(info_label)

        content_layout.addLayout(options_section)

        main_layout.addWidget(content_frame, 1)

        # Bottom navigation bar
        nav_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancel_button")
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        nav_layout.addWidget(self.cancel_btn)

        nav_layout.addStretch()

        self.next_btn = QPushButton("Next: Verify Hardware >")
        self.next_btn.setObjectName("nav_button")
        self.next_btn.clicked.connect(self._on_next_clicked)
        nav_layout.addWidget(self.next_btn)

        main_layout.addLayout(nav_layout)

    def _load_config(self):
        """Load configuration into UI."""
        self.platform_type_combo.setCurrentText(self.config.platform_type)
        self.platform_id_edit.setText(self.config.platform_id)

        # Load cameras
        self.camera_table.clear_all()
        for camera in self.config.cameras:
            self.camera_table.add_camera_row(camera, self.base_path)

        # Add default cameras if none exist
        if not self.config.cameras:
            for _ in range(4):
                self._on_add_camera()

    def _on_add_camera(self):
        """Add a new camera to the configuration."""
        camera = self.config.add_camera()
        self.camera_table.add_camera_row(camera, self.base_path)

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

    def _reload_camera_table(self):
        """Reload the camera table from config."""
        self.camera_table.clear_all()
        for camera in self.config.cameras:
            self.camera_table.add_camera_row(camera, self.base_path)

    def _on_next_clicked(self):
        """Handle Next button click."""
        # Validate inputs
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
                cam.camera_type = data['camera_type']
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

    def refresh_intrinsic_status(self):
        """Refresh the intrinsic calibration status for all cameras."""
        for row, camera in enumerate(self.config.cameras):
            has_intrinsic = camera.has_intrinsic_calibration(self.base_path)
            self.camera_table.update_intrinsic_status(
                row, has_intrinsic, camera.intrinsic_file_path
            )

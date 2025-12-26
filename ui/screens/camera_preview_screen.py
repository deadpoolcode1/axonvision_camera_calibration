"""
Camera Preview Screen (Screen 3)

Hardware verification and intrinsic calibration:
- Shows all cameras in a read-only table
- Intrinsic calibration button for each camera
- Block progression if any camera missing intrinsic calibration
"""

from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from ..styles import COLORS
from ..data_models import PlatformConfiguration

from ..dialogs import IntrinsicCalibrationDialog


class CameraPreviewScreen(QWidget):
    """
    Camera Preview Screen

    Step 2 of 6 in the calibration workflow (Hardware Verification).
    Displays camera list in a read-only table with intrinsic calibration buttons.
    """

    cancel_requested = Signal()
    next_requested = Signal(PlatformConfiguration)

    def __init__(self, config: PlatformConfiguration = None, parent=None):
        super().__init__(parent)
        self.config = config or PlatformConfiguration()
        self.base_path = "."
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

        title = QLabel("Intrinsic Calibration")
        title.setObjectName("title")
        header_layout.addWidget(title)

        step_label = QLabel("Step 2 of 6")
        step_label.setObjectName("step_indicator")
        header_layout.addWidget(step_label)

        main_layout.addLayout(header_layout)

        # Instructions
        instructions_frame = QFrame()
        instructions_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['table_header']};
                border-radius: 6px;
                padding: 15px;
            }}
        """)
        instructions_layout = QVBoxLayout(instructions_frame)

        instructions_text = QLabel(
            "Complete intrinsic calibration for all cameras before proceeding.\n"
            "Click the 'Calibrate' button for each camera that needs calibration."
        )
        instructions_text.setWordWrap(True)
        instructions_text.setStyleSheet(f"color: {COLORS['text_dark']}; font-size: 14px;")
        instructions_layout.addWidget(instructions_text)

        main_layout.addWidget(instructions_frame)

        # Camera table
        table_frame = QFrame()
        table_frame.setObjectName("card")
        table_layout = QVBoxLayout(table_frame)

        table_header = QLabel("Camera List")
        table_header.setObjectName("section_header")
        table_layout.addWidget(table_header)

        self.camera_table = QTableWidget()
        self._setup_camera_table()
        table_layout.addWidget(self.camera_table, 1)

        main_layout.addWidget(table_frame, 1)

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
        self.next_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.next_btn)

        main_layout.addLayout(nav_layout)

    def _setup_camera_table(self):
        """Setup the camera table with columns."""
        columns = ['#', 'Camera ID', 'Type', 'Model', 'Position', 'IP Address', 'Intrinsic Status', 'Action']
        self.camera_table.setColumnCount(len(columns))
        self.camera_table.setHorizontalHeaderLabels(columns)

        header = self.camera_table.horizontalHeader()
        # Use ResizeToContents for auto-width based on text length
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # #
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Camera ID
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Model
        header.setSectionResizeMode(4, QHeaderView.Stretch)           # Position - stretch to fill
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # IP Address
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Intrinsic Status
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Action

        self.camera_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.camera_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.camera_table.verticalHeader().setVisible(False)
        self.camera_table.setAlternatingRowColors(True)

        # Disable selection to gray out
        self.camera_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.camera_table.setFocusPolicy(Qt.NoFocus)

    def _populate_camera_table(self):
        """Populate the camera table with camera data."""
        self.camera_table.setRowCount(0)

        for camera in self.config.cameras:
            row = self.camera_table.rowCount()
            self.camera_table.insertRow(row)

            # Camera number
            num_item = QTableWidgetItem(str(camera.camera_number))
            num_item.setTextAlignment(Qt.AlignCenter)
            num_item.setFlags(num_item.flags() & ~Qt.ItemIsSelectable)
            self.camera_table.setItem(row, 0, num_item)

            # Camera ID
            id_item = QTableWidgetItem(camera.camera_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsSelectable)
            self.camera_table.setItem(row, 1, id_item)

            # Type
            type_item = QTableWidgetItem(camera.camera_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsSelectable)
            self.camera_table.setItem(row, 2, type_item)

            # Model
            model_item = QTableWidgetItem(camera.camera_model)
            model_item.setFlags(model_item.flags() & ~Qt.ItemIsSelectable)
            self.camera_table.setItem(row, 3, model_item)

            # Position
            pos_item = QTableWidgetItem(camera.mounting_position)
            pos_item.setFlags(pos_item.flags() & ~Qt.ItemIsSelectable)
            self.camera_table.setItem(row, 4, pos_item)

            # IP Address
            ip_item = QTableWidgetItem(camera.ip_address)
            ip_item.setFlags(ip_item.flags() & ~Qt.ItemIsSelectable)
            self.camera_table.setItem(row, 5, ip_item)

            # Intrinsic status
            has_intrinsic = camera.has_intrinsic_calibration(self.base_path)
            if has_intrinsic:
                status_item = QTableWidgetItem("✓ Calibrated")
                status_item.setForeground(Qt.darkGreen)
            else:
                status_item = QTableWidgetItem("✗ Not Calibrated")
                status_item.setForeground(Qt.red)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsSelectable)
            self.camera_table.setItem(row, 6, status_item)

            # Calibrate button
            calibrate_btn = QPushButton("Calibrate")
            calibrate_btn.setObjectName("calibrate_button")
            calibrate_btn.setToolTip(f"Run intrinsic calibration for {camera.camera_id}")
            calibrate_btn.clicked.connect(
                lambda checked, cam_num=camera.camera_number: self._on_calibrate_clicked(cam_num)
            )

            # Create a container widget for the button
            btn_container = QWidget()
            btn_layout = QHBoxLayout(btn_container)
            btn_layout.setContentsMargins(5, 2, 5, 2)
            btn_layout.addWidget(calibrate_btn)
            self.camera_table.setCellWidget(row, 7, btn_container)

            self.camera_table.setRowHeight(row, 45)

    def _on_calibrate_clicked(self, camera_number: int):
        """Handle calibrate button click for a camera."""
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
            # Refresh the table to update intrinsic status
            self._populate_camera_table()

    def _on_cancel_clicked(self):
        """Handle Cancel/Back button click."""
        self.cancel_requested.emit()

    def _on_next_clicked(self):
        """Handle Next button click."""
        # Check if all cameras have intrinsic calibration
        missing_intrinsics = [
            c.camera_id for c in self.config.cameras
            if not c.has_intrinsic_calibration(self.base_path)
        ]

        if missing_intrinsics:
            QMessageBox.warning(
                self,
                "Intrinsic Calibration Required",
                f"The following cameras are missing intrinsic calibration:\n\n"
                + "\n".join(f"  • {cam_id}" for cam_id in missing_intrinsics) +
                "\n\nPlease complete intrinsic calibration for all cameras before proceeding."
            )
            return

        # All cameras calibrated, proceed
        QMessageBox.information(
            self,
            "Next Step: Extrinsic Calibration",
            f"Platform: {self.config.platform_type} - {self.config.platform_id}\n"
            f"Cameras verified: {len(self.config.cameras)}\n\n"
            "All cameras have intrinsic calibration.\n"
            "Proceeding to extrinsic calibration."
        )

        self.next_requested.emit(self.config)

    def set_config(self, config: PlatformConfiguration):
        """Set a new configuration and update UI."""
        self.config = config
        self._populate_camera_table()

    def set_base_path(self, path: str):
        """Set the base path."""
        self.base_path = path

    def showEvent(self, event):
        """Refresh table when screen is shown."""
        super().showEvent(event)
        self._populate_camera_table()

    def hideEvent(self, event):
        """Handle screen hide."""
        super().hideEvent(event)

"""
Camera Preview Screen (Screen 3)

Displays camera previews and allows mounting position adjustments:
- Shows all configured cameras with live preview
- Allows user to change mounting positions
- Ping test for each camera on entry
- Navigate back to Screen 2 if any camera fails connectivity
"""

from pathlib import Path
import subprocess
import platform
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QGridLayout, QSizePolicy, QMessageBox,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

from ..styles import COLORS
from ..data_models import PlatformConfiguration, MOUNTING_POSITIONS


class CameraPreviewCard(QFrame):
    """
    A card widget displaying a single camera preview with mounting position selector.
    """

    mounting_position_changed = Signal(int, str)  # camera_number, new_position

    def __init__(self, camera_number: int, camera_id: str, ip_address: str,
                 mounting_position: str, parent=None):
        super().__init__(parent)
        self.camera_number = camera_number
        self.camera_id = camera_id
        self.ip_address = ip_address
        self.mounting_position = mounting_position
        self.ping_status = None  # None = not tested, True = ok, False = failed

        self._setup_ui()

    def _setup_ui(self):
        """Setup the card UI."""
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
        layout.setSpacing(10)

        # Header with camera info
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

        # Camera ID and IP
        info_label = QLabel(f"ID: {self.camera_id}  |  IP: {self.ip_address}")
        info_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        layout.addWidget(info_label)

        # Preview area (placeholder for now - could add live preview later)
        self.preview_frame = QFrame()
        self.preview_frame.setFixedSize(280, 180)
        self.preview_frame.setStyleSheet(f"""
            background-color: {COLORS['background']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        """)

        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setAlignment(Qt.AlignCenter)

        preview_placeholder = QLabel("Camera Preview")
        preview_placeholder.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 14px;
            font-style: italic;
        """)
        preview_placeholder.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_placeholder)

        layout.addWidget(self.preview_frame, alignment=Qt.AlignCenter)

        # Mounting Position selector
        position_layout = QHBoxLayout()
        position_label = QLabel("Mounting Position:")
        position_label.setStyleSheet(f"font-weight: bold;")
        position_layout.addWidget(position_label)

        self.position_combo = QComboBox()
        self.position_combo.addItems(MOUNTING_POSITIONS)
        self.position_combo.setCurrentText(self.mounting_position)
        self.position_combo.currentTextChanged.connect(self._on_position_changed)
        self.position_combo.setMinimumWidth(150)
        position_layout.addWidget(self.position_combo)

        position_layout.addStretch()
        layout.addLayout(position_layout)

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
            self.status_label.setText("\u2713 Connected")
            self.status_label.setStyleSheet(f"""
                color: {COLORS['success']};
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                background-color: #E8F5E9;
                border-radius: 4px;
            """)
        else:
            self.status_label.setText("\u2717 Not Detected")
            self.status_label.setStyleSheet(f"""
                color: {COLORS['danger']};
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                background-color: #FFEBEE;
                border-radius: 4px;
            """)

    def set_ping_status(self, status: bool):
        """Set the ping status and update display."""
        self.ping_status = status
        self._update_status_display()

    def _on_position_changed(self, new_position: str):
        """Handle mounting position change."""
        self.mounting_position = new_position
        self.mounting_position_changed.emit(self.camera_number, new_position)

    def get_mounting_position(self) -> str:
        """Get current mounting position."""
        return self.position_combo.currentText()


class CameraPreviewScreen(QWidget):
    """
    Camera Preview Screen

    Step 2 of 6 in the calibration workflow (Hardware Verification).
    Displays camera previews and allows mounting position adjustment.
    """

    # Signals
    cancel_requested = Signal()  # Go back to Screen 2
    next_requested = Signal(PlatformConfiguration)  # Proceed to next step

    def __init__(self, config: PlatformConfiguration = None, parent=None):
        super().__init__(parent)
        self.config = config or PlatformConfiguration()
        self.base_path = "."
        self.camera_cards = []
        self.ping_failed_cameras = []

        self._setup_ui()

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(20)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)

        screen_label = QLabel("Screen 3: Camera Preview")
        screen_label.setObjectName("screen_indicator")
        header_layout.addWidget(screen_label)

        title = QLabel("Hardware Verification")
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
        self.status_message.setStyleSheet(f"font-size: 14px;")
        status_layout.addWidget(self.status_message, 1)

        main_layout.addWidget(self.status_frame)

        # Instructions
        instructions = QLabel(
            "Review camera configuration and adjust mounting positions if needed. "
            "All cameras must be connected to proceed."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
        main_layout.addWidget(instructions)

        # Scrollable camera grid area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("background-color: transparent;")

        self.cameras_container = QWidget()
        self.cameras_layout = QGridLayout(self.cameras_container)
        self.cameras_layout.setSpacing(20)
        self.cameras_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll_area.setWidget(self.cameras_container)
        main_layout.addWidget(scroll_area, 1)

        # Bottom navigation bar
        nav_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("< Back to Configuration")
        self.cancel_btn.setObjectName("cancel_button")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        # Prevent space key from triggering when focused
        self.cancel_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.cancel_btn)

        nav_layout.addStretch()

        self.next_btn = QPushButton("Next: Extrinsic Calibration >")
        self.next_btn.setObjectName("nav_button")
        self.next_btn.clicked.connect(self._on_next_clicked)
        self.next_btn.setEnabled(False)  # Disabled until all cameras verified
        # Prevent space key from triggering when focused
        self.next_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.next_btn)

        main_layout.addLayout(nav_layout)

    def set_config(self, config: PlatformConfiguration):
        """Set a new configuration and update UI."""
        self.config = config
        self._rebuild_camera_grid()

    def set_base_path(self, path: str):
        """Set the base path."""
        self.base_path = path

    def _rebuild_camera_grid(self):
        """Rebuild the camera preview grid from configuration."""
        # Clear existing cards
        for card in self.camera_cards:
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
        max_cols = 2  # 2 cameras per row

        for camera in self.config.cameras:
            card = CameraPreviewCard(
                camera_number=camera.camera_number,
                camera_id=camera.camera_id,
                ip_address=camera.ip_address,
                mounting_position=camera.mounting_position,
                parent=self
            )
            card.mounting_position_changed.connect(self._on_mounting_position_changed)
            self.camera_cards.append(card)
            self.cameras_layout.addWidget(card, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Add stretch to push cards to top-left
        self.cameras_layout.setRowStretch(row + 1, 1)
        self.cameras_layout.setColumnStretch(max_cols, 1)

    def verify_cameras(self):
        """
        Verify all cameras are reachable via ping.
        Called when entering this screen.
        """
        self.ping_failed_cameras.clear()
        self.next_btn.setEnabled(False)

        # Update status to testing
        self._update_status("testing", "Testing camera connectivity...")

        # Test each camera
        all_passed = True
        for card in self.camera_cards:
            ping_ok = self._ping_device(card.ip_address)
            card.set_ping_status(ping_ok)

            if not ping_ok:
                all_passed = False
                self.ping_failed_cameras.append(card.camera_id)

        # Update status and enable/disable next button
        if all_passed:
            self._update_status("success", "All cameras connected and ready!")
            self.next_btn.setEnabled(True)
        else:
            failed_list = ", ".join(self.ping_failed_cameras)
            self._update_status(
                "error",
                f"Camera(s) not detected: {failed_list}. "
                "Please check connections and go back to verify configuration."
            )
            self.next_btn.setEnabled(False)

            # Show warning dialog
            self._show_camera_failure_dialog()

    def _ping_device(self, ip_address: str) -> bool:
        """Ping a device to check if it's reachable."""
        if not ip_address:
            return False

        try:
            # Determine ping command based on OS
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            timeout_param = '-w' if platform.system().lower() == 'windows' else '-W'

            # Run ping with 1 packet and 2 second timeout
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
                QFrame {{
                    background-color: {COLORS['table_header']};
                    border-radius: 6px;
                    padding: 10px;
                }}
            """)
            self.status_icon.setText("\u23F3")  # Hourglass
            self.status_icon.setStyleSheet(f"font-size: 20px;")
        elif status_type == "success":
            self.status_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: #E8F5E9;
                    border-radius: 6px;
                    padding: 10px;
                }}
            """)
            self.status_icon.setText("\u2713")
            self.status_icon.setStyleSheet(f"font-size: 20px; color: {COLORS['success']};")
        elif status_type == "error":
            self.status_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: #FFEBEE;
                    border-radius: 6px;
                    padding: 10px;
                }}
            """)
            self.status_icon.setText("\u2717")
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
            "Go back to the configuration screen to verify settings."
        )

    def _on_mounting_position_changed(self, camera_number: int, new_position: str):
        """Handle mounting position change for a camera."""
        # Update the configuration
        for camera in self.config.cameras:
            if camera.camera_number == camera_number:
                camera.mounting_position = new_position
                break

    def _on_cancel_clicked(self):
        """Handle Cancel/Back button click."""
        self.cancel_requested.emit()

    def _on_next_clicked(self):
        """Handle Next button click."""
        # Update config with any mounting position changes
        self._update_config_from_ui()

        # For now, show info about next steps since screen 4 doesn't exist yet
        QMessageBox.information(
            self,
            "Next Step: Extrinsic Calibration",
            f"Platform: {self.config.platform_type} - {self.config.platform_id}\n"
            f"Cameras verified: {len(self.config.cameras)}\n\n"
            "Next step will be Extrinsic Calibration.\n\n"
            "This screen is coming soon:\n"
            "- Step 3: Extrinsic Calibration\n"
            "- Step 4: Validation\n"
            "- Step 5: Report Generation"
        )

        # Emit signal with updated config
        self.next_requested.emit(self.config)

    def _update_config_from_ui(self):
        """Update the configuration from current UI state."""
        for card in self.camera_cards:
            for camera in self.config.cameras:
                if camera.camera_number == card.camera_number:
                    camera.mounting_position = card.get_mounting_position()
                    break

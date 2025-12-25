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
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QGridLayout, QSizePolicy, QMessageBox,
    QScrollArea, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

import numpy as np
import cv2

from ..styles import COLORS
from ..data_models import PlatformConfiguration, MOUNTING_POSITIONS

# Import camera streaming from intrinsic_calibration module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from intrinsic_calibration import NetworkCameraSource
from ins_reader import INSSerialReader, MockINSReader


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

        # Camera streaming components
        self.camera_source: Optional[NetworkCameraSource] = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self._is_streaming = False

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

        # Preview area - QLabel for live video display
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

    def start_streaming(self):
        """Start camera streaming."""
        if self._is_streaming:
            return

        if not self.ip_address:
            self.video_label.setText("No IP address")
            return

        self.video_label.setText("Connecting...")

        try:
            self.camera_source = NetworkCameraSource(
                ip=self.ip_address,
                api_port=5000,
                multicast_host="239.255.0.1",
                stream_port=5010,
                timeout=10.0
            )

            if self.camera_source.connect():
                self._is_streaming = True
                self.timer.start(33)  # ~30 FPS
            else:
                error_msg = self.camera_source.last_error or "Connection failed"
                self.video_label.setText(f"Error:\n{error_msg[:30]}...")
                self.video_label.setStyleSheet(f"""
                    background-color: #1a1a1a;
                    border: 1px solid {COLORS['danger']};
                    border-radius: 4px;
                    color: {COLORS['danger']};
                    font-size: 10px;
                """)
        except Exception as e:
            self.video_label.setText(f"Error:\n{str(e)[:30]}...")

    def stop_streaming(self):
        """Stop camera streaming."""
        self.timer.stop()
        self._is_streaming = False

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
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale frame to fit label (280x180)
        h, w = rgb_frame.shape[:2]
        target_w, target_h = 280, 180

        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = rgb_frame.shape[:2]

        # Create QImage
        bytes_per_line = 3 * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Display
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)


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

        # INS reader for real-time data
        self.ins_reader = None
        self.ins_timer = QTimer()
        self.ins_timer.timeout.connect(self._update_ins_display)
        self.ins_connected = False

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

        # Top action bar with instructions and refresh button
        action_bar = QHBoxLayout()

        # Instructions
        instructions = QLabel(
            "Review camera configuration and adjust mounting positions if needed. "
            "All cameras must be connected to proceed."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic; font-size: 14px;")
        action_bar.addWidget(instructions, 1)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh Discovery")
        self.refresh_btn.setObjectName("add_button")
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
            }}
        """)
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        action_bar.addWidget(self.refresh_btn)

        main_layout.addLayout(action_bar)

        # INS Data Panel
        ins_panel = QFrame()
        ins_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['white']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        ins_layout = QVBoxLayout(ins_panel)
        ins_layout.setSpacing(10)

        # INS Header with connection status
        ins_header_layout = QHBoxLayout()
        ins_title = QLabel("Real-Time INS Data")
        ins_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['primary']};")
        ins_header_layout.addWidget(ins_title)

        self.ins_status_label = QLabel("Not Connected")
        self.ins_status_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 12px;
            padding: 4px 8px;
            background-color: {COLORS['background']};
            border-radius: 4px;
        """)
        ins_header_layout.addStretch()
        ins_header_layout.addWidget(self.ins_status_label)

        self.connect_ins_btn = QPushButton("Connect INS")
        self.connect_ins_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                padding: 6px 12px;
                font-size: 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
        """)
        self.connect_ins_btn.clicked.connect(self._toggle_ins_connection)
        ins_header_layout.addWidget(self.connect_ins_btn)

        ins_layout.addLayout(ins_header_layout)

        # Data display in horizontal layout
        data_layout = QHBoxLayout()
        data_layout.setSpacing(30)

        # LLA (Latitude, Longitude, Altitude)
        lla_group = QFrame()
        lla_group.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['table_header']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        lla_layout = QVBoxLayout(lla_group)
        lla_layout.setSpacing(5)

        lla_title = QLabel("Location (LLA)")
        lla_title.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {COLORS['text_dark']};")
        lla_layout.addWidget(lla_title)

        # Latitude
        lat_layout = QHBoxLayout()
        lat_label = QLabel("Latitude:")
        lat_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']};")
        lat_label.setFixedWidth(80)
        self.lat_value = QLabel("---.------")
        self.lat_value.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']}; font-family: monospace;")
        lat_layout.addWidget(lat_label)
        lat_layout.addWidget(self.lat_value)
        lat_layout.addStretch()
        lla_layout.addLayout(lat_layout)

        # Longitude
        lon_layout = QHBoxLayout()
        lon_label = QLabel("Longitude:")
        lon_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']};")
        lon_label.setFixedWidth(80)
        self.lon_value = QLabel("---.------")
        self.lon_value.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']}; font-family: monospace;")
        lon_layout.addWidget(lon_label)
        lon_layout.addWidget(self.lon_value)
        lon_layout.addStretch()
        lla_layout.addLayout(lon_layout)

        # Altitude
        alt_layout = QHBoxLayout()
        alt_label = QLabel("Altitude:")
        alt_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']};")
        alt_label.setFixedWidth(80)
        self.alt_value = QLabel("-----.-- m")
        self.alt_value.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']}; font-family: monospace;")
        alt_layout.addWidget(alt_label)
        alt_layout.addWidget(self.alt_value)
        alt_layout.addStretch()
        lla_layout.addLayout(alt_layout)

        data_layout.addWidget(lla_group, 1)

        # Orientation (Yaw, Pitch, Roll)
        orientation_group = QFrame()
        orientation_group.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['table_header']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        orientation_layout = QVBoxLayout(orientation_group)
        orientation_layout.setSpacing(5)

        orientation_title = QLabel("Orientation (YPR)")
        orientation_title.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {COLORS['text_dark']};")
        orientation_layout.addWidget(orientation_title)

        # Yaw
        yaw_layout = QHBoxLayout()
        yaw_label = QLabel("Yaw:")
        yaw_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']};")
        yaw_label.setFixedWidth(80)
        self.yaw_value = QLabel("----.---")
        self.yaw_value.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']}; font-family: monospace;")
        yaw_layout.addWidget(yaw_label)
        yaw_layout.addWidget(self.yaw_value)
        yaw_layout.addStretch()
        orientation_layout.addLayout(yaw_layout)

        # Pitch
        pitch_layout = QHBoxLayout()
        pitch_label = QLabel("Pitch:")
        pitch_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']};")
        pitch_label.setFixedWidth(80)
        self.pitch_value = QLabel("----.---")
        self.pitch_value.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']}; font-family: monospace;")
        pitch_layout.addWidget(pitch_label)
        pitch_layout.addWidget(self.pitch_value)
        pitch_layout.addStretch()
        orientation_layout.addLayout(pitch_layout)

        # Roll
        roll_layout = QHBoxLayout()
        roll_label = QLabel("Roll:")
        roll_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']};")
        roll_label.setFixedWidth(80)
        self.roll_value = QLabel("----.---")
        self.roll_value.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']}; font-family: monospace;")
        roll_layout.addWidget(roll_label)
        roll_layout.addWidget(self.roll_value)
        roll_layout.addStretch()
        orientation_layout.addLayout(roll_layout)

        data_layout.addWidget(orientation_group, 1)

        ins_layout.addLayout(data_layout)

        main_layout.addWidget(ins_panel)

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

        # Start video streaming for cameras that passed ping
        self.start_all_streams()

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
        self.stop_all_streams()
        self.cancel_requested.emit()

    def _on_next_clicked(self):
        """Handle Next button click."""
        # Stop streams before navigating
        self.stop_all_streams()

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

    def start_all_streams(self):
        """Start video streaming for all camera cards that passed ping."""
        for card in self.camera_cards:
            if card.ping_status:  # Only start if ping passed
                card.start_streaming()

    def stop_all_streams(self):
        """Stop video streaming for all camera cards."""
        for card in self.camera_cards:
            card.stop_streaming()

    def hideEvent(self, event):
        """Stop all streams when screen is hidden."""
        self.stop_all_streams()
        self._disconnect_ins()
        super().hideEvent(event)

    def _on_refresh_clicked(self):
        """Handle refresh button click - re-verify all cameras."""
        # Stop existing streams
        self.stop_all_streams()

        # Reset camera status
        for card in self.camera_cards:
            card.ping_status = None
            card.video_label.setText("Connecting...")

        # Update status message
        self._update_status("testing", "Refreshing camera discovery...")

        # Re-verify cameras
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self.verify_cameras)

    def _toggle_ins_connection(self):
        """Toggle INS connection on/off."""
        if self.ins_connected:
            self._disconnect_ins()
        else:
            self._connect_ins()

    def _connect_ins(self):
        """Connect to INS device."""
        try:
            # Try to connect to real INS
            self.ins_reader = INSSerialReader()
            if self.ins_reader.connect():
                self.ins_connected = True
                self.ins_timer.start(100)  # Update every 100ms
                self._update_ins_status("connected")
                self.connect_ins_btn.setText("Disconnect INS")
                self.connect_ins_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS['danger']};
                        color: white;
                        padding: 6px 12px;
                        font-size: 12px;
                        border-radius: 4px;
                    }}
                    QPushButton:hover {{
                        background-color: {COLORS['danger_hover']};
                    }}
                """)
            else:
                # Fall back to mock INS for demo purposes
                self.ins_reader = MockINSReader(yaw=45.0, pitch=2.5, roll=-1.0)
                self.ins_reader.connect()
                self.ins_connected = True
                self.ins_timer.start(100)
                self._update_ins_status("mock")
                self.connect_ins_btn.setText("Disconnect INS")
                self.connect_ins_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS['danger']};
                        color: white;
                        padding: 6px 12px;
                        font-size: 12px;
                        border-radius: 4px;
                    }}
                    QPushButton:hover {{
                        background-color: {COLORS['danger_hover']};
                    }}
                """)
        except Exception as e:
            QMessageBox.warning(
                self,
                "INS Connection Failed",
                f"Failed to connect to INS device:\n{str(e)}\n\n"
                "Please check:\n"
                "  1. INS device is connected via USB\n"
                "  2. Serial port permissions are set correctly\n"
                "  3. INS device is powered on"
            )

    def _disconnect_ins(self):
        """Disconnect from INS device."""
        self.ins_timer.stop()
        if self.ins_reader:
            self.ins_reader.disconnect()
            self.ins_reader = None
        self.ins_connected = False
        self._update_ins_status("disconnected")
        self.connect_ins_btn.setText("Connect INS")
        self.connect_ins_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                padding: 6px 12px;
                font-size: 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
        """)

        # Reset display values
        self.lat_value.setText("---.------")
        self.lon_value.setText("---.------")
        self.alt_value.setText("-----.-- m")
        self.yaw_value.setText("----.---")
        self.pitch_value.setText("----.---")
        self.roll_value.setText("----.---")

    def _update_ins_status(self, status: str):
        """Update INS connection status label."""
        if status == "connected":
            self.ins_status_label.setText("Connected")
            self.ins_status_label.setStyleSheet(f"""
                color: {COLORS['success']};
                font-size: 12px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #E8F5E9;
                border-radius: 4px;
            """)
        elif status == "mock":
            self.ins_status_label.setText("Mock Data")
            self.ins_status_label.setStyleSheet(f"""
                color: {COLORS['warning']};
                font-size: 12px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #FFF8E1;
                border-radius: 4px;
            """)
        else:
            self.ins_status_label.setText("Not Connected")
            self.ins_status_label.setStyleSheet(f"""
                color: {COLORS['text_muted']};
                font-size: 12px;
                padding: 4px 8px;
                background-color: {COLORS['background']};
                border-radius: 4px;
            """)

    def _update_ins_display(self):
        """Update INS data display with latest reading."""
        if not self.ins_reader:
            return

        reading = self.ins_reader.get_latest()
        if reading:
            # Update LLA
            self.lat_value.setText(f"{reading.latitude:+011.6f}")
            self.lon_value.setText(f"{reading.longitude:+012.6f}")
            self.alt_value.setText(f"{reading.altitude:+08.2f} m")

            # Update orientation
            self.yaw_value.setText(f"{reading.yaw:+08.3f}")
            self.pitch_value.setText(f"{reading.pitch:+08.3f}")
            self.roll_value.setText(f"{reading.roll:+08.3f}")

"""
Intrinsic Calibration Dialog

Interactive dialog for performing intrinsic camera calibration with live preview.
Allows user to capture calibration images by pressing Space and cancel with ESC.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QMessageBox, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QKeyEvent

from ..styles import COLORS

# Import calibration classes from intrinsic_calibration module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from intrinsic_calibration import (
    ChArUcoBoardConfig, ChArUcoDetector, IntrinsicCalibrator, NetworkCameraSource
)


class IntrinsicCalibrationDialog(QDialog):
    """
    Dialog for performing intrinsic camera calibration with live video preview.

    Features:
    - Live video feed from network camera
    - Real-time ChArUco detection overlay
    - Space key to capture calibration images
    - ESC key to cancel calibration
    - Progress bar showing capture progress
    - Success/failure notifications
    """

    calibration_completed = Signal(bool, str)  # success, message

    # Calibration parameters
    MIN_IMAGES = 10
    TARGET_IMAGES = 25
    MIN_CORNERS = 6

    def __init__(self, camera_id: str, ip_address: str, base_path: str = ".", parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.ip_address = ip_address
        self.base_path = base_path

        # Calibration components
        self.board_config = ChArUcoBoardConfig()
        self.detector = ChArUcoDetector(self.board_config)
        self.calibrator = IntrinsicCalibrator(self.board_config, camera_id)
        self.camera_source: Optional[NetworkCameraSource] = None

        # State
        self.captured_count = 0
        self.is_capturing = False
        self.last_detection_valid = False
        self.calibration_successful = False

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self._setup_ui()
        self.setModal(True)

    def _setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle(f"Intrinsic Calibration - {self.camera_id}")
        self.setMinimumSize(900, 700)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)

        title = QLabel(f"Intrinsic Calibration: {self.camera_id}")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['primary']};")
        header_layout.addWidget(title)

        subtitle = QLabel(f"Camera IP: {self.ip_address}")
        subtitle.setStyleSheet(f"font-size: 14px; color: {COLORS['text_muted']};")
        header_layout.addWidget(subtitle)

        main_layout.addLayout(header_layout)

        # Instructions card
        instructions_frame = QFrame()
        instructions_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['table_header']};
                border: 1px solid {COLORS['primary']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        instructions_layout = QVBoxLayout(instructions_frame)
        instructions_layout.setContentsMargins(15, 10, 15, 10)

        instructions = QLabel(
            "<b>Instructions:</b><br>"
            "1. Hold the ChArUco calibration board in view of the camera<br>"
            "2. Press <b>SPACE</b> to capture an image when the board is detected (green overlay)<br>"
            "3. Move the board to different positions and angles for best results<br>"
            "4. Capture at least 10 images (25 recommended) for accurate calibration<br>"
            "5. Press <b>ESC</b> to cancel at any time"
        )
        instructions.setStyleSheet(f"color: {COLORS['text_dark']}; font-size: 13px;")
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)

        main_layout.addWidget(instructions_frame)

        # Video preview area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        self.video_label.setText("Connecting to camera...")
        self.video_label.setStyleSheet(self.video_label.styleSheet() + f"color: {COLORS['white']};")
        main_layout.addWidget(self.video_label, 1)

        # Status and progress area
        status_layout = QHBoxLayout()

        # Detection status
        self.detection_label = QLabel("Detection: Waiting...")
        self.detection_label.setStyleSheet(f"font-size: 14px; color: {COLORS['text_muted']};")
        status_layout.addWidget(self.detection_label)

        status_layout.addStretch()

        # Capture count
        self.capture_label = QLabel(f"Captured: 0 / {self.TARGET_IMAGES}")
        self.capture_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {COLORS['primary']};")
        status_layout.addWidget(self.capture_label)

        main_layout.addLayout(status_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.TARGET_IMAGES)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m images")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                background-color: {COLORS['background']};
                text-align: center;
                height: 25px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 3px;
            }}
        """)
        main_layout.addWidget(self.progress_bar)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("Cancel (ESC)")
        self.cancel_btn.setObjectName("cancel_button")
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        button_layout.addStretch()

        self.capture_btn = QPushButton("Capture (SPACE)")
        self.capture_btn.setObjectName("nav_button")
        self.capture_btn.clicked.connect(self._on_capture)
        self.capture_btn.setEnabled(False)
        button_layout.addWidget(self.capture_btn)

        self.calibrate_btn = QPushButton("Run Calibration")
        self.calibrate_btn.setObjectName("primary_button")
        self.calibrate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['text_muted']};
            }}
        """)
        self.calibrate_btn.clicked.connect(self._on_run_calibration)
        self.calibrate_btn.setEnabled(False)
        button_layout.addWidget(self.calibrate_btn)

        main_layout.addLayout(button_layout)

    def showEvent(self, event):
        """Start camera when dialog is shown."""
        super().showEvent(event)
        self._connect_camera()

    def closeEvent(self, event):
        """Cleanup when dialog is closed."""
        self._stop_camera()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key presses for capture (Space) and cancel (ESC)."""
        if event.key() == Qt.Key_Space:
            self._on_capture()
        elif event.key() == Qt.Key_Escape:
            self._on_cancel()
        else:
            super().keyPressEvent(event)

    def _connect_camera(self):
        """Connect to the network camera."""
        self.video_label.setText("Connecting to camera...")

        try:
            self.camera_source = NetworkCameraSource(
                ip=self.ip_address,
                api_port=5000,
                multicast_host="239.255.0.1",
                stream_port=5010
            )

            if self.camera_source.connect():
                self.is_capturing = True
                self.timer.start(33)  # ~30 FPS
            else:
                self._show_connection_error("Failed to connect to camera stream")

        except Exception as e:
            self._show_connection_error(f"Camera connection error: {str(e)}")

    def _show_connection_error(self, message: str):
        """Show connection error in the video area."""
        self.video_label.setText(f"Connection Error:\n{message}\n\nPlease verify the camera IP and try again.")
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: 2px solid {COLORS['danger']};
                border-radius: 6px;
                color: {COLORS['danger']};
                font-size: 14px;
            }}
        """)

    def _stop_camera(self):
        """Stop camera capture and release resources."""
        self.timer.stop()
        self.is_capturing = False
        if self.camera_source:
            self.camera_source.release()
            self.camera_source = None

    def _update_frame(self):
        """Update video frame and run detection."""
        if not self.camera_source:
            return

        frame = self.camera_source.get_image()
        if frame is None:
            return

        # Run ChArUco detection
        corners, ids, annotated = self.detector.detect(frame)

        # Check if detection is valid (enough corners)
        if corners is not None and len(corners) >= self.MIN_CORNERS:
            self.last_detection_valid = True
            corner_count = len(corners)
            self.detection_label.setText(f"Detection: {corner_count} corners detected")
            self.detection_label.setStyleSheet(f"font-size: 14px; color: {COLORS['success']}; font-weight: bold;")
            self.capture_btn.setEnabled(True)

            # Draw green border to indicate valid detection
            cv2.rectangle(annotated, (5, 5), (annotated.shape[1]-5, annotated.shape[0]-5),
                         (0, 255, 0), 3)
        else:
            self.last_detection_valid = False
            corner_count = len(corners) if corners is not None else 0
            self.detection_label.setText(f"Detection: {corner_count} corners (need {self.MIN_CORNERS}+)")
            self.detection_label.setStyleSheet(f"font-size: 14px; color: {COLORS['warning']};")
            self.capture_btn.setEnabled(False)

        # Convert frame to QPixmap and display
        self._display_frame(annotated)

    def _display_frame(self, frame: np.ndarray):
        """Convert OpenCV frame to QPixmap and display."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale frame to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        h, w = rgb_frame.shape[:2]

        # Calculate scale to fit
        scale_w = label_size.width() / w
        scale_h = label_size.height() / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale

        if scale < 1.0:
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

    def _on_capture(self):
        """Handle capture button/space press."""
        if not self.last_detection_valid or not self.camera_source:
            return

        # Get current frame and detect again
        frame = self.camera_source.get_image()
        if frame is None:
            return

        corners, ids, annotated = self.detector.detect(frame)

        if corners is None or len(corners) < self.MIN_CORNERS:
            return

        # Add detection to calibrator
        image_size = (frame.shape[1], frame.shape[0])
        if self.calibrator.add_detection(corners, ids, image_size):
            self.captured_count += 1
            self._update_capture_display()

            # Flash green to indicate successful capture
            self.video_label.setStyleSheet(f"""
                QLabel {{
                    background-color: #1a1a1a;
                    border: 3px solid {COLORS['success']};
                    border-radius: 6px;
                }}
            """)
            QTimer.singleShot(200, self._reset_video_border)

    def _reset_video_border(self):
        """Reset video label border after flash."""
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)

    def _update_capture_display(self):
        """Update capture count and progress bar."""
        self.capture_label.setText(f"Captured: {self.captured_count} / {self.TARGET_IMAGES}")
        self.progress_bar.setValue(self.captured_count)

        # Enable calibration button when minimum images reached
        if self.captured_count >= self.MIN_IMAGES:
            self.calibrate_btn.setEnabled(True)
            if self.captured_count == self.MIN_IMAGES:
                self.calibrate_btn.setText(f"Run Calibration ({self.captured_count} images)")
            else:
                self.calibrate_btn.setText(f"Run Calibration ({self.captured_count} images)")

    def _on_run_calibration(self):
        """Run the calibration with captured images."""
        if self.captured_count < self.MIN_IMAGES:
            QMessageBox.warning(
                self,
                "Not Enough Images",
                f"Please capture at least {self.MIN_IMAGES} images.\n"
                f"Currently captured: {self.captured_count}"
            )
            return

        # Stop camera updates during calibration
        self.timer.stop()
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setText("Calibrating...")
        self.capture_btn.setEnabled(False)

        # Update UI to show calibration in progress
        self.video_label.setText("Running calibration...\nThis may take a moment.")
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: 2px solid {COLORS['primary']};
                border-radius: 6px;
                color: {COLORS['white']};
                font-size: 16px;
            }}
        """)

        # Process events to update UI
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        # Run calibration
        result = self.calibrator.calibrate(min_images=self.MIN_IMAGES)

        if result is not None and result.get('rms_error', float('inf')) < 2.0:
            # Calibration successful - save to file
            output_dir = Path(self.base_path) / "camera_intrinsic"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"camera_intrinsics_{self.camera_id}.json"

            try:
                self.calibrator.save_to_json(str(output_path))
                self.calibration_successful = True

                rms_error = result.get('rms_error', 0)
                QMessageBox.information(
                    self,
                    "Calibration Successful",
                    f"Intrinsic calibration for {self.camera_id} completed successfully!\n\n"
                    f"Images used: {self.captured_count}\n"
                    f"RMS Error: {rms_error:.4f} pixels\n\n"
                    f"Calibration saved to:\n{output_path}"
                )

                self.calibration_completed.emit(True, f"Calibration saved to {output_path}")
                self.accept()

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Calibration succeeded but failed to save:\n{str(e)}"
                )
                self._resume_capture()
        else:
            # Calibration failed
            rms_error = result.get('rms_error', float('inf')) if result else float('inf')
            QMessageBox.warning(
                self,
                "Calibration Failed",
                f"Calibration failed or produced poor results.\n\n"
                f"RMS Error: {rms_error:.4f} pixels (should be < 2.0)\n\n"
                "Please capture more images with the board in different\n"
                "positions and angles, then try again."
            )
            self._resume_capture()

    def _resume_capture(self):
        """Resume capture mode after failed calibration."""
        self.calibrate_btn.setEnabled(self.captured_count >= self.MIN_IMAGES)
        self.calibrate_btn.setText(f"Run Calibration ({self.captured_count} images)")
        self.timer.start(33)

    def _on_cancel(self):
        """Handle cancel button or ESC key."""
        if self.captured_count > 0:
            reply = QMessageBox.question(
                self,
                "Cancel Calibration",
                f"You have captured {self.captured_count} images.\n\n"
                "Are you sure you want to cancel?\n"
                "All captured images will be discarded.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self._stop_camera()
        self.calibration_completed.emit(False, "Calibration cancelled by user")
        self.reject()

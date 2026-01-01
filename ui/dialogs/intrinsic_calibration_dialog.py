"""
Intrinsic Calibration Dialog

Interactive dialog for performing intrinsic camera calibration with live preview.
Allows user to capture calibration images by pressing Space and cancel with ESC.

Features:
- Real-time FOV coverage percentage display
- Active guidance suggesting next best capture movement
- Pass/Fail gating with actionable failure messages
- Automatic image persistence for debugging
- PDF report generation with detailed metrics
"""

import numpy as np
import cv2
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QMessageBox, QFrame, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Signal, QEvent
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QColor, QPainter, QBrush, QPen

from ..styles import COLORS

# Import calibration classes from intrinsic_calibration module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from intrinsic_calibration import (
    ChArUcoBoardConfig, ChArUcoDetector, IntrinsicCalibrator, NetworkCameraSource,
    CoverageAnalyzer, CalibrationReportGenerator, HAS_REPORTLAB
)


class FOVGridWidget(QLabel):
    """Widget that displays a 3x3 FOV coverage grid visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(90, 60)
        self.grid_data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def update_grid(self, grid_data: List[List[int]]):
        """Update the grid data and repaint."""
        self.grid_data = grid_data
        self.update()

    def paintEvent(self, event):
        """Draw the 3x3 FOV grid with coverage coloring."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cell_width = self.width() // 3
        cell_height = self.height() // 3

        for row in range(3):
            for col in range(3):
                x = col * cell_width
                y = row * cell_height
                count = self.grid_data[row][col]

                # Color based on coverage count
                if count == 0:
                    color = QColor(200, 50, 50, 150)  # Red - not covered
                elif count == 1:
                    color = QColor(255, 200, 50, 150)  # Yellow - minimal
                else:
                    color = QColor(50, 200, 50, 150)  # Green - good coverage

                painter.fillRect(x, y, cell_width - 1, cell_height - 1, QBrush(color))
                painter.setPen(QPen(Qt.white, 1))
                painter.drawRect(x, y, cell_width - 1, cell_height - 1)

        painter.end()


class IntrinsicCalibrationDialog(QDialog):
    """
    Dialog for performing intrinsic camera calibration with live video preview.

    Features:
    - Live video feed from network camera
    - Real-time ChArUco detection overlay
    - Real-time FOV coverage percentage display
    - Active guidance suggesting next capture movement
    - Space key to capture calibration images
    - ESC key to cancel calibration
    - Progress bar showing capture progress
    - Pass/Fail gating with actionable failure messages
    - Automatic image persistence for debugging
    - PDF report generation
    """

    calibration_completed = Signal(bool, str)  # success, message

    # Calibration parameters
    MIN_IMAGES = 10
    TARGET_IMAGES = 25
    MIN_CORNERS = 6

    # Quality thresholds
    MAX_RMS_ERROR = 1.0  # Maximum acceptable RMS error in pixels
    MIN_FOV_COVERAGE = 60.0  # Minimum FOV coverage percentage

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

        # Image storage for persistence and PDF report
        self.captured_images: List[Dict[str, Any]] = []
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directories
        self.images_dir = Path(self.base_path) / "calibration_images" / f"{camera_id}_{self.session_timestamp}"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self._setup_ui()
        self.setModal(True)

        # Install event filter on all buttons to intercept SPACE key
        # This prevents SPACE from clicking buttons (which would close dialog)
        self.cancel_btn.installEventFilter(self)
        self.retry_btn.installEventFilter(self)
        self.capture_btn.installEventFilter(self)
        self.calibrate_btn.installEventFilter(self)

    def _setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle(f"Intrinsic Calibration - {self.camera_id}")
        self.setMinimumSize(900, 700)
        # Enable proper window controls (minimize, maximize, close)
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowCloseButtonHint |
            Qt.WindowMinMaxButtonsHint
        )

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

        # Coverage and guidance section (enhanced)
        coverage_frame = QFrame()
        coverage_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        coverage_main_layout = QHBoxLayout(coverage_frame)
        coverage_main_layout.setContentsMargins(10, 8, 10, 8)
        coverage_main_layout.setSpacing(15)

        # Left side: FOV Grid visualization
        fov_section = QVBoxLayout()
        fov_section.setSpacing(4)

        fov_header = QLabel("FOV Coverage")
        fov_header.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {COLORS['text_muted']};")
        fov_section.addWidget(fov_header)

        self.fov_grid_widget = FOVGridWidget()
        fov_section.addWidget(self.fov_grid_widget)

        # FOV percentage display
        self.fov_percent_label = QLabel("0%")
        self.fov_percent_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['danger']};")
        self.fov_percent_label.setAlignment(Qt.AlignCenter)
        fov_section.addWidget(self.fov_percent_label)

        coverage_main_layout.addLayout(fov_section)

        # Vertical separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet(f"color: {COLORS['border']};")
        coverage_main_layout.addWidget(separator)

        # Right side: Guidance and metrics
        guidance_section = QVBoxLayout()
        guidance_section.setSpacing(6)

        # Next action guidance (prominent)
        guidance_header = QLabel("Next Action")
        guidance_header.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {COLORS['text_muted']};")
        guidance_section.addWidget(guidance_header)

        self.guidance_label = QLabel("Start by capturing the target in center of frame")
        self.guidance_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {COLORS['primary']};
            background-color: rgba(23, 162, 184, 0.1);
            padding: 8px;
            border-radius: 4px;
        """)
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setMinimumHeight(50)
        guidance_section.addWidget(self.guidance_label)

        # Coverage metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)

        # Viewpoint diversity
        viewpoint_section = QVBoxLayout()
        viewpoint_section.setSpacing(2)
        viewpoint_label = QLabel("Viewpoints")
        viewpoint_label.setStyleSheet(f"font-size: 10px; color: {COLORS['text_muted']};")
        viewpoint_section.addWidget(viewpoint_label)
        self.viewpoint_count_label = QLabel("0 / 3")
        self.viewpoint_count_label.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLORS['danger']};")
        viewpoint_section.addWidget(self.viewpoint_count_label)
        metrics_layout.addLayout(viewpoint_section)

        # Scale coverage
        scale_section = QVBoxLayout()
        scale_section.setSpacing(2)
        scale_label = QLabel("Distances")
        scale_label.setStyleSheet(f"font-size: 10px; color: {COLORS['text_muted']};")
        scale_section.addWidget(scale_label)
        self.scale_label = QLabel("C:0 M:0 F:0")
        self.scale_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
        scale_section.addWidget(self.scale_label)
        metrics_layout.addLayout(scale_section)

        # Tilt coverage
        tilt_section = QVBoxLayout()
        tilt_section.setSpacing(2)
        tilt_label_header = QLabel("Tilts")
        tilt_label_header.setStyleSheet(f"font-size: 10px; color: {COLORS['text_muted']};")
        tilt_section.addWidget(tilt_label_header)
        self.tilt_label = QLabel("None")
        self.tilt_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
        tilt_section.addWidget(self.tilt_label)
        metrics_layout.addLayout(tilt_section)

        metrics_layout.addStretch()
        guidance_section.addLayout(metrics_layout)

        # Coverage status indicator
        self.coverage_status_label = QLabel("")
        self.coverage_status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
        guidance_section.addWidget(self.coverage_status_label)

        coverage_main_layout.addLayout(guidance_section, 1)

        main_layout.addWidget(coverage_frame)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("Cancel (ESC)")
        self.cancel_btn.setObjectName("cancel_button")
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.setDefault(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        # Retry button (shown when connection fails)
        self.retry_btn = QPushButton("Retry Connection")
        self.retry_btn.setAutoDefault(False)
        self.retry_btn.setDefault(False)
        self.retry_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: {COLORS['text_dark']};
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #E0A800;
            }}
        """)
        self.retry_btn.clicked.connect(self._on_retry_connection)
        self.retry_btn.hide()  # Initially hidden
        button_layout.addWidget(self.retry_btn)

        button_layout.addStretch()

        self.capture_btn = QPushButton("Capture (SPACE)")
        self.capture_btn.setObjectName("nav_button")
        self.capture_btn.setAutoDefault(False)
        self.capture_btn.setDefault(False)
        self.capture_btn.clicked.connect(self._on_capture)
        self.capture_btn.setEnabled(False)
        button_layout.addWidget(self.capture_btn)

        self.calibrate_btn = QPushButton("Run Calibration")
        self.calibrate_btn.setObjectName("primary_button")
        self.calibrate_btn.setAutoDefault(False)
        self.calibrate_btn.setDefault(False)
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

    def eventFilter(self, watched, event):
        """Filter SPACE key from buttons to prevent accidental dialog close."""
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            # Intercept SPACE on any button - redirect to capture
            self._on_capture()
            return True  # Event consumed, don't let button process it
        return super().eventFilter(watched, event)

    def event(self, event):
        """Intercept key events before they reach buttons."""
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space:
                self._on_capture()
                return True  # Event fully handled, don't propagate to buttons
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key presses for cancel (ESC)."""
        if event.key() == Qt.Key_Escape:
            self._on_cancel()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _connect_camera(self):
        """Connect to the network camera."""
        self.video_label.setText("Connecting to camera...\nStopping any existing streams...")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            self.camera_source = NetworkCameraSource(
                ip=self.ip_address,
                api_port=5000,
                multicast_host="239.255.0.1",
                stream_port=5010,
                timeout=10.0  # Increased timeout for reliability
            )

            self.video_label.setText(f"Starting camera stream from {self.ip_address}...")
            QApplication.processEvents()

            if self.camera_source.connect():
                self.is_capturing = True
                self.timer.start(33)  # ~30 FPS
            else:
                # Get detailed error message from the camera source
                error_msg = self.camera_source.last_error or "Failed to connect to camera stream"
                self._show_connection_error(error_msg)

        except Exception as e:
            self._show_connection_error(f"Camera connection error: {str(e)}")

    def _show_connection_error(self, message: str):
        """Show connection error in the video area."""
        self.video_label.setText(
            f"Connection Error:\n\n{message}\n\n"
            f"Troubleshooting:\n"
            f"• Verify the camera IP address ({self.ip_address})\n"
            f"• Check that the camera is powered on\n"
            f"• Ensure the camera API is running on port 5000\n"
            f"• Check network/firewall settings for multicast"
        )
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: 2px solid {COLORS['danger']};
                border-radius: 6px;
                color: {COLORS['danger']};
                font-size: 13px;
                padding: 20px;
            }}
        """)
        # Show retry button
        self.retry_btn.show()

    def _on_retry_connection(self):
        """Retry camera connection."""
        # Hide retry button and reset video label
        self.retry_btn.hide()
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
                color: {COLORS['white']};
            }}
        """)

        # Clean up any existing connection
        if self.camera_source:
            self.camera_source.release()
            self.camera_source = None

        # Try to connect again
        self._connect_camera()

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
        if not self.camera_source:
            return

        if not self.last_detection_valid:
            # Flash red border to indicate invalid capture attempt
            self.video_label.setStyleSheet(f"""
                QLabel {{
                    background-color: #1a1a1a;
                    border: 3px solid {COLORS['danger']};
                    border-radius: 6px;
                }}
            """)
            QTimer.singleShot(200, self._reset_video_border)
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

            # Save images for persistence and debugging
            self._save_captured_image(frame, annotated, corners, ids)

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

    def _save_captured_image(self, raw_frame: np.ndarray, annotated_frame: np.ndarray,
                             corners: np.ndarray, ids: np.ndarray):
        """Save captured image and metadata for persistence."""
        img_num = self.captured_count

        # Save raw and annotated images
        raw_path = self.images_dir / f"raw_{img_num:03d}.png"
        annotated_path = self.images_dir / f"annotated_{img_num:03d}.png"

        cv2.imwrite(str(raw_path), raw_frame)
        cv2.imwrite(str(annotated_path), annotated_frame)

        # Get coverage analysis for this capture
        coverage = self.calibrator.get_coverage_summary()

        # Store image metadata
        image_data = {
            'index': img_num,
            'raw_path': str(raw_path),
            'annotated_path': str(annotated_path),
            'num_corners': len(corners),
            'timestamp': datetime.now().isoformat(),
            'coverage_at_capture': {
                'fov_percent': coverage.get('fov_coverage_percent', 0),
                'viewpoint_diversity': coverage.get('viewpoint_diversity', 0)
            }
        }
        self.captured_images.append(image_data)

        # Save session metadata file
        self._save_session_metadata()

    def _save_session_metadata(self):
        """Save session metadata to JSON file."""
        coverage = self.calibrator.get_coverage_summary()

        metadata = {
            'camera_id': self.camera_id,
            'session_timestamp': self.session_timestamp,
            'ip_address': self.ip_address,
            'board_config': {
                'squares_x': self.board_config.squares_x,
                'squares_y': self.board_config.squares_y,
                'square_length': self.board_config.square_length,
                'marker_length': self.board_config.marker_length
            },
            'images': self.captured_images,
            'coverage_summary': {
                'total_captures': coverage.get('total', 0),
                'fov_coverage_percent': coverage.get('fov_coverage_percent', 0),
                'viewpoint_diversity': coverage.get('viewpoint_diversity', 0),
                'positions': coverage.get('positions', {}),
                'scales': coverage.get('scales', {}),
                'tilts': coverage.get('tilts', {})
            }
        }

        metadata_path = self.images_dir / "session_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

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
        """Update capture count, progress bar, and guidance."""
        self.capture_label.setText(f"Captured: {self.captured_count} / {self.TARGET_IMAGES}")
        self.progress_bar.setValue(self.captured_count)

        # Update capture guidance
        self._update_guidance()

        # Enable calibration button when minimum images reached
        if self.captured_count >= self.MIN_IMAGES:
            self.calibrate_btn.setEnabled(True)
            if self.captured_count == self.MIN_IMAGES:
                self.calibrate_btn.setText(f"Run Calibration ({self.captured_count} images)")
            else:
                self.calibrate_btn.setText(f"Run Calibration ({self.captured_count} images)")

    def _update_guidance(self):
        """Update the capture guidance based on current coverage."""
        # Get guidance and coverage from calibrator's coverage analyzer
        guidance = self.calibrator.get_capture_guidance()
        coverage = self.calibrator.get_coverage_summary()
        readiness = self.calibrator.coverage_analyzer.get_calibration_readiness()

        # Update FOV grid visualization
        fov_grid = coverage.get('fov_grid', [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.fov_grid_widget.update_grid(fov_grid)

        # Update FOV percentage with color coding
        fov_percent = coverage.get('fov_coverage_percent', 0)
        self.fov_percent_label.setText(f"{fov_percent:.0f}%")
        if fov_percent >= self.MIN_FOV_COVERAGE:
            self.fov_percent_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['success']};")
        elif fov_percent >= self.MIN_FOV_COVERAGE * 0.7:
            self.fov_percent_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['warning']};")
        else:
            self.fov_percent_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['danger']};")

        # Update viewpoint diversity
        viewpoint_diversity = coverage.get('viewpoint_diversity', 0)
        min_viewpoints = 3
        self.viewpoint_count_label.setText(f"{viewpoint_diversity} / {min_viewpoints}")
        if viewpoint_diversity >= min_viewpoints:
            self.viewpoint_count_label.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLORS['success']};")
        else:
            self.viewpoint_count_label.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLORS['danger']};")

        # Update scale (distance) coverage
        scales = coverage.get('scales', {})
        close_count = scales.get('close', 0)
        medium_count = scales.get('medium', 0)
        far_count = scales.get('far', 0)
        self.scale_label.setText(f"C:{close_count} M:{medium_count} F:{far_count}")

        # Update tilt coverage
        tilts = coverage.get('tilts', {})
        tilt_types = []
        for tilt, count in tilts.items():
            if count > 0 and tilt != 'level':
                tilt_types.append(tilt.replace('tilted_', '').capitalize())
        if tilt_types:
            self.tilt_label.setText(', '.join(tilt_types[:2]))
        else:
            self.tilt_label.setText("None yet")

        # Update guidance message
        if guidance:
            self.guidance_label.setText(f"→ {guidance}")
            self.guidance_label.setStyleSheet(f"""
                font-size: 14px;
                font-weight: bold;
                color: {COLORS['warning']};
                background-color: rgba(255, 193, 7, 0.15);
                padding: 8px;
                border-radius: 4px;
            """)
        elif readiness['ready']:
            self.guidance_label.setText("✓ Good coverage! Ready for calibration.")
            self.guidance_label.setStyleSheet(f"""
                font-size: 14px;
                font-weight: bold;
                color: {COLORS['success']};
                background-color: rgba(40, 167, 69, 0.15);
                padding: 8px;
                border-radius: 4px;
            """)
        else:
            self.guidance_label.setText("Continue capturing different positions and angles")
            self.guidance_label.setStyleSheet(f"""
                font-size: 14px;
                font-weight: bold;
                color: {COLORS['primary']};
                background-color: rgba(23, 162, 184, 0.1);
                padding: 8px;
                border-radius: 4px;
            """)

        # Update overall coverage status
        if readiness['issues']:
            self.coverage_status_label.setText(f"⚠ {readiness['issues'][0]}")
            self.coverage_status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['warning']};")
        elif readiness['warnings']:
            self.coverage_status_label.setText(f"💡 {readiness['warnings'][0]}")
            self.coverage_status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
        else:
            self.coverage_status_label.setText("")

    def _on_run_calibration(self):
        """Run the calibration with captured images and provide pass/fail gating."""
        if self.captured_count < self.MIN_IMAGES:
            QMessageBox.warning(
                self,
                "Not Enough Images",
                f"Please capture at least {self.MIN_IMAGES} images.\n"
                f"Currently captured: {self.captured_count}"
            )
            return

        # Check coverage readiness before calibration
        readiness = self.calibrator.coverage_analyzer.get_calibration_readiness()
        coverage = self.calibrator.get_coverage_summary()
        fov_percent = coverage.get('fov_coverage_percent', 0)
        viewpoint_diversity = coverage.get('viewpoint_diversity', 0)

        # Warn if coverage is poor but allow to proceed
        if not readiness['ready']:
            warning_msg = self._build_coverage_warning_message(readiness, coverage)
            reply = QMessageBox.warning(
                self,
                "Coverage Warning",
                warning_msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
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

        # Evaluate result with pass/fail criteria
        calibration_passed, failure_reasons = self._evaluate_calibration_result(result, coverage)

        if calibration_passed:
            # Calibration PASSED - save results
            output_dir = Path(self.base_path) / "camera_intrinsic"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"camera_intrinsics_{self.camera_id}.json"

            try:
                self.calibrator.save_to_json(str(output_path))
                self.calibration_successful = True

                # Generate PDF report
                pdf_path = self._generate_pdf_report(result, output_dir)

                # Update session metadata with results
                self._save_calibration_results(result, str(output_path), pdf_path)

                rms_error = result.get('rms_error', 0)

                # Show success message with detailed results
                success_msg = self._build_success_message(result, coverage, output_path, pdf_path)
                QMessageBox.information(self, "Calibration PASSED", success_msg)

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
            # Calibration FAILED - provide actionable guidance
            failure_msg = self._build_failure_message(result, coverage, failure_reasons)
            QMessageBox.warning(self, "Calibration FAILED", failure_msg)
            self._resume_capture()

    def _evaluate_calibration_result(self, result: Optional[Dict], coverage: Dict) -> Tuple[bool, List[str]]:
        """Evaluate calibration result and return pass/fail with reasons."""
        failure_reasons = []

        if result is None:
            failure_reasons.append("Calibration algorithm failed to converge")
            return False, failure_reasons

        rms_error = result.get('rms_error', float('inf'))
        fov_percent = coverage.get('fov_coverage_percent', 0)
        viewpoint_diversity = coverage.get('viewpoint_diversity', 0)

        # Check RMS error threshold
        if rms_error > self.MAX_RMS_ERROR:
            failure_reasons.append(f"High reprojection error ({rms_error:.3f} > {self.MAX_RMS_ERROR} pixels)")

        # Check FOV coverage (warning, not failure)
        if fov_percent < self.MIN_FOV_COVERAGE:
            failure_reasons.append(f"Insufficient FOV coverage ({fov_percent:.0f}% < {self.MIN_FOV_COVERAGE:.0f}%)")

        # Check for extremely high error (definite failure)
        if rms_error > 2.0:
            failure_reasons.append("Reprojection error exceeds maximum threshold")

        # Pass if RMS is acceptable (allow warning-level issues)
        passed = rms_error <= self.MAX_RMS_ERROR
        return passed, failure_reasons

    def _build_coverage_warning_message(self, readiness: Dict, coverage: Dict) -> str:
        """Build a warning message about coverage issues."""
        fov_percent = coverage.get('fov_coverage_percent', 0)
        viewpoint_diversity = coverage.get('viewpoint_diversity', 0)

        msg_parts = [
            "Coverage may be insufficient for optimal calibration:\n"
        ]

        for issue in readiness.get('issues', []):
            msg_parts.append(f"• {issue}")

        msg_parts.append(f"\nCurrent coverage: {fov_percent:.0f}% FOV, {viewpoint_diversity} viewpoints")
        msg_parts.append("\nDo you want to proceed anyway?")

        return "\n".join(msg_parts)

    def _build_success_message(self, result: Dict, coverage: Dict, output_path: Path,
                               pdf_path: Optional[str]) -> str:
        """Build a success message with calibration details."""
        rms_error = result.get('rms_error', 0)
        fov_percent = coverage.get('fov_coverage_percent', 0)
        viewpoint_diversity = coverage.get('viewpoint_diversity', 0)

        msg = (
            f"Intrinsic calibration for {self.camera_id} completed successfully!\n\n"
            f"RESULT: PASSED\n\n"
            f"Quality Metrics:\n"
            f"  • RMS Error: {rms_error:.4f} pixels (< {self.MAX_RMS_ERROR})\n"
            f"  • FOV Coverage: {fov_percent:.0f}%\n"
            f"  • Viewpoint Diversity: {viewpoint_diversity}\n"
            f"  • Images Used: {self.captured_count}\n\n"
            f"Output Files:\n"
            f"  • Calibration: {output_path.name}\n"
        )

        if pdf_path:
            msg += f"  • Report: {Path(pdf_path).name}\n"

        msg += f"\nImages saved to: {self.images_dir}"

        return msg

    def _build_failure_message(self, result: Optional[Dict], coverage: Dict,
                               failure_reasons: List[str]) -> str:
        """Build an actionable failure message."""
        fov_percent = coverage.get('fov_coverage_percent', 0)
        viewpoint_diversity = coverage.get('viewpoint_diversity', 0)
        rms_error = result.get('rms_error', float('inf')) if result else float('inf')

        msg = "Calibration FAILED\n\n"

        # Show failure reasons
        msg += "Issues detected:\n"
        for reason in failure_reasons:
            msg += f"  • {reason}\n"

        msg += f"\nCurrent metrics:\n"
        msg += f"  • RMS Error: {rms_error:.4f} pixels\n"
        msg += f"  • FOV Coverage: {fov_percent:.0f}%\n"
        msg += f"  • Viewpoint Diversity: {viewpoint_diversity}\n"
        msg += f"  • Images Captured: {self.captured_count}\n\n"

        # Provide actionable guidance
        msg += "Recommended actions:\n"

        if fov_percent < self.MIN_FOV_COVERAGE:
            uncovered = self.calibrator.coverage_analyzer.get_uncovered_fov_regions()
            if uncovered:
                regions = ', '.join(uncovered[:3])
                msg += f"  1. Capture images with target in: {regions}\n"
            else:
                msg += "  1. Capture additional images at different positions\n"

        if viewpoint_diversity < 3:
            msg += "  2. Vary distance (close/medium/far) and tilt angles\n"

        if rms_error > self.MAX_RMS_ERROR:
            msg += "  3. Ensure target is fully visible and in focus\n"
            msg += "  4. Avoid motion blur - hold target steady when capturing\n"
            msg += "  5. Check lighting - avoid reflections on target\n"

        msg += "\nTip: The FOV grid shows which areas need more coverage (red = missing)"

        return msg

    def _generate_pdf_report(self, result: Dict, output_dir: Path) -> Optional[str]:
        """Generate a PDF report for the calibration."""
        if not HAS_REPORTLAB:
            return None

        try:
            pdf_path = output_dir / f"calibration_report_{self.camera_id}_{self.session_timestamp}.pdf"
            report_gen = CalibrationReportGenerator(str(pdf_path))

            # Add captured images to report
            per_image_errors = result.get('per_image_errors', [])
            for i, img_data in enumerate(self.captured_images):
                annotated_path = img_data.get('annotated_path')
                if annotated_path and Path(annotated_path).exists():
                    annotated_img = cv2.imread(annotated_path)
                    if annotated_img is not None:
                        error = per_image_errors[i] if i < len(per_image_errors) else None
                        report_gen.add_image(
                            image=annotated_img,
                            image_name=f"Image {i+1}",
                            num_corners=img_data.get('num_corners', 0),
                            reprojection_error=error
                        )

            # Generate report
            report_gen.generate_report(result, self.board_config)
            return str(pdf_path)

        except Exception as e:
            print(f"Warning: Failed to generate PDF report: {e}")
            return None

    def _save_calibration_results(self, result: Dict, calibration_path: str,
                                   pdf_path: Optional[str]):
        """Save calibration results to session metadata."""
        coverage = self.calibrator.get_coverage_summary()

        results_data = {
            'status': 'PASSED',
            'rms_error': result.get('rms_error', 0),
            'calibration_file': calibration_path,
            'pdf_report': pdf_path,
            'final_coverage': {
                'fov_percent': coverage.get('fov_coverage_percent', 0),
                'viewpoint_diversity': coverage.get('viewpoint_diversity', 0),
                'total_images': self.captured_count
            }
        }

        # Update session metadata
        metadata_path = self.images_dir / "session_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata['calibration_results'] = results_data
        metadata['completion_timestamp'] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

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

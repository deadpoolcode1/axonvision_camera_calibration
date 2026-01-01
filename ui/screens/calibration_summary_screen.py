"""
Calibration Summary Screen (Final Screen)

Final summary of all calibration steps with pass/fail status.
Allows user to:
- View summarized results for all steps
- Redo specific steps that failed or need improvement
- Finish and write out final calibration YAML files
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import yaml
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QScrollArea, QGridLayout
)
from PySide6.QtCore import Qt, Signal

from ..styles import COLORS
from ..data_models import PlatformConfiguration

logger = logging.getLogger(__name__)


class StepStatusWidget(QFrame):
    """Widget displaying status of a calibration step."""

    redo_requested = Signal(str)  # step_name

    def __init__(self, step_name: str, step_title: str, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.step_title = step_title
        self._setup_ui()
        self.set_status("pending", {})

    def _setup_ui(self):
        """Setup the widget UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(8)

        # Header row
        header_layout = QHBoxLayout()

        # Step icon and title
        self.status_icon = QLabel("○")
        self.status_icon.setStyleSheet("font-size: 18px;")
        header_layout.addWidget(self.status_icon)

        self.title_label = QLabel(self.step_title)
        self.title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']};")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # Status badge
        self.status_badge = QLabel("Pending")
        self.status_badge.setStyleSheet(f"""
            background-color: {COLORS['text_muted']};
            color: white;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        """)
        header_layout.addWidget(self.status_badge)

        layout.addLayout(header_layout)

        # Details section
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(f"font-size: 12px; color: {COLORS['text_muted']};")
        layout.addWidget(self.details_label)

        # Metrics grid
        self.metrics_frame = QFrame()
        self.metrics_layout = QGridLayout(self.metrics_frame)
        self.metrics_layout.setContentsMargins(0, 5, 0, 5)
        self.metrics_layout.setSpacing(10)
        layout.addWidget(self.metrics_frame)
        self.metrics_frame.hide()  # Hidden until we have metrics

        # Action buttons row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.redo_btn = QPushButton("Redo Step")
        self.redo_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: {COLORS['text_dark']};
                padding: 6px 16px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #E0A800;
            }}
        """)
        self.redo_btn.clicked.connect(lambda: self.redo_requested.emit(self.step_name))
        self.redo_btn.hide()  # Only shown for completed steps
        button_layout.addWidget(self.redo_btn)

        layout.addLayout(button_layout)

    def set_status(self, status: str, metrics: Dict[str, Any]):
        """
        Set the step status and display metrics.

        Args:
            status: 'pending', 'passed', 'failed', 'warning'
            metrics: Dict of metrics to display
        """
        # Update status icon and badge
        if status == "passed":
            self.status_icon.setText("✓")
            self.status_icon.setStyleSheet(f"font-size: 18px; color: {COLORS['success']};")
            self.status_badge.setText("PASSED")
            self.status_badge.setStyleSheet(f"""
                background-color: {COLORS['success']};
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            """)
            self.redo_btn.show()
        elif status == "failed":
            self.status_icon.setText("✗")
            self.status_icon.setStyleSheet(f"font-size: 18px; color: {COLORS['danger']};")
            self.status_badge.setText("FAILED")
            self.status_badge.setStyleSheet(f"""
                background-color: {COLORS['danger']};
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            """)
            self.redo_btn.show()
            self.redo_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['danger']};
                    color: white;
                    padding: 6px 16px;
                    font-size: 12px;
                    font-weight: bold;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: #C82333;
                }}
            """)
            self.redo_btn.setText("Retry Step")
        elif status == "warning":
            self.status_icon.setText("⚠")
            self.status_icon.setStyleSheet(f"font-size: 18px; color: {COLORS['warning']};")
            self.status_badge.setText("WARNING")
            self.status_badge.setStyleSheet(f"""
                background-color: {COLORS['warning']};
                color: {COLORS['text_dark']};
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            """)
            self.redo_btn.show()
        else:  # pending
            self.status_icon.setText("○")
            self.status_icon.setStyleSheet(f"font-size: 18px; color: {COLORS['text_muted']};")
            self.status_badge.setText("PENDING")
            self.status_badge.setStyleSheet(f"""
                background-color: {COLORS['text_muted']};
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            """)
            self.redo_btn.hide()

        # Update metrics display
        self._update_metrics(metrics)

    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update the metrics grid."""
        # Clear existing metrics
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not metrics:
            self.metrics_frame.hide()
            return

        self.metrics_frame.show()
        col = 0
        for key, value in metrics.items():
            # Metric name
            name_label = QLabel(f"{key}:")
            name_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
            self.metrics_layout.addWidget(name_label, 0, col)

            # Metric value
            value_label = QLabel(str(value))
            value_label.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLORS['text_dark']};")
            self.metrics_layout.addWidget(value_label, 1, col)

            col += 1

    def set_details(self, details: str):
        """Set the details text."""
        self.details_label.setText(details)


class CalibrationSummaryScreen(QWidget):
    """
    Calibration Summary Screen

    Final screen showing all calibration step results with option to
    redo steps or finish and write calibration files.
    """

    cancel_requested = Signal()
    redo_step_requested = Signal(str)  # step_name
    finish_requested = Signal()

    def __init__(self, config: PlatformConfiguration = None, parent=None):
        super().__init__(parent)
        self.config = config or PlatformConfiguration()
        self.base_path = "."
        self.step_widgets: Dict[str, StepStatusWidget] = {}
        self.calibration_results: Dict[str, Dict] = {}
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(15)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)

        screen_label = QLabel("Final Step: Calibration Summary")
        screen_label.setObjectName("screen_indicator")
        header_layout.addWidget(screen_label)

        title = QLabel("Calibration Results")
        title.setObjectName("title")
        header_layout.addWidget(title)

        step_label = QLabel("Review results and finish calibration")
        step_label.setObjectName("step_indicator")
        header_layout.addWidget(step_label)

        main_layout.addLayout(header_layout)

        # Overall status banner
        self.status_banner = QFrame()
        self.status_banner.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['table_header']};
                border-radius: 6px;
                padding: 15px;
            }}
        """)
        banner_layout = QHBoxLayout(self.status_banner)

        self.overall_status_icon = QLabel("○")
        self.overall_status_icon.setStyleSheet("font-size: 32px;")
        banner_layout.addWidget(self.overall_status_icon)

        status_text_layout = QVBoxLayout()
        self.overall_status_label = QLabel("Calibration In Progress")
        self.overall_status_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['text_dark']};")
        status_text_layout.addWidget(self.overall_status_label)

        self.overall_details_label = QLabel("Complete all steps to finish calibration")
        self.overall_details_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']};")
        status_text_layout.addWidget(self.overall_details_label)

        banner_layout.addLayout(status_text_layout)
        banner_layout.addStretch()

        main_layout.addWidget(self.status_banner)

        # Scrollable steps area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        self.steps_layout = QVBoxLayout(scroll_content)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(12)

        # Add step widgets
        self._create_step_widgets()

        self.steps_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)

        # Bottom navigation bar
        nav_layout = QHBoxLayout()

        self.back_btn = QPushButton("< Back to Calibration")
        self.back_btn.setObjectName("cancel_button")
        self.back_btn.clicked.connect(self._on_back_clicked)
        self.back_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.back_btn)

        nav_layout.addStretch()

        self.finish_btn = QPushButton("Finish & Save YAML Files")
        self.finish_btn.setObjectName("primary_button")
        self.finish_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                padding: 12px 30px;
                font-size: 15px;
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
        self.finish_btn.clicked.connect(self._on_finish_clicked)
        self.finish_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.finish_btn)

        main_layout.addLayout(nav_layout)

    def _create_step_widgets(self):
        """Create step status widgets for each calibration step."""
        steps = [
            ("intrinsic", "Intrinsic Calibration"),
            ("extrinsic", "Extrinsic Calibration"),
        ]

        for step_name, step_title in steps:
            widget = StepStatusWidget(step_name, step_title)
            widget.redo_requested.connect(self._on_redo_requested)
            self.step_widgets[step_name] = widget
            self.steps_layout.addWidget(widget)

    def set_config(self, config: PlatformConfiguration):
        """Set the platform configuration."""
        self.config = config
        self._update_intrinsic_status()

    def set_base_path(self, path: str):
        """Set the base path for finding calibration files."""
        self.base_path = path

    def _update_intrinsic_status(self):
        """Update intrinsic calibration status for all cameras."""
        if "intrinsic" not in self.step_widgets:
            return

        widget = self.step_widgets["intrinsic"]

        # Check calibration status for all cameras
        all_calibrated = True
        calibrated_count = 0
        total_cameras = len(self.config.cameras)
        total_rms = 0.0
        total_fov = 0.0

        camera_results = []
        for camera in self.config.cameras:
            has_calib = camera.has_intrinsic_calibration(self.base_path)
            if has_calib:
                calibrated_count += 1
                # Try to load calibration data for metrics
                calib_path = Path(self.base_path) / camera.intrinsic_file_path
                if calib_path.exists():
                    try:
                        with open(calib_path) as f:
                            calib_data = json.load(f)
                        rms = calib_data.get('rms_error', 0)
                        total_rms += rms
                        camera_results.append({
                            'camera_id': camera.camera_id,
                            'rms_error': rms,
                            'status': 'passed'
                        })
                    except Exception:
                        camera_results.append({
                            'camera_id': camera.camera_id,
                            'status': 'passed'
                        })
            else:
                all_calibrated = False
                camera_results.append({
                    'camera_id': camera.camera_id,
                    'status': 'missing'
                })

        # Determine overall status
        if total_cameras == 0:
            widget.set_status("pending", {})
            widget.set_details("No cameras configured")
        elif all_calibrated:
            avg_rms = total_rms / calibrated_count if calibrated_count > 0 else 0
            metrics = {
                "Cameras": f"{calibrated_count}/{total_cameras}",
                "Avg RMS": f"{avg_rms:.4f} px"
            }
            widget.set_status("passed", metrics)
            widget.set_details(f"All {total_cameras} cameras calibrated successfully")
        else:
            missing_cameras = [c.camera_id for c in self.config.cameras
                             if not c.has_intrinsic_calibration(self.base_path)]
            metrics = {
                "Cameras": f"{calibrated_count}/{total_cameras}",
                "Missing": str(len(missing_cameras))
            }
            widget.set_status("failed", metrics)
            widget.set_details(f"Missing calibration: {', '.join(missing_cameras[:3])}")

        self.calibration_results['intrinsic'] = {
            'cameras': camera_results,
            'all_complete': all_calibrated,
            'count': calibrated_count,
            'total': total_cameras
        }

        # Update extrinsic status (placeholder for now)
        if "extrinsic" in self.step_widgets:
            extrinsic_widget = self.step_widgets["extrinsic"]
            extrinsic_widget.set_status("pending", {})
            extrinsic_widget.set_details("Extrinsic calibration not yet implemented")

        # Update overall status
        self._update_overall_status()

    def _update_overall_status(self):
        """Update the overall calibration status banner."""
        intrinsic_result = self.calibration_results.get('intrinsic', {})

        if intrinsic_result.get('all_complete', False):
            self.overall_status_icon.setText("✓")
            self.overall_status_icon.setStyleSheet(f"font-size: 32px; color: {COLORS['success']};")
            self.overall_status_label.setText("Ready to Finish")
            self.overall_details_label.setText("All intrinsic calibrations complete. Click 'Finish' to save YAML files.")
            self.finish_btn.setEnabled(True)
            self.status_banner.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba(40, 167, 69, 0.1);
                    border: 2px solid {COLORS['success']};
                    border-radius: 6px;
                    padding: 15px;
                }}
            """)
        else:
            missing = intrinsic_result.get('total', 0) - intrinsic_result.get('count', 0)
            self.overall_status_icon.setText("○")
            self.overall_status_icon.setStyleSheet(f"font-size: 32px; color: {COLORS['warning']};")
            self.overall_status_label.setText("Calibration Incomplete")
            self.overall_details_label.setText(f"{missing} camera(s) still need intrinsic calibration")
            self.finish_btn.setEnabled(False)
            self.status_banner.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['table_header']};
                    border-radius: 6px;
                    padding: 15px;
                }}
            """)

    def _on_redo_requested(self, step_name: str):
        """Handle redo request for a step."""
        self.redo_step_requested.emit(step_name)

    def _on_back_clicked(self):
        """Handle back button click."""
        self.cancel_requested.emit()

    def _on_finish_clicked(self):
        """Handle finish button click - save YAML files."""
        # Confirm before saving
        reply = QMessageBox.question(
            self,
            "Finish Calibration",
            "This will save the final calibration YAML files for all cameras.\n\n"
            "Do you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        # Generate and save YAML files
        try:
            saved_files = self._save_yaml_files()

            if saved_files:
                success_msg = "Calibration files saved successfully!\n\n"
                success_msg += "Files created:\n"
                for f in saved_files[:5]:  # Show first 5 files
                    success_msg += f"  • {Path(f).name}\n"
                if len(saved_files) > 5:
                    success_msg += f"  ... and {len(saved_files) - 5} more\n"
                success_msg += f"\nOutput directory: {Path(self.base_path) / 'camera_calibration'}"

                QMessageBox.information(self, "Calibration Complete", success_msg)
                self.finish_requested.emit()
            else:
                QMessageBox.warning(
                    self,
                    "No Files Saved",
                    "No calibration files were saved. Please check that all cameras are calibrated."
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save calibration files:\n{str(e)}"
            )

    def _save_yaml_files(self) -> List[str]:
        """Save calibration data to YAML files."""
        saved_files = []

        output_dir = Path(self.base_path) / "camera_calibration"
        output_dir.mkdir(parents=True, exist_ok=True)

        for camera in self.config.cameras:
            # Load JSON calibration data
            json_path = Path(self.base_path) / camera.intrinsic_file_path
            if not json_path.exists():
                continue

            try:
                with open(json_path) as f:
                    calib_data = json.load(f)

                # Convert to YAML format
                yaml_data = {
                    'camera_id': camera.camera_id,
                    'camera_type': camera.camera_type,
                    'camera_model': camera.camera_model,
                    'mounting_position': camera.mounting_position,
                    'calibration': {
                        'intrinsic': {
                            'camera_matrix': calib_data.get('camera_matrix'),
                            'distortion_coefficients': calib_data.get('distortion_coefficients'),
                            'image_size': calib_data.get('image_size'),
                            'rms_error': calib_data.get('rms_error'),
                            'calibration_date': calib_data.get('calibration_date')
                        }
                    },
                    'platform': {
                        'type': self.config.platform_type,
                        'id': self.config.platform_id
                    },
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'source_file': str(json_path)
                    }
                }

                # Save YAML file
                yaml_path = output_dir / f"{camera.camera_id}_calibration.yaml"
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

                saved_files.append(str(yaml_path))
                logger.info(f"Saved calibration YAML: {yaml_path}")

            except Exception as e:
                logger.error(f"Failed to save YAML for {camera.camera_id}: {e}")

        # Also save a combined platform calibration file
        if saved_files:
            try:
                platform_data = {
                    'platform': {
                        'type': self.config.platform_type,
                        'id': self.config.platform_id,
                        'cameras_count': len(self.config.cameras)
                    },
                    'cameras': [c.camera_id for c in self.config.cameras],
                    'calibration_files': [Path(f).name for f in saved_files],
                    'generated_at': datetime.now().isoformat()
                }

                platform_path = output_dir / f"platform_{self.config.platform_id}_calibration.yaml"
                with open(platform_path, 'w') as f:
                    yaml.dump(platform_data, f, default_flow_style=False, sort_keys=False)

                saved_files.append(str(platform_path))

            except Exception as e:
                logger.error(f"Failed to save platform YAML: {e}")

        return saved_files

    def refresh(self):
        """Refresh the summary with current calibration status."""
        self._update_intrinsic_status()

    def showEvent(self, event):
        """Refresh when screen is shown."""
        super().showEvent(event)
        self.refresh()

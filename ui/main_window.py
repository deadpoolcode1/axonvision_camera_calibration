"""
Main Window for Camera Calibration Tool

Central controller managing screen navigation and application state.
"""

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QPushButton, QMenu, QLabel, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction

from . import __version__
from .styles import MAIN_STYLESHEET
from config import edgesa_config
from .data_models import (
    CalibrationDataStore, CalibrationSession, PlatformConfiguration
)
from .screens.login_screen import LoginScreen
from .screens.welcome_screen import WelcomeScreen
from .screens.platform_config_screen import PlatformConfigScreen
from .screens.camera_preview_screen import CameraPreviewScreen
from .screens.calibration_summary_screen import CalibrationSummaryScreen
from .dialogs.log_viewer_dialog import LogViewerDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window.

    Manages navigation between screens and overall application state.
    """

    def __init__(self):
        super().__init__()

        # Determine base path (where the app is run from)
        self.base_path = str(Path.cwd())

        # Build settings dict for components that need simulation mode info
        self._settings = self._build_settings_dict()

        # Initialize data store with settings for mock sync
        self.data_store = CalibrationDataStore(settings=self._settings)

        # Current session state
        self.current_session = None
        self.current_config = None
        self.current_user = None

        self._setup_ui()
        self._connect_signals()

    def _build_settings_dict(self) -> dict:
        """Build settings dict from edgesa_config for components that need it."""
        return {
            "simulation": {
                "enabled": edgesa_config.simulation.enabled,
                "persistence": {
                    "state_file": edgesa_config.simulation.persistence.state_file,
                    "filesystem_path": edgesa_config.simulation.persistence.filesystem_path,
                }
            }
        }

    def _setup_ui(self):
        """Setup the main window UI."""
        self.setWindowTitle(f"AxonVision Camera Calibration Tool v{__version__}")
        self.setMinimumSize(1100, 800)
        self.resize(1280, 900)

        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

        # Central widget with stacked layout for screens
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Hamburger menu bar
        menu_bar = QHBoxLayout()
        menu_bar.setContentsMargins(10, 5, 10, 5)

        self.hamburger_btn = QPushButton("≡")  # Hamburger icon (≡ renders better than ☰)
        self.hamburger_btn.setFixedSize(40, 40)
        self.hamburger_btn.setToolTip("Application menu - access logs, settings, and help")
        self.hamburger_btn.setStyleSheet("""
            QPushButton {
                font-size: 32px;
                font-weight: 900;
                color: #ffffff;
                background-color: #17a2b8;
                border: 2px solid #138496;
                border-radius: 6px;
                padding-bottom: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
                color: #ffffff;
                border-color: #0d6d7e;
            }
            QPushButton:pressed {
                background-color: #0d6d7e;
                color: #ffffff;
            }
        """)
        self.hamburger_btn.clicked.connect(self._show_hamburger_menu)
        menu_bar.addWidget(self.hamburger_btn)

        menu_bar.addStretch()

        layout.addLayout(menu_bar)

        # Simulation mode banner (shown when simulation is enabled)
        self._setup_simulation_banner(layout)

        self.screen_stack = QStackedWidget()
        layout.addWidget(self.screen_stack)

        # Create screens
        self.login_screen = LoginScreen()
        self.welcome_screen = WelcomeScreen(self.data_store)
        self.platform_config_screen = PlatformConfigScreen(settings=self._settings)
        self.platform_config_screen.set_base_path(self.base_path)
        self.camera_preview_screen = CameraPreviewScreen()
        self.camera_preview_screen.set_base_path(self.base_path)
        self.summary_screen = CalibrationSummaryScreen()
        self.summary_screen.set_base_path(self.base_path)

        # Add screens to stack
        self.screen_stack.addWidget(self.login_screen)  # Index 0
        self.screen_stack.addWidget(self.welcome_screen)  # Index 1
        self.screen_stack.addWidget(self.platform_config_screen)  # Index 2
        self.screen_stack.addWidget(self.camera_preview_screen)  # Index 3
        self.screen_stack.addWidget(self.summary_screen)  # Index 4

        # Start on login screen
        self.screen_stack.setCurrentIndex(0)

    def _setup_simulation_banner(self, layout: QVBoxLayout):
        """Setup the simulation mode warning banner."""
        if not edgesa_config.simulation.enabled:
            return

        # Create banner frame
        self.simulation_banner = QFrame()
        self.simulation_banner.setObjectName("simulationBanner")
        self.simulation_banner.setStyleSheet("""
            QFrame#simulationBanner {
                background-color: #fff3cd;
                border: 2px solid #ffc107;
                border-radius: 6px;
                padding: 8px;
                margin: 5px 10px;
            }
        """)

        banner_layout = QHBoxLayout(self.simulation_banner)
        banner_layout.setContentsMargins(15, 8, 15, 8)

        # Warning icon
        icon_label = QLabel("⚠")
        icon_label.setStyleSheet("font-size: 20px; color: #856404;")
        banner_layout.addWidget(icon_label)

        # Banner text - don't show YAML device count, it's misleading
        # Simulation mode mocks settings/config APIs, video streams from real cameras
        mock_server = edgesa_config.simulation.mock_server
        text_label = QLabel(
            f"<b>SIMULATION MODE ACTIVE</b> - Using mock device APIs "
            f"(Discovery: {mock_server.host}:{mock_server.discovery_port}, "
            f"Device API: {mock_server.host}:{mock_server.device_api_port})"
        )
        text_label.setStyleSheet("color: #856404; font-size: 12px;")
        text_label.setWordWrap(True)
        banner_layout.addWidget(text_label, 1)

        # Close button (optional - just hides the banner)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #856404;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #533f03;
            }
        """)
        close_btn.setToolTip("Hide this banner (simulation mode remains active)")
        close_btn.clicked.connect(lambda: self.simulation_banner.hide())
        banner_layout.addWidget(close_btn)

        layout.addWidget(self.simulation_banner)

        # Log simulation mode startup
        logger.warning(
            f"Application started in SIMULATION MODE - "
            f"Mock server expected at {mock_server.discovery_url}"
        )

        # Schedule health check after UI is rendered
        QTimer.singleShot(500, self._check_simulation_server_health)

    def _check_simulation_server_health(self):
        """Check if simulation mock server is reachable and warn user if not."""
        if not edgesa_config.simulation.enabled:
            return

        from services.device_config_service import DeviceConfigService

        service = DeviceConfigService()
        is_healthy, message = service.check_discovery_health()

        if not is_healthy:
            mock_server = edgesa_config.simulation.mock_server
            QMessageBox.warning(
                self,
                "Simulation Server Not Running",
                f"<b>Simulation mode is enabled but the mock server is not reachable.</b><br><br>"
                f"Expected server at: <code>{mock_server.discovery_url}</code><br><br>"
                f"Error: {message}<br><br>"
                f"<b>To start the mock server, run:</b><br>"
                f"<code>python -m simulation.server</code><br><br>"
                f"Camera connections will fail until the server is running."
            )
            logger.error(
                f"Simulation mock server not reachable at {mock_server.discovery_url}: {message}"
            )
        else:
            logger.info(f"Simulation mock server health check passed: {message}")

    def _connect_signals(self):
        """Connect screen signals to handlers."""
        # Login screen signals
        self.login_screen.login_successful.connect(self._on_login_successful)

        # Welcome screen signals
        self.welcome_screen.start_new_calibration.connect(self._on_start_new)
        self.welcome_screen.load_existing_calibration.connect(self._on_load_existing)
        self.welcome_screen.open_settings.connect(self._on_open_settings)

        # Platform config screen signals
        self.platform_config_screen.cancel_requested.connect(self._on_cancel_to_welcome)
        self.platform_config_screen.next_requested.connect(self._on_platform_config_next)
        self.platform_config_screen.config_changed.connect(self._on_config_changed)

        # Camera preview screen signals
        self.camera_preview_screen.cancel_requested.connect(self._on_camera_preview_cancel)
        self.camera_preview_screen.next_requested.connect(self._on_camera_preview_next)
        self.camera_preview_screen.camera_removed.connect(self._on_camera_removed_preview)

        # Summary screen signals
        self.summary_screen.cancel_requested.connect(self._on_summary_cancel)
        self.summary_screen.redo_step_requested.connect(self._on_summary_redo_step)
        self.summary_screen.finish_requested.connect(self._on_calibration_finished)

    def _on_login_successful(self, username: str):
        """Handle successful login."""
        self.current_user = username
        self.setWindowTitle(f"AxonVision Camera Calibration Tool v{__version__} - {username}")
        logger.info(f"User '{username}' logged in successfully")
        self.screen_stack.setCurrentWidget(self.welcome_screen)

    def _on_start_new(self):
        """Handle Start New Calibration from welcome screen."""
        # Create fresh configuration with no default cameras
        # User can add cameras as needed using "+ Add Camera" button
        self.current_config = PlatformConfiguration()

        # Update platform config screen and navigate
        self.platform_config_screen.set_config(self.current_config)
        self.screen_stack.setCurrentWidget(self.platform_config_screen)

    def _on_load_existing(self, session_id: str):
        """Handle Load Existing Calibration from welcome screen."""
        if session_id:
            # Load specific session
            session = self.data_store.get_session(session_id)
            if session and session.platform_config:
                self.current_config = session.platform_config
            else:
                QMessageBox.warning(
                    self,
                    "Session Not Found",
                    f"Could not load session: {session_id}"
                )
                return
        else:
            # Load latest configuration
            if self.data_store.last_platform_config:
                self.current_config = self.data_store.last_platform_config
            else:
                # No existing config, start fresh
                QMessageBox.information(
                    self,
                    "No Previous Configuration",
                    "No previous calibration found. Starting with default configuration."
                )
                self._on_start_new()
                return

        # Update platform config screen and navigate
        self.platform_config_screen.set_config(self.current_config)
        self.screen_stack.setCurrentWidget(self.platform_config_screen)

    def _on_open_settings(self):
        """Handle Settings button from welcome screen."""
        QMessageBox.information(
            self,
            "Settings",
            "Settings panel will be implemented in a future update.\n\n"
            "Configure:\n"
            "- Default camera IP range\n"
            "- ChArUco board parameters\n"
            "- Output directories\n"
            "- Network camera settings"
        )

    def _on_cancel_to_welcome(self):
        """Return to welcome screen."""
        self.screen_stack.setCurrentWidget(self.welcome_screen)
        self.welcome_screen.refresh()

    def _on_config_changed(self, config: PlatformConfiguration):
        """Handle config changes (camera added/removed) - save immediately.

        This ensures that if the user navigates back to welcome screen and
        loads the configuration again, the changes are preserved.
        """
        self.current_config = config
        self.data_store.last_platform_config = config
        self.data_store.save()

    def _on_platform_config_next(self, config: PlatformConfiguration):
        """Handle Next from platform configuration screen."""
        self.current_config = config

        # Save configuration as last used
        self.data_store.last_platform_config = config
        self.data_store.save()

        # Navigate to camera preview screen
        self.camera_preview_screen.set_config(config)
        self.screen_stack.setCurrentWidget(self.camera_preview_screen)

        # Verify cameras after screen is shown (with slight delay for UI update)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self.camera_preview_screen.verify_cameras)

    def _on_camera_preview_cancel(self):
        """Handle Cancel/Back from camera preview screen - return to platform config."""
        self.screen_stack.setCurrentWidget(self.platform_config_screen)

    def _on_camera_preview_next(self, config: PlatformConfiguration):
        """Handle Next from camera preview screen - navigate to summary."""
        self.current_config = config

        # Save updated configuration
        self.data_store.last_platform_config = config
        self.data_store.save()

        # Navigate to summary screen
        self.summary_screen.set_config(config)
        self.screen_stack.setCurrentWidget(self.summary_screen)

    def _on_summary_cancel(self):
        """Handle Cancel/Back from summary screen - return to camera preview."""
        self.screen_stack.setCurrentWidget(self.camera_preview_screen)

    def _on_summary_redo_step(self, step_name: str):
        """Handle redo step request from summary screen."""
        if step_name == "intrinsic":
            # Go back to camera preview screen for intrinsic calibration
            self.screen_stack.setCurrentWidget(self.camera_preview_screen)
            self.camera_preview_screen.verify_cameras()
        elif step_name == "extrinsic":
            # Future: Go to extrinsic calibration screen
            QMessageBox.information(
                self,
                "Extrinsic Calibration",
                "Extrinsic calibration is not yet implemented.\n"
                "This feature will be available in a future update."
            )

    def _on_calibration_finished(self):
        """Handle calibration finish - return to welcome screen."""
        QMessageBox.information(
            self,
            "Calibration Complete",
            f"Platform: {self.current_config.platform_type} - {self.current_config.platform_id}\n\n"
            f"All calibration files have been saved.\n"
            f"You can now use these files for camera operation."
        )
        self.screen_stack.setCurrentWidget(self.welcome_screen)
        self.welcome_screen.refresh()

    def _on_camera_removed_preview(self, camera_index: int):
        """Handle camera removal from preview screen - save updated config."""
        self.current_config = self.camera_preview_screen.config
        self.data_store.last_platform_config = self.current_config
        self.data_store.save()
        logger.info(f"Camera at index {camera_index} removed from configuration")

        # Also update the platform config screen in case user navigates back
        self.platform_config_screen.set_config(self.current_config)

    def _show_hamburger_menu(self):
        """Show the hamburger menu with application options."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #e3f2fd;
            }
        """)

        # View Logs action
        view_logs_action = QAction("View Logs", self)
        view_logs_action.setToolTip("Open the log viewer to see application logs")
        view_logs_action.triggered.connect(self._open_log_viewer)
        menu.addAction(view_logs_action)

        menu.addSeparator()

        # About action
        about_action = QAction(f"About v{__version__}", self)
        about_action.setToolTip("Show application version and information")
        about_action.triggered.connect(self._show_about)
        menu.addAction(about_action)

        # Show menu at button position
        menu.exec(self.hamburger_btn.mapToGlobal(self.hamburger_btn.rect().bottomLeft()))

    def _open_log_viewer(self):
        """Open the log viewer dialog."""
        dialog = LogViewerDialog(parent=self)
        dialog.exec()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            f"About AxonVision Calibration Tool",
            f"<h3>AxonVision Camera Calibration Tool</h3>"
            f"<p>Version: {__version__}</p>"
            f"<p>A Qt-based graphical interface for camera calibration workflow.</p>"
            f"<p>Features:</p>"
            f"<ul>"
            f"<li>Intrinsic camera calibration</li>"
            f"<li>Extrinsic camera calibration</li>"
            f"<li>Multi-camera platform support</li>"
            f"<li>Real-time sensor data display</li>"
            f"</ul>"
        )

    def closeEvent(self, event):
        """Handle window close event."""
        # Save any pending data
        self.data_store.save()
        event.accept()

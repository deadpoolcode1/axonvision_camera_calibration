"""
Main Window for Camera Calibration Tool

Central controller managing screen navigation and application state.
"""

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QPushButton, QMenu
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from . import __version__
from .styles import MAIN_STYLESHEET
from .data_models import (
    CalibrationDataStore, CalibrationSession, PlatformConfiguration
)
from .screens.login_screen import LoginScreen
from .screens.welcome_screen import WelcomeScreen
from .screens.platform_config_screen import PlatformConfigScreen
from .screens.camera_preview_screen import CameraPreviewScreen
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

        # Initialize data store
        self.data_store = CalibrationDataStore()

        # Current session state
        self.current_session = None
        self.current_config = None
        self.current_user = None

        self._setup_ui()
        self._connect_signals()

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

        self.hamburger_btn = QPushButton("☰")  # Hamburger icon
        self.hamburger_btn.setFixedSize(40, 40)
        self.hamburger_btn.setToolTip("Application menu - access logs, settings, and help")
        self.hamburger_btn.setStyleSheet("""
            QPushButton {
                font-size: 28px;
                font-weight: bold;
                color: #333333;
                background-color: #e8f4f8;
                border: 2px solid #17a2b8;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #17a2b8;
                color: white;
            }
            QPushButton:pressed {
                background-color: #138496;
                color: white;
            }
        """)
        self.hamburger_btn.clicked.connect(self._show_hamburger_menu)
        menu_bar.addWidget(self.hamburger_btn)

        menu_bar.addStretch()

        layout.addLayout(menu_bar)

        self.screen_stack = QStackedWidget()
        layout.addWidget(self.screen_stack)

        # Create screens
        self.login_screen = LoginScreen()
        self.welcome_screen = WelcomeScreen(self.data_store)
        self.platform_config_screen = PlatformConfigScreen()
        self.platform_config_screen.set_base_path(self.base_path)
        self.camera_preview_screen = CameraPreviewScreen()
        self.camera_preview_screen.set_base_path(self.base_path)

        # Add screens to stack
        self.screen_stack.addWidget(self.login_screen)  # Index 0
        self.screen_stack.addWidget(self.welcome_screen)  # Index 1
        self.screen_stack.addWidget(self.platform_config_screen)  # Index 2
        self.screen_stack.addWidget(self.camera_preview_screen)  # Index 3

        # Start on login screen
        self.screen_stack.setCurrentIndex(0)

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
        """Handle Next from camera preview screen."""
        self.current_config = config

        # Save updated configuration
        self.data_store.last_platform_config = config
        self.data_store.save()

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

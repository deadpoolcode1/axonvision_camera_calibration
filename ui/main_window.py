"""
Main Window for Camera Calibration Tool

Central controller managing screen navigation and application state.
"""

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt

from .styles import MAIN_STYLESHEET
from .data_models import (
    CalibrationDataStore, CalibrationSession, PlatformConfiguration
)
from .screens.login_screen import LoginScreen
from .screens.welcome_screen import WelcomeScreen
from .screens.platform_config_screen import PlatformConfigScreen
from .screens.camera_preview_screen import CameraPreviewScreen


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
        self.setWindowTitle("AxonVision Camera Calibration Tool")
        self.setMinimumSize(1100, 800)
        self.resize(1280, 900)

        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

        # Central widget with stacked layout for screens
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

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

        # Camera preview screen signals
        self.camera_preview_screen.cancel_requested.connect(self._on_camera_preview_cancel)
        self.camera_preview_screen.next_requested.connect(self._on_camera_preview_next)

    def _on_login_successful(self, username: str):
        """Handle successful login."""
        self.current_user = username
        self.setWindowTitle(f"AxonVision Camera Calibration Tool - {username}")
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

    def closeEvent(self, event):
        """Handle window close event."""
        # Save any pending data
        self.data_store.save()
        event.accept()

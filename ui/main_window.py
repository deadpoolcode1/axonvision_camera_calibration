"""
Main Window for Camera Calibration Tool

Central controller managing screen navigation and application state.
"""

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QPushButton, QMenu, QFrame, QLabel
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction

from . import __version__
from .styles import MAIN_STYLESHEET, COLORS
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
        logger.info("MainWindow initialized")

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

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar with hamburger menu
        top_bar = QFrame()
        top_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['primary']};
                padding: 5px;
            }}
        """)
        top_bar.setFixedHeight(40)

        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(10, 0, 10, 0)

        # Hamburger menu button
        self.menu_btn = QPushButton("☰")
        self.menu_btn.setToolTip("Open application menu for logs, settings, and more")
        self.menu_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: white;
                font-size: 20px;
                font-weight: bold;
                border: none;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
                border-radius: 4px;
            }}
        """)
        self.menu_btn.clicked.connect(self._show_hamburger_menu)
        top_bar_layout.addWidget(self.menu_btn)

        # Title label
        title_label = QLabel(f"AxonVision v{__version__}")
        title_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        title_label.setToolTip("AxonVision Camera Calibration Tool")
        top_bar_layout.addWidget(title_label)

        top_bar_layout.addStretch()

        main_layout.addWidget(top_bar)

        # Screen stack
        self.screen_stack = QStackedWidget()
        main_layout.addWidget(self.screen_stack, 1)

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

    def _show_hamburger_menu(self):
        """Show the hamburger menu."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: white;
                border: 1px solid {COLORS['border']};
                padding: 5px;
            }}
            QMenu::item {{
                padding: 8px 20px;
            }}
            QMenu::item:selected {{
                background-color: {COLORS['table_header']};
            }}
        """)

        # View Logs action
        view_logs_action = QAction("View Logs", self)
        view_logs_action.setToolTip("Open the log viewer to see application logs")
        view_logs_action.triggered.connect(self._open_log_viewer)
        menu.addAction(view_logs_action)

        menu.addSeparator()

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        menu.addAction(about_action)

        # Show menu below button
        menu.exec(self.menu_btn.mapToGlobal(self.menu_btn.rect().bottomLeft()))

    def _open_log_viewer(self):
        """Open the log viewer dialog."""
        logger.info("Opening log viewer")
        dialog = LogViewerDialog(self)
        dialog.exec()

    def _show_about(self):
        """Show about dialog."""
        try:
            from core.logging_config import get_request_id
            request_id = get_request_id()
        except ImportError:
            request_id = "N/A"

        QMessageBox.about(
            self,
            "About AxonVision Camera Calibration",
            f"AxonVision Camera Calibration Tool\n"
            f"Version: {__version__}\n\n"
            f"Multi-camera intrinsic and extrinsic calibration\n"
            f"with INS integration.\n\n"
            f"Session ID: {request_id}\n\n"
            f"© 2024 AxonVision"
        )

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

    def _on_login_successful(self, username: str):
        """Handle successful login."""
        self.current_user = username
        self.setWindowTitle(f"AxonVision Camera Calibration Tool v{__version__} - {username}")
        logger.info(f"User logged in: {username}")
        self.screen_stack.setCurrentWidget(self.welcome_screen)

    def _on_start_new(self):
        """Handle Start New Calibration from welcome screen."""
        logger.info("Starting new calibration session")
        # Create fresh configuration with no default cameras
        # User can add cameras as needed using "+ Add Camera" button
        self.current_config = PlatformConfiguration()

        # Update platform config screen and navigate
        self.platform_config_screen.set_config(self.current_config)
        self.screen_stack.setCurrentWidget(self.platform_config_screen)

    def _on_load_existing(self, session_id: str):
        """Handle Load Existing Calibration from welcome screen."""
        if session_id:
            logger.info(f"Loading existing session: {session_id}")
            # Load specific session
            session = self.data_store.get_session(session_id)
            if session and session.platform_config:
                self.current_config = session.platform_config
            else:
                logger.warning(f"Session not found: {session_id}")
                QMessageBox.warning(
                    self,
                    "Session Not Found",
                    f"Could not load session: {session_id}\n\n"
                    "Possible solutions:\n"
                    "  1. Check if the session file exists\n"
                    "  2. Try starting a new calibration\n"
                    "  3. Verify the session ID is correct"
                )
                return
        else:
            # Load latest configuration
            if self.data_store.last_platform_config:
                logger.info("Loading last platform configuration")
                self.current_config = self.data_store.last_platform_config
            else:
                # No existing config, start fresh
                logger.info("No previous configuration found, starting new")
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
        logger.info("Settings panel requested")
        QMessageBox.information(
            self,
            "Settings",
            "Settings panel will be implemented in a future update.\n\n"
            "Configure:\n"
            "  • Default camera IP range\n"
            "  • ChArUco board parameters\n"
            "  • Output directories\n"
            "  • Network camera settings\n"
            "  • Logging preferences"
        )

    def _on_cancel_to_welcome(self):
        """Return to welcome screen."""
        logger.debug("Returning to welcome screen")
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
        logger.debug("Configuration saved")

    def _on_platform_config_next(self, config: PlatformConfiguration):
        """Handle Next from platform configuration screen."""
        self.current_config = config
        logger.info(f"Platform configured: {config.platform_id} with {len(config.cameras)} cameras")

        # Save configuration as last used
        self.data_store.last_platform_config = config
        self.data_store.save()

        # Navigate to camera preview screen
        self.camera_preview_screen.set_config(config)
        self.screen_stack.setCurrentWidget(self.camera_preview_screen)

        # Verify cameras after screen is shown (with slight delay for UI update)
        QTimer.singleShot(100, self.camera_preview_screen.verify_cameras)

    def _on_camera_preview_cancel(self):
        """Handle Cancel/Back from camera preview screen - return to platform config."""
        logger.debug("Returning to platform configuration")
        self.screen_stack.setCurrentWidget(self.platform_config_screen)

    def _on_camera_preview_next(self, config: PlatformConfiguration):
        """Handle Next from camera preview screen."""
        self.current_config = config
        logger.info("Camera preview completed, proceeding to next step")

        # Save updated configuration
        self.data_store.last_platform_config = config
        self.data_store.save()

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Application closing")
        # Save any pending data
        self.data_store.save()
        event.accept()

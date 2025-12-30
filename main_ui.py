#!/usr/bin/env python3
"""
AxonVision Camera Calibration Tool - GUI Application

Main entry point for the Qt-based graphical interface.

Usage:
    python main_ui.py
"""

import sys
import os

# Set environment variables before Qt import to avoid conflicts
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11 backend

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from ui import __version__
from ui.main_window import MainWindow
from core.logging_config import setup_logging, get_logger, get_request_id


def print_startup_banner(version: str, request_id: str) -> None:
    """Print the startup banner with version and request ID."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           AxonVision Camera Calibration Tool                 ║
║                      v{version}                                 ║
║                                                              ║
║  Request ID: {request_id}          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main entry point for the GUI application."""
    # Initialize logging first
    setup_logging()
    logger = get_logger(__name__)

    # Print startup banner
    request_id = get_request_id()
    print_startup_banner(__version__, request_id)

    logger.info("Application starting")
    logger.info(f"Version: {__version__}")
    logger.info(f"Request ID: {request_id}")

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AxonVision Camera Calibration")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AxonVision")

    logger.info("Qt application created")

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("Main window displayed")

    # Run event loop
    exit_code = app.exec()
    logger.info(f"Application exiting with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

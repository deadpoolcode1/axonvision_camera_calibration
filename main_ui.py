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

from ui.main_window import MainWindow
from ui import __version__
from config.logging_config import setup_logging, print_startup_banner, get_request_id


def main():
    """Main entry point for the GUI application."""
    # Initialize logging
    setup_logging(
        log_level=os.environ.get('LOG_LEVEL', 'DEBUG'),
        log_to_console=True,
        log_to_file=True,
        colored_console=True
    )

    # Print startup banner
    print_startup_banner(version=__version__)

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AxonVision Camera Calibration")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AxonVision")

    # Create and show main window
    window = MainWindow()
    window.setWindowTitle(f"AxonVision Camera Calibration Tool v{__version__}")
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

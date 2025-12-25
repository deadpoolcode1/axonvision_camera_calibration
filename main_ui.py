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


def main():
    """Main entry point for the GUI application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AxonVision Camera Calibration")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AxonVision")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

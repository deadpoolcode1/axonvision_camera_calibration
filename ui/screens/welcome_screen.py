"""
Welcome/Start Screen (Screen 1)

Main entry point for the calibration tool with options to:
- Start a new calibration (goes through all selection and options)
- Load an existing calibration (uses latest camera naming, IPs, definitions)
- View recent calibrations
"""

import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont

from .. import __version__
from ..styles import COLORS, get_status_style
from ..data_models import CalibrationDataStore, CalibrationSession

logger = logging.getLogger(__name__)


class CameraIcon(QWidget):
    """Custom camera icon widget."""

    def __init__(self, size: int = 80, parent=None):
        super().__init__(parent)
        self.size = size
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw circle background
        painter.setBrush(QColor(COLORS['primary']))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.size, self.size)

        # Draw camera body (rectangle)
        painter.setBrush(QColor('white'))
        body_w = self.size * 0.4
        body_h = self.size * 0.5
        body_x = (self.size - body_w) / 2
        body_y = (self.size - body_h) / 2 + 5
        painter.drawRoundedRect(int(body_x), int(body_y), int(body_w), int(body_h), 3, 3)

        # Draw lens circle
        lens_r = self.size * 0.12
        lens_x = self.size / 2
        lens_y = body_y + body_h * 0.45
        painter.setBrush(QColor(COLORS['primary']))
        painter.drawEllipse(int(lens_x - lens_r), int(lens_y - lens_r), int(lens_r * 2), int(lens_r * 2))


class RecentCalibrationItem(QFrame):
    """Widget for displaying a single recent calibration entry."""

    clicked = Signal(str)  # Emits session_id when clicked

    def __init__(self, session: CalibrationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.setObjectName("recent_item")
        self.setCursor(Qt.PointingHandCursor)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # Platform ID
        id_label = QLabel(self.session.platform_id)
        id_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_dark']};")
        id_label.setFixedWidth(120)

        # Timestamp
        time_label = QLabel(self.session.timestamp)
        time_label.setStyleSheet(f"color: {COLORS['text_muted']};")

        # Status with icon
        status_text = self.session.status
        if status_text.lower() == 'passed':
            status_icon = "\u2713"  # Checkmark
        elif status_text.lower() == 'warning':
            status_icon = "\u26A0"  # Warning triangle
        else:
            status_icon = "\u2717"  # X mark

        status_label = QLabel(f"{status_icon} {status_text}")
        status_label.setStyleSheet(get_status_style(status_text))
        status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_label.setFixedWidth(100)

        layout.addWidget(id_label)
        layout.addWidget(time_label, 1)
        layout.addWidget(status_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.session.session_id)
        super().mousePressEvent(event)


class WelcomeScreen(QWidget):
    """
    Welcome/Start Screen

    Provides options to start new calibration or load existing one,
    and shows recent calibration history.
    """

    # Signals
    start_new_calibration = Signal()
    load_existing_calibration = Signal(str)  # Emits session_id or empty for latest
    open_settings = Signal()

    def __init__(self, data_store: CalibrationDataStore, parent=None):
        super().__init__(parent)
        self.data_store = data_store
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 20)
        main_layout.setSpacing(0)

        # Screen indicator (top)
        screen_label = QLabel("Screen 1: Welcome / Start Screen")
        screen_label.setObjectName("screen_indicator")
        screen_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(screen_label)

        main_layout.addSpacing(20)

        # Center content area
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setAlignment(Qt.AlignHCenter)
        center_layout.setSpacing(10)

        # Camera icon
        icon = CameraIcon(80)
        icon_container = QHBoxLayout()
        icon_container.addStretch()
        icon_container.addWidget(icon)
        icon_container.addStretch()
        center_layout.addLayout(icon_container)

        center_layout.addSpacing(10)

        # Title
        title = QLabel("Camera Calibration Tool")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Multi-Camera INS Integration System")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(subtitle)

        center_layout.addSpacing(30)

        # Buttons container
        buttons_container = QFrame()
        buttons_container.setObjectName("card")
        buttons_container.setFixedWidth(500)
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(30, 30, 30, 30)
        buttons_layout.setSpacing(15)

        # Start New Calibration button
        self.new_btn = QPushButton("\u25B6  Start New Calibration")
        self.new_btn.setObjectName("primary_button")
        self.new_btn.setToolTip("Start a fresh calibration session.\n"
                                "Configure platform type, add cameras, and perform calibration.")
        self.new_btn.clicked.connect(self._on_new_calibration)
        buttons_layout.addWidget(self.new_btn)

        # Load Existing Calibration button
        self.load_btn = QPushButton("\U0001F4C2  Load Existing Calibration")
        self.load_btn.setObjectName("secondary_button")
        self.load_btn.setToolTip("Load the most recent calibration configuration.\n"
                                 "Restores previous camera definitions and settings.")
        self.load_btn.clicked.connect(self._on_load_existing)
        buttons_layout.addWidget(self.load_btn)

        buttons_container_layout = QHBoxLayout()
        buttons_container_layout.addStretch()
        buttons_container_layout.addWidget(buttons_container)
        buttons_container_layout.addStretch()
        center_layout.addLayout(buttons_container_layout)

        center_layout.addSpacing(30)

        # Recent calibrations section
        recent_section = QWidget()
        recent_section.setFixedWidth(500)
        recent_layout = QVBoxLayout(recent_section)
        recent_layout.setContentsMargins(0, 0, 0, 0)
        recent_layout.setSpacing(10)

        recent_header = QLabel("Recent Calibrations:")
        recent_header.setStyleSheet(f"font-weight: bold; color: {COLORS['text_dark']};")
        recent_layout.addWidget(recent_header)

        # Recent items container
        self.recent_container = QFrame()
        self.recent_container.setObjectName("card")
        self.recent_items_layout = QVBoxLayout(self.recent_container)
        self.recent_items_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_items_layout.setSpacing(0)

        recent_layout.addWidget(self.recent_container)

        recent_section_layout = QHBoxLayout()
        recent_section_layout.addStretch()
        recent_section_layout.addWidget(recent_section)
        recent_section_layout.addStretch()
        center_layout.addLayout(recent_section_layout)

        main_layout.addWidget(center_widget)
        main_layout.addStretch()

        # Bottom bar
        bottom_layout = QHBoxLayout()

        # Settings button
        self.settings_btn = QPushButton("\u2699 Settings")
        self.settings_btn.setObjectName("settings_button")
        self.settings_btn.setToolTip("Open application settings.\n"
                                      "Configure default IPs, ChArUco board parameters, and output directories.")
        self.settings_btn.clicked.connect(self.open_settings.emit)
        bottom_layout.addWidget(self.settings_btn)

        bottom_layout.addStretch()

        # Version label
        version_label = QLabel(f"Version {__version__}")
        version_label.setObjectName("version")
        version_label.setToolTip("AxonVision Camera Calibration Tool version")
        bottom_layout.addWidget(version_label)

        main_layout.addLayout(bottom_layout)

        # Load recent calibrations
        self._load_recent_calibrations()

    def _load_recent_calibrations(self):
        """Load and display recent calibration sessions."""
        # Clear existing items
        while self.recent_items_layout.count():
            item = self.recent_items_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get recent sessions
        sessions = self.data_store.get_recent_sessions(5)

        if not sessions:
            # Show placeholder
            placeholder = QLabel("No recent calibrations")
            placeholder.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 20px;")
            placeholder.setAlignment(Qt.AlignCenter)
            self.recent_items_layout.addWidget(placeholder)
        else:
            for session in sessions:
                item = RecentCalibrationItem(session)
                item.clicked.connect(self._on_recent_clicked)
                self.recent_items_layout.addWidget(item)

    def _on_new_calibration(self):
        """Handle Start New Calibration button click."""
        self.start_new_calibration.emit()

    def _on_load_existing(self):
        """Handle Load Existing Calibration button click."""
        # Load with latest configuration
        self.load_existing_calibration.emit("")

    def _on_recent_clicked(self, session_id: str):
        """Handle click on recent calibration item."""
        self.load_existing_calibration.emit(session_id)

    def refresh(self):
        """Refresh the recent calibrations list."""
        self._load_recent_calibrations()

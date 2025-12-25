"""
Login Screen (Screen 0)

User authentication screen that appears before the welcome screen.
- Username and password fields
- Login button
- Error message display
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor

from ..styles import COLORS


class LockIcon(QWidget):
    """Custom lock icon widget for login screen."""

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

        # Draw lock body (rectangle)
        painter.setBrush(QColor('white'))
        body_w = self.size * 0.4
        body_h = self.size * 0.35
        body_x = (self.size - body_w) / 2
        body_y = (self.size - body_h) / 2 + 8
        painter.drawRoundedRect(int(body_x), int(body_y), int(body_w), int(body_h), 3, 3)

        # Draw lock shackle (arc)
        painter.setPen(QColor('white'))
        painter.setBrush(Qt.NoBrush)
        pen = painter.pen()
        pen.setWidth(4)
        painter.setPen(pen)

        shackle_w = self.size * 0.25
        shackle_h = self.size * 0.2
        shackle_x = (self.size - shackle_w) / 2
        shackle_y = body_y - shackle_h + 2

        painter.drawArc(
            int(shackle_x), int(shackle_y),
            int(shackle_w), int(shackle_h * 2),
            0, 180 * 16  # Start angle and span in 1/16th degrees
        )


class LoginScreen(QWidget):
    """
    Login Screen

    Entry point for user authentication before accessing the calibration tool.
    """

    # Signals
    login_successful = Signal(str)  # Emits username on successful login

    # Valid credentials (in production, this would be server-side)
    VALID_USERS = {
        "admin": "admin123",
        "operator": "operator123",
        "technician": "tech123",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 20)
        main_layout.setSpacing(0)

        main_layout.addStretch(1)

        # Center content area
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setAlignment(Qt.AlignHCenter)
        center_layout.setSpacing(15)

        # Lock icon
        icon = LockIcon(100)
        icon_container = QHBoxLayout()
        icon_container.addStretch()
        icon_container.addWidget(icon)
        icon_container.addStretch()
        center_layout.addLayout(icon_container)

        center_layout.addSpacing(20)

        # Title
        title = QLabel("AxonVision Calibration")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Please log in to continue")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(subtitle)

        center_layout.addSpacing(30)

        # Login form container
        form_container = QFrame()
        form_container.setObjectName("card")
        form_container.setFixedWidth(400)
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(30, 30, 30, 30)
        form_layout.setSpacing(20)

        # Username field
        username_layout = QVBoxLayout()
        username_layout.setSpacing(8)
        username_label = QLabel("Username")
        username_label.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {COLORS['text_dark']};")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setMinimumHeight(45)
        self.username_input.setStyleSheet(f"""
            QLineEdit {{
                padding: 12px 16px;
                font-size: 16px;
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['primary']};
            }}
            QLineEdit:hover {{
                border-color: {COLORS['primary_dark']};
            }}
        """)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        form_layout.addLayout(username_layout)

        # Password field
        password_layout = QVBoxLayout()
        password_layout.setSpacing(8)
        password_label = QLabel("Password")
        password_label.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {COLORS['text_dark']};")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(45)
        self.password_input.setStyleSheet(f"""
            QLineEdit {{
                padding: 12px 16px;
                font-size: 16px;
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['primary']};
            }}
            QLineEdit:hover {{
                border-color: {COLORS['primary_dark']};
            }}
        """)
        self.password_input.returnPressed.connect(self._on_login_clicked)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        form_layout.addLayout(password_layout)

        # Error message label (hidden by default)
        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"""
            color: {COLORS['danger']};
            font-size: 14px;
            padding: 10px;
            background-color: #FFEBEE;
            border-radius: 4px;
        """)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        form_layout.addWidget(self.error_label)

        form_layout.addSpacing(10)

        # Login button
        self.login_btn = QPushButton("Login")
        self.login_btn.setObjectName("primary_button")
        self.login_btn.setMinimumHeight(50)
        self.login_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 6px;
                padding: 12px 40px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
            QPushButton:pressed {{
                background-color: #1e7e34;
            }}
        """)
        self.login_btn.clicked.connect(self._on_login_clicked)
        form_layout.addWidget(self.login_btn)

        # Center the form
        form_container_layout = QHBoxLayout()
        form_container_layout.addStretch()
        form_container_layout.addWidget(form_container)
        form_container_layout.addStretch()
        center_layout.addLayout(form_container_layout)

        main_layout.addWidget(center_widget)
        main_layout.addStretch(2)

        # Version label at bottom
        version_layout = QHBoxLayout()
        version_layout.addStretch()
        version_label = QLabel("Version 1.0.0")
        version_label.setObjectName("version")
        version_layout.addWidget(version_label)
        main_layout.addLayout(version_layout)

    def _on_login_clicked(self):
        """Handle login button click."""
        username = self.username_input.text().strip()
        password = self.password_input.text()

        # Validate input
        if not username:
            self._show_error("Please enter your username")
            self.username_input.setFocus()
            return

        if not password:
            self._show_error("Please enter your password")
            self.password_input.setFocus()
            return

        # Check credentials
        if username in self.VALID_USERS and self.VALID_USERS[username] == password:
            self._hide_error()
            self.login_successful.emit(username)
        else:
            self._show_error("Invalid username or password")
            self.password_input.clear()
            self.password_input.setFocus()

    def _show_error(self, message: str):
        """Show error message."""
        self.error_label.setText(message)
        self.error_label.show()

    def _hide_error(self):
        """Hide error message."""
        self.error_label.hide()

    def reset(self):
        """Reset the login form."""
        self.username_input.clear()
        self.password_input.clear()
        self._hide_error()
        self.username_input.setFocus()

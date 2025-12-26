"""
Login Screen

User authentication screen for the calibration tool.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor

from ..styles import COLORS


class LoginIcon(QWidget):
    """Custom user/login icon widget."""

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

        # Draw user icon (head circle)
        painter.setBrush(QColor('white'))
        head_r = self.size * 0.15
        head_x = self.size / 2
        head_y = self.size * 0.35
        painter.drawEllipse(int(head_x - head_r), int(head_y - head_r), int(head_r * 2), int(head_r * 2))

        # Draw body (arc/semicircle)
        body_w = self.size * 0.4
        body_h = self.size * 0.25
        body_x = (self.size - body_w) / 2
        body_y = self.size * 0.55
        painter.drawEllipse(int(body_x), int(body_y), int(body_w), int(body_h))


class LoginScreen(QWidget):
    """
    Login/Authentication Screen

    Provides user authentication before accessing the calibration tool.
    """

    # Signals
    login_successful = Signal(str)  # Emits username on successful login

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(0)

        main_layout.addStretch(1)

        # Center content area
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setAlignment(Qt.AlignHCenter)
        center_layout.setSpacing(15)

        # Login icon
        icon = LoginIcon(100)
        icon_container = QHBoxLayout()
        icon_container.addStretch()
        icon_container.addWidget(icon)
        icon_container.addStretch()
        center_layout.addLayout(icon_container)

        center_layout.addSpacing(10)

        # Title
        title = QLabel("AxonVision Calibration")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Please sign in to continue")
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
        username_layout.setSpacing(5)
        username_label = QLabel("Username")
        username_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_dark']};")
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Enter your username")
        self.username_edit.setToolTip("Enter your username or email")
        self.username_edit.returnPressed.connect(self._focus_password)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_edit)
        form_layout.addLayout(username_layout)

        # Password field
        password_layout = QVBoxLayout()
        password_layout.setSpacing(5)
        password_label = QLabel("Password")
        password_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_dark']};")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter your password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setToolTip("Enter your password")
        self.password_edit.returnPressed.connect(self._on_login_clicked)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_edit)
        form_layout.addLayout(password_layout)

        form_layout.addSpacing(10)

        # Error message label (hidden by default)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet(f"""
            color: {COLORS['danger']};
            font-weight: bold;
            padding: 10px;
            background-color: #FFEBEE;
            border-radius: 4px;
        """)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.hide()
        form_layout.addWidget(self.error_label)

        # Login button
        self.login_btn = QPushButton("Sign In")
        self.login_btn.setObjectName("primary_button")
        self.login_btn.setToolTip("Click to sign in")
        self.login_btn.clicked.connect(self._on_login_clicked)
        form_layout.addWidget(self.login_btn)

        form_container_layout = QHBoxLayout()
        form_container_layout.addStretch()
        form_container_layout.addWidget(form_container)
        form_container_layout.addStretch()
        center_layout.addLayout(form_container_layout)

        main_layout.addWidget(center_widget)

        main_layout.addStretch(2)

        # Bottom bar with version
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        version_label = QLabel("Version 1.0.0")
        version_label.setObjectName("version")
        bottom_layout.addWidget(version_label)

        main_layout.addLayout(bottom_layout)

    def _focus_password(self):
        """Move focus to password field."""
        self.password_edit.setFocus()

    def _on_login_clicked(self):
        """Handle login button click."""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()

        # Validate inputs
        if not username:
            self._show_error("Please enter your username")
            self.username_edit.setFocus()
            return

        if not password:
            self._show_error("Please enter your password")
            self.password_edit.setFocus()
            return

        # Simple authentication check
        # In production, this would validate against a backend/database
        if self._authenticate(username, password):
            self.error_label.hide()
            self.login_successful.emit(username)
        else:
            self._show_error("Invalid username or password")
            self.password_edit.clear()
            self.password_edit.setFocus()

    def _authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate user credentials.

        In production, this should validate against a secure backend.
        Currently accepts any non-empty username/password for demo purposes.
        """
        # Demo authentication - accepts any credentials with at least 3 characters each
        # Replace with actual authentication logic in production
        if len(username) >= 3 and len(password) >= 3:
            return True

        # Or use specific demo credentials
        demo_users = {
            'admin': 'admin123',
            'operator': 'operator123',
            'demo': 'demo'
        }
        return demo_users.get(username) == password

    def _show_error(self, message: str):
        """Show error message."""
        self.error_label.setText(message)
        self.error_label.show()

    def clear_form(self):
        """Clear the login form."""
        self.username_edit.clear()
        self.password_edit.clear()
        self.error_label.hide()
        self.username_edit.setFocus()

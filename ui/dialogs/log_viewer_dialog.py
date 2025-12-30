"""
Log Viewer Dialog

Provides a dialog for viewing application logs with filtering and search capabilities.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QLineEdit, QFrame, QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat

from ..styles import COLORS


class LogViewerDialog(QDialog):
    """
    Dialog for viewing and filtering application logs.

    Features:
    - Real-time log viewing
    - Filter by severity level
    - Search within logs
    - Copy log entries to clipboard
    """

    # Log level colors
    LEVEL_COLORS = {
        'DEBUG': '#17a2b8',     # Cyan
        'INFO': '#28a745',      # Green
        'WARNING': '#ffc107',   # Yellow
        'ERROR': '#dc3545',     # Red
        'CRITICAL': '#6f42c1',  # Purple
    }

    def __init__(self, log_file: str = "logs/calibration.log", parent=None):
        super().__init__(parent)
        self.log_file = log_file
        self.current_filter = "ALL"
        self.search_text = ""
        self._all_logs: List[str] = []

        self._setup_ui()
        self._load_logs()

        # Set up auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._load_logs)
        self.refresh_timer.start(2000)  # Refresh every 2 seconds

    def _setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Application Logs")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("Application Logs")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['primary']};")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Manually refresh log display")
        refresh_btn.clicked.connect(self._load_logs)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Filter controls
        filter_frame = QFrame()
        filter_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['table_header']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        filter_layout = QHBoxLayout(filter_frame)

        # Severity filter
        filter_label = QLabel("Filter by Level:")
        filter_layout.addWidget(filter_label)

        self.level_combo = QComboBox()
        self.level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_combo.setToolTip("Filter logs by severity level")
        self.level_combo.currentTextChanged.connect(self._on_filter_changed)
        self.level_combo.setFixedWidth(120)
        filter_layout.addWidget(self.level_combo)

        filter_layout.addSpacing(20)

        # Search
        search_label = QLabel("Search:")
        filter_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search text...")
        self.search_input.setToolTip("Search within log messages")
        self.search_input.textChanged.connect(self._on_search_changed)
        self.search_input.setFixedWidth(200)
        filter_layout.addWidget(self.search_input)

        filter_layout.addStretch()

        # Clear button
        clear_btn = QPushButton("Clear Filter")
        clear_btn.setToolTip("Clear all filters and show all logs")
        clear_btn.clicked.connect(self._clear_filters)
        filter_layout.addWidget(clear_btn)

        layout.addWidget(filter_frame)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.log_display, 1)

        # Status bar
        status_layout = QHBoxLayout()

        self.status_label = QLabel("0 log entries")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        # Copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.setToolTip("Copy all displayed logs to clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        status_layout.addWidget(copy_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        status_layout.addWidget(close_btn)

        layout.addLayout(status_layout)

    def _load_logs(self):
        """Load logs from file."""
        log_path = Path(self.log_file)
        if not log_path.exists():
            self._all_logs = ["No log file found. Logs will appear here once the application starts logging."]
        else:
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    self._all_logs = f.readlines()
            except Exception as e:
                self._all_logs = [f"Error reading log file: {e}"]

        self._apply_filters()

    def _on_filter_changed(self, level: str):
        """Handle filter level change."""
        self.current_filter = level
        self._apply_filters()

    def _on_search_changed(self, text: str):
        """Handle search text change."""
        self.search_text = text.lower()
        self._apply_filters()

    def _clear_filters(self):
        """Clear all filters."""
        self.level_combo.setCurrentText("ALL")
        self.search_input.clear()
        self.current_filter = "ALL"
        self.search_text = ""
        self._apply_filters()

    def _apply_filters(self):
        """Apply current filters to logs and display."""
        filtered_logs = []

        for line in self._all_logs:
            # Apply level filter
            if self.current_filter != "ALL":
                if f" - {self.current_filter} - " not in line:
                    continue

            # Apply search filter
            if self.search_text and self.search_text not in line.lower():
                continue

            filtered_logs.append(line)

        self._display_logs(filtered_logs)
        self.status_label.setText(f"{len(filtered_logs)} of {len(self._all_logs)} log entries")

    def _display_logs(self, logs: List[str]):
        """Display logs with syntax highlighting."""
        self.log_display.clear()

        for line in logs:
            # Determine color based on log level
            color = "#d4d4d4"  # Default gray
            for level, level_color in self.LEVEL_COLORS.items():
                if f" - {level} - " in line:
                    color = level_color
                    break

            # Format and append
            cursor = self.log_display.textCursor()
            cursor.movePosition(QTextCursor.End)

            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cursor.setCharFormat(fmt)
            cursor.insertText(line)

        # Scroll to bottom
        self.log_display.moveCursor(QTextCursor.End)

    def _copy_to_clipboard(self):
        """Copy displayed logs to clipboard."""
        text = self.log_display.toPlainText()
        QApplication.clipboard().setText(text)

    def closeEvent(self, event):
        """Stop refresh timer when dialog closes."""
        self.refresh_timer.stop()
        super().closeEvent(event)

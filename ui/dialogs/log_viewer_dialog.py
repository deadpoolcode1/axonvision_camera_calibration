"""
Log Viewer Dialog

Provides a searchable, filterable log viewer accessible from the hamburger menu.
"""

import logging
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QLineEdit, QFrame, QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCharFormat, QColor, QFont

from ..styles import COLORS

try:
    from core.logging_config import get_log_records, clear_log_records
except ImportError:
    def get_log_records():
        return []

    def clear_log_records():
        pass


class LogViewerDialog(QDialog):
    """Dialog for viewing application logs with filtering and search."""

    LOG_LEVEL_COLORS = {
        'DEBUG': '#6C757D',
        'INFO': '#2E86AB',
        'WARNING': '#FFC107',
        'ERROR': '#DC3545',
        'CRITICAL': '#9B2335',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Logs")
        self.setMinimumSize(800, 500)
        self.resize(900, 600)

        self._setup_ui()
        self._refresh_logs()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_logs)
        self.refresh_timer.start(2000)  # Refresh every 2 seconds

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header with controls
        header_layout = QHBoxLayout()

        # Log level filter
        level_label = QLabel("Filter by level:")
        self.level_combo = QComboBox()
        self.level_combo.addItems(['All', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.level_combo.setCurrentText('All')
        self.level_combo.setToolTip("Filter logs by severity level")
        self.level_combo.currentTextChanged.connect(self._refresh_logs)
        header_layout.addWidget(level_label)
        header_layout.addWidget(self.level_combo)

        header_layout.addSpacing(20)

        # Search box
        search_label = QLabel("Search:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to search logs...")
        self.search_edit.setToolTip("Search for text in log messages")
        self.search_edit.textChanged.connect(self._refresh_logs)
        header_layout.addWidget(search_label)
        header_layout.addWidget(self.search_edit, 1)

        header_layout.addSpacing(20)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Manually refresh the log display")
        refresh_btn.clicked.connect(self._refresh_logs)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Log display area
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        self.log_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 10px;
            }}
        """)
        layout.addWidget(self.log_display, 1)

        # Footer with stats and actions
        footer_layout = QHBoxLayout()

        self.stats_label = QLabel("0 log entries")
        self.stats_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        footer_layout.addWidget(self.stats_label)

        footer_layout.addStretch()

        # Copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.setToolTip("Copy all displayed logs to clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        footer_layout.addWidget(copy_btn)

        # Clear button
        clear_btn = QPushButton("Clear Logs")
        clear_btn.setToolTip("Clear the log buffer (logs are still saved to file)")
        clear_btn.clicked.connect(self._clear_logs)
        footer_layout.addWidget(clear_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer_layout.addWidget(close_btn)

        layout.addLayout(footer_layout)

    def _refresh_logs(self):
        """Refresh the log display with current filters."""
        records = get_log_records()

        # Apply level filter
        selected_level = self.level_combo.currentText()
        if selected_level != 'All':
            level_order = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            min_level_idx = level_order.index(selected_level)
            records = [
                r for r in records
                if level_order.index(r.get('level', 'INFO')) >= min_level_idx
            ]

        # Apply search filter
        search_text = self.search_edit.text().lower()
        if search_text:
            records = [
                r for r in records
                if search_text in r.get('message', '').lower()
            ]

        # Format and display logs
        self.log_display.clear()
        for record in records:
            level = record.get('level', 'INFO')
            color = self.LOG_LEVEL_COLORS.get(level, '#D4D4D4')
            timestamp = record.get('timestamp', '')
            message = record.get('message', '')

            # Use HTML for colored output
            html = f'<span style="color: {color};">[{timestamp}] [{level}]</span> {message}<br>'
            self.log_display.insertHtml(html)

        # Update stats
        self.stats_label.setText(f"{len(records)} log entries")

        # Scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _copy_to_clipboard(self):
        """Copy displayed logs to clipboard."""
        text = self.log_display.toPlainText()
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def _clear_logs(self):
        """Clear the log buffer."""
        clear_log_records()
        self._refresh_logs()

    def closeEvent(self, event):
        """Stop timer when dialog is closed."""
        self.refresh_timer.stop()
        super().closeEvent(event)

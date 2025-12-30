"""
Logging Configuration Module

Provides standardized logging with:
- Consistent log format across all modules
- Request ID tracking for debugging
- File and console output
- Log rotation

Usage:
    from core.logging_config import setup_logging, get_logger

    # Setup at application startup
    setup_logging()

    # Get a logger in any module
    logger = get_logger(__name__)
    logger.info("Application started")
"""

import logging
import os
import sys
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Global request ID for the application session
_request_id: str = ""

# Log storage for UI log viewer
_log_records: list = []
_max_log_records: int = 1000


def generate_request_id() -> str:
    """Generate a new UUID request ID."""
    global _request_id
    _request_id = str(uuid.uuid4())
    return _request_id


def get_request_id() -> str:
    """Get the current request ID."""
    global _request_id
    if not _request_id:
        generate_request_id()
    return _request_id


class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class LogRecordStore(logging.Handler):
    """Handler that stores log records for UI display."""

    def emit(self, record: logging.LogRecord) -> None:
        global _log_records, _max_log_records
        try:
            msg = self.format(record)
            _log_records.append({
                'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'message': msg,
                'filename': record.filename,
                'lineno': record.lineno,
            })
            # Keep only the last N records
            if len(_log_records) > _max_log_records:
                _log_records = _log_records[-_max_log_records:]
        except Exception:
            pass


def get_log_records() -> list:
    """Get stored log records for UI display."""
    return _log_records.copy()


def clear_log_records() -> None:
    """Clear stored log records."""
    global _log_records
    _log_records = []


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    max_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """
    Setup application logging with standardized format.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        log_file: Log filename (default: axonvision.log)
        max_size_mb: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    # Try to load from config
    try:
        from config import get_config
        config = get_config()
        level = config.get('logging.level', level)
        log_dir = log_dir or config.get('logging.file.path', 'logs')
        log_file = log_file or config.get('logging.file.filename', 'axonvision.log')
        max_size_mb = config.get('logging.file.max_size_mb', max_size_mb)
        backup_count = config.get('logging.file.backup_count', backup_count)
    except ImportError:
        pass

    # Generate request ID for this session
    generate_request_id()

    # Create log directory if needed
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

    # Standardized log format
    log_format = (
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - "
        "[request_id=%(request_id)s] - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add request ID filter
    request_id_filter = RequestIdFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_dir and log_file:
        file_path = Path(log_dir) / log_file
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_id_filter)
        root_logger.addHandler(file_handler)

    # Add log record store for UI
    store_handler = LogRecordStore()
    store_handler.setFormatter(formatter)
    store_handler.addFilter(request_id_filter)
    root_logger.addHandler(store_handler)

    # Log startup message
    root_logger.info(f"Logging initialized - level={level}, request_id={get_request_id()}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

"""
Logging Configuration Module

Provides standardized logging with:
- Consistent format across the application
- File and console output
- Log rotation
- Request ID tracking
- Colored console output (optional)
"""

import logging
import logging.handlers
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


# Global request ID for the current session
_request_id: Optional[str] = None


def get_request_id() -> str:
    """Get the current request ID, generating one if needed."""
    global _request_id
    if _request_id is None:
        _request_id = str(uuid.uuid4())
    return _request_id


def set_request_id(request_id: str) -> None:
    """Set the current request ID."""
    global _request_id
    _request_id = request_id


class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for console output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        result = super().format(record)

        # Restore original level name
        record.levelname = levelname
        return result


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_to_console: Optional[bool] = None,
    log_to_file: Optional[bool] = None,
    max_file_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    colored_console: Optional[bool] = None
) -> logging.Logger:
    """
    Setup application-wide logging configuration.

    Settings are loaded from config/settings.yaml by default.
    Parameters passed to this function override YAML settings.
    Environment variables (LOG_LEVEL, LOG_FILE) override all other settings.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default from YAML: logs/calibration.log)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        max_file_size_mb: Maximum size of log file before rotation (default from YAML: 1MB)
        backup_count: Number of backup log files to keep (default from YAML: 5)
        colored_console: Whether to use colored output in console

    Returns:
        Configured root logger
    """
    # Import config here to avoid circular imports
    from . import config

    # Use YAML config as defaults, then override with function parameters
    log_level = log_level or config.log_level
    log_file = log_file or config.log_file_path
    log_to_console = log_to_console if log_to_console is not None else config.log_console_enabled
    log_to_file = log_to_file if log_to_file is not None else config.log_file_enabled
    max_file_size_mb = max_file_size_mb if max_file_size_mb is not None else config.log_max_size_mb
    backup_count = backup_count if backup_count is not None else config.log_backup_count
    colored_console = colored_console if colored_console is not None else config.log_console_colored

    # Override with environment variables (highest priority)
    log_level = os.environ.get('LOG_LEVEL', log_level).upper()
    log_file = os.environ.get('LOG_FILE', log_file)

    # Create logs directory if needed
    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Log format
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - [request_id=%(request_id)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add request ID filter
    request_id_filter = RequestIdFilter()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))

        if colored_console and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)

        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(request_id_filter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(request_id_filter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def print_startup_banner(version: str = "0.9.0", port: Optional[int] = None) -> None:
    """
    Print a startup banner with application information.

    Args:
        version: Application version
        port: Running port (optional)
    """
    logger = get_logger(__name__)
    request_id = get_request_id()

    banner = f"""
============================================
  AxonVision Camera Calibration Tool v{version}
  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Request ID: {request_id}
{"  Port: " + str(port) if port else ""}
============================================
"""

    # Log to both console and file
    for line in banner.strip().split('\n'):
        logger.info(line)

"""
Core Module for AxonVision Camera Calibration

Contains core infrastructure components: exceptions, logging, and utilities.
"""

from .exceptions import (
    ApplicationException,
    CameraNotFoundException,
    InvalidConfigurationError,
    CalibrationError,
    DiscoveryServiceError,
    NetworkError,
)
from .logging_config import setup_logging, get_logger, get_request_id

__all__ = [
    'ApplicationException',
    'CameraNotFoundException',
    'InvalidConfigurationError',
    'CalibrationError',
    'DiscoveryServiceError',
    'NetworkError',
    'setup_logging',
    'get_logger',
    'get_request_id',
]

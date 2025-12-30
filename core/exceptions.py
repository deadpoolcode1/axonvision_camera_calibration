"""
Custom Domain Exception Classes

Provides a structured exception hierarchy for the AxonVision camera calibration system.
All domain-specific exceptions inherit from ApplicationException.

Usage:
    from core.exceptions import CameraNotFoundException, CalibrationError

    # Raise with context
    raise CameraNotFoundException(
        camera_id="cam_1",
        message="Camera not found on network"
    )

    # Catch and handle
    try:
        perform_calibration()
    except CalibrationError as e:
        logger.error(f"Calibration failed: {e}")
        show_error_dialog(str(e), e.resolution_steps)
"""

from typing import List, Optional


class ApplicationException(Exception):
    """
    Base exception class for all application-specific errors.

    Provides:
    - Meaningful error messages with context
    - Resolution steps for user guidance
    - Structured error information for logging
    """

    def __init__(
        self,
        message: str,
        resolution_steps: Optional[List[str]] = None,
        details: Optional[dict] = None
    ):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            resolution_steps: List of steps the user can take to resolve the issue
            details: Additional context for logging/debugging
        """
        super().__init__(message)
        self.message = message
        self.resolution_steps = resolution_steps or []
        self.details = details or {}

    def __str__(self) -> str:
        return self.message

    def get_user_message(self) -> str:
        """Get a formatted message suitable for display to users."""
        msg = self.message
        if self.resolution_steps:
            msg += "\n\nPossible solutions:\n"
            for i, step in enumerate(self.resolution_steps, 1):
                msg += f"  {i}. {step}\n"
        return msg


class CameraNotFoundException(ApplicationException):
    """
    Raised when a camera cannot be found or connected.

    Common causes:
    - Camera is powered off
    - Network connectivity issues
    - Incorrect IP address configuration
    """

    def __init__(
        self,
        camera_id: str = "",
        ip_address: str = "",
        message: str = "Camera not found",
        resolution_steps: Optional[List[str]] = None
    ):
        default_steps = [
            "Check that the camera is powered on",
            "Verify the camera IP address is correct",
            "Ensure the camera is connected to the network",
            "Try refreshing the camera list",
        ]
        super().__init__(
            message=message,
            resolution_steps=resolution_steps or default_steps,
            details={"camera_id": camera_id, "ip_address": ip_address}
        )
        self.camera_id = camera_id
        self.ip_address = ip_address


class InvalidConfigurationError(ApplicationException):
    """
    Raised when configuration validation fails.

    Common causes:
    - Missing required fields
    - Invalid field values
    - Conflicting configuration options
    """

    def __init__(
        self,
        field: str = "",
        value: str = "",
        message: str = "Invalid configuration",
        resolution_steps: Optional[List[str]] = None
    ):
        default_steps = [
            "Review the configuration values",
            "Check for missing required fields",
            "Ensure values are within valid ranges",
        ]
        super().__init__(
            message=message,
            resolution_steps=resolution_steps or default_steps,
            details={"field": field, "value": value}
        )
        self.field = field
        self.value = value


class CalibrationError(ApplicationException):
    """
    Raised when camera calibration fails.

    Common causes:
    - Insufficient calibration images
    - Poor image quality
    - ChArUco board not detected
    - Camera movement during capture
    """

    def __init__(
        self,
        camera_id: str = "",
        calibration_type: str = "intrinsic",
        message: str = "Calibration failed",
        resolution_steps: Optional[List[str]] = None
    ):
        default_steps = [
            "Ensure the ChArUco board is fully visible in the frame",
            "Improve lighting conditions",
            "Hold the camera steady during capture",
            "Try capturing more calibration images",
        ]
        super().__init__(
            message=message,
            resolution_steps=resolution_steps or default_steps,
            details={"camera_id": camera_id, "calibration_type": calibration_type}
        )
        self.camera_id = camera_id
        self.calibration_type = calibration_type


class DiscoveryServiceError(ApplicationException):
    """
    Raised when camera discovery service fails.

    Common causes:
    - Network interface not available
    - Multicast not supported
    - Firewall blocking discovery
    """

    def __init__(
        self,
        service_type: str = "",
        message: str = "Discovery service error",
        resolution_steps: Optional[List[str]] = None
    ):
        default_steps = [
            "Check network interface connectivity",
            "Verify multicast is enabled on the network",
            "Check firewall settings for discovery ports",
            "Try manual camera IP configuration",
        ]
        super().__init__(
            message=message,
            resolution_steps=resolution_steps or default_steps,
            details={"service_type": service_type}
        )
        self.service_type = service_type


class NetworkError(ApplicationException):
    """
    Raised when network communication fails.

    Common causes:
    - Network timeout
    - Connection refused
    - DNS resolution failure
    """

    def __init__(
        self,
        host: str = "",
        port: int = 0,
        message: str = "Network error",
        resolution_steps: Optional[List[str]] = None
    ):
        default_steps = [
            "Check network connectivity",
            "Verify the target host is reachable",
            "Check if the service is running on the target port",
            "Review firewall settings",
        ]
        super().__init__(
            message=message,
            resolution_steps=resolution_steps or default_steps,
            details={"host": host, "port": port}
        )
        self.host = host
        self.port = port

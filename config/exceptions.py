"""
Custom Exception Classes

Provides domain-specific exception hierarchy for better error handling
and user-friendly error messages.
"""

from typing import Optional, List


class ApplicationException(Exception):
    """
    Base exception class for all application errors.

    Provides common functionality for error handling including:
    - User-friendly error messages
    - Suggested resolution steps
    - Error context preservation
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        resolution_steps: Optional[List[str]] = None
    ):
        """
        Initialize the exception.

        Args:
            message: User-friendly error message
            details: Technical details for logging
            resolution_steps: List of suggested steps to resolve the error
        """
        super().__init__(message)
        self.message = message
        self.details = details
        self.resolution_steps = resolution_steps or []

    def get_user_message(self) -> str:
        """Get a formatted message suitable for displaying to users."""
        parts = [self.message]
        if self.resolution_steps:
            parts.append("\nPossible solutions:")
            for i, step in enumerate(self.resolution_steps, 1):
                parts.append(f"  {i}. {step}")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.message


class CameraNotFoundException(ApplicationException):
    """Raised when a camera cannot be found or accessed."""

    def __init__(
        self,
        camera_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[str] = None
    ):
        if camera_id and ip_address:
            message = f"Camera '{camera_id}' at {ip_address} not found"
        elif camera_id:
            message = f"Camera '{camera_id}' not found"
        elif ip_address:
            message = f"Camera at {ip_address} not found"
        else:
            message = "Camera not found"

        resolution_steps = [
            "Verify the camera is powered on",
            "Check the network cable connection",
            "Confirm the IP address is correct",
            "Ensure the camera is on the same network subnet",
            "Try pinging the camera IP address"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )
        self.camera_id = camera_id
        self.ip_address = ip_address


class InvalidConfigurationError(ApplicationException):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        config_item: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[str] = None
    ):
        if config_item and reason:
            message = f"Invalid configuration for '{config_item}': {reason}"
        elif config_item:
            message = f"Invalid configuration for '{config_item}'"
        elif reason:
            message = f"Invalid configuration: {reason}"
        else:
            message = "Invalid configuration"

        resolution_steps = [
            "Review the configuration settings",
            "Check the settings.yaml file for errors",
            "Verify all required fields are filled",
            "Reset to default configuration if needed"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )
        self.config_item = config_item
        self.reason = reason


class CalibrationError(ApplicationException):
    """Raised when calibration process fails."""

    def __init__(
        self,
        stage: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[str] = None
    ):
        if stage and reason:
            message = f"Calibration failed during {stage}: {reason}"
        elif stage:
            message = f"Calibration failed during {stage}"
        elif reason:
            message = f"Calibration failed: {reason}"
        else:
            message = "Calibration process failed"

        resolution_steps = [
            "Ensure adequate lighting conditions",
            "Hold the calibration board steady during capture",
            "Cover the entire frame with board positions",
            "Check that the ChArUco board is clearly visible",
            "Try increasing the number of calibration images",
            "Verify camera focus is set correctly"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )
        self.stage = stage
        self.reason = reason


class DiscoveryServiceError(ApplicationException):
    """Raised when discovery service communication fails."""

    def __init__(
        self,
        service: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[str] = None
    ):
        if service and reason:
            message = f"Discovery service error ({service}): {reason}"
        elif service:
            message = f"Discovery service error: {service}"
        elif reason:
            message = f"Discovery service error: {reason}"
        else:
            message = "Discovery service communication failed"

        resolution_steps = [
            "Check network connectivity",
            "Verify the discovery service is running",
            "Check firewall settings",
            "Restart the discovery service",
            "Check service logs for more details"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )
        self.service = service
        self.reason = reason


class StreamingError(ApplicationException):
    """Raised when video streaming fails."""

    def __init__(
        self,
        camera_id: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[str] = None
    ):
        if camera_id and reason:
            message = f"Streaming error for camera '{camera_id}': {reason}"
        elif camera_id:
            message = f"Streaming error for camera '{camera_id}'"
        elif reason:
            message = f"Streaming error: {reason}"
        else:
            message = "Video streaming failed"

        resolution_steps = [
            "Check camera connection status",
            "Verify multicast network configuration",
            "Restart the camera stream",
            "Check available network bandwidth",
            "Verify GStreamer is properly installed"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )
        self.camera_id = camera_id
        self.reason = reason


class INSConnectionError(ApplicationException):
    """Raised when INS hardware connection fails."""

    def __init__(
        self,
        port: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[str] = None
    ):
        if port and reason:
            message = f"INS connection error on {port}: {reason}"
        elif port:
            message = f"INS connection error on {port}"
        elif reason:
            message = f"INS connection error: {reason}"
        else:
            message = "INS hardware connection failed"

        resolution_steps = [
            "Verify the serial cable is connected",
            "Check the serial port name (e.g., /dev/ttyUSB0)",
            "Ensure the INS device is powered on",
            "Verify no other application is using the port",
            "Check serial port permissions"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )
        self.port = port
        self.reason = reason


class ConfigurationLoadError(ApplicationException):
    """Raised when configuration loading or validation fails."""

    def __init__(
        self,
        source: Optional[str] = None,
        reason: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[str] = None
    ):
        if field and reason:
            message = f"Configuration error for '{field}': {reason}"
        elif source and reason:
            message = f"Failed to load configuration from {source}: {reason}"
        elif source:
            message = f"Failed to load configuration from {source}"
        elif reason:
            message = f"Configuration error: {reason}"
        else:
            message = "Configuration loading failed"

        resolution_steps = [
            "Check that the configuration file exists and is readable",
            "Verify the YAML syntax is correct",
            "Ensure all required fields are present",
            "Check that .env file format is valid (KEY=value)",
            "Review environment variable overrides",
            "Reset to default configuration if needed"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )
        self.source = source
        self.reason = reason
        self.field = field


class ConfigurationValidationError(ApplicationException):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        errors: Optional[List[str]] = None,
        details: Optional[str] = None
    ):
        self.errors = errors or []

        if len(self.errors) == 1:
            message = f"Configuration validation failed: {self.errors[0]}"
        elif len(self.errors) > 1:
            message = f"Configuration validation failed with {len(self.errors)} errors"
        else:
            message = "Configuration validation failed"

        resolution_steps = [
            "Review the validation errors listed below",
            "Check that all required fields have valid values",
            "Ensure IP addresses are in valid format",
            "Verify port numbers are within valid range (1-65535)",
            "Check that device hostnames match expected patterns"
        ]

        super().__init__(
            message=message,
            details=details,
            resolution_steps=resolution_steps
        )

    def get_user_message(self) -> str:
        """Get a formatted message with all validation errors."""
        parts = [self.message]
        if self.errors:
            parts.append("\nValidation errors:")
            for i, error in enumerate(self.errors, 1):
                parts.append(f"  {i}. {error}")
        if self.resolution_steps:
            parts.append("\nPossible solutions:")
            for i, step in enumerate(self.resolution_steps, 1):
                parts.append(f"  {i}. {step}")
        return "\n".join(parts)

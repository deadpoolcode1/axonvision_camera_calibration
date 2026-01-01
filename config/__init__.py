"""
Configuration Management Module

Provides centralized configuration with YAML file and environment variable support.

This module includes:
- Config: General application configuration (camera calibration tool)
- EdgeSAConfig: EdgeSA distributed surveillance system configuration

Usage:
    # For camera calibration tool config
    from config import config
    app_name = config.app_name

    # For EdgeSA configuration
    from config import edgesa_config
    discovery_url = edgesa_config.discovery.base_url
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """
    Centralized configuration manager.

    Loads settings from YAML file and allows environment variable overrides.
    Environment variables take precedence over YAML settings.
    """

    _instance: Optional['Config'] = None
    _config: dict = {}

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_dir = Path(__file__).parent
        config_file = config_dir / "settings.yaml"

        if config_file.exists():
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Supports dot notation for nested keys (e.g., 'app.version').
        Environment variables can override any setting using uppercase
        with underscores (e.g., APP_VERSION for 'app.version').

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        # Check environment variable first (highest priority)
        env_key = key.upper().replace('.', '_')
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)

        # Navigate nested config
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable string to appropriate type."""
        # Handle boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Handle numeric
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        return value

    @property
    def app_name(self) -> str:
        return self.get('app.name', 'AxonVision Camera Calibration Tool')

    @property
    def app_version(self) -> str:
        return self.get('app.version', '0.9.0')

    @property
    def log_level(self) -> str:
        return self.get('app.log_level', 'INFO')

    @property
    def max_cameras(self) -> int:
        return self.get('camera.max_cameras', 6)

    @property
    def max_ai_central_cameras(self) -> int:
        return self.get('camera.max_ai_central_cameras', 1)

    # Logging configuration properties
    @property
    def log_file_enabled(self) -> bool:
        return self.get('logging.file.enabled', True)

    @property
    def log_file_path(self) -> str:
        return self.get('logging.file.path', 'logs/calibration.log')

    @property
    def log_max_size_mb(self) -> int:
        """Maximum log file size in MB before rotation (default: 1MB)."""
        return self.get('logging.file.max_size_mb', 1)

    @property
    def log_backup_count(self) -> int:
        """Number of backup log files to keep."""
        return self.get('logging.file.backup_count', 5)

    @property
    def log_console_enabled(self) -> bool:
        return self.get('logging.console.enabled', True)

    @property
    def log_console_colored(self) -> bool:
        return self.get('logging.console.colored', True)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()


# Singleton instance
config = Config()

# EdgeSA configuration exports (lazy-loaded)
from .edgesa_config import EdgeSAConfig, edgesa_config, get_edgesa_config
from .exceptions import (
    ConfigurationLoadError,
    ConfigurationValidationError
)

__all__ = [
    'Config',
    'config',
    'EdgeSAConfig',
    'edgesa_config',
    'get_edgesa_config',
    'ConfigurationLoadError',
    'ConfigurationValidationError'
]

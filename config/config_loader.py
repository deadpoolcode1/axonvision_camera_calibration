"""
Configuration Loader

Loads configuration from YAML files with environment variable support.
Supports environment-specific overrides (development, production).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """
    Configuration manager that loads settings from YAML files.

    Supports:
    - Default configuration (config/default.yaml)
    - Environment-specific overrides (config/development.yaml, config/production.yaml)
    - Environment variable overrides (AXONVISION_* prefix)

    Usage:
        from config import get_config
        config = get_config()

        # Access nested values
        max_cameras = config.get('camera.max_cameras')
        log_level = config.get('logging.level')
    """

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from YAML files."""
        config_dir = Path(__file__).parent

        # Load default configuration
        default_path = config_dir / 'default.yaml'
        if default_path.exists():
            with open(default_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}

        # Determine environment
        env = os.environ.get('AXONVISION_ENV', 'development').lower()

        # Load environment-specific overrides
        env_path = config_dir / f'{env}.yaml'
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, env_config)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides (AXONVISION_* prefix)."""
        env_mappings = {
            'AXONVISION_LOG_LEVEL': 'logging.level',
            'AXONVISION_LOG_PATH': 'logging.file.path',
            'AXONVISION_MAX_CAMERAS': 'camera.max_cameras',
            'AXONVISION_CONNECTION_TIMEOUT': 'network.connection_timeout',
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config_path, value)

    def _set_nested(self, path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert value to appropriate type
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

        current[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            path: Dot-separated path to the config value (e.g., 'logging.level')
            default: Default value if path not found

        Returns:
            The configuration value or default
        """
        keys = path.split('.')
        current = self._config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config.copy()

    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = {}
        self._load_config()


# Singleton accessor
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

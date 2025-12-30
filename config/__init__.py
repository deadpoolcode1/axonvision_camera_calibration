"""
Configuration Module for AxonVision Camera Calibration

Provides centralized configuration management with YAML files and environment variable support.
"""

from .config_loader import Config, get_config

__all__ = ['Config', 'get_config']

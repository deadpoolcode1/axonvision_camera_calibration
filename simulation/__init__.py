"""
EdgeSA Device Simulation Module

Provides mock implementations of the EdgeSA device APIs for development and testing
when real hardware is not available. Video streams still come from real cameras.

Components:
- DeviceStateManager: Manages simulated device state (mode, config, etc.)
- MockDiscoveryService: Simulates the Service Discovery API
- MockDeviceAPI: Simulates device REST API endpoints
- MockSFTPServer: Simulates SFTP file operations

Usage:
    # Run the mock server
    python -m simulation.server

    # Or programmatically
    from simulation import start_mock_server
    start_mock_server()
"""

from .device_state import DeviceStateManager, SimulatedDevice
from .server import start_mock_server, is_mock_server_running

__all__ = [
    'DeviceStateManager',
    'SimulatedDevice',
    'start_mock_server',
    'is_mock_server_running',
]

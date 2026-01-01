"""
Device Configuration Service

Provides a unified interface for device configuration operations.
Routes requests to either real devices or the mock server based on simulation mode.

Usage:
    from services import DeviceConfigService
    from config import edgesa_config

    service = DeviceConfigService(edgesa_config)

    # Discovery
    devices = service.discover_devices()

    # Device configuration
    service.set_pipeline_mode("192.168.1.10", "worker")
    service.set_manager_connection("192.168.1.10", "192.168.1.30")
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

logger = logging.getLogger(__name__)


class DeviceConfigService:
    """
    Service for configuring EdgeSA devices.

    Automatically routes requests to mock server when simulation mode is enabled.
    """

    def __init__(self, config=None):
        """
        Initialize the device configuration service.

        Args:
            config: EdgeSAConfig instance (uses singleton if not provided)
        """
        if config is None:
            from config import edgesa_config
            config = edgesa_config

        self._config = config
        self._simulation_mode = config.simulation.enabled
        self._mock_server = config.simulation.mock_server if self._simulation_mode else None
        self._timeout = config.device_api.timeout

        if self._simulation_mode:
            logger.info("DeviceConfigService initialized in SIMULATION mode")
        else:
            logger.info("DeviceConfigService initialized in PRODUCTION mode")

    @property
    def is_simulation_mode(self) -> bool:
        """Check if running in simulation mode."""
        return self._simulation_mode

    def _get_discovery_url(self) -> str:
        """Get the base URL for discovery service."""
        if self._simulation_mode:
            return self._mock_server.discovery_url
        return self._config.discovery.base_url

    def _get_device_url(self, device_ip: str, endpoint: str) -> str:
        """Get the full URL for a device endpoint."""
        if self._simulation_mode:
            # Route through mock server with device IP header
            base_url = self._mock_server.device_api_url
            return f"{base_url}{endpoint}"
        else:
            # Direct to device
            port = self._config.device_api.port
            return f"http://{device_ip}:{port}{endpoint}"

    def _get_headers(self, device_ip: Optional[str] = None) -> Dict[str, str]:
        """Get request headers, including device IP for simulation mode."""
        headers = {"Content-Type": "application/json"}
        if self._simulation_mode and device_ip:
            headers["X-Device-IP"] = device_ip
        return headers

    # =========================================================================
    # Discovery API
    # =========================================================================

    def check_discovery_health(self) -> Tuple[bool, str]:
        """
        Check if the discovery service is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            url = f"{self._get_discovery_url()}/v1/health"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            status = data.get("status", "unknown")
            return status == "healthy", f"Discovery service status: {status}"
        except ConnectionError:
            return False, "Discovery service not reachable"
        except Timeout:
            return False, "Discovery service timeout"
        except Exception as e:
            return False, f"Discovery service error: {e}"

    def discover_devices(self, timeout_ms: int = 5000) -> List[Dict[str, Any]]:
        """
        Discover EdgeSA devices on the network.

        Args:
            timeout_ms: mDNS scan timeout in milliseconds

        Returns:
            List of discovered devices
        """
        try:
            url = f"{self._get_discovery_url()}/v1/discovery"
            params = {"timeout_ms": timeout_ms}
            response = requests.get(url, params=params, timeout=timeout_ms / 1000 + 5)
            response.raise_for_status()
            data = response.json()
            devices = data.get("devices", [])
            logger.info(f"Discovered {len(devices)} devices")
            return devices
        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            raise

    def provision_ssh_keys(
        self,
        targets: List[str],
        username: str = "nvidia",
        password: str = "nvidia"
    ) -> Dict[str, Any]:
        """
        Provision SSH keys on target devices.

        Args:
            targets: List of device IP addresses
            username: SSH username
            password: SSH password

        Returns:
            Provisioning results
        """
        try:
            url = f"{self._get_discovery_url()}/v1/ssh-keys"
            payload = {
                "targets": [{"ip": ip} for ip in targets],
                "credentials": {"username": username, "password": password}
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"SSH provisioning failed: {e}")
            raise

    # =========================================================================
    # Pipeline Mode API (threecluster only)
    # =========================================================================

    def get_pipeline_mode(self, device_ip: str) -> Dict[str, Any]:
        """
        Get the current pipeline mode of a device.

        Args:
            device_ip: Device IP address

        Returns:
            Mode information dict
        """
        try:
            endpoint = self._config.device_api.get_endpoint("pipeline_mode")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            response = requests.get(url, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get pipeline mode failed for {device_ip}: {e}")
            raise

    def set_pipeline_mode(self, device_ip: str, mode: str) -> Dict[str, Any]:
        """
        Set the pipeline mode of a device.

        Args:
            device_ip: Device IP address
            mode: "worker" or "manager"

        Returns:
            Result dict
        """
        try:
            endpoint = self._config.device_api.get_endpoint("pipeline_mode")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            payload = {"mode": mode}
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Set pipeline mode for {device_ip}: {mode}")
            return result
        except Exception as e:
            logger.error(f"Set pipeline mode failed for {device_ip}: {e}")
            raise

    # =========================================================================
    # Manager Connection API (worker mode)
    # =========================================================================

    def get_manager_connection(self, device_ip: str) -> Dict[str, Any]:
        """
        Get the manager connection settings for a worker device.

        Args:
            device_ip: Device IP address

        Returns:
            Manager connection info
        """
        try:
            endpoint = self._config.device_api.get_endpoint("manager_connection")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            response = requests.get(url, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get manager connection failed for {device_ip}: {e}")
            raise

    def set_manager_connection(self, device_ip: str, manager_ip: str) -> Dict[str, Any]:
        """
        Set the manager connection for a worker device.

        Args:
            device_ip: Device IP address
            manager_ip: Manager IP address

        Returns:
            Result dict
        """
        try:
            endpoint = self._config.device_api.get_endpoint("manager_connection")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            payload = {"manager_ip": manager_ip}
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Set manager connection for {device_ip}: {manager_ip}")
            return result
        except Exception as e:
            logger.error(f"Set manager connection failed for {device_ip}: {e}")
            raise

    # =========================================================================
    # Sensors Config API (manager mode)
    # =========================================================================

    def get_sensors_config(self, device_ip: str) -> Dict[str, Any]:
        """
        Get the sensors configuration from a manager device.

        Args:
            device_ip: Device IP address

        Returns:
            Sensors configuration
        """
        try:
            endpoint = self._config.device_api.get_endpoint("sensors_config")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            response = requests.get(url, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get sensors config failed for {device_ip}: {e}")
            raise

    def set_sensors_config(
        self,
        device_ip: str,
        sensors_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set the sensors configuration on a manager device.

        Args:
            device_ip: Device IP address
            sensors_config: Sensor configuration dict

        Returns:
            Result dict
        """
        try:
            endpoint = self._config.device_api.get_endpoint("sensors_config")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            response = requests.post(url, json=sensors_config, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Set sensors config for {device_ip}: {len(sensors_config)} sensors")
            return result
        except Exception as e:
            logger.error(f"Set sensors config failed for {device_ip}: {e}")
            raise

    def get_all_sensor_urls(self, device_ip: str) -> Dict[str, Any]:
        """
        Get all sensor URLs from a manager device.

        Args:
            device_ip: Device IP address

        Returns:
            Sensor URL map
        """
        try:
            endpoint = self._config.device_api.get_endpoint("all_sensor_urls")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            response = requests.get(url, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get sensor URLs failed for {device_ip}: {e}")
            raise

    # =========================================================================
    # Stream Configuration
    # =========================================================================

    def get_stream_id(self, device_ip: str, slot: int = 0) -> str:
        """
        Get the stream ID for a device slot.

        Args:
            device_ip: Device IP address
            slot: Camera slot (0-2)

        Returns:
            Stream ID string
        """
        try:
            endpoint = self._config.device_api.get_endpoint("streams", slot=slot)
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            response = requests.get(url, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("stream_id", "")
        except Exception as e:
            logger.error(f"Get stream ID failed for {device_ip} slot {slot}: {e}")
            raise

    # =========================================================================
    # Position Configuration
    # =========================================================================

    def set_position(self, device_ip: str, position: str) -> Dict[str, Any]:
        """
        Set the camera position for a worker device.

        Args:
            device_ip: Device IP address
            position: Position string (e.g., "Front Left")

        Returns:
            Result dict
        """
        try:
            endpoint = self._config.device_api.get_endpoint("position")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            payload = {"position": position}
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Set position for {device_ip}: {position}")
            return result
        except Exception as e:
            logger.error(f"Set position failed for {device_ip}: {e}")
            raise

    # =========================================================================
    # Health Check
    # =========================================================================

    def check_device_health(self, device_ip: str) -> Tuple[bool, str]:
        """
        Check if a device is healthy.

        Args:
            device_ip: Device IP address

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            endpoint = self._config.device_api.get_endpoint("health")
            url = self._get_device_url(device_ip, endpoint)
            headers = self._get_headers(device_ip)
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            status = data.get("status", "unknown")
            return status == "healthy", f"Device {device_ip}: {status}"
        except ConnectionError:
            return False, f"Device {device_ip} not reachable"
        except Timeout:
            return False, f"Device {device_ip} timeout"
        except Exception as e:
            return False, f"Device {device_ip} error: {e}"

    # =========================================================================
    # File Operations (SFTP simulation via HTTP or direct)
    # =========================================================================

    def write_device_file(
        self,
        device_ip: str,
        remote_path: str,
        content: str
    ) -> bool:
        """
        Write a file to a device.

        In simulation mode, uses HTTP endpoint.
        In production mode, would use SFTP (not implemented here).

        Args:
            device_ip: Device IP address
            remote_path: Remote file path
            content: File content

        Returns:
            True if successful
        """
        if self._simulation_mode:
            try:
                url = f"{self._mock_server.device_api_url}/sftp/write"
                headers = self._get_headers(device_ip)
                payload = {"path": remote_path, "content": content}
                response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
                response.raise_for_status()
                logger.info(f"Wrote file to {device_ip}:{remote_path}")
                return True
            except Exception as e:
                logger.error(f"Write file failed for {device_ip}:{remote_path}: {e}")
                return False
        else:
            # Production mode would use paramiko SFTP
            logger.warning("SFTP not implemented for production mode")
            return False

    def read_device_file(
        self,
        device_ip: str,
        remote_path: str
    ) -> Optional[str]:
        """
        Read a file from a device.

        In simulation mode, uses HTTP endpoint.
        In production mode, would use SFTP (not implemented here).

        Args:
            device_ip: Device IP address
            remote_path: Remote file path

        Returns:
            File content or None if failed
        """
        if self._simulation_mode:
            try:
                url = f"{self._mock_server.device_api_url}/sftp/read"
                headers = self._get_headers(device_ip)
                params = {"path": remote_path}
                response = requests.get(url, params=params, headers=headers, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()
                return data.get("content")
            except Exception as e:
                logger.error(f"Read file failed for {device_ip}:{remote_path}: {e}")
                return None
        else:
            # Production mode would use paramiko SFTP
            logger.warning("SFTP not implemented for production mode")
            return None

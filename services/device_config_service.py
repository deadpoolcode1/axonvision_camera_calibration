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

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

logger = logging.getLogger(__name__)

# Enable debug logging for API calls
DEBUG_API_CALLS = True


def _log_request(method: str, url: str, headers: Dict, body: Any = None, device_ip: str = None):
    """Log outgoing API request with full details (DEBUG level)."""
    if not DEBUG_API_CALLS:
        return

    log_msg = f"\n{'='*70}\n"
    log_msg += f"📤 OUTGOING REQUEST\n"
    log_msg += f"{'='*70}\n"
    log_msg += f"  Method:     {method}\n"
    log_msg += f"  URL:        {url}\n"
    if device_ip:
        log_msg += f"  Target IP:  {device_ip}\n"
    log_msg += f"  Headers:    {headers}\n"
    if body:
        try:
            if isinstance(body, dict):
                log_msg += f"  Body:\n{json.dumps(body, indent=4, default=str)}\n"
            else:
                log_msg += f"  Body: {body}\n"
        except Exception:
            log_msg += f"  Body: {body}\n"
    log_msg += f"{'='*70}"
    logger.debug(log_msg)


def _log_response(response: requests.Response, duration_ms: float, device_ip: str = None):
    """Log incoming API response with full details (DEBUG level)."""
    if not DEBUG_API_CALLS:
        return

    log_msg = f"\n{'='*70}\n"
    log_msg += f"📥 INCOMING RESPONSE\n"
    log_msg += f"{'='*70}\n"
    if device_ip:
        log_msg += f"  Target IP:  {device_ip}\n"
    log_msg += f"  Status:     {response.status_code} {response.reason}\n"
    log_msg += f"  Duration:   {duration_ms:.2f}ms\n"

    try:
        resp_json = response.json()
        log_msg += f"  Response Body:\n{json.dumps(resp_json, indent=4, default=str)}\n"
    except Exception:
        # Not JSON, show text
        text = response.text[:500] if response.text else "(empty)"
        log_msg += f"  Response Body: {text}\n"

    log_msg += f"{'='*70}"
    logger.debug(log_msg)


def _log_error(method: str, url: str, error: Exception, device_ip: str = None):
    """Log API request error (DEBUG level for details, ERROR for the error itself)."""
    if not DEBUG_API_CALLS:
        return

    log_msg = f"\n{'='*70}\n"
    log_msg += f"❌ REQUEST FAILED\n"
    log_msg += f"{'='*70}\n"
    log_msg += f"  Method:     {method}\n"
    log_msg += f"  URL:        {url}\n"
    if device_ip:
        log_msg += f"  Target IP:  {device_ip}\n"
    log_msg += f"  Error:      {type(error).__name__}: {error}\n"
    log_msg += f"{'='*70}"
    logger.debug(log_msg)


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
        url = f"{self._get_discovery_url()}/v1/health"
        headers = {"Content-Type": "application/json"}

        try:
            _log_request("GET", url, headers)
            start_time = time.time()
            response = requests.get(url, timeout=5)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms)

            response.raise_for_status()
            data = response.json()
            status = data.get("status", "unknown")
            return status == "healthy", f"Discovery service status: {status}"
        except ConnectionError as e:
            _log_error("GET", url, e)
            return False, "Discovery service not reachable"
        except Timeout as e:
            _log_error("GET", url, e)
            return False, "Discovery service timeout"
        except Exception as e:
            _log_error("GET", url, e)
            return False, f"Discovery service error: {e}"

    def discover_devices(self, timeout_ms: int = 5000) -> List[Dict[str, Any]]:
        """
        Discover EdgeSA devices on the network.

        Args:
            timeout_ms: mDNS scan timeout in milliseconds

        Returns:
            List of discovered devices
        """
        url = f"{self._get_discovery_url()}/v1/discovery"
        params = {"timeout_ms": timeout_ms}
        headers = {"Content-Type": "application/json"}

        try:
            _log_request("GET", f"{url}?timeout_ms={timeout_ms}", headers)
            start_time = time.time()
            response = requests.get(url, params=params, timeout=timeout_ms / 1000 + 5)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms)

            response.raise_for_status()
            data = response.json()
            devices = data.get("devices", [])
            logger.info(f"Discovered {len(devices)} devices")
            return devices
        except Exception as e:
            _log_error("GET", url, e)
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
        url = f"{self._get_discovery_url()}/v1/ssh-keys"
        headers = {"Content-Type": "application/json"}
        payload = {
            "targets": [{"ip": ip} for ip in targets],
            "credentials": {"username": username, "password": password}
        }

        try:
            _log_request("POST", url, headers, payload)
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            _log_error("POST", url, e)
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
        endpoint = self._config.device_api.get_endpoint("pipeline_mode")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)

        try:
            _log_request("GET", url, headers, device_ip=device_ip)
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            _log_error("GET", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("pipeline_mode")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)
        payload = {"mode": mode}

        try:
            _log_request("POST", url, headers, payload, device_ip=device_ip)
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            result = response.json()
            logger.info(f"Set pipeline mode for {device_ip}: {mode}")
            return result
        except Exception as e:
            _log_error("POST", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("manager_connection")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)

        try:
            _log_request("GET", url, headers, device_ip=device_ip)
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            _log_error("GET", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("manager_connection")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)
        payload = {"manager_ip": manager_ip}

        try:
            _log_request("POST", url, headers, payload, device_ip=device_ip)
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            result = response.json()
            logger.info(f"Set manager connection for {device_ip}: {manager_ip}")
            return result
        except Exception as e:
            _log_error("POST", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("sensors_config")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)

        try:
            _log_request("GET", url, headers, device_ip=device_ip)
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            _log_error("GET", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("sensors_config")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)

        try:
            _log_request("POST", url, headers, sensors_config, device_ip=device_ip)
            start_time = time.time()
            response = requests.post(url, json=sensors_config, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            result = response.json()
            logger.info(f"Set sensors config for {device_ip}: {len(sensors_config)} sensors")
            return result
        except Exception as e:
            _log_error("POST", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("all_sensor_urls")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)

        try:
            _log_request("GET", url, headers, device_ip=device_ip)
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            _log_error("GET", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("streams", slot=slot)
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)

        try:
            _log_request("GET", url, headers, device_ip=device_ip)
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            data = response.json()
            return data.get("stream_id", "")
        except Exception as e:
            _log_error("GET", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("position")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)
        payload = {"position": position}

        try:
            _log_request("POST", url, headers, payload, device_ip=device_ip)
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            result = response.json()
            logger.info(f"Set position for {device_ip}: {position}")
            return result
        except Exception as e:
            _log_error("POST", url, e, device_ip=device_ip)
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
        endpoint = self._config.device_api.get_endpoint("health")
        url = self._get_device_url(device_ip, endpoint)
        headers = self._get_headers(device_ip)

        try:
            _log_request("GET", url, headers, device_ip=device_ip)
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=5)
            duration_ms = (time.time() - start_time) * 1000
            _log_response(response, duration_ms, device_ip=device_ip)

            response.raise_for_status()
            data = response.json()
            status = data.get("status", "unknown")
            return status == "healthy", f"Device {device_ip}: {status}"
        except ConnectionError as e:
            _log_error("GET", url, e, device_ip=device_ip)
            return False, f"Device {device_ip} not reachable"
        except Timeout as e:
            _log_error("GET", url, e, device_ip=device_ip)
            return False, f"Device {device_ip} timeout"
        except Exception as e:
            _log_error("GET", url, e, device_ip=device_ip)
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
            url = f"{self._mock_server.device_api_url}/sftp/write"
            headers = self._get_headers(device_ip)
            payload = {"path": remote_path, "content": content}

            try:
                _log_request("POST", url, headers, {"path": remote_path, "content": f"<{len(content)} bytes>"}, device_ip=device_ip)
                start_time = time.time()
                response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
                duration_ms = (time.time() - start_time) * 1000
                _log_response(response, duration_ms, device_ip=device_ip)

                response.raise_for_status()
                logger.info(f"Wrote file to {device_ip}:{remote_path}")
                return True
            except Exception as e:
                _log_error("POST", url, e, device_ip=device_ip)
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
            url = f"{self._mock_server.device_api_url}/sftp/read"
            headers = self._get_headers(device_ip)
            params = {"path": remote_path}

            try:
                _log_request("GET", f"{url}?path={remote_path}", headers, device_ip=device_ip)
                start_time = time.time()
                response = requests.get(url, params=params, headers=headers, timeout=self._timeout)
                duration_ms = (time.time() - start_time) * 1000
                _log_response(response, duration_ms, device_ip=device_ip)

                response.raise_for_status()
                data = response.json()
                return data.get("content")
            except Exception as e:
                _log_error("GET", url, e, device_ip=device_ip)
                logger.error(f"Read file failed for {device_ip}:{remote_path}: {e}")
                return None
        else:
            # Production mode would use paramiko SFTP
            logger.warning("SFTP not implemented for production mode")
            return None

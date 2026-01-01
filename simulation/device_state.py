"""
Device State Manager

Manages the simulated state for all mock devices including:
- Device mode (worker/manager)
- Manager connection settings
- Sensor configuration
- Position settings
- SSH provisioning status
- Virtual filesystem for SFTP
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Stream configuration for a device slot."""
    slot: int
    stream_id: str
    mjpeg_vis_port: int
    mjpeg_thermal_port: int
    mpegts_port: int


@dataclass
class SensorEntry:
    """A single sensor entry in the sensors_config."""
    sensor_name: str
    endpoint: str
    port: int
    sensor_type: str  # "vis", "thermal", "day", "night"


@dataclass
class SimulatedDevice:
    """
    Represents a simulated EdgeSA device with all its state.

    Attributes:
        ip: Device IP address (real camera IP for video)
        hostname: Device hostname (e.g., "smartcluster-001")
        device_type: Device type ("smartcluster", "threecluster", "aicentral")
        mode: Current mode ("worker" or "manager")
        manager_ip: IP of the manager (for workers)
        position: Camera position string
        sensors_config: Sensor configuration (for managers)
        ssh_configured: Whether SSH key is installed
        streams: Stream configurations per slot
    """
    ip: str
    hostname: str
    device_type: str
    mode: str = "worker"
    manager_ip: Optional[str] = None
    position: Optional[str] = None
    sensors_config: Dict[str, SensorEntry] = field(default_factory=dict)
    ssh_configured: bool = False
    streams: List[StreamConfig] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        """Initialize streams based on device type."""
        if not self.streams:
            self._init_streams()

        # Set default mode based on device type
        if self.device_type == "aicentral":
            self.mode = "manager"
        elif self.device_type == "smartcluster":
            self.mode = "worker"
        # threecluster can be either, default to worker

    def _init_streams(self):
        """Initialize stream configurations based on device type."""
        if self.device_type == "smartcluster":
            self.streams = [
                StreamConfig(
                    slot=0,
                    stream_id=str(uuid.uuid4()),
                    mjpeg_vis_port=8080,
                    mjpeg_thermal_port=9080,
                    mpegts_port=5005
                )
            ]
        elif self.device_type == "threecluster":
            self.streams = [
                StreamConfig(
                    slot=0,
                    stream_id=str(uuid.uuid4()),
                    mjpeg_vis_port=8079,
                    mjpeg_thermal_port=9079,
                    mpegts_port=5004
                ),
                StreamConfig(
                    slot=1,
                    stream_id=str(uuid.uuid4()),
                    mjpeg_vis_port=8080,
                    mjpeg_thermal_port=9080,
                    mpegts_port=5005
                ),
                StreamConfig(
                    slot=2,
                    stream_id=str(uuid.uuid4()),
                    mjpeg_vis_port=8081,
                    mjpeg_thermal_port=9081,
                    mpegts_port=5006
                ),
            ]
        # aicentral has no local streams

    def can_switch_mode(self) -> bool:
        """Check if this device can switch between worker/manager modes."""
        return self.device_type == "threecluster"

    def is_manager_eligible(self) -> bool:
        """Check if this device can be a manager."""
        return self.device_type in ("threecluster", "aicentral")

    def get_available_modes(self) -> List[str]:
        """Get list of modes this device supports."""
        if self.device_type == "aicentral":
            return ["manager"]
        elif self.device_type == "smartcluster":
            return ["worker"]
        else:  # threecluster
            return ["worker", "manager"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "ip": self.ip,
            "hostname": self.hostname,
            "device_type": self.device_type,
            "mode": self.mode,
            "manager_ip": self.manager_ip,
            "position": self.position,
            "ssh_configured": self.ssh_configured,
            "last_updated": self.last_updated,
            "streams": [
                {
                    "slot": s.slot,
                    "stream_id": s.stream_id,
                    "mjpeg_vis_port": s.mjpeg_vis_port,
                    "mjpeg_thermal_port": s.mjpeg_thermal_port,
                    "mpegts_port": s.mpegts_port
                }
                for s in self.streams
            ],
            "sensors_config": {
                k: {
                    "sensor_name": v.sensor_name,
                    "endpoint": v.endpoint,
                    "port": v.port,
                    "type": v.sensor_type
                }
                for k, v in self.sensors_config.items()
            } if self.sensors_config else {}
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulatedDevice':
        """Create from dictionary."""
        streams = [
            StreamConfig(
                slot=s["slot"],
                stream_id=s["stream_id"],
                mjpeg_vis_port=s["mjpeg_vis_port"],
                mjpeg_thermal_port=s["mjpeg_thermal_port"],
                mpegts_port=s["mpegts_port"]
            )
            for s in data.get("streams", [])
        ]

        sensors_config = {}
        for k, v in data.get("sensors_config", {}).items():
            sensors_config[k] = SensorEntry(
                sensor_name=v["sensor_name"],
                endpoint=v["endpoint"],
                port=v["port"],
                sensor_type=v.get("type", "vis")
            )

        return cls(
            ip=data["ip"],
            hostname=data["hostname"],
            device_type=data["device_type"],
            mode=data.get("mode", "worker"),
            manager_ip=data.get("manager_ip"),
            position=data.get("position"),
            sensors_config=sensors_config,
            ssh_configured=data.get("ssh_configured", False),
            streams=streams,
            last_updated=data.get("last_updated", datetime.utcnow().isoformat() + "Z")
        )


class DeviceStateManager:
    """
    Manages state for all simulated devices.

    Provides thread-safe access to device state and optional
    persistence to disk.
    """

    def __init__(
        self,
        state_file: Optional[str] = None,
        filesystem_path: Optional[str] = None,
        initial_devices: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the device state manager.

        Args:
            state_file: Path to JSON file for persisting state
            filesystem_path: Path for virtual filesystem storage
            initial_devices: List of device configurations to initialize
        """
        self._devices: Dict[str, SimulatedDevice] = {}
        self._lock = Lock()
        self._state_file = state_file
        self._filesystem_path = filesystem_path

        # Create filesystem directory if specified
        if filesystem_path:
            Path(filesystem_path).mkdir(parents=True, exist_ok=True)

        # Load existing state or initialize from config
        if state_file and Path(state_file).exists():
            self._load_state()
        elif initial_devices:
            self._init_devices(initial_devices)

    def _init_devices(self, devices_config: List[Dict[str, Any]]) -> None:
        """Initialize devices from configuration."""
        for dev_config in devices_config:
            device = SimulatedDevice(
                ip=dev_config["ip"],
                hostname=dev_config["hostname"],
                device_type=dev_config["type"],
                mode=dev_config.get("initial_mode", "worker")
            )
            self._devices[dev_config["ip"]] = device
            logger.info(f"Initialized simulated device: {device.hostname} ({device.ip})")

        self._save_state()

    def _load_state(self) -> None:
        """Load device state from file."""
        try:
            with open(self._state_file, 'r') as f:
                data = json.load(f)

            for ip, device_data in data.get("devices", {}).items():
                self._devices[ip] = SimulatedDevice.from_dict(device_data)
                logger.debug(f"Loaded device state: {ip}")

            logger.info(f"Loaded state for {len(self._devices)} devices from {self._state_file}")
        except Exception as e:
            logger.error(f"Failed to load state from {self._state_file}: {e}")

    def _save_state(self) -> None:
        """Save device state to file."""
        if not self._state_file:
            return

        try:
            # Ensure directory exists
            Path(self._state_file).parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "devices": {
                    ip: device.to_dict()
                    for ip, device in self._devices.items()
                }
            }

            with open(self._state_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved state for {len(self._devices)} devices")
        except Exception as e:
            logger.error(f"Failed to save state to {self._state_file}: {e}")

    def get_device(self, ip: str) -> Optional[SimulatedDevice]:
        """Get a device by IP address."""
        with self._lock:
            return self._devices.get(ip)

    def get_all_devices(self) -> List[SimulatedDevice]:
        """Get all devices."""
        with self._lock:
            return list(self._devices.values())

    def add_device(self, device: SimulatedDevice) -> None:
        """Add or update a device."""
        with self._lock:
            device.last_updated = datetime.utcnow().isoformat() + "Z"
            self._devices[device.ip] = device
            self._save_state()

    def update_device(self, ip: str, **updates) -> Optional[SimulatedDevice]:
        """Update device properties."""
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return None

            for key, value in updates.items():
                if hasattr(device, key):
                    setattr(device, key, value)

            device.last_updated = datetime.utcnow().isoformat() + "Z"
            self._save_state()
            return device

    def set_mode(self, ip: str, mode: str) -> Dict[str, Any]:
        """
        Set device mode (worker/manager).

        Returns status dict with result.
        """
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            if not device.can_switch_mode():
                if device.mode == mode:
                    return {
                        "status": "already_in_mode",
                        "mode": mode,
                        "available_modes": device.get_available_modes()
                    }
                return {
                    "status": "error",
                    "detail": f"Device {device.hostname} cannot switch modes"
                }

            if mode not in ("worker", "manager"):
                return {"status": "error", "detail": f"Invalid mode: {mode}"}

            previous_mode = device.mode
            if previous_mode == mode:
                return {
                    "status": "already_in_mode",
                    "mode": mode,
                    "available_modes": device.get_available_modes()
                }

            device.mode = mode
            device.last_updated = datetime.utcnow().isoformat() + "Z"
            self._save_state()

            logger.info(f"Device {device.hostname} mode changed: {previous_mode} -> {mode}")
            return {
                "status": "success",
                "mode": mode,
                "previous_mode": previous_mode,
                "available_modes": device.get_available_modes()
            }

    def set_manager_connection(self, ip: str, manager_ip: str) -> Dict[str, Any]:
        """Set the manager IP for a worker device."""
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            if device.mode != "worker":
                return {
                    "status": "error",
                    "detail": f"Endpoint only available in worker mode. Current mode: {device.mode}"
                }

            device.manager_ip = manager_ip
            device.last_updated = datetime.utcnow().isoformat() + "Z"
            self._save_state()

            # Return list of "updated components"
            components = ["stream_component_central_1", "output_component_message"]
            logger.info(f"Device {device.hostname} manager_ip set to {manager_ip}")
            return {
                "status": "success",
                "manager_ip": manager_ip,
                "updated_components": components
            }

    def get_manager_connection(self, ip: str) -> Dict[str, Any]:
        """Get manager connection info for a worker device."""
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            if device.mode != "worker":
                return {
                    "status": "error",
                    "detail": f"Endpoint only available in worker mode. Current mode: {device.mode}"
                }

            return {
                "status": "success",
                "manager_ip": device.manager_ip,
                "components": [
                    {"name": "stream_component_central_1", "direction": "input",
                     "description": "Receives data from manager"},
                    {"name": "output_component_message", "direction": "output",
                     "description": "Sends results to manager"}
                ]
            }

    def set_sensors_config(self, ip: str, sensors_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set sensors configuration for a manager device."""
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            if device.mode != "manager":
                return {
                    "status": "error",
                    "detail": f"Endpoint only available in manager mode. Current mode: {device.mode}"
                }

            # Convert to SensorEntry objects
            device.sensors_config = {}
            for sensor_id, sensor_data in sensors_config.items():
                device.sensors_config[sensor_id] = SensorEntry(
                    sensor_name=sensor_data["sensor_name"],
                    endpoint=sensor_data["endpoint"],
                    port=sensor_data["port"],
                    sensor_type=sensor_data.get("type", "vis")
                )

            device.last_updated = datetime.utcnow().isoformat() + "Z"
            self._save_state()

            # Calculate worker URLs from sensor endpoints
            worker_ips = set()
            for entry in device.sensors_config.values():
                worker_ips.add(entry.endpoint)
            worker_urls = [f"http://{ip}:5000/" for ip in sorted(worker_ips)]

            logger.info(f"Device {device.hostname} sensors_config updated with {len(sensors_config)} sensors")
            return {
                "status": "success",
                "sensors_config": sensors_config,
                "worker_urls": worker_urls
            }

    def get_sensors_config(self, ip: str) -> Dict[str, Any]:
        """Get sensors configuration for a manager device."""
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            if device.mode != "manager":
                return {
                    "status": "error",
                    "detail": f"Endpoint only available in manager mode. Current mode: {device.mode}"
                }

            sensors_config = {}
            worker_ips = set()
            for sensor_id, entry in device.sensors_config.items():
                sensors_config[sensor_id] = {
                    "sensor_name": entry.sensor_name,
                    "endpoint": entry.endpoint,
                    "port": entry.port,
                    "type": entry.sensor_type
                }
                worker_ips.add(entry.endpoint)

            worker_urls = [f"http://{ip}:5000/" for ip in sorted(worker_ips)]

            return {
                "status": "success",
                "sensors_config": sensors_config,
                "worker_urls": worker_urls
            }

    def get_stream_id(self, ip: str, slot: int) -> Dict[str, Any]:
        """Get stream ID for a device slot."""
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            for stream in device.streams:
                if stream.slot == slot:
                    return {
                        "status": "success",
                        "slot": slot,
                        "stream_id": stream.stream_id
                    }

            return {"status": "error", "detail": f"Slot {slot} not found on device {ip}"}

    def set_ssh_configured(self, ip: str, configured: bool = True) -> Dict[str, Any]:
        """Mark device as SSH configured."""
        with self._lock:
            device = self._devices.get(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            device.ssh_configured = configured
            device.last_updated = datetime.utcnow().isoformat() + "Z"
            self._save_state()

            return {"status": "success", "ssh_configured": configured}

    # =========================================================================
    # Virtual Filesystem Operations (for SFTP simulation)
    # =========================================================================

    def _get_device_fs_path(self, ip: str) -> Path:
        """Get the filesystem path for a device."""
        if not self._filesystem_path:
            raise ValueError("Filesystem path not configured")
        return Path(self._filesystem_path) / ip

    def write_file(self, ip: str, remote_path: str, content: str) -> Dict[str, Any]:
        """Write a file to the virtual filesystem."""
        try:
            device = self.get_device(ip)
            if not device:
                return {"status": "error", "detail": f"Device {ip} not found"}

            # Create backup first
            self._backup_file(ip, remote_path)

            # Write the file
            fs_path = self._get_device_fs_path(ip)
            file_path = fs_path / remote_path.lstrip("/")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                f.write(content)

            logger.info(f"SFTP write to {ip}:{remote_path}")
            return {"status": "success", "path": remote_path}

        except Exception as e:
            logger.error(f"SFTP write failed for {ip}:{remote_path}: {e}")
            return {"status": "error", "detail": str(e)}

    def read_file(self, ip: str, remote_path: str) -> Dict[str, Any]:
        """Read a file from the virtual filesystem."""
        try:
            fs_path = self._get_device_fs_path(ip)
            file_path = fs_path / remote_path.lstrip("/")

            if not file_path.exists():
                return {"status": "error", "detail": f"File not found: {remote_path}"}

            with open(file_path, 'r') as f:
                content = f.read()

            return {"status": "success", "path": remote_path, "content": content}

        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def _backup_file(self, ip: str, remote_path: str) -> None:
        """Create a backup of an existing file before overwriting."""
        fs_path = self._get_device_fs_path(ip)
        file_path = fs_path / remote_path.lstrip("/")

        if not file_path.exists():
            return

        # Create backup in snapshots directory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_dir = fs_path / "etc" / "axon" / "snapshots" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_path = backup_dir / file_path.name
        with open(file_path, 'r') as src:
            with open(backup_path, 'w') as dst:
                dst.write(src.read())

        logger.debug(f"Created backup: {backup_path}")

    def list_files(self, ip: str, remote_path: str) -> Dict[str, Any]:
        """List files in a directory on the virtual filesystem."""
        try:
            fs_path = self._get_device_fs_path(ip)
            dir_path = fs_path / remote_path.lstrip("/")

            if not dir_path.exists():
                return {"status": "success", "path": remote_path, "files": []}

            files = []
            for item in dir_path.iterdir():
                files.append({
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0
                })

            return {"status": "success", "path": remote_path, "files": files}

        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def reset_all(self, initial_devices: Optional[List[Dict[str, Any]]] = None) -> None:
        """Reset all device state to initial configuration."""
        with self._lock:
            self._devices.clear()

            # Clear filesystem
            if self._filesystem_path and Path(self._filesystem_path).exists():
                import shutil
                shutil.rmtree(self._filesystem_path)
                Path(self._filesystem_path).mkdir(parents=True, exist_ok=True)

            # Re-initialize from config
            if initial_devices:
                self._init_devices(initial_devices)

            logger.info("Device state manager reset")

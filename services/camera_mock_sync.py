"""
Camera-Mock Device Synchronization Service

Links user-added cameras to simulated mock devices so that:
- When a camera is added, a mock device is created with the same IP
- When a camera is removed, the mock device is deleted
- Camera configuration changes sync to the mock device

This ensures simulation mode accurately reflects the user's actual camera setup.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Type mappings from UI camera types to simulation device types
CAMERA_TYPE_TO_DEVICE_TYPE = {
    "1:1": "smartcluster",
    "3:1 (worker)": "threecluster",
    "3:1 (manager)": "threecluster",
    "AI CENTRAL": "aicentral",
}

# Mode mappings from UI camera types
CAMERA_TYPE_TO_MODE = {
    "1:1": "worker",
    "3:1 (worker)": "worker",
    "3:1 (manager)": "manager",
    "AI CENTRAL": "manager",
}


class CameraMockSyncService:
    """
    Synchronizes camera configurations with mock devices.

    When simulation mode is enabled, this service ensures that mock devices
    are created/updated/deleted to match the cameras configured in the UI.
    """

    def __init__(self, state_manager=None, settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the sync service.

        Args:
            state_manager: DeviceStateManager instance (optional, will load if needed)
            settings: Application settings dict
        """
        self._state_manager = state_manager
        self._settings = settings or {}
        self._initialized = False

    def _ensure_state_manager(self):
        """Lazily initialize state manager if not provided."""
        if self._state_manager is not None:
            self._initialized = True
            return True

        try:
            # Try to get the shared state manager from the mock server
            from simulation.device_state import DeviceStateManager

            # Load persistence settings
            persistence_config = self._settings.get("simulation", {}).get("persistence", {})
            state_file = persistence_config.get("state_file", "./mock_data/device_states.json")
            fs_path = persistence_config.get("filesystem_path", "./mock_data/devices")

            self._state_manager = DeviceStateManager(
                state_file=state_file,
                filesystem_path=fs_path,
                initial_devices=[]  # Don't load hardcoded devices
            )
            self._initialized = True
            logger.info("CameraMockSyncService: Initialized state manager")
            return True
        except Exception as e:
            logger.warning(f"CameraMockSyncService: Could not initialize state manager: {e}")
            return False

    def is_simulation_enabled(self) -> bool:
        """Check if simulation mode is enabled in settings."""
        return self._settings.get("simulation", {}).get("enabled", False)

    def sync_camera_added(self, camera_data: Dict[str, Any]) -> bool:
        """
        Sync when a camera is added.

        Creates a corresponding mock device with the camera's configuration.

        Args:
            camera_data: Camera definition dict with keys:
                - ip_address: Camera IP
                - camera_type: "1:1", "3:1 (worker)", "3:1 (manager)", or "AI CENTRAL"
                - camera_id: Unique camera identifier
                - mounting_position: Position string

        Returns:
            True if mock device was created, False otherwise
        """
        if not self.is_simulation_enabled():
            logger.debug("Simulation mode disabled, skipping mock device creation")
            return False

        if not self._ensure_state_manager():
            logger.warning("Cannot sync camera: state manager not available")
            return False

        ip = camera_data.get("ip_address", "").strip()
        if not ip:
            logger.warning("Cannot create mock device: no IP address")
            return False

        camera_type = camera_data.get("camera_type", "1:1")
        camera_id = camera_data.get("camera_id", "")
        position = camera_data.get("mounting_position")

        logger.debug(f"sync_camera_added: ip={ip}, type={camera_type}, id={camera_id}, position={position}")

        # Map camera type to device type
        device_type = CAMERA_TYPE_TO_DEVICE_TYPE.get(camera_type, "smartcluster")
        mode = CAMERA_TYPE_TO_MODE.get(camera_type, "worker")

        # Generate hostname from camera ID or auto-generate
        if camera_id:
            # Use camera_id as base for hostname
            hostname = f"{device_type}-{camera_id.replace('cam_', '').zfill(3)}"
        else:
            # Auto-generate based on IP
            hostname = f"{device_type}-{ip.replace('.', '-')}"

        try:
            from simulation.device_state import SimulatedDevice

            # Check if device already exists
            existing = self._state_manager.get_device(ip)
            if existing:
                # Update existing device
                self._state_manager.update_device(
                    ip,
                    device_type=device_type,
                    mode=mode,
                    position=position,
                    hostname=hostname
                )
                logger.info(f"Updated mock device for camera: {ip} ({device_type})")
            else:
                # Create new device
                device = SimulatedDevice(
                    ip=ip,
                    hostname=hostname,
                    device_type=device_type,
                    mode=mode,
                    position=position
                )
                self._state_manager.add_device(device)
                logger.info(f"Created mock device for camera: {ip} ({device_type})")

            return True

        except Exception as e:
            logger.error(f"Failed to create mock device for {ip}: {e}")
            return False

    def sync_camera_removed(self, ip_address: str) -> bool:
        """
        Sync when a camera is removed.

        Deletes the corresponding mock device.

        Args:
            ip_address: IP address of the camera being removed

        Returns:
            True if mock device was removed, False otherwise
        """
        if not self.is_simulation_enabled():
            return False

        if not self._ensure_state_manager():
            logger.warning("Cannot sync camera removal: state manager not available")
            return False

        ip = ip_address.strip()
        if not ip:
            return False

        try:
            result = self._state_manager.remove_device(ip)
            if result:
                logger.info(f"Removed mock device for camera: {ip}")
            return result
        except Exception as e:
            logger.error(f"Failed to remove mock device for {ip}: {e}")
            return False

    def sync_camera_updated(self, camera_data: Dict[str, Any], old_ip: Optional[str] = None) -> bool:
        """
        Sync when a camera is updated.

        Updates the mock device configuration to match the camera.
        If IP changed, removes old device and creates new one.

        Args:
            camera_data: Updated camera definition
            old_ip: Previous IP address if it changed

        Returns:
            True if sync was successful
        """
        if not self.is_simulation_enabled():
            return False

        new_ip = camera_data.get("ip_address", "").strip()

        # If IP changed, remove old and create new
        if old_ip and old_ip != new_ip:
            self.sync_camera_removed(old_ip)

        return self.sync_camera_added(camera_data)

    def sync_all_cameras(self, cameras: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sync all cameras at once.

        This is useful for initial load or full refresh. It will:
        1. Create mock devices for all cameras
        2. Remove orphan mock devices that don't have corresponding cameras

        Args:
            cameras: List of camera definition dicts

        Returns:
            Dict with sync results:
                - created: number of devices created
                - updated: number of devices updated
                - removed: number of orphan devices removed
                - errors: list of error messages
        """
        if not self.is_simulation_enabled():
            return {"created": 0, "updated": 0, "removed": 0, "errors": []}

        if not self._ensure_state_manager():
            return {"created": 0, "updated": 0, "removed": 0, "errors": ["State manager not available"]}

        results = {"created": 0, "updated": 0, "removed": 0, "errors": []}

        # Get current camera IPs
        camera_ips = {cam.get("ip_address", "").strip() for cam in cameras if cam.get("ip_address")}

        # Get current mock device IPs
        existing_devices = self._state_manager.get_all_devices()
        existing_ips = {d.ip for d in existing_devices}

        # Create/update devices for all cameras
        for camera in cameras:
            ip = camera.get("ip_address", "").strip()
            if not ip:
                continue

            try:
                was_existing = self._state_manager.has_device(ip)
                if self.sync_camera_added(camera):
                    if was_existing:
                        results["updated"] += 1
                    else:
                        results["created"] += 1
            except Exception as e:
                results["errors"].append(f"Failed to sync {ip}: {e}")

        # Remove orphan devices (devices not in camera list)
        orphan_ips = existing_ips - camera_ips
        for ip in orphan_ips:
            try:
                if self.sync_camera_removed(ip):
                    results["removed"] += 1
            except Exception as e:
                results["errors"].append(f"Failed to remove orphan {ip}: {e}")

        logger.info(f"Camera sync complete: {results}")
        return results

    def get_mock_device(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """
        Get mock device info for a camera IP.

        Args:
            ip_address: Camera IP address

        Returns:
            Device dict or None if not found
        """
        if not self._ensure_state_manager():
            return None

        device = self._state_manager.get_device(ip_address)
        if device:
            return device.to_dict()
        return None

    def get_all_mock_devices(self) -> List[Dict[str, Any]]:
        """
        Get all mock devices.

        Returns:
            List of device dicts
        """
        if not self._ensure_state_manager():
            return []

        devices = self._state_manager.get_all_devices()
        return [d.to_dict() for d in devices]


# Global singleton instance
_sync_service: Optional[CameraMockSyncService] = None


def get_camera_mock_sync_service(settings: Optional[Dict[str, Any]] = None) -> CameraMockSyncService:
    """
    Get or create the global CameraMockSyncService instance.

    Args:
        settings: Application settings (used on first call to initialize)

    Returns:
        CameraMockSyncService instance
    """
    global _sync_service
    if _sync_service is None:
        _sync_service = CameraMockSyncService(settings=settings)
    return _sync_service


def set_camera_mock_sync_service(service: CameraMockSyncService):
    """
    Set a custom sync service instance (useful for testing or injection).

    Args:
        service: CameraMockSyncService instance to use
    """
    global _sync_service
    _sync_service = service

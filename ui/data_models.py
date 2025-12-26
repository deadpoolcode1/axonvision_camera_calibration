"""
Data Models for Camera Calibration UI

Defines data structures for calibration sessions, cameras, and platforms.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


# Mounting position options
MOUNTING_POSITIONS = [
    "N/A",  # Default - must be changed before proceeding
    "Front Center",
    "Front Left",
    "Front Right",
    "Rear Center",
    "Rear Left",
    "Rear Right",
    "Left Up",
    "Left Center",
    "Left Down",
    "Right Up",
    "Right Center",
    "Right Down",
]

# 3:1 cameras can only be in these positions
VALID_3_1_POSITIONS = ["Front Center", "Rear Center"]

# Camera type options
CAMERA_TYPES = ["AI CENTRAL", "1:1", "3:1 manager", "3:1 worker"]

# Maximum limits
MAX_CAMERAS = 6
MAX_AI_CENTRAL_CAMERAS = 1

# Camera model options
CAMERA_MODELS = ["IMX219", "IMX477", "IMX708", "OV5647", "Custom"]

# Platform type options
PLATFORM_TYPES = ["Type A", "Type B", "Type C", "Custom"]


@dataclass
class CameraDefinition:
    """Definition of a single camera in the system."""
    camera_number: int
    camera_type: str = "1:1"  # AI CENTRAL, 1:1, 3:1 manager, or 3:1 worker
    camera_model: str = "IMX219"
    mounting_position: str = "N/A"  # Default to N/A - must be set
    ip_address: str = "192.168.1.100"
    camera_id: str = ""  # Will be auto-generated if empty

    def __post_init__(self):
        if not self.camera_id:
            self.camera_id = f"cam_{self.camera_number}"

    @property
    def intrinsic_file_path(self) -> str:
        """Get the expected intrinsic calibration file path."""
        return f"camera_intrinsic/camera_intrinsics_{self.camera_id}.json"

    def has_intrinsic_calibration(self, base_path: str = ".") -> bool:
        """Check if intrinsic calibration exists for this camera."""
        full_path = Path(base_path) / self.intrinsic_file_path
        return full_path.exists()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraDefinition':
        # Filter to only include valid fields for CameraDefinition
        valid_fields = {'camera_number', 'camera_type', 'camera_model',
                        'mounting_position', 'ip_address', 'camera_id'}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class PlatformConfiguration:
    """Configuration for a calibration platform."""
    platform_type: str = "Type A"
    platform_id: str = ""
    cameras: List[CameraDefinition] = field(default_factory=list)

    def add_camera(self) -> CameraDefinition:
        """Add a new camera with the next available number."""
        next_num = len(self.cameras) + 1
        # Determine default IP based on camera count
        ip_suffix = 99 + next_num
        camera = CameraDefinition(
            camera_number=next_num,
            ip_address=f"192.168.1.{ip_suffix}"
        )
        self.cameras.append(camera)
        return camera

    def remove_camera(self, index: int) -> None:
        """Remove a camera by index."""
        if 0 <= index < len(self.cameras):
            self.cameras.pop(index)
            # Re-number remaining cameras
            for i, cam in enumerate(self.cameras):
                cam.camera_number = i + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'platform_type': self.platform_type,
            'platform_id': self.platform_id,
            'cameras': [cam.to_dict() for cam in self.cameras]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlatformConfiguration':
        config = cls(
            platform_type=data.get('platform_type', 'Type A'),
            platform_id=data.get('platform_id', '')
        )
        for cam_data in data.get('cameras', []):
            config.cameras.append(CameraDefinition.from_dict(cam_data))
        return config


@dataclass
class CalibrationSession:
    """A calibration session record."""
    session_id: str
    platform_id: str
    timestamp: str
    status: str  # 'Passed', 'Warning', 'Failed', 'In Progress'
    platform_config: Optional[PlatformConfiguration] = None
    results_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'session_id': self.session_id,
            'platform_id': self.platform_id,
            'timestamp': self.timestamp,
            'status': self.status,
            'results_path': self.results_path
        }
        if self.platform_config:
            data['platform_config'] = self.platform_config.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationSession':
        platform_config = None
        if 'platform_config' in data:
            platform_config = PlatformConfiguration.from_dict(data['platform_config'])
        return cls(
            session_id=data['session_id'],
            platform_id=data['platform_id'],
            timestamp=data['timestamp'],
            status=data['status'],
            platform_config=platform_config,
            results_path=data.get('results_path')
        )


class CalibrationDataStore:
    """Persistent storage for calibration sessions and configurations."""

    DEFAULT_FILE = "calibration_sessions.json"

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or self.DEFAULT_FILE
        self.sessions: List[CalibrationSession] = []
        self.last_platform_config: Optional[PlatformConfiguration] = None
        self._load()

    def _load(self) -> None:
        """Load data from storage file."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self.sessions = [
                CalibrationSession.from_dict(s)
                for s in data.get('sessions', [])
            ]

            if 'last_platform_config' in data and data['last_platform_config']:
                self.last_platform_config = PlatformConfiguration.from_dict(
                    data['last_platform_config']
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load calibration data: {e}")

    def save(self) -> None:
        """Save data to storage file."""
        data = {
            'sessions': [s.to_dict() for s in self.sessions],
            'last_platform_config': (
                self.last_platform_config.to_dict()
                if self.last_platform_config else None
            )
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_session(self, session: CalibrationSession) -> None:
        """Add a new calibration session."""
        self.sessions.insert(0, session)  # Most recent first
        if session.platform_config:
            self.last_platform_config = session.platform_config
        self.save()

    def update_session(self, session_id: str, status: str, results_path: Optional[str] = None) -> None:
        """Update an existing session."""
        for session in self.sessions:
            if session.session_id == session_id:
                session.status = status
                if results_path:
                    session.results_path = results_path
                break
        self.save()

    def get_recent_sessions(self, limit: int = 5) -> List[CalibrationSession]:
        """Get the most recent calibration sessions."""
        return self.sessions[:limit]

    def get_session(self, session_id: str) -> Optional[CalibrationSession]:
        """Get a specific session by ID."""
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        return None

    @staticmethod
    def generate_session_id(platform_id: str) -> str:
        """Generate a unique session ID based on platform ID."""
        # Count existing sessions with this platform ID pattern
        # Format: PQ4459-001, PQ4459-002, etc.
        return f"{platform_id}-{datetime.now().strftime('%H%M%S')}"

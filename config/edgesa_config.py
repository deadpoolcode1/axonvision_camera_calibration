"""
EdgeSA Configuration Loader

Provides configuration management for the EdgeSA distributed surveillance system.
Supports YAML file configuration with .env file and environment variable overrides.

Configuration Loading Priority (highest to lowest):
1. Environment variables (e.g., DISCOVERY_TIMEOUT_MS=10000)
2. .env file values
3. YAML configuration file
4. Built-in defaults

Usage:
    from config.edgesa_config import EdgeSAConfig, edgesa_config

    # Using the singleton instance
    timeout = edgesa_config.discovery.timeout_ms

    # Creating a custom instance with different config file
    custom_config = EdgeSAConfig(
        config_file="/path/to/custom.yaml",
        env_file="/path/to/.env"
    )
"""

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Optional dependency - support running without python-dotenv installed
try:
    from dotenv import load_dotenv, dotenv_values
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None
    dotenv_values = None

from .exceptions import ConfigurationLoadError, ConfigurationValidationError


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class DiscoveryConfig:
    """Discovery service configuration."""
    base_url: str = "http://localhost:8000"
    timeout_ms: int = 5000
    mdns_service_type: str = "_axon-vision._tcp.local."
    api_version: str = "v1"

    @property
    def health_endpoint(self) -> str:
        """Full URL for health check endpoint."""
        return f"{self.base_url}/{self.api_version}/health"

    @property
    def discovery_endpoint(self) -> str:
        """Full URL for discovery endpoint."""
        return f"{self.base_url}/{self.api_version}/discovery"

    @property
    def ssh_keys_endpoint(self) -> str:
        """Full URL for SSH keys endpoint."""
        return f"{self.base_url}/{self.api_version}/ssh-keys"


@dataclass
class StreamPorts:
    """Stream port configuration for a device slot."""
    mjpeg_vis: int = 8080
    mjpeg_thermal: int = 9080
    mpegts: int = 5005


@dataclass
class DeviceTypeConfig:
    """Configuration for a device type."""
    hostname_pattern: str
    hostname_prefix: str
    role: str  # "manager", "worker", or "switchable"
    can_switch_mode: bool = False
    max_input_streams: int = 1
    max_manager_streams: int = 0
    stream_ports: Dict[str, Any] = field(default_factory=dict)

    def matches_hostname(self, hostname: str) -> bool:
        """Check if a hostname matches this device type pattern."""
        import fnmatch
        # Remove .local suffix if present
        clean_hostname = hostname.replace(".local", "")
        pattern = self.hostname_pattern.replace("-*", "*").replace(".local", "")
        return fnmatch.fnmatch(clean_hostname, pattern)


@dataclass
class SSHConfig:
    """SSH provisioning configuration."""
    default_username: str = "nvidia"
    default_password: str = "nvidia"
    connect_timeout_ms: int = 5000
    auth_timeout_ms: int = 8000
    max_concurrency: int = 3


@dataclass
class DevicePathsConfig:
    """Device file path configuration."""
    base_path: str = "/etc/axon"
    mode_file: str = "mode.env"
    manager_address_file: str = "manager_address.env"
    position_file: str = "position.env"
    sensors_config_file: str = "sensors_config.yml"
    snapshots_dir: str = "snapshots"

    def get_full_path(self, filename: str) -> str:
        """Get full path to a configuration file on a device."""
        return f"{self.base_path}/{filename}"

    @property
    def mode_path(self) -> str:
        return self.get_full_path(self.mode_file)

    @property
    def manager_address_path(self) -> str:
        return self.get_full_path(self.manager_address_file)

    @property
    def position_path(self) -> str:
        return self.get_full_path(self.position_file)

    @property
    def sensors_config_path(self) -> str:
        return self.get_full_path(self.sensors_config_file)

    @property
    def snapshots_path(self) -> str:
        return self.get_full_path(self.snapshots_dir)


@dataclass
class DeviceAPIConfig:
    """Device REST API configuration."""
    port: int = 5000
    timeout: float = 10.0
    endpoints: Dict[str, str] = field(default_factory=lambda: {
        "pipeline_mode": "/api/pipeline_mode",
        "manager_connection": "/api/manager_connection",
        "streams": "/config/streams/{slot}/stream-id",
        "position": "/config/position",
        "sensors_config": "/api/sensors_config",
        "all_sensor_urls": "/api/all_sensor_urls",
        "health": "/health"
    })

    def get_endpoint(self, name: str, **kwargs) -> str:
        """Get an endpoint path, optionally formatting with kwargs."""
        endpoint = self.endpoints.get(name, "")
        if kwargs:
            endpoint = endpoint.format(**kwargs)
        return endpoint

    def get_url(self, ip: str, endpoint_name: str, **kwargs) -> str:
        """Get full URL for an endpoint on a specific device."""
        endpoint = self.get_endpoint(endpoint_name, **kwargs)
        return f"http://{ip}:{self.port}{endpoint}"


@dataclass
class PositionsConfig:
    """Position configuration and constraints."""
    available: List[str] = field(default_factory=list)
    threecluster_only: List[str] = field(default_factory=list)
    all_workers: List[str] = field(default_factory=list)

    def get_valid_positions(self, device_type: str) -> List[str]:
        """Get valid positions for a device type."""
        if device_type == "threecluster":
            return self.available
        elif device_type in ("smartcluster", "ai_central"):
            return self.all_workers
        return self.available


@dataclass
class BackupConfig:
    """Backup configuration."""
    local_path: str = "./backups/{hostname}/snapshots/{timestamp}"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    auto_backup: bool = True


@dataclass
class NetworkConfig:
    """Network configuration."""
    default_subnet: str = "192.168.1.0/24"
    multicast_base_address: str = "239.1.1.1"
    multicast_port_start: int = 5000


@dataclass
class RetryConfig:
    """Retry configuration for network operations."""
    max_attempts: int = 4
    backoff_base: int = 2
    max_backoff: int = 16

    def get_backoff_seconds(self, attempt: int) -> int:
        """Get backoff time for a given attempt number."""
        backoff = self.backoff_base ** attempt
        return min(backoff, self.max_backoff)


@dataclass
class ValidationConfig:
    """Validation patterns and rules."""
    ip_pattern: str = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    hostname_pattern: str = r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.local$"
    port_min: int = 1
    port_max: int = 65535
    stream_id_pattern: str = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

    def is_valid_ip(self, ip: str) -> bool:
        """Validate an IP address."""
        return bool(re.match(self.ip_pattern, ip))

    def is_valid_hostname(self, hostname: str) -> bool:
        """Validate a hostname."""
        return bool(re.match(self.hostname_pattern, hostname))

    def is_valid_port(self, port: int) -> bool:
        """Validate a port number."""
        return self.port_min <= port <= self.port_max

    def is_valid_stream_id(self, stream_id: str) -> bool:
        """Validate a stream ID (UUID format)."""
        return bool(re.match(self.stream_id_pattern, stream_id))


@dataclass
class LoggingConfig:
    """Logging configuration for EdgeSA operations."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/edgesa_config.log"
    file_max_size_mb: int = 5
    file_backup_count: int = 3


@dataclass
class MockServerConfig:
    """Mock server configuration."""
    host: str = "127.0.0.1"
    discovery_port: int = 8000
    device_api_port: int = 5000
    sftp_port: int = 2222

    @property
    def discovery_url(self) -> str:
        """Full URL for mock discovery service."""
        return f"http://{self.host}:{self.discovery_port}"

    @property
    def device_api_url(self) -> str:
        """Full URL for mock device API."""
        return f"http://{self.host}:{self.device_api_port}"


@dataclass
class SimulatedDeviceConfig:
    """Configuration for a simulated device."""
    ip: str
    hostname: str
    device_type: str
    initial_mode: str = "worker"


@dataclass
class SimulationPersistenceConfig:
    """Persistence configuration for simulation."""
    enabled: bool = True
    state_file: str = "./mock_data/device_states.json"
    filesystem_path: str = "./mock_data/devices"


@dataclass
class SimulationConfig:
    """
    Simulation mode configuration.

    When enabled, the application uses mock services instead of real device APIs.
    Video streams still come from real cameras.
    """
    enabled: bool = False
    mock_server: MockServerConfig = field(default_factory=MockServerConfig)
    devices: List[SimulatedDeviceConfig] = field(default_factory=list)
    persistence: SimulationPersistenceConfig = field(default_factory=SimulationPersistenceConfig)
    active_scenario: str = "happy_path"

    def get_device_config_list(self) -> List[Dict[str, Any]]:
        """Get devices as list of dicts for state manager initialization."""
        return [
            {
                "ip": d.ip,
                "hostname": d.hostname,
                "type": d.device_type,
                "initial_mode": d.initial_mode
            }
            for d in self.devices
        ]


# =============================================================================
# Main Configuration Class
# =============================================================================

class EdgeSAConfig:
    """
    EdgeSA Configuration Manager.

    Loads configuration from YAML file with support for .env file and
    environment variable overrides. Validates configuration on startup.

    Configuration Loading Priority (highest to lowest):
    1. Environment variables
    2. .env file values
    3. YAML configuration file
    4. Built-in defaults

    Attributes:
        discovery: Discovery service configuration
        device_types: Device type configurations (ai_central, smartcluster, threecluster)
        ssh: SSH provisioning configuration
        device_paths: Device file path configuration
        device_api: Device REST API configuration
        positions: Position configuration and constraints
        backup: Backup configuration
        network: Network configuration
        retry: Retry configuration
        validation: Validation patterns and rules
        logging: Logging configuration
    """

    _instance: Optional['EdgeSAConfig'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> 'EdgeSAConfig':
        """Singleton pattern - returns existing instance if already created."""
        if cls._instance is None or kwargs.get('force_new', False):
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        env_file: Optional[Union[str, Path]] = None,
        validate: bool = True,
        force_new: bool = False
    ):
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to YAML configuration file. Defaults to
                         config/edgesa_settings.yaml
            env_file: Path to .env file. Defaults to .env in current directory
            validate: Whether to validate configuration on load (default: True)
            force_new: Force creating a new instance (default: False)
        """
        if self._initialized and not force_new:
            return

        self._config_file = config_file
        self._env_file = env_file
        self._raw_config: Dict[str, Any] = {}
        self._env_values: Dict[str, str] = {}

        # Initialize configuration sections
        self.discovery = DiscoveryConfig()
        self.device_types: Dict[str, DeviceTypeConfig] = {}
        self.ssh = SSHConfig()
        self.device_paths = DevicePathsConfig()
        self.device_api = DeviceAPIConfig()
        self.positions = PositionsConfig()
        self.backup = BackupConfig()
        self.network = NetworkConfig()
        self.retry = RetryConfig()
        self.validation = ValidationConfig()
        self.logging = LoggingConfig()
        self.simulation = SimulationConfig()
        self.sensor_types: List[str] = ["vis", "thermal", "day", "night"]

        # Load configuration
        self._load_configuration()

        # Validate if requested
        if validate:
            self.validate()

        self._initialized = True

    def _load_configuration(self) -> None:
        """Load configuration from all sources."""
        # Load .env file first (lower priority than env vars)
        self._load_env_file()

        # Load YAML configuration
        self._load_yaml_config()

        # Apply configuration with environment overrides
        self._apply_configuration()

    def _load_env_file(self) -> None:
        """Load environment variables from .env file."""
        if not DOTENV_AVAILABLE:
            logger.debug("python-dotenv not installed, skipping .env file loading")
            return

        if self._env_file:
            env_path = Path(self._env_file)
        else:
            # Look for .env in current directory and config directory
            cwd_env = Path.cwd() / ".env"
            config_dir_env = Path(__file__).parent / ".env"

            if cwd_env.exists():
                env_path = cwd_env
            elif config_dir_env.exists():
                env_path = config_dir_env
            else:
                env_path = None

        if env_path and env_path.exists():
            try:
                # Load values from .env file (doesn't override existing env vars)
                load_dotenv(env_path, override=False)
                # Also store the raw values for reference
                self._env_values = dotenv_values(env_path)
                logger.info(f"Loaded environment from: {env_path}")
            except Exception as e:
                raise ConfigurationLoadError(
                    source=str(env_path),
                    reason=f"Failed to parse .env file: {e}",
                    details=str(e)
                )

    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        if self._config_file:
            config_path = Path(self._config_file)
        else:
            config_path = Path(__file__).parent / "edgesa_settings.yaml"

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}, using defaults")
            self._raw_config = {}
            return

        try:
            with open(config_path, 'r') as f:
                self._raw_config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationLoadError(
                source=str(config_path),
                reason=f"Invalid YAML syntax: {e}",
                details=str(e)
            )
        except IOError as e:
            raise ConfigurationLoadError(
                source=str(config_path),
                reason=f"Failed to read file: {e}",
                details=str(e)
            )

    def _get_value(
        self,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Get a configuration value with environment variable override.

        Priority: Environment variable > .env file > YAML > default

        Args:
            key: Dot-notation key (e.g., "discovery.timeout_ms")
            default: Default value if not found
            required: If True, raise error when value is missing

        Returns:
            Configuration value
        """
        # Build environment variable name (SECTION_SUBSECTION_KEY format)
        env_key = key.upper().replace('.', '_')

        # Check environment variable first (highest priority)
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return self._parse_value(env_value)

        # Navigate YAML config
        keys = key.split('.')
        value = self._raw_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                if required:
                    raise ConfigurationLoadError(
                        field=key,
                        reason=f"Required configuration field '{key}' is missing"
                    )
                return default

        return value

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate type."""
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

    def _apply_configuration(self) -> None:
        """Apply loaded configuration to dataclass instances."""
        # Discovery configuration
        self.discovery = DiscoveryConfig(
            base_url=self._get_value("discovery.base_url", "http://localhost:8000"),
            timeout_ms=self._get_value("discovery.timeout_ms", 5000),
            mdns_service_type=self._get_value("discovery.mdns_service_type", "_axon-vision._tcp.local."),
            api_version=self._get_value("discovery.api_version", "v1")
        )

        # Device types configuration
        self._load_device_types()

        # SSH configuration
        self.ssh = SSHConfig(
            default_username=self._get_value("ssh.default_username", "nvidia"),
            default_password=self._get_value("ssh.default_password", "nvidia"),
            connect_timeout_ms=self._get_value("ssh.connect_timeout_ms", 5000),
            auth_timeout_ms=self._get_value("ssh.auth_timeout_ms", 8000),
            max_concurrency=self._get_value("ssh.max_concurrency", 3)
        )

        # Device paths configuration
        files = self._get_value("device_paths.files", {})
        self.device_paths = DevicePathsConfig(
            base_path=self._get_value("device_paths.base_path", "/etc/axon"),
            mode_file=files.get("mode", "mode.env"),
            manager_address_file=files.get("manager_address", "manager_address.env"),
            position_file=files.get("position", "position.env"),
            sensors_config_file=files.get("sensors_config", "sensors_config.yml"),
            snapshots_dir=self._get_value("device_paths.snapshots_dir", "snapshots")
        )

        # Device API configuration
        endpoints = self._get_value("device_api.endpoints", {})
        self.device_api = DeviceAPIConfig(
            port=self._get_value("device_api.port", 5000),
            timeout=self._get_value("device_api.timeout", 10.0),
            endpoints=endpoints if endpoints else DeviceAPIConfig().endpoints
        )

        # Positions configuration
        constraints = self._get_value("positions.constraints", {})
        self.positions = PositionsConfig(
            available=self._get_value("positions.available", []),
            threecluster_only=constraints.get("threecluster_only", []),
            all_workers=constraints.get("all_workers", [])
        )

        # Backup configuration
        self.backup = BackupConfig(
            local_path=self._get_value("backup.local_path", "./backups/{hostname}/snapshots/{timestamp}"),
            timestamp_format=self._get_value("backup.timestamp_format", "%Y%m%d_%H%M%S"),
            auto_backup=self._get_value("backup.auto_backup", True)
        )

        # Network configuration
        multicast = self._get_value("network.multicast", {})
        self.network = NetworkConfig(
            default_subnet=self._get_value("network.default_subnet", "192.168.1.0/24"),
            multicast_base_address=multicast.get("base_address", "239.1.1.1"),
            multicast_port_start=multicast.get("port_start", 5000)
        )

        # Retry configuration
        self.retry = RetryConfig(
            max_attempts=self._get_value("retry.max_attempts", 4),
            backoff_base=self._get_value("retry.backoff_base", 2),
            max_backoff=self._get_value("retry.max_backoff", 16)
        )

        # Validation configuration
        self.validation = ValidationConfig(
            ip_pattern=self._get_value("validation.ip_pattern", ValidationConfig.ip_pattern),
            hostname_pattern=self._get_value("validation.hostname_pattern", ValidationConfig.hostname_pattern),
            port_min=self._get_value("validation.port_min", 1),
            port_max=self._get_value("validation.port_max", 65535),
            stream_id_pattern=self._get_value("validation.stream_id_pattern", ValidationConfig.stream_id_pattern)
        )

        # Logging configuration
        log_file = self._get_value("logging.file", {})
        self.logging = LoggingConfig(
            level=self._get_value("logging.level", "INFO"),
            format=self._get_value("logging.format", LoggingConfig.format),
            file_enabled=log_file.get("enabled", True),
            file_path=log_file.get("path", "logs/edgesa_config.log"),
            file_max_size_mb=log_file.get("max_size_mb", 5),
            file_backup_count=log_file.get("backup_count", 3)
        )

        # Sensor types
        self.sensor_types = self._get_value("sensor_types", ["vis", "thermal", "day", "night"])

        # Simulation configuration
        self._load_simulation_config()

    def _load_simulation_config(self) -> None:
        """Load simulation mode configuration."""
        sim_config = self._get_value("simulation", {})

        # Mock server configuration
        mock_server_config = sim_config.get("mock_server", {})
        mock_server = MockServerConfig(
            host=mock_server_config.get("host", "127.0.0.1"),
            discovery_port=mock_server_config.get("discovery_port", 8000),
            device_api_port=mock_server_config.get("device_api_port", 5000),
            sftp_port=mock_server_config.get("sftp_port", 2222)
        )

        # Simulated devices
        devices_config = sim_config.get("devices", [])
        devices = []
        for dev in devices_config:
            device_type = dev.get("type", "smartcluster")
            default_mode = "manager" if device_type == "aicentral" else "worker"
            devices.append(SimulatedDeviceConfig(
                ip=dev.get("ip", ""),
                hostname=dev.get("hostname", ""),
                device_type=device_type,
                initial_mode=dev.get("initial_mode", default_mode)
            ))

        # Persistence configuration
        persistence_config = sim_config.get("persistence", {})
        persistence = SimulationPersistenceConfig(
            enabled=persistence_config.get("enabled", True),
            state_file=persistence_config.get("state_file", "./mock_data/device_states.json"),
            filesystem_path=persistence_config.get("filesystem_path", "./mock_data/devices")
        )

        # Build simulation config
        self.simulation = SimulationConfig(
            enabled=self._get_value("simulation.enabled", False),
            mock_server=mock_server,
            devices=devices,
            persistence=persistence,
            active_scenario=sim_config.get("scenarios", {}).get("active_scenario", "happy_path")
        )

        if self.simulation.enabled:
            logger.info("Simulation mode is ENABLED")

    def _load_device_types(self) -> None:
        """Load device type configurations."""
        device_types_config = self._get_value("device_types", {})

        # Default device type configurations
        defaults = {
            "ai_central": DeviceTypeConfig(
                hostname_pattern="aicentral-*",
                hostname_prefix="aicentral",
                role="manager",
                can_switch_mode=False,
                max_input_streams=6
            ),
            "smartcluster": DeviceTypeConfig(
                hostname_pattern="smartcluster-*",
                hostname_prefix="smartcluster",
                role="worker",
                can_switch_mode=False,
                max_input_streams=1,
                stream_ports={
                    "mjpeg": {"vis": 8080, "thermal": 9080},
                    "mpegts": [5005]
                }
            ),
            "threecluster": DeviceTypeConfig(
                hostname_pattern="threecluster-*",
                hostname_prefix="threecluster",
                role="switchable",
                can_switch_mode=True,
                max_input_streams=3,
                max_manager_streams=3,
                stream_ports={
                    "slot_0": {"mjpeg": {"vis": 8079, "thermal": 9079}, "mpegts": 5004},
                    "slot_1": {"mjpeg": {"vis": 8080, "thermal": 9080}, "mpegts": 5005},
                    "slot_2": {"mjpeg": {"vis": 8081, "thermal": 9081}, "mpegts": 5006}
                }
            )
        }

        for device_type, default_config in defaults.items():
            type_config = device_types_config.get(device_type, {})

            self.device_types[device_type] = DeviceTypeConfig(
                hostname_pattern=type_config.get("hostname_pattern", default_config.hostname_pattern),
                hostname_prefix=type_config.get("hostname_prefix", default_config.hostname_prefix),
                role=type_config.get("role", default_config.role),
                can_switch_mode=type_config.get("can_switch_mode", default_config.can_switch_mode),
                max_input_streams=type_config.get("max_input_streams", default_config.max_input_streams),
                max_manager_streams=type_config.get("max_manager_streams", default_config.max_manager_streams),
                stream_ports=type_config.get("stream_ports", default_config.stream_ports)
            )

    def validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ConfigurationValidationError: If validation fails
        """
        errors: List[str] = []

        # Validate discovery configuration
        if not self.discovery.base_url:
            errors.append("discovery.base_url is required")
        if not 250 <= self.discovery.timeout_ms <= 15000:
            errors.append(f"discovery.timeout_ms must be between 250 and 15000, got {self.discovery.timeout_ms}")

        # Validate SSH configuration
        if not self.ssh.default_username:
            errors.append("ssh.default_username is required")
        if not 500 <= self.ssh.connect_timeout_ms <= 15000:
            errors.append(f"ssh.connect_timeout_ms must be between 500 and 15000, got {self.ssh.connect_timeout_ms}")
        if not 1000 <= self.ssh.auth_timeout_ms <= 30000:
            errors.append(f"ssh.auth_timeout_ms must be between 1000 and 30000, got {self.ssh.auth_timeout_ms}")
        if self.ssh.auth_timeout_ms < self.ssh.connect_timeout_ms:
            errors.append("ssh.auth_timeout_ms must be >= ssh.connect_timeout_ms")
        if not 1 <= self.ssh.max_concurrency <= 5:
            errors.append(f"ssh.max_concurrency must be between 1 and 5, got {self.ssh.max_concurrency}")

        # Validate device API configuration
        if not self.validation.is_valid_port(self.device_api.port):
            errors.append(f"device_api.port must be between 1 and 65535, got {self.device_api.port}")

        # Validate device types exist
        required_device_types = ["ai_central", "smartcluster", "threecluster"]
        for dt in required_device_types:
            if dt not in self.device_types:
                errors.append(f"device_types.{dt} configuration is missing")

        # Validate retry configuration
        if self.retry.max_attempts < 1:
            errors.append(f"retry.max_attempts must be >= 1, got {self.retry.max_attempts}")
        if self.retry.backoff_base < 1:
            errors.append(f"retry.backoff_base must be >= 1, got {self.retry.backoff_base}")

        if errors:
            raise ConfigurationValidationError(errors=errors)

        logger.info("Configuration validation passed")

    def get_device_type(self, hostname: str) -> Optional[str]:
        """
        Determine device type from hostname.

        Args:
            hostname: Device hostname (e.g., "smartcluster001.local")

        Returns:
            Device type key ("ai_central", "smartcluster", "threecluster") or None
        """
        for device_type, config in self.device_types.items():
            if config.matches_hostname(hostname):
                return device_type
        return None

    def get_stream_ports(self, device_type: str, slot: int = 0) -> StreamPorts:
        """
        Get stream ports for a device type and slot.

        Args:
            device_type: Device type key
            slot: Camera slot index (0-2 for threecluster, 0 for others)

        Returns:
            StreamPorts configuration
        """
        type_config = self.device_types.get(device_type)
        if not type_config:
            return StreamPorts()

        ports = type_config.stream_ports
        if device_type == "threecluster":
            slot_key = f"slot_{slot}"
            slot_config = ports.get(slot_key, {})
            mjpeg = slot_config.get("mjpeg", {})
            return StreamPorts(
                mjpeg_vis=mjpeg.get("vis", 8080),
                mjpeg_thermal=mjpeg.get("thermal", 9080),
                mpegts=slot_config.get("mpegts", 5005)
            )
        else:
            mjpeg = ports.get("mjpeg", {})
            mpegts_list = ports.get("mpegts", [5005])
            return StreamPorts(
                mjpeg_vis=mjpeg.get("vis", 8080),
                mjpeg_thermal=mjpeg.get("thermal", 9080),
                mpegts=mpegts_list[0] if mpegts_list else 5005
            )

    def is_manager_eligible(self, device_type: str) -> bool:
        """
        Check if a device type can serve as manager.

        Args:
            device_type: Device type key

        Returns:
            True if device can be manager
        """
        type_config = self.device_types.get(device_type)
        if not type_config:
            return False
        return type_config.role in ("manager", "switchable")

    def reload(self) -> None:
        """Reload configuration from files."""
        self._initialized = False
        self._load_configuration()
        self.validate()
        self._initialized = True
        logger.info("Configuration reloaded")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "discovery": {
                "base_url": self.discovery.base_url,
                "timeout_ms": self.discovery.timeout_ms,
                "mdns_service_type": self.discovery.mdns_service_type,
                "api_version": self.discovery.api_version
            },
            "ssh": {
                "default_username": self.ssh.default_username,
                "connect_timeout_ms": self.ssh.connect_timeout_ms,
                "auth_timeout_ms": self.ssh.auth_timeout_ms,
                "max_concurrency": self.ssh.max_concurrency
            },
            "device_paths": {
                "base_path": self.device_paths.base_path,
                "mode_file": self.device_paths.mode_file,
                "manager_address_file": self.device_paths.manager_address_file,
                "position_file": self.device_paths.position_file,
                "sensors_config_file": self.device_paths.sensors_config_file
            },
            "device_api": {
                "port": self.device_api.port,
                "timeout": self.device_api.timeout
            },
            "network": {
                "default_subnet": self.network.default_subnet,
                "multicast_base_address": self.network.multicast_base_address
            },
            "retry": {
                "max_attempts": self.retry.max_attempts,
                "backoff_base": self.retry.backoff_base,
                "max_backoff": self.retry.max_backoff
            }
        }

    def __repr__(self) -> str:
        return (
            f"EdgeSAConfig("
            f"discovery_url={self.discovery.base_url!r}, "
            f"device_types={list(self.device_types.keys())}, "
            f"ssh_user={self.ssh.default_username!r})"
        )


# =============================================================================
# Singleton Instance
# =============================================================================

# Create singleton instance on module load
# Set validate=False initially to allow importing without immediate validation
_edgesa_config: Optional[EdgeSAConfig] = None


def get_edgesa_config(
    config_file: Optional[Union[str, Path]] = None,
    env_file: Optional[Union[str, Path]] = None,
    validate: bool = True
) -> EdgeSAConfig:
    """
    Get the EdgeSA configuration singleton instance.

    Args:
        config_file: Path to YAML configuration file (only used on first call)
        env_file: Path to .env file (only used on first call)
        validate: Whether to validate configuration

    Returns:
        EdgeSAConfig singleton instance
    """
    global _edgesa_config
    if _edgesa_config is None:
        _edgesa_config = EdgeSAConfig(
            config_file=config_file,
            env_file=env_file,
            validate=validate
        )
    return _edgesa_config


# Lazy-loaded singleton for convenient import
# Usage: from config.edgesa_config import edgesa_config
class _EdgeSAConfigProxy:
    """Lazy proxy for EdgeSAConfig singleton."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_edgesa_config(), name)

    def __repr__(self) -> str:
        return repr(get_edgesa_config())


edgesa_config = _EdgeSAConfigProxy()

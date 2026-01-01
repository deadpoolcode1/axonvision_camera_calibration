"""
Mock Discovery Service

Simulates the Hardware Discovery & Provisioning Service API:
- GET /v1/health - Health check
- GET /v1/discovery - Device discovery (returns configured devices)
- POST /v1/ssh-keys - SSH key provisioning (simulated)
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field, field_validator
import re

from .device_state import DeviceStateManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["discovery"])

# Will be set by server.py
state_manager: Optional[DeviceStateManager] = None
active_scenario: str = "happy_path"


def set_state_manager(manager: DeviceStateManager) -> None:
    """Set the device state manager instance."""
    global state_manager
    state_manager = manager


def set_scenario(scenario: str) -> None:
    """Set the active error injection scenario."""
    global active_scenario
    active_scenario = scenario


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthCheck(BaseModel):
    """Health check for a service component."""
    name: str
    status: str  # "healthy" or "unhealthy"
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    requestId: str
    endpoint: str
    status: str
    version: str
    checks: List[HealthCheck]
    errors: List[str]


class DeviceStreams(BaseModel):
    """Stream ports for a device."""
    mjpeg: List[int]
    mpegts: List[int]


class DiscoveredDevice(BaseModel):
    """A discovered device."""
    id: str
    deviceType: str
    hostname: str
    ipAddress: str
    addresses: List[str]
    streams: DeviceStreams
    txt: Optional[Dict[str, str]] = None
    discoveredAt: str


class DiscoveryResponse(BaseModel):
    """Discovery scan response."""
    requestId: str
    endpoint: str
    scanTimeMs: int
    totalDevices: int
    devices: List[DiscoveredDevice]
    errors: List[str]


class SSHTarget(BaseModel):
    """SSH provisioning target."""
    ip: str

    @field_validator('ip')
    @classmethod
    def validate_ip(cls, v):
        # Basic IPv4 validation
        pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid IPv4 address: {v}")
        # Check for reserved addresses
        if v in ("0.0.0.0", "255.255.255.255"):
            raise ValueError(f"Reserved address not allowed: {v}")
        # Check for multicast
        first_octet = int(v.split('.')[0])
        if 224 <= first_octet <= 239:
            raise ValueError(f"Multicast address not allowed: {v}")
        return v


class SSHCredentials(BaseModel):
    """SSH credentials."""
    username: str = "nvidia"
    password: str = "nvidia"


class SSHOptions(BaseModel):
    """SSH connection options."""
    maxConcurrency: int = Field(default=3, ge=1, le=5)
    connectTimeoutMs: int = Field(default=5000, ge=500, le=15000)
    authTimeoutMs: int = Field(default=8000, ge=1000, le=30000)

    @field_validator('authTimeoutMs')
    @classmethod
    def validate_auth_timeout(cls, v, info):
        connect_timeout = info.data.get('connectTimeoutMs', 5000)
        if v < connect_timeout:
            raise ValueError("authTimeoutMs must be >= connectTimeoutMs")
        return v


class SSHKeysRequest(BaseModel):
    """SSH key provisioning request."""
    targets: List[SSHTarget] = Field(..., min_length=1, max_length=50)
    credentials: Optional[SSHCredentials] = None
    options: Optional[SSHOptions] = None

    @field_validator('targets')
    @classmethod
    def validate_unique_targets(cls, v):
        ips = [t.ip for t in v]
        if len(ips) != len(set(ips)):
            raise ValueError("Duplicate IP addresses are not allowed")
        return v


class SSHStep(BaseModel):
    """A step in the SSH provisioning process."""
    name: str
    status: str  # "ok", "failed", "skipped"
    durationMs: int


class SSHError(BaseModel):
    """SSH provisioning error."""
    code: str
    message: str


class SSHResult(BaseModel):
    """Result for a single SSH target."""
    target: str
    status: str  # "key_installed", "already_configured", "failed"
    error: Optional[SSHError] = None
    steps: List[SSHStep]


class SSHSummary(BaseModel):
    """Summary of SSH provisioning results."""
    totalTargets: int
    keyInstalled: int
    alreadyConfigured: int
    failed: int


class SSHKeysResponse(BaseModel):
    """SSH key provisioning response."""
    requestId: str
    endpoint: str
    summary: SSHSummary
    results: List[SSHResult]
    errors: List[str]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health status.

    Returns health status of all service components.
    """
    request_id = str(uuid.uuid4())

    checks = [
        HealthCheck(name="discovery_service", status="healthy", message=None),
        HealthCheck(name="ssh_key_manager", status="healthy", message=None)
    ]

    return HealthResponse(
        requestId=request_id,
        endpoint="/v1/health",
        status="healthy",
        version="1.0.0",
        checks=checks,
        errors=[]
    )


@router.get("/discovery", response_model=DiscoveryResponse)
async def discover_devices(
    timeout_ms: int = Query(default=5000, ge=250, le=15000),
    ip_subnet: Optional[str] = None,
    ip_range_start: Optional[str] = None,
    ip_range_end: Optional[str] = None,
    include_txt: bool = False
):
    """
    Scan network for EdgeSA devices via mDNS.

    Returns list of discovered devices with their configuration.
    In simulation mode, returns the pre-configured device list.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    # Simulate scan delay (proportional to timeout but capped)
    scan_delay = min(timeout_ms / 1000 * 0.1, 0.5)
    time.sleep(scan_delay)

    # Get devices from state manager
    all_devices = state_manager.get_all_devices()

    # Filter by subnet or IP range if specified
    filtered_devices = all_devices
    if ip_subnet:
        # Simple subnet filtering (for simulation, just check prefix)
        subnet_prefix = ip_subnet.rsplit('.', 1)[0]
        filtered_devices = [d for d in all_devices if d.ip.startswith(subnet_prefix)]
    elif ip_range_start and ip_range_end:
        # Range filtering
        def ip_to_int(ip):
            parts = ip.split('.')
            return sum(int(p) << (24 - 8 * i) for i, p in enumerate(parts))

        start_int = ip_to_int(ip_range_start)
        end_int = ip_to_int(ip_range_end)
        filtered_devices = [
            d for d in all_devices
            if start_int <= ip_to_int(d.ip) <= end_int
        ]

    # Build response devices
    devices = []
    for device in filtered_devices:
        # Determine stream ports based on device type
        if device.device_type == "smartcluster":
            mjpeg_ports = [8080, 9080]
            mpegts_ports = [5005]
        elif device.device_type == "threecluster":
            mjpeg_ports = [8079, 8080, 8081, 9079, 9080, 9081]
            mpegts_ports = [5004, 5005, 5006]
        else:  # aicentral
            mjpeg_ports = []
            mpegts_ports = []

        discovered = DiscoveredDevice(
            id=device.hostname.replace(".local", ""),
            deviceType=device.device_type,
            hostname=f"{device.hostname}.local" if not device.hostname.endswith(".local") else device.hostname,
            ipAddress=device.ip,
            addresses=[device.ip],
            streams=DeviceStreams(mjpeg=mjpeg_ports, mpegts=mpegts_ports),
            txt={"mode": device.mode, "ssh_configured": str(device.ssh_configured).lower()} if include_txt else None,
            discoveredAt=datetime.utcnow().isoformat() + "Z"
        )
        devices.append(discovered)

    scan_time_ms = int((time.time() - start_time) * 1000)

    logger.info(f"Discovery scan completed: {len(devices)} devices found in {scan_time_ms}ms")

    return DiscoveryResponse(
        requestId=request_id,
        endpoint="/v1/discovery",
        scanTimeMs=scan_time_ms,
        totalDevices=len(devices),
        devices=devices,
        errors=[]
    )


@router.post("/ssh-keys", response_model=SSHKeysResponse)
async def provision_ssh_keys(request: SSHKeysRequest):
    """
    Install SSH public key on target devices.

    Simulates the SSH key provisioning flow including:
    - Connection attempt
    - Key-based auth check
    - Password auth fallback
    - Key installation
    - Verification
    """
    request_id = str(uuid.uuid4())

    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    credentials = request.credentials or SSHCredentials()
    options = request.options or SSHOptions()

    results: List[SSHResult] = []
    key_installed = 0
    already_configured = 0
    failed = 0

    for target in request.targets:
        device = state_manager.get_device(target.ip)

        # Check scenario for failure injection
        if active_scenario == "partial_ssh_failure" and target.ip.endswith(".99"):
            # Simulate unreachable device
            results.append(SSHResult(
                target=target.ip,
                status="failed",
                error=SSHError(
                    code="SSH_UNREACHABLE",
                    message=f"Cannot connect to {target.ip}: Connection timed out"
                ),
                steps=[
                    SSHStep(name="resolve", status="ok", durationMs=0),
                    SSHStep(name="connect", status="failed", durationMs=options.connectTimeoutMs)
                ]
            ))
            failed += 1
            continue

        if device is None:
            # Device not in our list - simulate connection failure
            results.append(SSHResult(
                target=target.ip,
                status="failed",
                error=SSHError(
                    code="SSH_UNREACHABLE",
                    message=f"Cannot connect to {target.ip}: Connection refused"
                ),
                steps=[
                    SSHStep(name="resolve", status="ok", durationMs=0),
                    SSHStep(name="connect", status="failed", durationMs=options.connectTimeoutMs)
                ]
            ))
            failed += 1
            continue

        if device.ssh_configured:
            # Already has key - simulate quick success
            results.append(SSHResult(
                target=target.ip,
                status="already_configured",
                error=None,
                steps=[
                    SSHStep(name="resolve", status="ok", durationMs=0),
                    SSHStep(name="connect", status="ok", durationMs=110),
                    SSHStep(name="key_auth", status="ok", durationMs=0)
                ]
            ))
            already_configured += 1
        else:
            # Simulate full provisioning flow
            # Check credentials (for simulation, accept default nvidia/nvidia)
            if credentials.username == "nvidia" and credentials.password == "nvidia":
                # Successful provisioning
                state_manager.set_ssh_configured(target.ip, True)
                results.append(SSHResult(
                    target=target.ip,
                    status="key_installed",
                    error=None,
                    steps=[
                        SSHStep(name="resolve", status="ok", durationMs=0),
                        SSHStep(name="connect", status="ok", durationMs=120),
                        SSHStep(name="key_auth", status="failed", durationMs=0),
                        SSHStep(name="password_auth", status="ok", durationMs=210),
                        SSHStep(name="install_key", status="ok", durationMs=95),
                        SSHStep(name="verify_key_auth", status="ok", durationMs=160)
                    ]
                ))
                key_installed += 1
            else:
                # Wrong password
                results.append(SSHResult(
                    target=target.ip,
                    status="failed",
                    error=SSHError(
                        code="PASSWORD_AUTH_FAILED",
                        message=f"Password authentication failed for {credentials.username}@{target.ip}"
                    ),
                    steps=[
                        SSHStep(name="resolve", status="ok", durationMs=0),
                        SSHStep(name="connect", status="ok", durationMs=115),
                        SSHStep(name="key_auth", status="failed", durationMs=0),
                        SSHStep(name="password_auth", status="failed", durationMs=180)
                    ]
                ))
                failed += 1

    summary = SSHSummary(
        totalTargets=len(request.targets),
        keyInstalled=key_installed,
        alreadyConfigured=already_configured,
        failed=failed
    )

    logger.info(f"SSH provisioning: {key_installed} installed, {already_configured} existing, {failed} failed")

    return SSHKeysResponse(
        requestId=request_id,
        endpoint="/v1/ssh-keys",
        summary=summary,
        results=results,
        errors=[]
    )

"""
Mock Device API

Simulates the Edge360 Device REST API endpoints:
- GET/POST /api/pipeline_mode - Mode management (threecluster only)
- GET/POST /api/manager_connection - Worker configuration
- GET/POST /api/sensors_config - Manager configuration
- GET /api/all_sensor_urls - Sensor URL map
- GET /config/streams/{slot}/stream-id - Stream ID lookup
- POST /config/position - Position setting

The device IP is passed via X-Device-IP header or as a path prefix.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Path
from pydantic import BaseModel, Field

from .device_state import DeviceStateManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["device"])

# Will be set by server.py
state_manager: Optional[DeviceStateManager] = None


def set_state_manager(manager: DeviceStateManager) -> None:
    """Set the device state manager instance."""
    global state_manager
    state_manager = manager


def get_device_ip(x_device_ip: Optional[str] = None) -> str:
    """Extract device IP from header."""
    if not x_device_ip:
        raise HTTPException(
            status_code=400,
            detail="X-Device-IP header is required"
        )
    return x_device_ip


# =============================================================================
# Request/Response Models
# =============================================================================

class PipelineModeRequest(BaseModel):
    """Request to set pipeline mode."""
    mode: str = Field(..., pattern="^(worker|manager)$")


class PipelineModeResponse(BaseModel):
    """Pipeline mode response."""
    status: str
    mode: str
    previous_mode: Optional[str] = None
    available_modes: list


class ManagerConnectionRequest(BaseModel):
    """Request to set manager connection."""
    manager_ip: str


class ManagerConnectionResponse(BaseModel):
    """Manager connection response."""
    status: str
    manager_ip: Optional[str] = None
    updated_components: Optional[list] = None
    components: Optional[list] = None


class SensorConfig(BaseModel):
    """Single sensor configuration."""
    sensor_name: str
    endpoint: str
    port: int = Field(..., ge=1, le=65535)
    type: str = Field(default="vis", pattern="^(vis|thermal|day|night)$")


class SensorsConfigResponse(BaseModel):
    """Sensors configuration response."""
    status: str
    sensors_config: Optional[Dict[str, Any]] = None
    worker_urls: Optional[list] = None


class StreamIdResponse(BaseModel):
    """Stream ID response."""
    status: str
    slot: Optional[int] = None
    stream_id: Optional[str] = None


class PositionRequest(BaseModel):
    """Request to set position."""
    position: str


class PositionResponse(BaseModel):
    """Position response."""
    status: str
    position: Optional[str] = None


class HealthResponse(BaseModel):
    """Device health response."""
    status: str
    device_type: Optional[str] = None
    mode: Optional[str] = None
    uptime_seconds: int = 12345


# =============================================================================
# Pipeline Mode Endpoints (threecluster only)
# =============================================================================

@router.get("/api/pipeline_mode", response_model=PipelineModeResponse)
async def get_pipeline_mode(x_device_ip: str = Header(...)):
    """
    Get the current operating mode of the pipeline instance.

    Only available on threecluster devices.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    device = state_manager.get_device(x_device_ip)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device {x_device_ip} not found")

    return PipelineModeResponse(
        status="success",
        mode=device.mode,
        available_modes=device.get_available_modes()
    )


@router.post("/api/pipeline_mode", response_model=PipelineModeResponse)
async def set_pipeline_mode(
    request: PipelineModeRequest,
    x_device_ip: str = Header(...)
):
    """
    Switch the pipeline between manager and worker modes.

    Only available on threecluster devices.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.set_mode(x_device_ip, request.mode)

    if result.get("status") == "error":
        detail = result.get("detail", "Mode switch failed")
        if "cannot switch" in detail.lower():
            raise HTTPException(status_code=400, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    return PipelineModeResponse(
        status=result["status"],
        mode=result["mode"],
        previous_mode=result.get("previous_mode"),
        available_modes=result["available_modes"]
    )


# =============================================================================
# Manager Connection Endpoints (worker mode only)
# =============================================================================

@router.get("/api/manager_connection", response_model=ManagerConnectionResponse)
async def get_manager_connection(x_device_ip: str = Header(...)):
    """
    Get the current manager IP address.

    Only available in worker mode.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.get_manager_connection(x_device_ip)

    if result.get("status") == "error":
        detail = result.get("detail", "Failed to get manager connection")
        if "worker mode" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    return ManagerConnectionResponse(
        status=result["status"],
        manager_ip=result.get("manager_ip"),
        components=result.get("components")
    )


@router.post("/api/manager_connection", response_model=ManagerConnectionResponse)
async def set_manager_connection(
    request: ManagerConnectionRequest,
    x_device_ip: str = Header(...)
):
    """
    Set the manager IP address for worker communication.

    Only available in worker mode.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.set_manager_connection(x_device_ip, request.manager_ip)

    if result.get("status") == "error":
        detail = result.get("detail", "Failed to set manager connection")
        if "worker mode" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    return ManagerConnectionResponse(
        status=result["status"],
        manager_ip=result.get("manager_ip"),
        updated_components=result.get("updated_components")
    )


# =============================================================================
# Sensors Config Endpoints (manager mode only)
# =============================================================================

@router.get("/api/sensors_config", response_model=SensorsConfigResponse)
async def get_sensors_config(x_device_ip: str = Header(...)):
    """
    Get the current sensor configuration.

    Only available in manager mode.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.get_sensors_config(x_device_ip)

    if result.get("status") == "error":
        detail = result.get("detail", "Failed to get sensors config")
        if "manager mode" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    return SensorsConfigResponse(
        status=result["status"],
        sensors_config=result.get("sensors_config"),
        worker_urls=result.get("worker_urls")
    )


@router.post("/api/sensors_config", response_model=SensorsConfigResponse)
async def set_sensors_config(
    sensors_config: Dict[str, SensorConfig],
    x_device_ip: str = Header(...)
):
    """
    Set the sensor configuration.

    Only available in manager mode.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    # Convert to dict format expected by state manager
    config_dict = {
        sensor_id: {
            "sensor_name": sensor.sensor_name,
            "endpoint": sensor.endpoint,
            "port": sensor.port,
            "type": sensor.type
        }
        for sensor_id, sensor in sensors_config.items()
    }

    result = state_manager.set_sensors_config(x_device_ip, config_dict)

    if result.get("status") == "error":
        detail = result.get("detail", "Failed to set sensors config")
        if "manager mode" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    return SensorsConfigResponse(
        status=result["status"],
        sensors_config=result.get("sensors_config"),
        worker_urls=result.get("worker_urls")
    )


@router.get("/api/all_sensor_urls")
async def get_all_sensor_urls(x_device_ip: str = Header(...)):
    """
    Get all sensor URLs in simplified format.

    Only available in manager mode.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.get_sensors_config(x_device_ip)

    if result.get("status") == "error":
        detail = result.get("detail", "Failed to get sensor URLs")
        if "manager mode" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    # Transform to simplified URL format
    sensor_urls = {}
    for sensor_id, sensor_config in result.get("sensors_config", {}).items():
        sensor_name = sensor_config.get("sensor_name", sensor_id)
        sensor_urls[sensor_name] = {
            "sensor_id": sensor_id,
            "url": f"{sensor_config['endpoint']}:{sensor_config['port']}"
        }

    return sensor_urls


# =============================================================================
# Stream Configuration Endpoints
# =============================================================================

@router.get("/config/streams/{slot}/stream-id", response_model=StreamIdResponse)
async def get_stream_id(
    slot: int = Path(..., ge=0, le=2),
    x_device_ip: str = Header(...)
):
    """
    Get the stream ID for a specific camera slot.

    Available on worker devices (smartcluster: slot 0, threecluster: slots 0-2).
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.get_stream_id(x_device_ip, slot)

    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("detail"))

    return StreamIdResponse(
        status=result["status"],
        slot=result.get("slot"),
        stream_id=result.get("stream_id")
    )


# =============================================================================
# Position Endpoint
# =============================================================================

@router.post("/config/position", response_model=PositionResponse)
async def set_position(
    request: PositionRequest,
    x_device_ip: str = Header(...)
):
    """
    Set the camera position for a worker device.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    device = state_manager.get_device(x_device_ip)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device {x_device_ip} not found")

    if device.mode != "worker":
        raise HTTPException(
            status_code=503,
            detail=f"Endpoint only available in worker mode. Current mode: {device.mode}"
        )

    state_manager.update_device(x_device_ip, position=request.position)

    logger.info(f"Device {x_device_ip} position set to: {request.position}")

    return PositionResponse(
        status="success",
        position=request.position
    )


@router.get("/config/position", response_model=PositionResponse)
async def get_position(x_device_ip: str = Header(...)):
    """
    Get the camera position for a worker device.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    device = state_manager.get_device(x_device_ip)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device {x_device_ip} not found")

    return PositionResponse(
        status="success",
        position=device.position
    )


# =============================================================================
# Health Endpoint
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check(x_device_ip: str = Header(...)):
    """
    Device health check endpoint.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    device = state_manager.get_device(x_device_ip)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device {x_device_ip} not found")

    return HealthResponse(
        status="healthy",
        device_type=device.device_type,
        mode=device.mode,
        uptime_seconds=12345
    )


# =============================================================================
# SFTP Simulation Endpoints (for testing without real SFTP)
# =============================================================================

class FileWriteRequest(BaseModel):
    """Request to write a file."""
    path: str
    content: str


class FileResponse(BaseModel):
    """File operation response."""
    status: str
    path: Optional[str] = None
    content: Optional[str] = None
    detail: Optional[str] = None


@router.post("/sftp/write", response_model=FileResponse)
async def sftp_write_file(
    request: FileWriteRequest,
    x_device_ip: str = Header(...)
):
    """
    Simulate SFTP file write operation.

    This endpoint allows testing file writes without a real SFTP connection.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.write_file(x_device_ip, request.path, request.content)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("detail"))

    return FileResponse(
        status=result["status"],
        path=result.get("path")
    )


@router.get("/sftp/read", response_model=FileResponse)
async def sftp_read_file(
    path: str,
    x_device_ip: str = Header(...)
):
    """
    Simulate SFTP file read operation.

    This endpoint allows testing file reads without a real SFTP connection.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.read_file(x_device_ip, path)

    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("detail"))

    return FileResponse(
        status=result["status"],
        path=result.get("path"),
        content=result.get("content")
    )


@router.get("/sftp/list")
async def sftp_list_files(
    path: str = "/etc/axon",
    x_device_ip: str = Header(...)
):
    """
    Simulate SFTP directory listing.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    result = state_manager.list_files(x_device_ip, path)

    return result


# =============================================================================
# Video Stream Simulation Endpoints
# =============================================================================
# These endpoints mock the camera's video streaming API.
# In simulation mode, these return success without actually starting a stream.
# The real video comes from the actual camera's existing stream.

class StreamStartRequest(BaseModel):
    """Request to start video stream."""
    host: str = "239.255.0.1"
    port: int = 5010
    bitrate: int = 4000000


class StreamResponse(BaseModel):
    """Video stream response."""
    status: str
    message: Optional[str] = None


@router.post("/api/stream/start", response_model=StreamResponse)
async def start_stream(
    request: StreamStartRequest,
    x_device_ip: str = Header(...)
):
    """
    Mock endpoint to start camera video streaming.

    In simulation mode, this returns success immediately.
    The actual video stream should already be available from the real camera.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    device = state_manager.get_device(x_device_ip)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device {x_device_ip} not found")

    logger.info(
        f"[MOCK] Stream start requested for {x_device_ip}: "
        f"host={request.host}, port={request.port}, bitrate={request.bitrate}"
    )

    return StreamResponse(
        status="success",
        message=f"Stream started (mock) - connect to {request.host}:{request.port}"
    )


@router.post("/api/stream/stop", response_model=StreamResponse)
async def stop_stream(x_device_ip: str = Header(...)):
    """
    Mock endpoint to stop camera video streaming.

    In simulation mode, this returns success immediately.
    """
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    device = state_manager.get_device(x_device_ip)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device {x_device_ip} not found")

    logger.info(f"[MOCK] Stream stop requested for {x_device_ip}")

    return StreamResponse(
        status="success",
        message="Stream stopped (mock)"
    )

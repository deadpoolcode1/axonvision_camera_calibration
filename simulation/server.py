"""
Mock Server Runner

Runs all simulation services:
- Discovery Service (FastAPI) on configurable port (default: 8000)
- Device API (FastAPI) on configurable port (default: 5000)
- SFTP Server (paramiko) on configurable port (default: 2222)

Usage:
    # Run from command line
    python -m simulation.server

    # Or programmatically
    from simulation.server import start_mock_server
    start_mock_server()
"""

import argparse
import asyncio
import logging
import os
import signal
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Server state
_mock_server_running = False
_shutdown_event = threading.Event()


def is_mock_server_running() -> bool:
    """Check if the mock server is currently running."""
    return _mock_server_running


def load_simulation_config() -> Dict[str, Any]:
    """Load simulation configuration from edgesa_settings.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "edgesa_settings.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get("simulation", get_default_config())
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default simulation configuration."""
    return {
        "enabled": True,
        "mock_server": {
            "host": "127.0.0.1",
            "discovery_port": 8000,
            "device_api_port": 5000,
            "sftp_port": 2222
        },
        "devices": [
            {"ip": "192.168.1.10", "hostname": "smartcluster-001", "type": "smartcluster"},
            {"ip": "192.168.1.11", "hostname": "smartcluster-002", "type": "smartcluster"},
            {"ip": "192.168.1.20", "hostname": "threecluster-001", "type": "threecluster", "initial_mode": "worker"},
            {"ip": "192.168.1.30", "hostname": "aicentral-001", "type": "aicentral"},
        ],
        "persistence": {
            "enabled": True,
            "state_file": "./mock_data/device_states.json",
            "filesystem_path": "./mock_data/devices"
        },
        "scenarios": {
            "active_scenario": "happy_path"
        }
    }


def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def print_banner(config: Dict[str, Any], state_manager=None):
    """Print startup banner with server information."""
    mock_server = config.get("mock_server", {})
    host = mock_server.get("host", "127.0.0.1")
    discovery_port = mock_server.get("discovery_port", 8000)
    device_api_port = mock_server.get("device_api_port", 5000)
    sftp_port = mock_server.get("sftp_port", 2222)

    banner = f"""
================================================================================
                    EDGESA DEVICE SIMULATION SERVER
================================================================================

  SIMULATION MODE ACTIVE - Using mock device APIs

  Services:
    Discovery API:  http://{host}:{discovery_port}/v1/
    Device API:     http://{host}:{device_api_port}/
    SFTP Server:    sftp://{host}:{sftp_port}/

  Simulated Devices:
"""

    # Get devices from state manager if available (actual user-configured cameras)
    if state_manager:
        devices = state_manager.get_all_devices()
        if devices:
            for device in devices:
                banner += f"    - {device.hostname} ({device.device_type}) @ {device.ip} [{device.mode}]\n"
        else:
            banner += "    (No devices configured - add cameras in the UI)\n"
    else:
        # Fallback to config
        devices = config.get("devices", [])
        for device in devices:
            device_type = device.get("type", "unknown")
            hostname = device.get("hostname", "unknown")
            ip = device.get("ip", "unknown")
            mode = device.get("initial_mode", "worker" if device_type != "aicentral" else "manager")
            banner += f"    - {hostname} ({device_type}) @ {ip} [{mode}]\n"

    banner += f"""
  Data Directory:  {config.get('persistence', {}).get('filesystem_path', './mock_data/devices')}

  Press Ctrl+C to stop the server.
================================================================================
"""
    print(banner)


def run_fastapi_server(app, host: str, port: int, name: str):
    """Run a FastAPI application with uvicorn."""
    try:
        import uvicorn
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False
        )
        server = uvicorn.Server(config)
        logger.info(f"{name} starting on {host}:{port}")
        server.run()
    except Exception as e:
        logger.error(f"{name} failed: {e}")


def create_discovery_app(state_manager):
    """Create the FastAPI app for the discovery service."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    import json
    import time as time_module

    from .mock_discovery import router as discovery_router
    from . import mock_discovery

    app = FastAPI(
        title="EdgeSA Mock Discovery Service",
        description="Simulates the Hardware Discovery & Provisioning Service",
        version="1.0.0"
    )

    # Debug logging middleware (DEBUG level)
    class DebugLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start_time = time_module.time()

            # Log incoming request
            body = b""
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()

            # Format request log
            req_log = f"\n{'='*60}\n"
            req_log += f"📥 DISCOVERY API - INCOMING REQUEST\n"
            req_log += f"{'='*60}\n"
            req_log += f"  Method: {request.method}\n"
            req_log += f"  Path:   {request.url.path}\n"
            req_log += f"  Query:  {dict(request.query_params)}\n"

            if body:
                try:
                    body_json = json.loads(body)
                    req_log += f"  Body:   {json.dumps(body_json, indent=4)}\n"
                except:
                    req_log += f"  Body:   {body.decode('utf-8', errors='replace')}\n"

            logger.debug(req_log)

            # Get response
            response = await call_next(request)

            duration = (time_module.time() - start_time) * 1000

            # Log response
            resp_log = f"\n{'='*60}\n"
            resp_log += f"📤 DISCOVERY API - OUTGOING RESPONSE\n"
            resp_log += f"{'='*60}\n"
            resp_log += f"  Status:   {response.status_code}\n"
            resp_log += f"  Duration: {duration:.2f}ms\n"
            resp_log += f"{'='*60}\n"

            logger.debug(resp_log)

            return response

    app.add_middleware(DebugLoggingMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set state manager
    mock_discovery.set_state_manager(state_manager)

    app.include_router(discovery_router)

    @app.get("/")
    async def root():
        return {
            "service": "EdgeSA Mock Discovery Service",
            "version": "1.0.0",
            "simulation": True,
            "endpoints": ["/v1/health", "/v1/discovery", "/v1/ssh-keys"]
        }

    return app


def create_device_api_app(state_manager):
    """Create the FastAPI app for the device API."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    import json
    import time as time_module

    from .mock_device_api import router as device_router
    from . import mock_device_api

    app = FastAPI(
        title="EdgeSA Mock Device API",
        description="Simulates the Edge360 Device REST API",
        version="1.0.0"
    )

    # Debug logging middleware (DEBUG level)
    class DebugLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start_time = time_module.time()

            # Log incoming request
            body = b""
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()

            # Get device IP from header
            device_ip = request.headers.get("X-Device-IP", "N/A")

            # Format request log
            req_log = f"\n{'='*60}\n"
            req_log += f"📥 DEVICE API - INCOMING REQUEST\n"
            req_log += f"{'='*60}\n"
            req_log += f"  Method:      {request.method}\n"
            req_log += f"  Path:        {request.url.path}\n"
            req_log += f"  Target IP:   {device_ip}\n"
            req_log += f"  Query:       {dict(request.query_params)}\n"

            # Log important headers
            important_headers = {k: v for k, v in request.headers.items()
                               if k.lower() in ['x-device-ip', 'content-type']}
            req_log += f"  Headers:     {important_headers}\n"

            if body:
                try:
                    body_json = json.loads(body)
                    req_log += f"  Body:\n{json.dumps(body_json, indent=4)}\n"
                except:
                    req_log += f"  Body: {body.decode('utf-8', errors='replace')}\n"

            logger.debug(req_log)

            # Get response
            response = await call_next(request)

            duration = (time_module.time() - start_time) * 1000

            # Log response
            resp_log = f"\n{'='*60}\n"
            resp_log += f"📤 DEVICE API - OUTGOING RESPONSE\n"
            resp_log += f"{'='*60}\n"
            resp_log += f"  Target IP:  {device_ip}\n"
            resp_log += f"  Status:     {response.status_code}\n"
            resp_log += f"  Duration:   {duration:.2f}ms\n"
            resp_log += f"{'='*60}\n"

            logger.debug(resp_log)

            return response

    app.add_middleware(DebugLoggingMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set state manager
    mock_device_api.set_state_manager(state_manager)

    app.include_router(device_router)

    @app.get("/")
    async def root():
        return {
            "service": "EdgeSA Mock Device API",
            "version": "1.0.0",
            "simulation": True,
            "note": "Use X-Device-IP header to specify target device"
        }

    return app


def start_mock_server(
    config: Optional[Dict[str, Any]] = None,
    blocking: bool = True
) -> Optional[threading.Thread]:
    """
    Start the mock server with all services.

    Args:
        config: Simulation configuration (loads from file if not provided)
        blocking: If True, blocks until Ctrl+C. If False, runs in background thread.

    Returns:
        If blocking=False, returns the server thread. Otherwise None.
    """
    global _mock_server_running

    if config is None:
        config = load_simulation_config()

    mock_server = config.get("mock_server", {})
    host = mock_server.get("host", "127.0.0.1")
    discovery_port = mock_server.get("discovery_port", 8000)
    device_api_port = mock_server.get("device_api_port", 5000)
    sftp_port = mock_server.get("sftp_port", 2222)

    persistence = config.get("persistence", {})
    state_file = persistence.get("state_file") if persistence.get("enabled") else None
    fs_path = persistence.get("filesystem_path", "./mock_data/devices")

    devices_config = config.get("devices", [])

    # Check port availability
    ports_to_check = [
        (discovery_port, "Discovery API"),
        (device_api_port, "Device API"),
    ]

    for port, name in ports_to_check:
        if not check_port_available(host, port):
            logger.error(f"Port {port} ({name}) is already in use")
            print(f"ERROR: Port {port} ({name}) is already in use. Is another instance running?")
            return None

    # Initialize device state manager
    # If state file exists, use it (dynamic devices from UI take priority)
    # Otherwise fall back to devices from YAML config
    from .device_state import DeviceStateManager
    from pathlib import Path as PathLib

    # Check if we have persisted state
    has_persisted_state = state_file and PathLib(state_file).exists()

    if has_persisted_state:
        # Load from persisted state - these are the user's actual camera configs
        state_manager = DeviceStateManager(
            state_file=state_file,
            filesystem_path=fs_path,
            initial_devices=None  # Don't use YAML devices, use saved state
        )
        logger.info(f"Loaded devices from persisted state: {state_file}")

        # If no devices in state file, fall back to YAML config
        if not state_manager.get_all_devices():
            logger.info("No devices in state file, using YAML config as fallback")
            state_manager = DeviceStateManager(
                state_file=state_file,
                filesystem_path=fs_path,
                initial_devices=devices_config
            )
    else:
        # No persisted state - use YAML config as initial devices
        state_manager = DeviceStateManager(
            state_file=state_file,
            filesystem_path=fs_path,
            initial_devices=devices_config
        )

    # Create FastAPI apps
    discovery_app = create_discovery_app(state_manager)
    device_api_app = create_device_api_app(state_manager)

    def run_servers():
        global _mock_server_running
        _mock_server_running = True

        threads = []

        # Start Discovery API
        discovery_thread = threading.Thread(
            target=run_fastapi_server,
            args=(discovery_app, host, discovery_port, "Discovery API"),
            daemon=True
        )
        discovery_thread.start()
        threads.append(discovery_thread)

        # Start Device API
        device_api_thread = threading.Thread(
            target=run_fastapi_server,
            args=(device_api_app, host, device_api_port, "Device API"),
            daemon=True
        )
        device_api_thread.start()
        threads.append(device_api_thread)

        # Start SFTP server (optional, requires paramiko)
        try:
            from .mock_sftp import MockSFTPServerRunner
            sftp_server = MockSFTPServerRunner(
                state_manager=state_manager,
                host=host,
                port=sftp_port,
                fs_root=fs_path
            )
            sftp_server.start()
        except Exception as e:
            logger.warning(f"SFTP server not started: {e}")
            print(f"Note: SFTP server not started ({e}). Use HTTP endpoints instead.")

        # Wait for shutdown
        while _mock_server_running and not _shutdown_event.is_set():
            time.sleep(0.5)

    if blocking:
        print_banner(config, state_manager)

        def signal_handler(sig, frame):
            global _mock_server_running
            print("\nShutting down mock server...")
            _mock_server_running = False
            _shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        run_servers()
        return None
    else:
        thread = threading.Thread(target=run_servers, daemon=True)
        thread.start()
        time.sleep(1)  # Give servers time to start
        return thread


def stop_mock_server():
    """Stop the mock server."""
    global _mock_server_running
    _mock_server_running = False
    _shutdown_event.set()


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="EdgeSA Device Simulation Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m simulation.server                    # Run with default config
  python -m simulation.server --discovery-port 8001
  python -m simulation.server --reset           # Reset device state
"""
    )

    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from config or 127.0.0.1)"
    )
    parser.add_argument(
        "--discovery-port",
        type=int,
        default=None,
        help="Port for discovery service (default: from config or 8000)"
    )
    parser.add_argument(
        "--device-api-port",
        type=int,
        default=None,
        help="Port for device API (default: from config or 5000)"
    )
    parser.add_argument(
        "--sftp-port",
        type=int,
        default=None,
        help="Port for SFTP server (default: from config or 2222)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset device state to initial configuration"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging (DEBUG enabled by default)
    log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load config and apply overrides
    config = load_simulation_config()

    if args.host:
        config.setdefault("mock_server", {})["host"] = args.host
    if args.discovery_port:
        config.setdefault("mock_server", {})["discovery_port"] = args.discovery_port
    if args.device_api_port:
        config.setdefault("mock_server", {})["device_api_port"] = args.device_api_port
    if args.sftp_port:
        config.setdefault("mock_server", {})["sftp_port"] = args.sftp_port

    # Reset state if requested
    if args.reset:
        from .device_state import DeviceStateManager
        persistence = config.get("persistence", {})
        state_file = persistence.get("state_file")
        fs_path = persistence.get("filesystem_path", "./mock_data/devices")

        if state_file and Path(state_file).exists():
            Path(state_file).unlink()
            print(f"Removed state file: {state_file}")

        if fs_path and Path(fs_path).exists():
            import shutil
            shutil.rmtree(fs_path)
            print(f"Removed filesystem: {fs_path}")

        print("Device state reset complete.")

    # Start the server
    start_mock_server(config, blocking=True)


if __name__ == "__main__":
    main()

"""
Mock SFTP Server

Provides a simulated SFTP server for testing file operations on virtual devices.
Uses paramiko to implement the SSH/SFTP protocol.

Each device IP has its own virtual filesystem stored locally.
"""

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import paramiko - it's optional for the mock server
try:
    import paramiko
    from paramiko import (
        ServerInterface, SFTPServerInterface, SFTPServer,
        SFTPHandle, SFTPAttributes, SFTP_OK, SFTP_NO_SUCH_FILE,
        SFTP_PERMISSION_DENIED, AUTH_SUCCESSFUL, AUTH_FAILED,
        OPEN_SUCCEEDED, RSAKey
    )
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger.warning("paramiko not installed - SFTP simulation will use HTTP fallback")


from .device_state import DeviceStateManager


class MockSFTPHandle(SFTPHandle if PARAMIKO_AVAILABLE else object):
    """Handle for an open file in the mock SFTP server."""

    def __init__(self, file_path: Path, flags: int = 0):
        if PARAMIKO_AVAILABLE:
            super().__init__(flags)
        self.file_path = file_path
        self.flags = flags
        self._file = None

        # Determine mode based on flags
        if flags & os.O_WRONLY or flags & os.O_RDWR:
            if flags & os.O_APPEND:
                mode = 'ab'
            elif flags & os.O_TRUNC:
                mode = 'wb'
            else:
                mode = 'r+b' if file_path.exists() else 'wb'
        else:
            mode = 'rb'

        # Create parent directories if writing
        if 'w' in mode or 'a' in mode:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._file = open(file_path, mode)
        except Exception as e:
            logger.error(f"Failed to open {file_path}: {e}")
            raise

    def close(self):
        if self._file:
            self._file.close()
        if PARAMIKO_AVAILABLE:
            super().close()

    def read(self, offset: int, length: int) -> bytes:
        self._file.seek(offset)
        return self._file.read(length)

    def write(self, offset: int, data: bytes) -> int:
        self._file.seek(offset)
        self._file.write(data)
        self._file.flush()
        return SFTP_OK if PARAMIKO_AVAILABLE else 0

    def stat(self):
        if PARAMIKO_AVAILABLE:
            return SFTPAttributes.from_stat(os.stat(str(self.file_path)))
        return None


class MockSFTPServer(SFTPServerInterface if PARAMIKO_AVAILABLE else object):
    """SFTP server interface that maps to virtual device filesystems."""

    def __init__(self, server, device_ip: str, fs_root: Path):
        if PARAMIKO_AVAILABLE:
            super().__init__(server)
        self.device_ip = device_ip
        self.fs_root = fs_root
        self.fs_root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"SFTP server initialized for {device_ip} at {fs_root}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a remote path to local filesystem path."""
        # Normalize and make relative
        clean_path = path.lstrip('/')
        resolved = self.fs_root / clean_path
        # Security check - ensure we don't escape the root
        try:
            resolved.resolve().relative_to(self.fs_root.resolve())
        except ValueError:
            raise PermissionError(f"Path escapes root: {path}")
        return resolved

    def list_folder(self, path: str):
        """List contents of a directory."""
        try:
            local_path = self._resolve_path(path)
            if not local_path.exists():
                return SFTP_NO_SUCH_FILE if PARAMIKO_AVAILABLE else []

            if not local_path.is_dir():
                return SFTP_NO_SUCH_FILE if PARAMIKO_AVAILABLE else []

            result = []
            for item in local_path.iterdir():
                if PARAMIKO_AVAILABLE:
                    attr = SFTPAttributes.from_stat(item.stat())
                    attr.filename = item.name
                    result.append(attr)
                else:
                    result.append({"name": item.name, "is_dir": item.is_dir()})

            return result
        except Exception as e:
            logger.error(f"list_folder error: {e}")
            return SFTP_PERMISSION_DENIED if PARAMIKO_AVAILABLE else []

    def stat(self, path: str):
        """Get file/directory attributes."""
        try:
            local_path = self._resolve_path(path)
            if not local_path.exists():
                return SFTP_NO_SUCH_FILE if PARAMIKO_AVAILABLE else None
            if PARAMIKO_AVAILABLE:
                return SFTPAttributes.from_stat(local_path.stat())
            return {"size": local_path.stat().st_size}
        except Exception as e:
            logger.error(f"stat error: {e}")
            return SFTP_PERMISSION_DENIED if PARAMIKO_AVAILABLE else None

    def lstat(self, path: str):
        """Get file/directory attributes (no symlink following)."""
        return self.stat(path)

    def open(self, path: str, flags: int, attr):
        """Open a file for reading or writing."""
        try:
            local_path = self._resolve_path(path)
            handle = MockSFTPHandle(local_path, flags)
            return handle
        except Exception as e:
            logger.error(f"open error for {path}: {e}")
            return SFTP_PERMISSION_DENIED if PARAMIKO_AVAILABLE else None

    def remove(self, path: str):
        """Remove a file."""
        try:
            local_path = self._resolve_path(path)
            if not local_path.exists():
                return SFTP_NO_SUCH_FILE if PARAMIKO_AVAILABLE else -1
            local_path.unlink()
            return SFTP_OK if PARAMIKO_AVAILABLE else 0
        except Exception as e:
            logger.error(f"remove error: {e}")
            return SFTP_PERMISSION_DENIED if PARAMIKO_AVAILABLE else -1

    def rename(self, old_path: str, new_path: str):
        """Rename/move a file."""
        try:
            old_local = self._resolve_path(old_path)
            new_local = self._resolve_path(new_path)
            if not old_local.exists():
                return SFTP_NO_SUCH_FILE if PARAMIKO_AVAILABLE else -1
            new_local.parent.mkdir(parents=True, exist_ok=True)
            old_local.rename(new_local)
            return SFTP_OK if PARAMIKO_AVAILABLE else 0
        except Exception as e:
            logger.error(f"rename error: {e}")
            return SFTP_PERMISSION_DENIED if PARAMIKO_AVAILABLE else -1

    def mkdir(self, path: str, attr):
        """Create a directory."""
        try:
            local_path = self._resolve_path(path)
            local_path.mkdir(parents=True, exist_ok=True)
            return SFTP_OK if PARAMIKO_AVAILABLE else 0
        except Exception as e:
            logger.error(f"mkdir error: {e}")
            return SFTP_PERMISSION_DENIED if PARAMIKO_AVAILABLE else -1

    def rmdir(self, path: str):
        """Remove a directory."""
        try:
            local_path = self._resolve_path(path)
            if not local_path.exists():
                return SFTP_NO_SUCH_FILE if PARAMIKO_AVAILABLE else -1
            local_path.rmdir()
            return SFTP_OK if PARAMIKO_AVAILABLE else 0
        except Exception as e:
            logger.error(f"rmdir error: {e}")
            return SFTP_PERMISSION_DENIED if PARAMIKO_AVAILABLE else -1


class MockSSHServer(ServerInterface if PARAMIKO_AVAILABLE else object):
    """SSH server interface for handling authentication."""

    def __init__(self, device_ip: str, fs_root: Path, username: str = "nvidia", password: str = "nvidia"):
        self.device_ip = device_ip
        self.fs_root = fs_root
        self.expected_username = username
        self.expected_password = password
        self.authenticated = False

    def check_auth_password(self, username: str, password: str) -> int:
        if username == self.expected_username and password == self.expected_password:
            self.authenticated = True
            logger.info(f"SFTP auth successful for {username}@{self.device_ip}")
            return AUTH_SUCCESSFUL if PARAMIKO_AVAILABLE else 0
        logger.warning(f"SFTP auth failed for {username}@{self.device_ip}")
        return AUTH_FAILED if PARAMIKO_AVAILABLE else -1

    def check_auth_publickey(self, username: str, key) -> int:
        # For simulation, accept any key for the expected username
        if username == self.expected_username:
            self.authenticated = True
            logger.info(f"SFTP pubkey auth successful for {username}@{self.device_ip}")
            return AUTH_SUCCESSFUL if PARAMIKO_AVAILABLE else 0
        return AUTH_FAILED if PARAMIKO_AVAILABLE else -1

    def check_channel_request(self, kind: str, chanid: int) -> int:
        if kind == 'session':
            return OPEN_SUCCEEDED if PARAMIKO_AVAILABLE else 0
        return -1

    def get_allowed_auths(self, username: str) -> str:
        return "password,publickey"


class MockSFTPServerRunner:
    """
    Runs the mock SFTP server in a background thread.

    Handles multiple device IPs by routing based on the target address
    the client connects to (though in practice, we use a single port
    and the device IP is inferred from the SSH session or header).
    """

    def __init__(
        self,
        state_manager: DeviceStateManager,
        host: str = "127.0.0.1",
        port: int = 2222,
        fs_root: str = "./mock_data/devices"
    ):
        self.state_manager = state_manager
        self.host = host
        self.port = port
        self.fs_root = Path(fs_root)
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._host_key = None

        if PARAMIKO_AVAILABLE:
            # Generate a host key for the server
            self._host_key = RSAKey.generate(2048)

    def start(self):
        """Start the SFTP server in a background thread."""
        if not PARAMIKO_AVAILABLE:
            logger.warning("paramiko not available - SFTP server not started")
            logger.info("Use HTTP endpoints at /sftp/write, /sftp/read instead")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Mock SFTP server started on {self.host}:{self.port}")

    def stop(self):
        """Stop the SFTP server."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Mock SFTP server stopped")

    def _run(self):
        """Main server loop."""
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(5)
            self._server_socket.settimeout(1.0)

            while self._running:
                try:
                    client_socket, addr = self._server_socket.accept()
                    logger.debug(f"SFTP connection from {addr}")
                    # Handle in a new thread
                    handler = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr),
                        daemon=True
                    )
                    handler.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger.error(f"SFTP accept error: {e}")

        except Exception as e:
            logger.error(f"SFTP server error: {e}")
        finally:
            if self._server_socket:
                self._server_socket.close()

    def _handle_client(self, client_socket: socket.socket, addr: tuple):
        """Handle an incoming SFTP client connection."""
        transport = None
        try:
            transport = paramiko.Transport(client_socket)
            transport.add_server_key(self._host_key)

            # For simulation, we'll use a default device IP
            # In a real scenario, you might negotiate this differently
            # For now, we'll use the first device in the list
            devices = self.state_manager.get_all_devices()
            if devices:
                device_ip = devices[0].ip
            else:
                device_ip = "default"

            fs_root = self.fs_root / device_ip
            server = MockSSHServer(device_ip, fs_root)

            try:
                transport.start_server(server=server)
            except paramiko.SSHException as e:
                logger.error(f"SSH negotiation failed: {e}")
                return

            channel = transport.accept(20)
            if channel is None:
                logger.warning("No channel opened")
                return

            # Start SFTP subsystem
            sftp_server = MockSFTPServer(server, device_ip, fs_root)
            SFTPServer(channel, 'sftp', sftp_server)

        except Exception as e:
            logger.error(f"SFTP client handler error: {e}")
        finally:
            if transport:
                transport.close()


# Convenience function for use without full SFTP server
def write_device_file(
    state_manager: DeviceStateManager,
    device_ip: str,
    remote_path: str,
    content: str
) -> bool:
    """
    Write a file to a device's virtual filesystem.

    This is the recommended way to write files when not using the full SFTP server.
    """
    result = state_manager.write_file(device_ip, remote_path, content)
    return result.get("status") == "success"


def read_device_file(
    state_manager: DeviceStateManager,
    device_ip: str,
    remote_path: str
) -> Optional[str]:
    """
    Read a file from a device's virtual filesystem.

    This is the recommended way to read files when not using the full SFTP server.
    """
    result = state_manager.read_file(device_ip, remote_path)
    if result.get("status") == "success":
        return result.get("content")
    return None

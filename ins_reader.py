#!/usr/bin/env python3
"""
INS (Inertial Navigation System) Serial Reader
===============================================

Reads real-time INS data from a VectorNav INS device via serial port.
Parses VNINS NMEA-style sentences to extract yaw, pitch, roll orientation data.

Protocol Format:
$VNINS,timestamp,status1,status2,yaw,pitch,roll,lat,lon,alt,velN,velE,velD,attUnc,posUnc,velUnc*checksum

Example:
$VNINS,000420.076257,0000,0080,-061.612,+010.071,+087.994,+00.00000000,+000.00000000,+00000.000,+000.000,+000.000,+000.000,99.99,99.99,9.999*52

Usage:
    # Terminal test:
    stty -F /dev/ttyUSB0 115200 raw -echo && timeout 5 cat /dev/ttyUSB0

    # Python:
    from ins_reader import INSSerialReader
    with INSSerialReader('/dev/ttyUSB0') as ins:
        data = ins.get_latest()
        print(f"Yaw: {data.yaw}, Pitch: {data.pitch}, Roll: {data.roll}")
"""

import time
import threading
import re
from dataclasses import dataclass
from typing import Optional, Callable
from queue import Queue, Empty

# Try to import serial, provide helpful message if not available
try:
    import serial
except ImportError:
    serial = None
    print("WARNING: pyserial not installed. Install with: pip install pyserial")


@dataclass
class INSReading:
    """A single INS reading with orientation and status information."""
    yaw: float          # Heading in degrees (0=North, +East)
    pitch: float        # Pitch in degrees (+nose down in NED)
    roll: float         # Roll in degrees (+right wing down in NED)
    timestamp: float    # System timestamp when reading was received
    ins_timestamp: str  # Original INS timestamp from the message
    status1: str        # INS status word 1
    status2: str        # INS status word 2
    att_uncertainty: float = 99.99  # Attitude uncertainty in degrees
    pos_uncertainty: float = 99.99  # Position uncertainty in meters
    vel_uncertainty: float = 9.999  # Velocity uncertainty in m/s
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0

    def is_valid(self) -> bool:
        """Check if INS data is valid (has reasonable values)."""
        # Check if attitude uncertainty is within reasonable range
        # 99.99 often indicates no valid solution
        return self.att_uncertainty < 90.0

    def is_aligned(self) -> bool:
        """Check if INS is aligned and has good attitude solution."""
        # Typically, attitude uncertainty < 1 degree indicates good alignment
        return self.att_uncertainty < 5.0


class INSSerialReader:
    """
    Serial port reader for VectorNav INS devices.

    Reads VNINS sentences continuously in a background thread and provides
    the latest valid INS reading on demand.
    """

    # Default serial configuration
    DEFAULT_PORT = '/dev/ttyUSB0'
    DEFAULT_BAUDRATE = 115200
    DEFAULT_TIMEOUT = 1.0

    # VNINS sentence pattern (NMEA-style)
    # $VNINS,timestamp,status1,status2,yaw,pitch,roll,lat,lon,alt,velN,velE,velD,attUnc,posUnc,velUnc*checksum
    VNINS_PATTERN = re.compile(
        r'\$VNINS,'
        r'([^,]+),'          # timestamp
        r'([0-9A-Fa-f]+),'   # status1 (hex)
        r'([0-9A-Fa-fE]+),'  # status2 (hex, can have E prefix)
        r'([+-]?\d+\.?\d*),' # yaw
        r'([+-]?\d+\.?\d*),' # pitch
        r'([+-]?\d+\.?\d*),' # roll
        r'([+-]?\d+\.?\d*),' # latitude
        r'([+-]?\d+\.?\d*),' # longitude
        r'([+-]?\d+\.?\d*),' # altitude
        r'([+-]?\d+\.?\d*),' # velN
        r'([+-]?\d+\.?\d*),' # velE
        r'([+-]?\d+\.?\d*),' # velD
        r'([+-]?\d+\.?\d*),' # attUnc
        r'([+-]?\d+\.?\d*),' # posUnc
        r'([+-]?\d+\.?\d*)'  # velUnc
        r'\*([0-9A-Fa-f]{2})'  # checksum
    )

    def __init__(self, port: str = None, baudrate: int = None, timeout: float = None):
        """
        Initialize the INS serial reader.

        Args:
            port: Serial port path (default: /dev/ttyUSB0)
            baudrate: Baud rate (default: 115200)
            timeout: Read timeout in seconds (default: 1.0)
        """
        self.port = port or self.DEFAULT_PORT
        self.baudrate = baudrate or self.DEFAULT_BAUDRATE
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_reading: Optional[INSReading] = None
        self._lock = threading.Lock()
        self._callbacks: list = []
        self._read_count = 0
        self._error_count = 0
        self._last_error = None

    def connect(self) -> bool:
        """
        Open the serial port and start the reader thread.

        Returns:
            True if connection successful, False otherwise.
        """
        if serial is None:
            self._last_error = "pyserial not installed"
            return False

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # Flush any pending data
            self._serial.reset_input_buffer()

            # Start reader thread
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

            return True

        except serial.SerialException as e:
            self._last_error = f"Serial error: {e}"
            return False
        except Exception as e:
            self._last_error = f"Connection error: {e}"
            return False

    def disconnect(self):
        """Stop the reader thread and close the serial port."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._serial and self._serial.is_open:
            self._serial.close()

        self._serial = None
        self._thread = None

    def is_connected(self) -> bool:
        """Check if connected and reading data."""
        return self._running and self._serial is not None and self._serial.is_open

    def get_latest(self) -> Optional[INSReading]:
        """
        Get the most recent INS reading.

        Returns:
            The latest INSReading, or None if no data available.
        """
        with self._lock:
            return self._latest_reading

    def wait_for_reading(self, timeout: float = 5.0) -> Optional[INSReading]:
        """
        Wait for a new INS reading.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            An INSReading, or None if timeout.
        """
        start = time.time()
        initial_count = self._read_count

        while time.time() - start < timeout:
            if self._read_count > initial_count:
                return self.get_latest()
            time.sleep(0.01)

        return None

    def wait_for_valid_reading(self, timeout: float = 10.0) -> Optional[INSReading]:
        """
        Wait for a valid INS reading (with reasonable uncertainty).

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            A valid INSReading, or None if timeout.
        """
        start = time.time()

        while time.time() - start < timeout:
            reading = self.get_latest()
            if reading and reading.is_valid():
                return reading
            time.sleep(0.05)

        return None

    def add_callback(self, callback: Callable[[INSReading], None]):
        """Add a callback to be called on each new reading."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[INSReading], None]):
        """Remove a previously added callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_stats(self) -> dict:
        """Get reader statistics."""
        return {
            'read_count': self._read_count,
            'error_count': self._error_count,
            'last_error': self._last_error,
            'connected': self.is_connected(),
            'has_data': self._latest_reading is not None
        }

    def _reader_loop(self):
        """Background thread that continuously reads INS data."""
        buffer = ""

        while self._running and self._serial and self._serial.is_open:
            try:
                # Read available data
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    buffer += data.decode('ascii', errors='ignore')

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if line.startswith('$VNINS'):
                            reading = self._parse_vnins(line)
                            if reading:
                                with self._lock:
                                    self._latest_reading = reading
                                self._read_count += 1

                                # Call registered callbacks
                                for callback in self._callbacks:
                                    try:
                                        callback(reading)
                                    except Exception:
                                        pass
                else:
                    time.sleep(0.001)  # Small delay if no data

            except serial.SerialException as e:
                self._error_count += 1
                self._last_error = f"Serial error: {e}"
                time.sleep(0.1)
            except Exception as e:
                self._error_count += 1
                self._last_error = f"Reader error: {e}"
                time.sleep(0.1)

    def _parse_vnins(self, line: str) -> Optional[INSReading]:
        """
        Parse a VNINS sentence.

        Args:
            line: Raw VNINS sentence string.

        Returns:
            INSReading if parsing successful, None otherwise.
        """
        match = self.VNINS_PATTERN.match(line)
        if not match:
            # Try simpler parsing for partial messages
            return self._parse_vnins_simple(line)

        try:
            # Verify checksum
            if not self._verify_checksum(line):
                return None

            groups = match.groups()

            return INSReading(
                ins_timestamp=groups[0],
                status1=groups[1],
                status2=groups[2],
                yaw=float(groups[3]),
                pitch=float(groups[4]),
                roll=float(groups[5]),
                latitude=float(groups[6]),
                longitude=float(groups[7]),
                altitude=float(groups[8]),
                # vel_n, vel_e, vel_d at indices 9, 10, 11
                att_uncertainty=float(groups[12]),
                pos_uncertainty=float(groups[13]),
                vel_uncertainty=float(groups[14]),
                timestamp=time.time()
            )

        except (ValueError, IndexError) as e:
            self._error_count += 1
            self._last_error = f"Parse error: {e}"
            return None

    def _parse_vnins_simple(self, line: str) -> Optional[INSReading]:
        """
        Simple fallback parser for VNINS sentences.

        This handles cases where the regex might not match exactly.
        """
        try:
            # Remove checksum
            if '*' in line:
                line = line.split('*')[0]

            # Remove $VNINS, prefix
            if line.startswith('$VNINS,'):
                line = line[7:]
            else:
                return None

            parts = line.split(',')
            if len(parts) < 15:
                return None

            return INSReading(
                ins_timestamp=parts[0],
                status1=parts[1],
                status2=parts[2],
                yaw=float(parts[3]),
                pitch=float(parts[4]),
                roll=float(parts[5]),
                latitude=float(parts[6]) if parts[6] else 0.0,
                longitude=float(parts[7]) if parts[7] else 0.0,
                altitude=float(parts[8]) if parts[8] else 0.0,
                att_uncertainty=float(parts[12]) if len(parts) > 12 else 99.99,
                pos_uncertainty=float(parts[13]) if len(parts) > 13 else 99.99,
                vel_uncertainty=float(parts[14]) if len(parts) > 14 else 9.999,
                timestamp=time.time()
            )

        except (ValueError, IndexError):
            return None

    def _verify_checksum(self, line: str) -> bool:
        """
        Verify NMEA-style checksum.

        The checksum is XOR of all bytes between $ and * (exclusive).
        """
        try:
            if '*' not in line:
                return False

            msg_part, checksum_hex = line.rsplit('*', 1)

            # Remove leading $
            if msg_part.startswith('$'):
                msg_part = msg_part[1:]

            # Calculate XOR checksum
            calculated = 0
            for char in msg_part:
                calculated ^= ord(char)

            expected = int(checksum_hex[:2], 16)
            return calculated == expected

        except (ValueError, IndexError):
            return False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


class MockINSReader:
    """
    Mock INS reader for testing without hardware.

    Returns simulated INS data that can be configured.
    """

    def __init__(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0):
        """
        Initialize mock reader with fixed values.

        Args:
            yaw: Fixed yaw value in degrees.
            pitch: Fixed pitch value in degrees.
            roll: Fixed roll value in degrees.
        """
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_latest(self) -> Optional[INSReading]:
        if not self._connected:
            return None
        return INSReading(
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            timestamp=time.time(),
            ins_timestamp="000000.000000",
            status1="0000",
            status2="0000",
            att_uncertainty=0.5,  # Simulate good alignment
            pos_uncertainty=0.02,
            vel_uncertainty=0.01
        )

    def wait_for_reading(self, timeout: float = 5.0) -> Optional[INSReading]:
        return self.get_latest()

    def wait_for_valid_reading(self, timeout: float = 10.0) -> Optional[INSReading]:
        return self.get_latest()

    def set_values(self, yaw: float, pitch: float, roll: float):
        """Update the mock values."""
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


def test_ins_connection(port: str = '/dev/ttyUSB0', duration: float = 5.0) -> bool:
    """
    Test INS connection and print readings for a duration.

    Args:
        port: Serial port path.
        duration: Test duration in seconds.

    Returns:
        True if readings were received, False otherwise.
    """
    print(f"\nTesting INS connection on {port}...")
    print("-" * 50)

    reader = INSSerialReader(port=port)

    if not reader.connect():
        print(f"ERROR: Failed to connect - {reader._last_error}")
        return False

    print(f"Connected to {port} at {reader.baudrate} baud")
    print(f"Listening for {duration} seconds...\n")

    start = time.time()
    readings_received = 0

    try:
        while time.time() - start < duration:
            reading = reader.get_latest()

            if reading and readings_received < reader._read_count:
                readings_received = reader._read_count
                status = "VALID" if reading.is_valid() else "INIT"
                print(f"[{readings_received:4d}] Yaw: {reading.yaw:8.3f}  "
                      f"Pitch: {reading.pitch:8.3f}  Roll: {reading.roll:8.3f}  "
                      f"AttUnc: {reading.att_uncertainty:5.2f}  [{status}]")

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        reader.disconnect()

    print("-" * 50)
    stats = reader.get_stats()
    print(f"Total readings: {stats['read_count']}")
    print(f"Errors: {stats['error_count']}")

    return stats['read_count'] > 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='INS Serial Reader Test')
    parser.add_argument('--port', default='/dev/ttyUSB0',
                        help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Test duration in seconds (default: 10)')

    args = parser.parse_args()

    success = test_ins_connection(args.port, args.duration)
    exit(0 if success else 1)

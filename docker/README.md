# AxonVision Camera Calibration Tool - Docker

Docker container for the AxonVision Camera Calibration Tool with PySide6 GUI.

## Prerequisites

- Docker Engine 20.10+ with Compose plugin
- X11 display server (Linux with X11 or XWayland)
- For INS support: serial port access (`/dev/ttyUSB0`)

## Quick Start

Run these commands from the **project root** directory:

```bash
# 1. Make the run script executable
chmod +x docker/run.sh

# 2. Build and run
./docker/run.sh build
```

The GUI window will appear on your desktop.

## Usage

All commands should be run from the **project root**:

```bash
# Start the application
./docker/run.sh

# Build and start (first time or after code changes)
./docker/run.sh build

# Start in background
./docker/run.sh detach

# View logs
./docker/run.sh logs

# Open shell in container
./docker/run.sh shell

# Stop the application
./docker/run.sh stop

# Check status
./docker/run.sh status

# Clean up (remove container and image)
./docker/run.sh clean
```

## Manual Docker Commands

If you prefer not to use the run script:

```bash
# Allow X11 access
xhost +local:docker

# Build and run (from project root)
docker compose -f docker/docker-compose.yml up --build

# Run in background
docker compose -f docker/docker-compose.yml up -d

# Stop
docker compose -f docker/docker-compose.yml down
```

## Project Structure

```
project-root/
├── docker/
│   ├── Dockerfile           # Container image definition
│   ├── docker-compose.yml   # Container orchestration
│   ├── requirements.txt     # Python dependencies (for Docker)
│   ├── run.sh              # Startup script with X11 setup
│   └── README.md           # This file
│
├── main_ui.py              # Main application entry point
├── ui/                     # UI module
├── config/                 # Config module
├── camera_streaming.py
├── extrinsic_calibration.py
├── intrinsic_calibration.py
├── ins_reader.py
├── requirements.txt        # Main dependencies
│
├── calibration_data/       # Mounted volume for calibration data
├── output/                 # Mounted volume for output files
└── logs/                   # Mounted volume for logs
```

## Configuration

### Serial Port Access (INS)

To enable INS serial port access, uncomment the devices section in `docker/docker-compose.yml`:

```yaml
devices:
  - /dev/ttyUSB0:/dev/ttyUSB0
  - /dev/ttyUSB1:/dev/ttyUSB1  # Add more as needed
```

### Network Camera Access

The container uses `network_mode: host` for direct network access to cameras.
If you need isolated networking, modify `docker/docker-compose.yml`:

```yaml
# Remove network_mode: host and add:
ports:
  - "5000:5000"   # Camera API
  - "5010:5010"   # RTP stream
networks:
  default:
    driver: bridge
```

### Persistent Data

Three directories are mounted for data persistence:

| Host Directory | Container Path | Purpose |
|----------------|----------------|---------|
| `./calibration_data` | `/app/calibration_data` | Calibration files |
| `./output` | `/app/output` | Generated reports |
| `./logs` | `/app/logs` | Application logs |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY` | `:0` | X11 display |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `QT_QPA_PLATFORM` | `xcb` | Qt platform plugin |

## Troubleshooting

### GUI doesn't appear

1. Ensure X11 server is running
2. Run `xhost +local:docker` to allow Docker access
3. Check DISPLAY variable: `echo $DISPLAY`

### "Cannot connect to X server"

```bash
# Reset X11 permissions
xhost +local:docker
export DISPLAY=:0
./docker/run.sh
```

### Serial port permission denied

```bash
# Add user to dialout group
sudo usermod -aG dialout $USER
# Log out and back in, then retry
```

### Container exits immediately

Check logs for errors:

```bash
./docker/run.sh logs
```

### OpenCV/Qt conflicts

The Dockerfile sets environment variables to prevent Qt/GTK conflicts.
If issues persist, try:

```bash
docker compose -f docker/docker-compose.yml run --rm -e QT_DEBUG_PLUGINS=1 axonvision
```

## Building for Different Platforms

For ARM64 (Jetson, Raspberry Pi):

```bash
docker buildx build --platform linux/arm64 -f docker/Dockerfile -t axonvision-calibration:arm64 .
```

## License

Proprietary - AxonVision

# AxonVision Camera Calibration Tool - Docker

Docker container for the AxonVision Camera Calibration Tool with PySide6 GUI.

## Prerequisites

- Docker and Docker Compose installed
- X11 display server (Linux with X11 or XWayland)
- For INS support: serial port access (`/dev/ttyUSB0`)

## Quick Start

```bash
# 1. Clone/copy your application files to this directory
#    Ensure your project structure includes:
#    - main_ui.py
#    - ui/ (module directory)
#    - config/ (module directory)
#    - *.py (calibration modules)

# 2. Make the run script executable
chmod +x run.sh

# 3. Build and run
./run.sh build
```

## Usage

```bash
# Start the application
./run.sh

# Start in background
./run.sh detach

# View logs
./run.sh logs

# Open shell in container
./run.sh shell

# Stop the application
./run.sh stop

# Clean up (remove container and image)
./run.sh clean
```

## Manual Docker Commands

If you prefer not to use the run script:

```bash
# Allow X11 access
xhost +local:docker

# Build image
docker-compose build

# Run container
docker-compose up

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

## Project Structure

```
axonvision-docker/
├── Dockerfile           # Container image definition
├── docker-compose.yml   # Container orchestration
├── requirements.txt     # Python dependencies
├── run.sh              # Startup script with X11 setup
├── .dockerignore       # Files to exclude from build
├── README.md           # This file
│
├── main_ui.py          # Main application entry point
├── ui/                 # UI module (copy your files here)
├── config/             # Config module (copy your files here)
├── camera_streaming.py
├── extrinsic_calibration.py
├── intrinsic_calibration.py
├── ins_reader.py
│
├── calibration_data/   # Mounted volume for calibration data
└── output/             # Mounted volume for output files
```

## Configuration

### Serial Port Access (INS)

The container is configured to access `/dev/ttyUSB0` for the INS device.
To add more serial ports, edit `docker-compose.yml`:

```yaml
devices:
  - /dev/ttyUSB0:/dev/ttyUSB0
  - /dev/ttyUSB1:/dev/ttyUSB1  # Add more as needed
```

### Network Camera Access

The container uses `network_mode: host` for direct network access to cameras.
If you need isolated networking, modify `docker-compose.yml`:

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

Two directories are mounted for data persistence:

- `./calibration_data` → `/app/calibration_data` - Calibration files
- `./output` → `/app/output` - Generated reports and exports

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
./run.sh
```

### Serial port permission denied

```bash
# Add user to dialout group
sudo usermod -aG dialout $USER
# Log out and back in, then retry
```

### OpenCV/Qt conflicts

The Dockerfile sets environment variables to prevent Qt/GTK conflicts.
If issues persist, try:

```bash
docker-compose run --rm -e QT_DEBUG_PLUGINS=1 axonvision
```

### Container exits immediately

Check logs for errors:

```bash
docker-compose logs
```

## Building for Different Platforms

For ARM64 (Jetson, Raspberry Pi):

```bash
docker buildx build --platform linux/arm64 -t axonvision-calibration:arm64 .
```

## License

Proprietary - AxonVision

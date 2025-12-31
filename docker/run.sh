#!/bin/bash
#
# AxonVision Camera Calibration Tool - Docker Run Script
#
# This script sets up X11 forwarding and launches the container.
# Can be run from project root or docker directory.
#
# Usage:
#   ./docker/run.sh              # Run with docker-compose
#   ./docker/run.sh build        # Build and run
#   ./docker/run.sh shell        # Open shell in container
#   ./docker/run.sh stop         # Stop container

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is the parent of the docker directory
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root for consistent paths
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

# Detect docker compose command (V2 vs V1)
detect_compose_cmd() {
    if docker compose version &>/dev/null; then
        COMPOSE_CMD="docker compose -f docker/docker-compose.yml"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose -f docker/docker-compose.yml"
    else
        log_error "Docker Compose not found. Please install Docker with Compose plugin."
        echo ""
        echo "Install Docker:"
        echo "  Ubuntu/Debian: sudo apt install docker.io docker-compose-v2"
        echo "  Or visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    log_info "Using: $COMPOSE_CMD"
}

# Setup X11 authentication for Docker
setup_x11() {
    log_info "Setting up X11 authentication..."

    # Allow local connections to X server
    if command -v xhost &>/dev/null; then
        xhost +local:docker 2>/dev/null || true
    else
        log_warn "xhost not found - X11 access may not work"
    fi

    # Export display if not set
    export DISPLAY="${DISPLAY:-:0}"

    # Create Xauthority file for docker if needed
    XAUTH_FILE="/tmp/.docker.xauth"
    if [ ! -f "$XAUTH_FILE" ]; then
        touch "$XAUTH_FILE"
        if command -v xauth &>/dev/null; then
            xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_FILE" nmerge - 2>/dev/null || true
        fi
    fi
    export XAUTHORITY="$XAUTH_FILE"

    log_info "X11 setup complete (DISPLAY=$DISPLAY)"
}

# Check if required devices exist
check_devices() {
    if [ -e /dev/ttyUSB0 ]; then
        log_info "INS serial port detected: /dev/ttyUSB0"
    else
        log_warn "INS serial port /dev/ttyUSB0 not found (optional)"
    fi
}

# Create data directories
create_directories() {
    mkdir -p calibration_data output logs
    log_info "Data directories ready"
}

# Initialize
log_header "AxonVision Camera Calibration Tool"
detect_compose_cmd

# Main command handling
case "${1:-run}" in
    build)
        log_info "Building and starting AxonVision..."
        setup_x11
        create_directories
        check_devices
        $COMPOSE_CMD up --build
        ;;

    run)
        log_info "Starting AxonVision..."
        setup_x11
        create_directories
        check_devices
        $COMPOSE_CMD up
        ;;

    detach|background|-d)
        log_info "Starting AxonVision in background..."
        setup_x11
        create_directories
        check_devices
        $COMPOSE_CMD up -d
        log_info "Container started. Use './docker/run.sh logs' to view output."
        ;;

    stop)
        log_info "Stopping AxonVision..."
        $COMPOSE_CMD down
        ;;

    restart)
        log_info "Restarting AxonVision..."
        $COMPOSE_CMD restart
        ;;

    logs)
        $COMPOSE_CMD logs -f
        ;;

    shell)
        log_info "Opening shell in container..."
        setup_x11
        $COMPOSE_CMD exec axonvision /bin/bash || \
            $COMPOSE_CMD run --rm axonvision /bin/bash
        ;;

    clean)
        log_info "Cleaning up Docker resources..."
        $COMPOSE_CMD down --rmi local --volumes
        log_info "Cleanup complete"
        ;;

    status)
        $COMPOSE_CMD ps
        ;;

    *)
        echo "AxonVision Camera Calibration Tool - Docker Runner"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  run        Start the application (default)"
        echo "  build      Build image and start"
        echo "  detach     Start in background"
        echo "  stop       Stop the container"
        echo "  restart    Restart the container"
        echo "  logs       Show container logs"
        echo "  shell      Open bash shell in container"
        echo "  status     Show container status"
        echo "  clean      Remove container and image"
        echo ""
        echo "Examples:"
        echo "  ./docker/run.sh build     # First time: build and run"
        echo "  ./docker/run.sh           # Run (after build)"
        echo "  ./docker/run.sh stop      # Stop the container"
        echo ""
        exit 1
        ;;
esac

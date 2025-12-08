# AxonVision Camera Calibration CLI
# Multi-stage build for optimized image size

FROM python:3.11-slim-bookworm AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including GStreamer and OpenCV requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GStreamer core and plugins
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    # GStreamer development (for OpenCV GStreamer backend)
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    # X11 for GUI support (optional, for preview windows)
    libx11-6 \
    libxcb1 \
    # Network tools for diagnostics
    iputils-ping \
    net-tools \
    # Build essentials (for some pip packages)
    build-essential \
    pkg-config \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY *.png ./  2>/dev/null || true

# Environment variables for OpenCV and Qt
ENV QT_QPA_PLATFORM=xcb
ENV OPENCV_VIDEOIO_PRIORITY_QT=0
ENV PYTHONUNBUFFERED=1

# Expose API port (if running as server)
EXPOSE 5000

# Default command - show help
CMD ["python3", "camera_streaming.py", "--help"]

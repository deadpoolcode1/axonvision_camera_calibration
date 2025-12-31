# AxonVision Camera Calibration Tool - Docker Image
# Base image with Ubuntu 22.04 for Qt/X11 compatibility
FROM ubuntu:22.04

LABEL maintainer="AxonVision"
LABEL description="AxonVision Camera Calibration Tool with PySide6 GUI"
LABEL version="1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Qt/X11 environment variables
ENV QT_QPA_PLATFORM=xcb
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0
ENV GDK_BACKEND=x11
ENV OPENCV_VIDEOIO_PRIORITY_QT=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    # Qt/X11 dependencies
    libxcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglib2.0-0 \
    libfontconfig1 \
    libdbus-1-3 \
    x11-utils \
    # GStreamer for camera streaming
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    # Networking tools
    iputils-ping \
    net-tools \
    curl \
    # Serial port support
    udev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash axonvision && \
    usermod -aG dialout axonvision

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership to non-root user
RUN chown -R axonvision:axonvision /app

# Switch to non-root user
USER axonvision

# Default command
CMD ["python3", "main_ui.py"]

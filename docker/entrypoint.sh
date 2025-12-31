#!/bin/bash
#
# AxonVision Docker Entrypoint Script
#
# This script fixes permissions on mounted volumes before running the app.
# It runs as root to fix permissions, then switches to the axonvision user.
#

set -e

# Ensure mounted directories exist and have correct ownership
# These directories are mounted from the host and may have wrong permissions
chown -R axonvision:axonvision /app/calibration_data /app/output /app/logs 2>/dev/null || true

# Execute the command as the axonvision user with virtual environment activated
exec su -s /bin/bash axonvision -c "source /app/venv/bin/activate && exec $*"

# Camera Calibration System

Complete calibration system for multi-camera INS integration. Determines both intrinsic (internal) and extrinsic (position/orientation) camera parameters.

## Overview

| Calibration | What it finds | When to do |
|-------------|---------------|------------|
| **Intrinsic** | Focal length, distortion, principal point | Once per camera (or after lens change) |
| **Extrinsic** | Camera position AND orientation | Once per camera installation |

## Coordinate System (NED)

```
        +X (Forward)
            |
            |
  -Y <------o------> +Y
 (Left)     |      (Right)
            |
            v
        +Z (Down)
        
Origin: IMU center
```

**Important:** Z is DOWN, so heights above ground are NEGATIVE Z values.
- Camera 1.5m above ground: Z = -1.5m
- Board center 0.9m above ground: Z = -0.9m

## Equipment Required

### ChArUco Calibration Board

- **Size:** 8x8 squares, 0.88m x 0.88m total
- **Square size:** 11cm
- **Marker size:** 8.5cm
- **Dictionary:** ARUCO_6X6_250
- **Material:** RIGID (foam board, aluminum composite, or acrylic)

### Other Equipment

| Item | Purpose | Accuracy |
|------|---------|----------|
| Board stand (adjustable height) | Hold board steady | - |
| RTK GPS receiver #1 | Vehicle/IMU antenna position | ~1-2cm |
| RTK GPS receiver #2 | Board antenna position (mounted on board) | ~1-2cm |
| Tape measure | Camera position prior estimate | ~10-20cm |
| Inclinometer/protractor | Camera angle prior estimate | ~5° |
| INS unit | Provide IMU orientation | - |
| Laptop | Run calibration software | - |

### Technician Roles

| Role | Location | Responsibilities |
|------|----------|------------------|
| **T1** (Computer) | At laptop | View camera feed, record data, run software |
| **T2** (Field) | At board/vehicle | Position equipment, take measurements, report values |

---

## Part 1: Intrinsic Calibration

Finds camera internal parameters. **Single operator, no INS needed.**

### Procedure

1. Take 15-20 images of the ChArUco board
2. Vary distance (1-5m), angles (up to 45 deg tilt), and positions (all corners)
3. Run calibration

```bash
python3 intrinsic_calibration.py \
    --images intrinsic_images/ \
    --output camera_intrinsics.json
```

### Expected Results

```
RMS reprojection error: 0.32 px  <-- Should be < 0.5 px
```

---

## Part 2: Extrinsic Calibration (Dual RTK + Bundle Adjustment)

Finds camera position AND orientation relative to **IMU** using **dual RTK measurements** and **bundle adjustment optimization**.

**Key advantages:**
- Full 6-DOF estimation (position + orientation)
- Position accuracy: ~2-3cm relative to IMU
- Orientation accuracy: <0.5°
- No manual distance/height measurements needed
- Proper SE(3) math (not angle averaging)

### How It Works

1. Mount RTK antenna on calibration board
2. Vehicle RTK antenna measures IMU position in world
3. Board RTK antenna measures board position in world
4. INS provides IMU attitude (yaw/pitch/roll)
5. Bundle adjustment optimizes T_IC (IMU→Camera transform) to minimize:
   - Reprojection error (detected vs projected corners)
   - RTK position errors (board + vehicle)
   - IMU attitude error
   - Camera prior constraints

```
World Frame (NED)
       |
       v
   [RTK positions]
       |
   T_WI (from INS + vehicle RTK)
       |
       v
   IMU Frame ----T_IC----> Camera Frame
       |                        |
       |                    [detects board]
       |                        |
   T_WB (from board RTK)        |
       |                        v
       v                   T_CB (from PnP)
   Board Frame <----------------+
```

---

### PHASE 1: Vehicle & RTK Setup

*Do once per calibration session.*

#### Step 1.1: Position Vehicle

| Who | Action |
|-----|--------|
| T1 | Park vehicle on level ground with clear sky view |
| T1 | Start INS, wait for alignment (yaw/pitch/roll stable) |
| T2 | Verify RTK base station has clear sky view |

#### Step 1.2: Verify RTK Fix

| Who | Action |
|-----|--------|
| T2 | Check vehicle RTK antenna has RTK fix |
| T2 | Check board RTK antenna has RTK fix |
| T1 | Record INS yaw/pitch/roll readings |

```
Example INS readings:
  Yaw: 45.0° (vehicle heading)
  Pitch: 0.0°
  Roll: 0.0°
```

---

### PHASE 2: Camera Prior Measurements

*Do for each camera. These are rough estimates that the optimizer will refine.*

#### Step 2.1: Measure Camera Position (Rough Estimate)

| Who | Action |
|-----|--------|
| T2 | Use tape measure from IMU center to camera lens center |
| T1 | Record as `camera_prior_position = [X, Y, Z]` |

**Directions (relative to IMU):**
- X: Forward (camera forward of IMU = positive)
- Y: Right (camera right of IMU = positive)
- Z: Down (camera below IMU = positive, camera above = negative)

```
Example: Camera is 50cm forward, 20cm right, 30cm ABOVE IMU
camera_prior_position = [0.50, 0.20, -0.30]
```

**Accuracy:** ±15cm is fine. Optimizer will refine this.

#### Step 2.2: Estimate Camera Orientation

| Who | Action |
|-----|--------|
| T1 | Estimate angles from mounting specs or visual inspection |
| T1 | Record as `camera_prior_orientation = [azimuth, elevation, roll]` |

**Angles:**
- Azimuth: 0°=forward, +90°=right, -90°=left, ±180°=rear
- Elevation: 0°=horizontal, positive=looking down
- Roll: usually 0° (camera upright)

```
Example: Camera points 15° right and 5° down
camera_prior_orientation = [15.0, 5.0, 0.0]
```

**Accuracy:** ±5° is fine. Optimizer will refine this.

---

### PHASE 3: Data Collection

*Collect 5-8 measurements per camera, varying board position.*

#### Step 3.1: Position Board

| Who | Action |
|-----|--------|
| T2 | Place board on stand, ~2.5-5m from camera |
| T2 | Hold board approximately vertical, facing camera |
| T2 | Vary position: distance, lateral offset, AND height |
| T1 | Verify board is visible in camera FOV |

**Important:** Vary all three dimensions for good observability:
- Distance: 2.5m to 5m
- Lateral: left/center/right within FOV
- Height: vary by ±30cm between positions

```
Good position variety:
  Pos 1: 2.9m, lateral -0.5m, height -0.2m
  Pos 2: 3.3m, lateral -0.2m, height -0.1m
  Pos 3: 3.7m, lateral  0.0m, height  0.0m
  Pos 4: 4.1m, lateral +0.2m, height +0.1m
  Pos 5: 4.5m, lateral +0.5m, height +0.2m
```

#### Step 3.2: Record RTK Readings

| Who | Action |
|-----|--------|
| T2 | Read board RTK position (N, E, D) |
| T2 | Read vehicle RTK position (N, E, D) |
| T1 | Record both RTK readings |

```
Example readings:
  Board RTK:   N=3.090m, E=-1.666m, D=-1.426m
  Vehicle RTK: N=0.012m, E=-0.007m, D=-1.492m
```

#### Step 3.3: Capture Image

| Who | Action |
|-----|--------|
| T1 | Verify camera preview: board visible, no blur, no glare |
| T1 | Verify corner detection: "81 corners detected" |
| T1 | Record INS yaw/pitch/roll |
| T1 | Capture and confirm: "reproj error < 1px" |

**Repeat for 5-8 positions**

---

### PHASE 4: Run Calibration

| Who | Action |
|-----|--------|
| T1 | Verify all measurements recorded (min 5 positions) |
| T1 | Run bundle adjustment optimization |
| T1 | Verify results meet specifications |
| T1 | Save to JSON file |

---

### Expected Results

```
COMPUTED CAMERA-TO-IMU TRANSFORM:
  Position (in IMU frame):
    Forward (X): +51.42 cm
    Right (Y):   +18.38 cm
    Down (Z):    -29.04 cm
  Orientation:
    Azimuth:     +15.35°
    Elevation:   +4.89°
    Roll:        +0.19°

COMPARISON WITH GROUND TRUTH:
  Position error: 2.36 cm   <-- Should be < 5cm
  Azimuth error:  +0.35°    <-- Should be < 1°
  Elevation error: -0.11°   <-- Should be < 1°

IMPROVEMENT OVER MANUAL MEASUREMENT:
  Position: 9.9cm → 2.4cm
  Azimuth:  2.0° → 0.3°
  Elevation: 1.0° → 0.1°

Quality Metrics:
  Mean reprojection error: 0.29 px  <-- Should be < 0.5px
  Optimization converged: True
```

### Interpreting Results

| Parameter | Meaning | Example |
|-----------|---------|---------|
| Position [0.5, 0.2, -0.3] | Camera is 50cm forward, 20cm right, 30cm above IMU | Roof-mounted camera |
| Azimuth +15° | Camera points 15° RIGHT of IMU forward | Right-facing camera |
| Azimuth -45° | Camera points 45° LEFT of IMU forward | Left-facing camera |
| Elevation +5° | Camera points 5° DOWN from horizontal | Typical downward tilt |
| Elevation -5° | Camera points 5° UP from horizontal | Upward-looking camera |
| Roll 0° | Camera "up" aligns with IMU "up" | Normal mounting |

---

## Multi-Camera Calibration

Each camera is calibrated **independently** using the same vehicle RTK antenna.

**Critical:** All cameras use the **SAME vehicle RTK antenna** and **SAME INS**. This ensures all cameras are referenced to a unified IMU coordinate system.

```
              IMU Center (origin)
                   |
    +--------------+--------------+
    |              |              |
    v              v              v
 Camera 1      Camera 2      Camera 3
 (T_IC_1)      (T_IC_2)      (T_IC_3)
```

**Procedure:**

1. Set up vehicle with INS running and vehicle RTK antenna fixed
2. For each camera:
   - Measure camera prior position/orientation relative to IMU
   - Collect 5-8 board positions with dual RTK readings
   - Run calibration
   - Save JSON file
3. Each camera produces its own JSON file with T_IC (IMU→Camera transform)

**Result:** All cameras share the same IMU coordinate system.

---

## Output Format

### Extrinsics JSON

```json
{
  "camera_id": "camera_front",
  "imu_to_camera_transform": {
    "rotation_matrix": [[...], [...], [...]],
    "translation_vector": [0.5142, 0.1838, -0.2904],
    "euler_angles": {
      "azimuth": 15.35,
      "elevation": 4.89,
      "roll": 0.19
    },
    "transform_matrix_4x4": [[...], [...], [...], [...]]
  },
  "imu_world_pose": {
    "rotation_matrix": [[...], [...], [...]],
    "translation_vector": [0.0, 0.0, -1.5],
    "euler_angles_ypr_deg": [45.0, 0.0, 0.0]
  },
  "coordinate_system": {
    "frame": "NED",
    "origin": "IMU center",
    "note": "T_IC transforms points from IMU frame to camera frame"
  },
  "quality_metrics": {
    "num_measurements": 6,
    "optimization_converged": true,
    "final_cost": 838.78,
    "mean_reproj_error_px": 0.286,
    "max_reproj_error_px": 0.45
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

**Key outputs:**
- `imu_to_camera_transform`: The 6-DOF transform T_IC (what you need for sensor fusion)
- `translation_vector`: Camera position in IMU frame [X, Y, Z] in meters
- `euler_angles`: Camera pointing direction relative to IMU forward
- `quality_metrics`: Optimization convergence and reprojection error

**Usage:** To transform a point from IMU frame to camera frame:
```
p_camera = R_IC @ p_imu + t_IC
```

---

## Accuracy Summary

| Parameter | Accuracy | Limited By |
|-----------|----------|------------|
| Position (vs IMU) | ~2-3cm | RTK accuracy, optimization |
| Azimuth | <0.5° | Optimization, corner detection |
| Elevation | <0.5° | Optimization, corner detection |
| Roll | <1° | Board levelness assumption |

---

## Troubleshooting

### "No corners detected"
- Board too far (>5m) or too close (<1.5m)
- Poor lighting or glare on board
- Board out of focus
- Board tilted too much (should be nearly vertical)

### High position error (>5cm)
- RTK not in fix mode (check for RTK fix on both antennas)
- Insufficient position variety (vary distance, lateral, AND height)
- Fewer than 5 measurements
- Camera prior estimate too far off (>20cm)

### High orientation error (>1°)
- Board not held approximately vertical during captures
- INS providing incorrect orientation (check alignment)
- Prior orientation estimate too far off (>10°)
- Insufficient measurements

### Optimization fails to converge
- Check that prior position/orientation estimates are reasonable
- Ensure board is visible and detected in all captures
- Verify RTK readings are consistent
- Need at least 5 measurements with good variety

### "Board not visible in camera FOV"
- Board position outside camera field of view
- Adjust board position (closer or more centered)

---

## Quick Reference

```
PHASE 1 (per session):
  [T1] Park vehicle, start INS, wait for alignment
  [T2] Verify RTK fix on both antennas (vehicle + board)
  [T1] Record INS yaw/pitch/roll

PHASE 2 (per camera):
  [T2] Measure IMU -> camera position with tape
  [T1] Record camera_prior_position = [X, Y, Z]
  [T1] Estimate camera_prior_orientation = [az, el, roll]

PHASE 3 (per camera, 5-8 positions):
  [T2] Place board at position (vary distance, lateral, height)
  [T2] Read board RTK (N, E, D) -> T1 records
  [T2] Read vehicle RTK (N, E, D) -> T1 records
  [T1] Verify 81 corners detected
  [T1] Record INS, capture image
  [T1] Verify reproj error < 1px

PHASE 4:
  [T1] Run calibration, verify specs, save JSON
```

---

## Usage

### Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or manually:
pip install opencv-contrib-python==4.8.1.78 "numpy<2" scipy requests
```

### GStreamer Setup (for Network Camera)

For network camera streaming, install GStreamer:

```bash
# Ubuntu/Debian
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad gstreamer1.0-libav

# Verify installation
gst-launch-1.0 --version
```

### Intrinsic Calibration (once per camera)

```bash
# With real images
python3 intrinsic_calibration.py --images intrinsic_images/ --output camera_intrinsics.json

# With network camera (H265/RTP stream)
python3 intrinsic_calibration.py --network-camera --output camera_intrinsics.json

# Synthetic demo (no hardware)
python3 intrinsic_calibration.py --synthetic --output demo_intrinsics.json
```

---

## Network Camera Streaming

The system supports network cameras with H265/RTP multicast streaming.

### Camera API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `http://<ip>:5000/` | GET | Camera web interface |
| `/api/stream/start` | POST | Start streaming (JSON: `{"host": "239.255.0.1", "port": 5010}`) |
| `/api/stream/stop` | POST | Stop streaming |

### Camera Streaming CLI

```bash
# Interactive menu
python3 camera_streaming.py

# Test camera connectivity (ping, API, multicast, GStreamer)
python3 camera_streaming.py test

# Start camera preview (GStreamer window)
python3 camera_streaming.py preview

# Start camera preview (OpenCV window - allows capture)
python3 camera_streaming.py preview --opencv

# Capture single frame
python3 camera_streaming.py capture -o snapshot.png

# Run calibration with live camera
python3 camera_streaming.py calibrate

# Start stream only (for external tools)
python3 camera_streaming.py start

# Stop stream
python3 camera_streaming.py stop
```

### Custom Camera IP

```bash
# Use different camera IP/ports
python3 camera_streaming.py test --ip 192.168.1.100 --api-port 5000

# Full custom configuration
python3 camera_streaming.py preview \
    --ip 192.168.1.100 \
    --api-port 5000 \
    --multicast 239.255.0.2 \
    --stream-port 5020
```

### GStreamer Pipeline (Manual)

```bash
# Start stream via API first
curl -X POST http://10.100.102.222:5000/api/stream/start \
    -H "Content-Type: application/json" \
    -d '{"host": "239.255.0.1", "port": 5010}'

# View with GStreamer
gst-launch-1.0 udpsrc address=239.255.0.1 port=5010 \
    caps="application/x-rtp,media=video,encoding-name=H265,payload=96" ! \
    rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! autovideosink
```

### Connectivity Diagnostics

The `test` command runs comprehensive diagnostics:

```
============================================================
  Camera Connectivity Diagnostics
============================================================

[*] Pinging 10.100.102.222...
[OK] Camera reachable at 10.100.102.222
      Latency: 1.2ms

[*] Checking port 5000...
[OK] Port 5000 is open

[*] Testing API at http://10.100.102.222:5000...
[OK] Camera API is responding

[*] Checking multicast support...
[OK] Multicast supported for 239.255.0.1

[*] Checking GStreamer installation...
[OK] GStreamer ready: gst-launch-1.0 version 1.20.3

============================================================
  Diagnostics Summary
============================================================
  Tests Passed: 5/5
[OK] All tests passed! Camera is ready.
```

### Calibration with Network Camera

```bash
# Step 1: Test connectivity first
python3 camera_streaming.py test

python3 camera_streaming.py --ip 10.1.13.37

# Step 2: Run intrinsic calibration
python3 intrinsic_calibration.py --network-camera -o camera_intrinsics.json

# Or use the integrated calibration mode
python3 camera_streaming.py calibrate
```

### Extrinsic Calibration (once per installation)

```bash
# Demo mode (no hardware needed) - shows full technician workflow
python3 extrinsic_calibration.py --demo -n 7 -o demo_extrinsics.json

# Demo with your intrinsics file
python3 extrinsic_calibration.py --demo -i camera_intrinsics.json -n 7 -o camera_extrinsics.json
```

### View Results

```bash
cat demo_extrinsics.json
```

### Complete Demo Workflow (No Hardware Needed)

```bash
# Step 1: Generate synthetic intrinsics
python3 intrinsic_calibration.py --synthetic -o demo_intrinsics.json

# Step 2: Run extrinsic calibration demo (shows technician workflow)
python3 extrinsic_calibration.py --demo -i demo_intrinsics.json -n 7 -o demo_extrinsics.json

# Step 3: Review results
cat demo_extrinsics.json
```

### Demo vs Real Calibration

| Step | Demo Mode | Real Calibration |
|------|-----------|------------------|
| Vehicle RTK | Simulated (~1.5cm noise) | Real RTK antenna on vehicle |
| Board RTK | Simulated (~1.5cm noise) | Real RTK antenna on board |
| Board placement | Simulated | Technician places board |
| Camera prior | Simulated (~10cm, ~2° error) | Measured with tape/inclinometer |
| INS | Simulated | Real INS data |
| Images | Synthetic ChArUco rendering | Real camera images |

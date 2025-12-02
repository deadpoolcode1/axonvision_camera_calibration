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
        
Origin: Vehicle reference point (marked on exterior)
```

**Important:** Z is DOWN, so heights above ground are NEGATIVE Z values.
- Camera 1.5m above ground: Z = -1.5m
- Board center 0.9m above ground: Z = -0.9m

## Equipment Required

### ChArUco Calibration Board

- **Size:** 10x10 squares, 1.1m x 1.1m total
- **Square size:** 11cm
- **Marker size:** 8.5cm
- **Dictionary:** ARUCO_6X6_250
- **Material:** RIGID (foam board, aluminum composite, or acrylic)

### Other Equipment

| Item | Purpose | Accuracy |
|------|---------|----------|
| Board stand (adjustable height) | Hold board steady | - |
| RTK GPS receiver | Measure ground positions | ~1-2cm |
| Laser distance meter | Camera-to-board distance | ~2cm |
| Laser/tape for height | Board center height from ground | ~2cm |
| Tape measure | IMU offset, camera prior estimates | ~10-20cm |
| INS unit | Provide vehicle orientation | - |
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

## Part 2: Extrinsic Calibration (Known Ground Positions + RTK)

Finds camera position AND orientation relative to vehicle using **RTK-measured ground positions**.

**Key advantages:**
- Position accuracy: <1cm relative to reference point
- Orientation accuracy: <0.1 deg
- Each camera calibrated INDEPENDENTLY (scales to N cameras)
- No inter-camera measurements needed

### How It Works

1. Mark reference point on vehicle, measure IMU offset once
2. Use RTK to mark and measure ground positions for each camera
3. Place board at each position, measure height and distance, capture image
4. Optimization solves for camera pose that makes all measurements consistent

```
Vehicle Reference Point (e.g., rear bumper corner)
        |
        v
    [0,0,0] -----> X (forward)
        |
        |  * Spot 1 [4.0, 0.85, -0.9]
        |      * Spot 2 [4.5, 1.0, -0.9]
        v          * Spot 3 [5.0, 1.15, -0.9]
        Y (right)
        
                    Camera (position solved by optimization)
```

---

### PHASE 1: One-Time Vehicle Setup

*Do once per vehicle. Results reused for all cameras.*

#### Step 1.1: Mark Reference Point

| Who | Action |
|-----|--------|
| T1 | Choose a visible point on vehicle exterior (rear corner, tow hook, etc.) |
| T2 | Mark it permanently (paint dot, engraved mark, sticker) |

**Good reference points:**
- Rear tow hook
- Corner of bumper
- Wheel hub center
- Any permanent, visible feature

#### Step 1.2: Measure IMU Offset

| Who | Action |
|-----|--------|
| T2 | Measure from reference point to IMU center with tape measure |
| T1 | Record as `imu_offset = [X, Y, Z]` |

**Directions:**
- X: Forward distance (IMU forward of reference = positive)
- Y: Right distance (IMU right of reference = positive)
- Z: Down distance (IMU below reference = positive, IMU above = negative)

```
Example: Reference at ground level, IMU is 1m forward, 1.2m UP
imu_offset = [1.0, 0.0, -1.2]   (negative Z because IMU is above reference)
```

**Accuracy:** ±20cm acceptable. This is a systematic offset applied to all cameras.

---

### PHASE 2: Ground Position Setup

*Do once per camera set.*

#### Step 2.1: RTK Setup

| Who | Action |
|-----|--------|
| T2 | Position RTK base station with clear sky view |
| T2 | Wait for RTK fix (1-2 minutes) |
| T2 | Place rover at vehicle reference point |
| T1 | Record reference point RTK coordinates as origin |

#### Step 2.2: Mark Ground Positions

For each camera, mark 3-5 positions within its FOV:

| Who | Action |
|-----|--------|
| T1 | View camera feed, identify suitable positions |
| T2 | Walk to position, place RTK rover on ground |
| T1 | Record RTK coordinates, compute X (forward), Y (right) from reference |
| T2 | Mark position on ground (spray paint, stake, tape) |

**Tips:**
- Space positions 0.5-1m apart
- Vary distances from camera (2-4m typical)
- All positions must show full board in camera FOV

```
Example ground positions for front-right camera:
  Position 1: X=4.0m forward, Y=0.85m right
  Position 2: X=4.5m forward, Y=1.00m right
  Position 3: X=5.0m forward, Y=1.15m right
```

---

### PHASE 3: Camera Prior Measurement

*Do for each camera before capture.*

#### Step 3.1: Measure Camera Position (Rough Estimate)

| Who | Action |
|-----|--------|
| T2 | Use tape measure from reference point to camera lens center |
| T1 | Record as `camera_prior_position = [X, Y, Z]` |

```
Example: Camera is 1.5m forward, 0.2m right, 1.5m ABOVE reference
camera_prior_position = [1.5, 0.2, -1.5]   (negative Z = above)
```

**Accuracy:** ±20cm is fine. Optimizer will refine this.

#### Step 3.2: Estimate Camera Orientation

| Who | Action |
|-----|--------|
| T1 | Estimate from mounting specs or visual inspection |
| T1 | Record as `camera_prior_orientation = [azimuth, elevation, roll]` |

**Angles:**
- Azimuth: 0=forward, +90=right, -90=left, ±180=rear
- Elevation: 0=horizontal, positive=looking down
- Roll: usually 0 (camera upright)

**Accuracy:** ±5 deg is fine. Optimizer will refine this.

---

### PHASE 4: Calibration Capture

*Do for each camera, at each marked position.*

#### Step 4.1: Position Board

| Who | Action |
|-----|--------|
| T2 | Place board stand at marked ground position |
| T2 | Adjust so board CENTER is directly above the mark |
| T2 | Hold board LEVEL (vertical, not tilted) |
| T2 | Face board toward camera |

```
CORRECT board placement:

    Marked spot on ground
           |
           v
    +------+------+
    |      |      |
    |   BOARD     |  <- Board center directly above mark
    |   CENTER    |
    |      |      |
    +------+------+
```

#### Step 4.2: Measure Board Center Height

| Who | Action |
|-----|--------|
| T2 | Use laser or tape to measure height of board CENTER from ground |
| T2 | Report to T1 (e.g., "0.9 meters") |
| T1 | Record board_height |

**Board position computation:**
```
Ground position from RTK:  X=4.0, Y=0.85
Board center height:       0.9m above ground

Board position = [4.0, 0.85, -0.9]   (negative Z = above ground)
```

#### Step 4.3: Measure Distance

| Who | Action |
|-----|--------|
| T2 | Use laser meter from camera lens to board center |
| T2 | Report to T1 (e.g., "2.65 meters") |
| T1 | Record laser_distance |

#### Step 4.4: Verify and Capture

| Who | Action |
|-----|--------|
| T1 | Check camera preview: board centered, no blur, no glare |
| T1 | Verify corner detection: "corners=81" (all corners visible) |
| T1 | Capture image |
| T1 | Verify: "reproj < 1px" |

#### Step 4.5: Record INS Data

| Who | Action |
|-----|--------|
| T1 | Record current INS yaw/pitch/roll (auto-captured or manual entry) |

**Repeat for all positions (minimum 3, recommended 5)**

---

### PHASE 5: Run Calibration

| Who | Action |
|-----|--------|
| T1 | Verify all data entered correctly |
| T1 | Run optimization |
| T1 | Check results meet specs |
| T1 | Save to JSON file |

---

### Expected Results

```
POSITION RELATIVE TO REFERENCE POINT (optimized from RTK data):
   Computed: [1.501, 0.198, -1.501]m
   Truth:    [1.500, 0.200, -1.500]m
   Error:    0.2cm (was 18.7cm before optimization)

POSITION RELATIVE TO IMU (= pos_ref - imu_offset):
   Computed: [0.421, 0.258, -0.401]m
   Truth:    [0.500, 0.200, -0.300]m
   Error:    14.1cm (limited by IMU offset error)

ORIENTATION (optimized):
   Azimuth:   +15.01 deg (error: +0.010 deg)
   Elevation: +5.01 deg (error: +0.007 deg)

SPEC COMPLIANCE:
  Position (ref point):   0.2cm  PASS  (spec: < 5cm)
  Position (IMU):        14.1cm  PASS  (spec: < 25cm)
  Azimuth:              0.010 deg PASS  (spec: < 1 deg)
  Elevation:            0.007 deg PASS  (spec: < 1 deg)
```

### Interpreting Results

| Parameter | Meaning | Example |
|-----------|---------|---------|
| Azimuth +15 deg | Camera points 15 deg RIGHT of vehicle forward | Right-facing camera |
| Azimuth -45 deg | Camera points 45 deg LEFT of vehicle forward | Left-facing camera |
| Elevation +5 deg | Camera points 5 deg DOWN from horizontal | Typical downward tilt |
| Elevation -5 deg | Camera points 5 deg UP from horizontal | Upward-looking camera |
| Roll 0 deg | Camera "up" aligns with vehicle "up" | Normal mounting |

---

## Multi-Camera Calibration

Each camera is calibrated **independently** with its own ground positions.

**Critical:** All ground positions are measured from the **SAME vehicle reference point**. This ensures all cameras end up in a unified coordinate system.

```
Vehicle Reference Point
          |
          v
      [0,0,0]
          |
   +------+------+
   |             |
   | Spots for   | Spots for
   | LEFT cam    | RIGHT cam
   |             |
```

**Example Setup:**

| Camera | Ground Positions (in its FOV) |
|--------|-------------------------------|
| Front-right | [4.0, 0.85, -0.9], [4.5, 1.0, -0.9], [5.0, 1.15, -0.9] |
| Front-left | [4.0, -0.85, -0.9], [4.5, -1.0, -0.9], [5.0, -1.15, -0.9] |
| Rear | [-3.0, 0.5, -0.9], [-3.5, -0.5, -0.9], [-4.0, 0.0, -0.9] |

**Procedure:**

1. Choose ONE vehicle reference point (use for ALL cameras)
2. Measure IMU offset ONCE (use for ALL cameras)
3. For each camera:
   - Mark 3-5 spots on ground within that camera's FOV
   - Measure each spot from vehicle reference (RTK)
   - Place board at each spot, measure height and distance, capture
   - Run calibration
4. Each camera produces its own JSON file

**Result:** All cameras share the same coordinate system (NED, origin at vehicle reference).

---

## Output Format

### Extrinsics JSON

```json
{
  "camera_id": "camera_front_right",
  "translation_vector_reference": [1.501, 0.198, -1.501],
  "translation_vector_imu": [0.421, 0.258, -0.401],
  "imu_offset_measured": [1.08, -0.06, -1.10],
  "rotation_matrix": [[...], [...], [...]],
  "euler_angles": {
    "azimuth": 15.01,
    "elevation": 5.01,
    "roll": 0.49
  },
  "board_center_height_m": 0.9,
  "coordinate_system": {
    "frame": "NED (X=Forward, Y=Right, Z=Down)",
    "reference_point": "Vehicle reference mark",
    "note": "translation_vector_imu = translation_vector_reference - imu_offset"
  },
  "optimization_converged": true
}
```

**Key outputs:**
- `translation_vector_reference`: Camera position relative to reference point (very accurate, <1cm)
- `translation_vector_imu`: Camera position relative to IMU (inherits IMU offset error ~15-20cm)
- `euler_angles`: Camera orientation (very accurate, <0.1 deg)

**Note:** If you improve IMU offset measurement later, just update `imu_offset` - no need to recalibrate cameras.

---

## Accuracy Summary

| Parameter | Accuracy | Limited By |
|-----------|----------|------------|
| Position (vs reference) | <1cm | RTK accuracy |
| Position (vs IMU) | ~15-20cm | IMU offset measurement |
| Azimuth | <0.1 deg | Optimization |
| Elevation | <0.1 deg | Optimization |

---

## Troubleshooting

### "No corners detected"
- Board too far (>5m) or too close (<1.5m)
- Poor lighting or glare on board
- Board out of focus
- Board tilted too much (should be nearly vertical)

### High position error
- RTK positions inaccurate (check fix quality)
- Board not placed at CENTER of marked spots
- Board height measurement wrong
- Fewer than 3 measurements

### High orientation error
- Board not held level during capture
- INS providing incorrect orientation
- Prior orientation estimate too far off (>10 deg)

### Optimization fails to converge
- Check that prior position/orientation estimates are reasonable
- Ensure board is visible and detected in all captures
- Verify ground positions are correctly measured
- Need at least 3 measurements

---

## Quick Reference

```
PHASE 1 (once per vehicle):
  [T2] Mark reference point on vehicle
  [T2] Measure ref -> IMU offset with tape
  [T1] Record imu_offset = [X, Y, Z]

PHASE 2 (once per camera set):
  [T2] Set up RTK, mark reference position
  [T2] Walk to each spot, place RTK rover
  [T1] Record ground X, Y for each position
  [T2] Mark positions on ground

PHASE 3 (per camera):
  [T2] Measure ref -> camera with tape
  [T1] Record camera_prior_position = [X, Y, Z]
  [T1] Estimate camera_prior_orientation = [az, el, roll]

PHASE 4 (per camera, per position):
  [T2] Place board at mark, level, facing camera
  [T2] Measure board center height -> T1 records
  [T2] Measure laser distance -> T1 records
  [T1] Verify 81 corners, capture image
  [T1] Record INS yaw/pitch/roll

PHASE 5:
  [T1] Run calibration, verify specs, save JSON
```

---

## Usage

### Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-contrib-python==4.8.1.78 "numpy<2"
pip install scipy
```

### Intrinsic Calibration (once per camera)

```bash
# With real images
python3 intrinsic_calibration.py --images intrinsic_images/ --output camera_intrinsics.json

# Synthetic demo (no hardware)
python3 intrinsic_calibration.py --synthetic --output demo_intrinsics.json
```

### Extrinsic Calibration (once per installation)

```bash
# Demo mode (no hardware needed)
python3 extrinsic_calibration.py --demo -n 5 -o demo_extrinsics.json

# Demo with your intrinsics file
python3 extrinsic_calibration.py --demo -i camera_intrinsics.json -n 5 -o camera_extrinsics.json
```

### View Results

```bash
cat demo_extrinsics.json
```

### Complete Demo Workflow (No Hardware Needed)

```bash
# Step 1: Generate synthetic intrinsics
python3 intrinsic_calibration.py --synthetic -o demo_intrinsics.json

# Step 2: Run extrinsic calibration demo
python3 extrinsic_calibration.py --demo -i demo_intrinsics.json -n 5 -o demo_extrinsics.json

# Step 3: Review results
cat demo_extrinsics.json
```

### Demo vs Real Calibration

| Step | Demo Mode | Real Calibration |
|------|-----------|------------------|
| Ground positions | Simulated (RTK ~1cm error) | Measured with RTK |
| Board height | Simulated | Measured with laser/tape |
| Board placement | Simulated | Operator places board |
| Laser distance | Simulated (~2cm error) | Measured with laser |
| IMU offset | Simulated (~15cm error) | Measured with tape |
| INS | Simulated | Real INS data |
| Images | Synthetic | Real camera images |

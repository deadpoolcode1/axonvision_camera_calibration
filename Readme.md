# Camera Calibration System

Multi-camera calibration system with IMU integration for vehicle/platform applications.

## Overview

This system performs two-phase camera calibration:
1. **Intrinsic Calibration** - Camera internal parameters (focal length, principal point, distortion)
2. **Extrinsic Calibration** - Camera pose relative to IMU reference frame

## Requirements

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install opencv-contrib-python==4.8.1.78 "numpy<2"
```

---

## ChArUco Board Specifications

For vehicle cameras at 2-8m distances:

| Parameter | Value |
|-----------|-------|
| Size | 1.1m × 1.1m (110cm × 110cm) |
| Pattern | 10×10 ChArUco squares |
| Square size | 11cm |
| Marker size | 8.5cm |
| Dictionary | DICT_6X6_250 |
| Material | **Rigid** aluminum composite or foam board |

⚠️ **Board MUST be rigid** - paper or flexible material will cause calibration errors!

### Printing the Board

```python
import cv2
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((10, 10), 0.11, 0.085, aruco_dict)
board_image = board.generateImage((4400, 4400))  # High resolution for printing
cv2.imwrite("charuco_board_10x10.png", board_image)
```

Print at **exactly 110cm × 110cm** on rigid material. Mark the center with tape.

---

## Intrinsic Calibration

Determines camera internal parameters (focal length, principal point, distortion).

### Usage

```bash
# Synthetic mode (testing/development)
python3 intrinsic_calibration.py --synthetic --camera-id camera_1 -o camera_1_intrinsics.json

# Real camera mode (production)
python3 intrinsic_calibration.py --camera-index 0 --camera-id camera_1 -o camera_1_intrinsics.json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera-id` | camera_1 | Camera identifier |
| `--output, -o` | camera_intrinsics.json | Output JSON file |
| `--num-images, -n` | 25 | Number of calibration images |
| `--synthetic` | off | Use synthetic images for testing |
| `--camera-index` | 0 | Camera device index (real mode) |

### Output Format

```json
{
  "camera_id": "camera_1",
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "image_size": [1920, 1080],
  "rms_error": 0.48,
  "calibration_date": "2025-11-30 08:53:00"
}
```

### Quality Metrics

| RMS Error | Quality | Action |
|-----------|---------|--------|
| < 0.5 px | Good | Accept |
| 0.5 - 1.0 px | Acceptable | Consider recalibration |
| > 1.0 px | Poor | Recalibrate |

---

## Extrinsic Calibration

Determines camera position and orientation relative to the IMU/world reference frame.

### Usage

```bash
# Demo mode (interactive training with synthetic images)
python3 extrinsic_calibration.py --demo \
    --intrinsics camera_1_intrinsics.json \
    --camera-id camera_1 \
    --output camera_1_extrinsics.json \
    --num-positions 7

# Synthetic mode (automated algorithm testing)
python3 extrinsic_calibration.py --synthetic \
    --intrinsics camera_1_intrinsics.json \
    --camera-id camera_1 \
    --output camera_1_extrinsics.json \
    --num-positions 10

# Real camera mode (production)
python3 extrinsic_calibration.py \
    --intrinsics camera_1_intrinsics.json \
    --camera-id camera_1 \
    --output camera_1_extrinsics.json \
    --num-positions 7
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--intrinsics, -i` | (required) | Path to intrinsics JSON |
| `--camera-id` | camera_1 | Camera identifier |
| `--output, -o` | camera_extrinsics.json | Output JSON file |
| `--num-positions, -n` | 7 | Target number of board positions |
| `--synthetic` | off | Automated test mode |
| `--demo` | off | Interactive demo with synthetic images |
| `--min-measurements` | 3 | Minimum before quality check |

### Adaptive Stopping

The calibration automatically checks quality after every measurement (once minimum is reached):

```
📈 QUALITY CHECK:
   Azimuth std:   0.32° ✓
   Elevation std: 0.28° ✓
   Reproj error:  0.45 px
   → Quality is GOOD - can finalize calibration

🎯 Quality threshold met!
   Continue to 7 measurements or finalize now? [c]ontinue/[f]inalize: 
```

**Quality thresholds:**

| Metric | Good (can stop) | Acceptable | Poor (keep going) |
|--------|-----------------|------------|-------------------|
| Azimuth std | < 0.5° | < 1.0° | ≥ 1.0° |
| Elevation std | < 0.5° | < 1.0° | ≥ 1.0° |
| Reproj error | < 1.0 px | < 1.5 px | ≥ 1.5 px |

**Workflow:**
- After 3+ measurements, quality is checked
- If quality is GOOD → option to finalize early
- If quality is POOR at target → option to take more measurements
- System suggests when to stop or continue

### Output Format

```json
{
  "camera_id": "camera_1",
  "rotation_matrix": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
  "translation_vector": [x, y, z],
  "euler_angles": {
    "azimuth": 45.0,
    "elevation": -10.0,
    "roll": 0.5
  },
  "quality_metrics": {
    "num_measurements": 7,
    "mean_reproj_error_px": 0.45,
    "azimuth_std_deg": 0.3,
    "elevation_std_deg": 0.2
  },
  "calibration_date": "2025-11-30 10:30:00"
}
```

### Quality Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Reprojection error | < 0.5 px | 0.5 - 1.5 px | > 1.5 px |
| Azimuth std | < 0.3° | 0.3 - 1.0° | > 1.0° |
| Elevation std | < 0.3° | 0.3 - 1.0° | > 1.0° |
| Position error | < 5% | 5 - 10% | > 10% |

---

## Coordinate System

```
World/IMU Frame:
  • Origin: IMU sensor location
  • X-axis: FORWARD (direction vehicle faces)
  • Y-axis: RIGHT (passenger side)
  • Z-axis: UP (toward sky)

Camera Frame:
  • Z-axis: Optical axis (forward, out of lens)
  • X-axis: Right (image horizontal)
  • Y-axis: Down (image vertical)

Euler Angles:
  • Azimuth: Camera heading (0° = forward/+X, 90° = right/+Y)
  • Elevation: Angle from horizontal (negative = looking down)
  • Roll: Rotation around optical axis
```

```
                TOP VIEW
                
        +X (Forward)
             ↑
             │
             │    Camera optical axis
             │        ↗
             │      ↗  azimuth angle
             │    ↗
             │  ↗
  ───────────●─────────→ +Y (Right)
            IMU
           Origin      ◎ Camera
```

---

## Extrinsic Calibration Procedure (Two-Operator)

### Personnel Required

| Role | Responsibilities |
|------|------------------|
| **Operator A** (at board) | Holds/positions board, measures position, reports measurements |
| **Operator B** (at monitor) | Watches camera preview, directs positioning, captures images, records data |

### Equipment Checklist

- [ ] ChArUco board (1.1m × 1.1m, rigid, center marked)
- [ ] Board stand or tripod (to hold board vertical)
- [ ] Laser distance meter (preferred) or tape measure
- [ ] Monitor/laptop showing camera live preview
- [ ] Communication (radio or line-of-sight voice)
- [ ] Clipboard for recording measurements

### Procedure Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXTRINSIC CALIBRATION WORKFLOW                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐         Voice/Radio          ┌──────────────┐    │
│  │              │ ◄──────────────────────────► │              │    │
│  │  OPERATOR A  │                              │  OPERATOR B  │    │
│  │  (at board)  │                              │ (at monitor) │    │
│  │              │                              │              │    │
│  └──────┬───────┘                              └──────┬───────┘    │
│         │                                             │            │
│         ▼                                             ▼            │
│  • Position board                              • Watch preview     │
│  • Measure X, Y, Z                             • Guide centering   │
│  • Hold board steady                           • Capture image     │
│  • Report measurements                         • Record data       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Procedure

#### Phase 1: Setup

1. **Position equipment:**
   - Operator B sets up monitor with camera live preview
   - Operator A takes board and measuring equipment to field

2. **Establish communication:**
   - Test voice/radio communication between operators
   - Agree on terminology: "left/right" = Operator A's left/right

3. **Identify IMU origin:**
   - Mark IMU location clearly (this is the measurement origin)
   - All X, Y, Z measurements are FROM this point

#### Phase 2: Find Optical Axis (First Measurement)

This is the most critical step - it establishes the camera direction.

```
OPERATOR B                              OPERATOR A
──────────────────────────────────────────────────────────────────────
"I don't see the board yet"             [Walks into expected camera
                                         field of view, ~4m away]

"I see you! Move LEFT..."               [Moves left]

"More... more... STOP!"                 [Stops]

"Now move FORWARD a bit..."             [Moves forward]

"STOP! You're centered. Hold there."    [Holds position]

"What's your position?"                 [Measures from IMU]
                                        "X = 3.2, Y = 2.9, Z = 0.5"

[Records: X=3.2, Y=2.9, Z=0.5]
[Calculates yaw ≈ 225°]
[Captures image]

"Good! That's position 1. The yaw       [Remembers: stay on this line,
is 225°, use that for all."             always face 225°]
──────────────────────────────────────────────────────────────────────
```

**Calculating Yaw:**
- Yaw = direction FROM board TO camera
- If board is at (3.2, 2.9) and camera/IMU is at (0.5, 0.8):
  - Direction = atan2(0.8 - 2.9, 0.5 - 3.2) = atan2(-2.1, -2.7) ≈ 218° ≈ **225°**

#### Phase 3: Remaining Measurements (6 more positions)

For each measurement, Operator A moves along the optical axis (closer/further) and varies height:

```
OPERATOR B                              OPERATOR A
──────────────────────────────────────────────────────────────────────
MEASUREMENT 2:
"Walk STRAIGHT BACK about 1 meter"      [Walks back along same line]

"Little RIGHT... stop, you're           [Adjusts until centered]
centered again."

"Raise the board about 30cm"            [Raises board]

"What's your position?"                 "X = 4.0, Y = 3.6, Z = 0.8"
                                        "Yaw still 225°"

[Captures image]
"Good! Position 2 done."
──────────────────────────────────────────────────────────────────────
MEASUREMENT 3:
"Move back another meter,               [Moves back, lowers board]
lower the board this time"

"Adjust LEFT slightly..."               [Adjusts]

"Good, what's your position?"           "X = 4.9, Y = 4.4, Z = 0.3"

[Captures image]
──────────────────────────────────────────────────────────────────────
... repeat for measurements 4-7 ...
──────────────────────────────────────────────────────────────────────
```

#### Measurement Pattern

| # | Distance | Height | Notes |
|---|----------|--------|-------|
| 1 | 3-4m | Baseline | Find optical axis, establish yaw |
| 2 | 4-5m | +30cm | Walk back, raise board |
| 3 | 5-6m | -20cm | Walk back, lower board |
| 4 | 3-4m | -30cm | Come forward, low position |
| 5 | 4-5m | Baseline | Mid distance, normal height |
| 6 | 5-6m | +30cm | Far, high position |
| 7 | 6-7m | Baseline | Furthest position |

### Critical Rules

```
╔══════════════════════════════════════════════════════════════════════╗
║                        ⚠️  CRITICAL RULES                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. KEEP BOARD CENTERED IN IMAGE                                     ║
║     • Operator B must guide Operator A until board is centered       ║
║     • "Centered" = board in middle of frame, not off to side         ║
║                                                                      ║
║  2. USE SAME YAW FOR ALL MEASUREMENTS                                ║
║     • Calculate yaw from first (centered) position                   ║
║     • Operator A keeps board facing same direction every time        ║
║                                                                      ║
║  3. VARY DISTANCE AND HEIGHT ONLY                                    ║
║     • Move closer/further along optical axis                         ║
║     • Move up/down (±30cm)                                           ║
║     • Do NOT move left/right of optical axis                         ║
║                                                                      ║
║  4. MEASURE FROM IMU ORIGIN                                          ║
║     • All X, Y, Z measurements are from IMU location                 ║
║     • Measure to board CENTER (mark it with tape)                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Visual Guide: Correct vs Incorrect

```
CAMERA VIEW - CORRECT (board centered):

    ┌─────────────────────────────────────┐
    │                                     │
    │           ╔═══════════╗             │
    │           ║           ║             │
    │           ║   BOARD   ║  ← GOOD!    │
    │           ║           ║             │
    │           ╚═══════════╝             │
    │                                     │
    └─────────────────────────────────────┘


CAMERA VIEW - INCORRECT (board off to side):

    ┌─────────────────────────────────────┐
    │                                     │
    │  ╔═══════════╗                      │
    │  ║           ║                      │
    │  ║   BOARD   ║  ← BAD! Will cause   │
    │  ║           ║    calibration       │
    │  ╚═══════════╝    errors!           │
    │                                     │
    └─────────────────────────────────────┘
```

```
TOP VIEW - CORRECT (positions along optical axis):

                    Camera optical axis
                          ↓
        ═══       ═══       ═══       ═══
        M1        M2        M3        M4
        (3m)      (4m)      (5m)      (6m)
          \         \         \         \
           \         \         \         \
            \─────────\─────────\─────────\ optical axis
             \         \         \         \
              ◎ Camera

    ✓ All boards along same line (optical axis)
    ✓ All boards face same direction (same yaw)
    ✓ Only distance varies


TOP VIEW - INCORRECT (positions scattered):

                    
        ═══                   ═══
        M1                    M3
                    
                ═══       
                M2       ═══
                         M4
              ◎ Camera

    ✗ Boards at different angles
    ✗ Not along optical axis
    ✗ Will cause large errors!
```

### Measurement Recording Sheet

```
EXTRINSIC CALIBRATION - DATA SHEET

Camera ID: ________________     Date: ________________
Operator A: _______________     Operator B: _______________

IMU Origin marked at: _________________________________

YAW ANGLE (from first measurement): ______°

┌─────┬────────┬────────┬────────┬─────────┬──────────┬─────────┐
│  #  │ X (m)  │ Y (m)  │ Z (m)  │ Yaw (°) │ Centered?│ Corners │
├─────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤
│  1  │        │        │        │         │  Y / N   │         │
├─────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤
│  2  │        │        │        │         │  Y / N   │         │
├─────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤
│  3  │        │        │        │         │  Y / N   │         │
├─────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤
│  4  │        │        │        │         │  Y / N   │         │
├─────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤
│  5  │        │        │        │         │  Y / N   │         │
├─────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤
│  6  │        │        │        │         │  Y / N   │         │
├─────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤
│  7  │        │        │        │         │  Y / N   │         │
└─────┴────────┴────────┴────────┴─────────┴──────────┴─────────┘

Notes:
_____________________________________________________________
_____________________________________________________________
```

---

## Multi-Camera Calibration

For systems with N cameras (with or without overlapping fields of view):

```bash
# Camera 1
python3 intrinsic_calibration.py --camera-index 0 -o cam1_intrinsics.json
python3 extrinsic_calibration.py -i cam1_intrinsics.json -o cam1_extrinsics.json

# Camera 2
python3 intrinsic_calibration.py --camera-index 1 -o cam2_intrinsics.json
python3 extrinsic_calibration.py -i cam2_intrinsics.json -o cam2_extrinsics.json

# Camera 3 (etc.)
python3 intrinsic_calibration.py --camera-index 2 -o cam3_intrinsics.json
python3 extrinsic_calibration.py -i cam3_intrinsics.json -o cam3_extrinsics.json
```

Each camera is calibrated independently to the same IMU origin, so:
- Overlapping cameras: will see same world points consistently
- Non-overlapping cameras: can still transform to common world frame
- All cameras share the same coordinate system

---

## Troubleshooting

### Intrinsic Calibration

| Problem | Cause | Solution |
|---------|-------|----------|
| High RMS (>1px) | Poor corner detection | Better lighting, sharper focus |
| Few corners detected | Board too far/angled | Move closer, face camera |
| Unstable focal length | Insufficient coverage | Capture more varied poses |

### Extrinsic Calibration

| Problem | Cause | Solution |
|---------|-------|----------|
| High azimuth std (>1°) | Board not centered | Re-do with better centering |
| High elevation std (>1°) | Inconsistent height measurement | Use laser distance meter |
| Large position error | Wrong IMU origin | Verify measurement reference point |
| Detection failures | Lighting/blur | Improve conditions, hold board steady |
| Results don't converge | Different yaw angles used | Use SAME yaw for all measurements |

### Common Mistakes

| Mistake | Why It's Bad | How to Fix |
|---------|--------------|------------|
| Board off-center in image | Adds systematic angle error | Always center board in preview |
| Different yaw each measurement | Inconsistent geometry | Use same yaw from first measurement |
| Measuring to board corner | Wrong reference point | Always measure to board CENTER |
| Board not vertical | Invalid pose assumption | Use level, keep board plumb |
| Moving laterally instead of along axis | Breaks centering assumption | Move only closer/further and up/down |

---

## Expected Accuracy

### With Proper Technique

| Metric | Expected Value |
|--------|----------------|
| Position error | < 5 cm (< 5% of distance) |
| Azimuth error | < 0.5° |
| Elevation error | < 0.5° |
| Roll error | < 1° |

### Factors Affecting Accuracy

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Board centering | High | Use live preview, two operators |
| Measurement precision | High | Use laser distance meter |
| Board rigidity | Medium | Use aluminum composite or thick foam board |
| Yaw consistency | High | Calculate once, use for all |
| Number of measurements | Medium | Take 7-10 measurements |

---

## File Structure

```
project/
├── intrinsic_calibration.py    # Intrinsic calibration module
├── extrinsic_calibration.py    # Extrinsic calibration module
├── camera_1_intrinsics.json    # Camera 1 intrinsic parameters
├── camera_1_extrinsics.json    # Camera 1 extrinsic parameters
├── camera_2_intrinsics.json    # Camera 2 intrinsic parameters
├── camera_2_extrinsics.json    # Camera 2 extrinsic parameters
└── README.md                   # This file
```

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────────┐
│              EXTRINSIC CALIBRATION QUICK REFERENCE                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  COORDINATE SYSTEM:                                                │
│    X = Forward    Y = Right    Z = Up    Origin = IMU              │
│                                                                    │
│  TWO OPERATORS:                                                    │
│    A = At board (position, measure, hold)                          │
│    B = At monitor (guide, capture, record)                         │
│                                                                    │
│  PROCEDURE:                                                        │
│    1. Center board in image (B guides A)                           │
│    2. Measure position, calculate yaw                              │
│    3. Capture image                                                │
│    4. Move along optical axis (closer/further, up/down)            │
│    5. Keep same yaw, re-center, measure, capture                   │
│    6. Repeat for 7 positions total                                 │
│                                                                    │
│  CRITICAL:                                                         │
│    ✓ Always center board in image                                  │
│    ✓ Same yaw for all measurements                                 │
│    ✓ Vary distance (3-7m) and height (±30cm)                       │
│    ✗ Don't move board left/right of optical axis                   │
│                                                                    │
│  QUALITY CHECK:                                                    │
│    Azimuth std < 1°     Elevation std < 1°     Reproj < 1.5px     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

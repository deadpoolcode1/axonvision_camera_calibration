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

## Intrinsic Calibration

### Usage

```bash
# Synthetic mode (testing/development)
python3 intrinsic_calibration.py --synthetic --camera-id camera_1 -o camera_1_intrinsics.json

# Real camera mode (production)
python3 intrinsic_calibration.py --camera-index 0 --camera-id camera_1 -o camera_1_intrinsics.json

# With image saving for review
python3 intrinsic_calibration.py --synthetic --camera-id camera_1 -o camera_1_intrinsics.json \
    --save-images --image-dir calibration_images/
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera-id` | camera_1 | Camera identifier in output |
| `--output, -o` | camera_intrinsics.json | Output JSON file |
| `--num-images, -n` | 25 | Number of calibration images |
| `--synthetic` | off | Use synthetic images for testing |
| `--camera-index` | 0 | Camera device index (real mode) |
| `--save-images` | off | Save annotated calibration images |
| `--image-dir` | calibration_images | Directory for saved images |
| `--board-squares-x` | 10 | ChArUco board squares (horizontal) |
| `--board-squares-y` | 10 | ChArUco board squares (vertical) |
| `--board-square-size` | 0.11 | Square size in meters |

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
# Demo mode (interactive workflow with synthetic images)
python3 extrinsic_calibration.py --demo \
    --intrinsics camera_1_intrinsics.json \
    --camera-id camera_1 \
    --output camera_1_extrinsics.json \
    --num-positions 7

# Synthetic mode (automated testing)
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
    --num-positions 7 \
    --image-dir calibration_images/
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--intrinsics, -i` | (required) | Path to intrinsics JSON from phase 1 |
| `--camera-id` | camera_1 | Camera identifier |
| `--output, -o` | camera_extrinsics.json | Output JSON file |
| `--num-positions, -n` | 7 | Number of board positions |
| `--synthetic` | off | Automated test with synthetic data |
| `--demo` | off | Interactive demo with synthetic images |
| `--image-dir` | none | Directory containing calibration images |

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
    "azimuth_std_deg": 0.8,
    "elevation_std_deg": 0.4
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

### Coordinate System

```
World/IMU Frame:
  • Origin: IMU sensor location
  • X-axis: FORWARD (direction vehicle faces)
  • Y-axis: RIGHT (passenger side)
  • Z-axis: UP (toward sky)

Camera Frame:
  • Z-axis: Optical axis (forward, out of lens)
  • X-axis: Right (in image horizontal direction)
  • Y-axis: Down (in image vertical direction)

Euler Angles:
  • Azimuth: Camera heading (0° = forward, 90° = right)
  • Elevation: Angle from horizontal (negative = looking down)
  • Roll: Rotation around optical axis
```

### Operator Workflow

The extrinsic calibration requires an operator to place a ChArUco board at multiple positions and measure:

1. **Board CENTER position** (X, Y, Z in meters relative to IMU origin)
2. **Board YAW angle** (direction from board toward camera, in degrees)

```
                    TOP VIEW
                    
        +X (Forward)
             ↑
             │
   Board ═══ │         Measure X (forward distance)
   (facing   │         ←─────────────────────→
    camera)  │
        ↘    │
     yaw ↘   │
           ↘ │
  ───────────●─────────→ +Y (Right)
            IMU        
           Origin    ◎ Camera
                    (looking at board)
```

**For each measurement:**
1. Place board VERTICAL, markers facing camera, 3-7m away
2. Measure board center position (X, Y, Z) from IMU origin
3. Measure board yaw (direction from board to camera)
4. Capture image
5. Software detects corners and computes camera pose

**Best practices:**
- Take 7-10 measurements at different positions
- Vary distance: 3m, 4m, 5m, 6m, 7m
- Vary lateral position: left, center, right of camera view
- Vary height: above, at, below camera level

---

## Synthetic vs Real Mode

### What Synthetic Mode Emulates

Both calibrations generate realistic test data by:

1. **Rendering actual ChArUco board** - Uses OpenCV to generate the real board pattern
2. **Perspective projection** - Warps board image using camera projection model with distortion
3. **Varied poses** - Simulates board at different distances, angles, and positions
4. **Noise injection** - Adds realistic pixel noise to images
5. **Real detection pipeline** - Uses identical OpenCV detection code as production

### Delta to Reality

| Aspect | Synthetic | Real | Impact |
|--------|-----------|------|--------|
| Detection code | Identical | Identical | None |
| Calibration algorithm | Identical | Identical | None |
| Corner sub-pixel refinement | Identical | Identical | None |
| Image noise | Gaussian ~3px | Sensor-specific | Minimal |
| Lighting variation | None | Present | May need exposure control |
| Board flatness | Perfect | May have slight warp | Minimal if board is rigid |
| Motion blur | None | Possible | Reject blurry images |
| Focus | Perfect | May vary | Ensure sharp focus |
| Measurement accuracy | ±5mm, ±2° | Operator skill | Main error source |

### Expected Accuracy

**Intrinsic calibration:**
- RMS error: 0.3 - 0.8 pixels
- Focal length accuracy: ±0.5%
- Principal point accuracy: ±5 pixels

**Extrinsic calibration (with careful measurements):**
- Position error: < 5% of distance
- Azimuth error: < 1°
- Elevation error: < 1°

---

## ChArUco Board Specifications

For vehicle cameras at 2-8m distances:

- **Size**: 1.1m × 1.1m (110cm × 110cm)
- **Pattern**: 10×10 ChArUco squares
- **Square size**: 11cm
- **Marker size**: 8.5cm
- **Dictionary**: DICT_6X6_250
- **Material**: Rigid aluminum composite or foam board (must be flat!)

### Printing the Board

Generate the board image:
```python
import cv2
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((10, 10), 0.11, 0.085, aruco_dict)
board_image = board.generateImage((4400, 4400))  # 4x print resolution
cv2.imwrite("charuco_board_10x10.png", board_image)
```

Print at exactly 110cm × 110cm on rigid material.

---

## Calibration Best Practices

### Image Capture

1. **Coverage** - Board should appear in all regions of image (center, corners, edges)
2. **Distance** - Vary from near (2-3m) to far (6-8m)
3. **Angles** - Tilt board up to 30-45° in various directions
4. **Focus** - Ensure sharp corners (avoid motion blur)
5. **Lighting** - Even illumination, avoid reflections

### Measurement Accuracy (Extrinsic)

1. **Mark board center** - Put tape at exact center for consistent measurement
2. **Use laser distance meter** - More accurate than tape measure
3. **Use angle reference** - Compass or known alignment marks for yaw
4. **Keep board vertical** - Use a level or plumb line
5. **Measure from IMU** - Not from camera or vehicle corner

### Number of Measurements

| Calibration | Minimum | Recommended |
|-------------|---------|-------------|
| Intrinsic | 10 images | 20-30 images |
| Extrinsic | 3 positions | 7-10 positions |

---

## File Structure

```
sw/
├── intrinsic_calibration.py    # Intrinsic calibration module
├── extrinsic_calibration.py    # Extrinsic calibration module
├── camera_1_intrinsics.json    # Intrinsic output (camera 1)
├── camera_1_extrinsics.json    # Extrinsic output (camera 1)
├── camera_2_intrinsics.json    # Intrinsic output (camera 2)
├── camera_2_extrinsics.json    # Extrinsic output (camera 2)
├── calibration_images/         # Saved detection images (optional)
└── README.md
```

---

## Complete Calibration Workflow

```bash
# 1. Intrinsic calibration (once per camera, or when lens changes)
python3 intrinsic_calibration.py --synthetic \
    --camera-id camera_1 \
    -o camera_1_intrinsics.json

# 2. Extrinsic calibration (when camera is mounted on vehicle)
python3 extrinsic_calibration.py --demo \
    --intrinsics camera_1_intrinsics.json \
    --camera-id camera_1 \
    --output camera_1_extrinsics.json \
    --num-positions 7

# 3. Verify results
cat camera_1_extrinsics.json
```

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
| High position std | Inconsistent measurements | Use laser distance meter |
| High azimuth std | Yaw measurement error | Use compass/angle reference |
| Detection failures | Lighting/blur | Improve conditions |
| Results don't converge | Wrong coordinate system | Verify X=forward, Y=right, Z=up |

---

## Next Steps

- [x] Intrinsic calibration
- [x] Extrinsic calibration
- [ ] Multi-camera validation
- [ ] GUI for guided calibration
- [ ] Real-time pose verification

## License

Proprietary - Kamacode

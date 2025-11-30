# Camera Calibration System

Multi-camera calibration system with IMU integration for vehicle/platform applications.

## Overview

This system performs two-phase camera calibration:
1. **Intrinsic Calibration** - Camera internal parameters (focal length, principal point, distortion)
2. **Extrinsic Calibration** - Camera pose relative to IMU reference frame (coming soon)

## Requirements

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install opencv-contrib-python==4.8.1.78 "numpy<2"
```

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

## Synthetic vs Real Mode

### What Synthetic Mode Emulates

The synthetic mode generates realistic test data by:

1. **Rendering actual ChArUco board** - Uses OpenCV to generate the real board pattern
2. **Perspective projection** - Warps board image using camera projection model with distortion
3. **Varied poses** - Simulates board at different distances (3-7m), angles (±30°), and image positions
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

### What Changes for Real Camera

Only **one line** changes conceptually - the image source:

```python
# Synthetic (testing)
image_source = SyntheticImageSource(board_config, camera_config, num_images)

# Real (production)  
image_source = RealCameraSource(camera_index=0)
```

Everything else - detection, calibration, validation, output - is identical.

### Expected Real-World Results

With proper capture technique:
- RMS error: 0.3 - 0.8 pixels (similar to synthetic)
- Focal length accuracy: ±0.5%
- Principal point accuracy: ±5 pixels
- Distortion coefficients: Properly characterized

## ChArUco Board Specifications

For vehicle cameras at 2-8m distances:

- **Size**: 1.2m × 1.2m recommended
- **Pattern**: 10×10 ChArUco squares
- **Square size**: 11cm
- **Dictionary**: DICT_6X6_250
- **Material**: Rigid aluminum composite or plywood (must be flat)

## Calibration Best Practices

### Image Capture

1. **Coverage** - Board should appear in all regions of image (center, corners, edges)
2. **Distance** - Vary from near (2-3m) to far (6-8m)
3. **Angles** - Tilt board up to 30-45° in various directions
4. **Focus** - Ensure sharp corners (avoid motion blur)
5. **Lighting** - Even illumination, avoid reflections

### Number of Images

- Minimum: 10 images
- Recommended: 20-30 images
- More images improve accuracy but with diminishing returns

## File Structure

```
sw/
├── intrinsic_calibration.py   # Main calibration module
├── camera_1_intrinsics.json   # Output (camera 1)
├── camera_2_intrinsics.json   # Output (camera 2)
├── calibration_images/        # Saved detection images (optional)
└── README.md
```

## Next Steps

- [ ] Extrinsic calibration (camera-to-IMU transform)
- [ ] Multi-camera validation
- [ ] GUI for guided calibration

## License

Proprietary - Kamacode

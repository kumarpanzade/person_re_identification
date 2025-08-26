# Squash Player Tracking: Motion Tracking Improvements

## Problem Fixed

Fixed the issue where bounding boxes would lag behind fast-moving players, causing tracking errors where the box stays in place while the player moves out of it.

## Key Improvements

### 1. Enhanced Motion Prediction Model

- **Velocity History**: Tracks maintain a history of recent velocity measurements
- **Weighted Velocity Averaging**: Newer velocities have more influence on predictions
- **Acceleration Detection**: System now detects and accounts for acceleration
- **Center-based Tracking**: Tracks player center positions instead of just corners

### 2. Adaptive Tracking Parameters

- **Lower Base IoU Threshold**: Reduced from 0.2 to 0.15 to better handle fast movements
- **Dynamic Thresholds**: IoU threshold automatically lowers for players not seen recently
- **Distance-based Matching**: Improved distance metric for matching when IoU fails

### 3. Kalman-like Motion Smoothing

- **Velocity Smoothing**: Implemented exponential smoothing for velocity estimation
- **Multiple History Points**: Uses up to 5 previous positions for better predictions
- **Predictive Scaling**: Faster movements get higher predictive scaling

## Technical Details

1. **Velocity Model**: Now uses a multi-point velocity history with weighted averaging
2. **Acceleration Component**: Added acceleration terms to handle rapid direction changes
3. **Adaptive Thresholds**: IoU threshold decreases by 0.02 for each frame a track is missed

## Usage

The improved tracking is now built into the system. For best results with fast-moving players:

```bash
python run_tracker.py --video your_squash_video.mp4 --reid-thresh 0.35
```

## Limitations

- May still struggle with extremely rapid camera movements
- Very sudden direction changes might cause brief tracking lag
- Performance impact is minimal (approximately 5-10% slower processing)
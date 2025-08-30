# Squash Player Tracker

A computer vision system for tracking squash players throughout match videos.

## Features

- **Player Tracking**: Continuously tracks players throughout the entire match
- **Player Identification**: Assigns P1/P2/P3 identifiers to players based on position
- **Player Re-identification**: Maintains player identity even during occlusions or rapid movements
- **Summary Report**: Generates a summary image with player identities

## Architecture

The system combines multiple state-of-the-art computer vision models:

1. **Person Detection**: YOLOv8 - Detects players in each frame with high accuracy 
2. **Person Re-identification**: OSNet - Maintains player identity based on appearance
3. **Tracking**: ByteTrack-inspired algorithm - Tracks players through occlusions and fast movements

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/squash_player_tracker.git
cd squash_player_tracker

# Install dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
python run_tracker.py --video path/to/squash_video.mp4 --output results.mp4
```

### Advanced Options

```bash
python run_tracker.py --video path/to/squash_video.mp4 --output results.mp4 \
    --det-conf 0.5 \
    --reid-thresh 0.6 \
    --no-display
```

### Command Line Arguments

- `--video`: Path to input video file (required)
- `--output`: Path to save output video (optional)
- `--no-display`: Disable display of processed frames
- `--det-conf`: Person detection confidence threshold (default: 0.5)
- `--reid-thresh`: Re-ID similarity threshold (default: 0.5)
- `--cpu`: Use CPU instead of GPU for inference

## How It Works

1. **Player Detection**:
   - Players are detected in each frame using YOLOv8
   - Players are assigned P1/P2/P3 identifiers based on their position on the court

2. **Tracking Phase**:
   - ByteTrack-inspired algorithm tracks players across frames
   - OSNet maintains player identity through appearance features
   - Each player maintains their assigned identity throughout the match

3. **Output**:
   - Processed video with player identities and tracking visualization
   - Summary image showing player identities

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- OpenCV 4.8+
- CUDA-capable GPU (recommended)

## Model Credits

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- OSNet: [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)
- ByteTrack: [ByteTrack](https://github.com/ifzhang/ByteTrack)
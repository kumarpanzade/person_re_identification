# Squash Player Tracker

A computer vision system for identifying and tracking squash players throughout match videos.

## Features

- **Face Recognition**: Identifies players at the beginning of the match via facial recognition
- **Player Tracking**: Continuously tracks players throughout the entire match
- **Fallback ID**: Assigns P1/P2 identifiers when faces cannot be recognized
- **Player Re-identification**: Maintains player identity even during occlusions or rapid movements
- **Summary Report**: Generates a summary image with player identities

## Architecture

The system combines multiple state-of-the-art computer vision models:

1. **Face Recognition**: InsightFace (ArcFace) - Recognizes player faces at match start
2. **Person Detection**: YOLOv8 - Detects players in each frame with high accuracy 
3. **Person Re-identification**: OSNet - Maintains player identity based on appearance
4. **Tracking**: ByteTrack-inspired algorithm - Tracks players through occlusions and fast movements

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
    --id-frames 100 \
    --face-conf 0.6 \
    --det-conf 0.5 \
    --reid-thresh 0.6 \
    --no-display
```

### Command Line Arguments

- `--video`: Path to input video file (required)
- `--output`: Path to save output video (optional)
- `--no-display`: Disable display of processed frames
- `--id-frames`: Number of frames for identification phase (default: 50)
- `--face-conf`: Face recognition confidence threshold (default: 0.5)
- `--det-conf`: Person detection confidence threshold (default: 0.5)
- `--reid-thresh`: Re-ID similarity threshold (default: 0.5)
- `--cpu`: Use CPU instead of GPU for inference

## How It Works

1. **Identification Phase**:
   - At the beginning of the video, the system tries to recognize player faces
   - If faces are recognized, players are assigned their names
   - If faces cannot be recognized, players are assigned P1/P2 identifiers

2. **Tracking Phase**:
   - Players are detected in each frame using YOLOv8
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
- InsightFace: [DeepInsight](https://github.com/deepinsight/insightface)
- OSNet: [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)
- ByteTrack: [ByteTrack](https://github.com/ifzhang/ByteTrack)
# Squash Player Tracker - Improved Version

This version fixes issues with multiple ID assignments for the same player in squash videos.

## Key Improvements

1. **Improved Re-identification**:
   - Multiple feature samples per player
   - Historical feature comparison
   - Lower matching threshold (0.35) for better recall

2. **Enhanced Tracking**:
   - Longer track lifetimes (max_age=60)
   - Combined IoU and center distance for matching
   - Tracks player movement patterns

3. **Squash-Specific Optimizations**:
   - Enforces exactly 2 players (squash standard)
   - Appearance consistency checks
   - Prioritizes high-confidence detections

## Usage

To run with optimal settings for squash videos:

```bash
python run_tracker.py --video path/to/squash_video.mp4 --output results.mp4 --reid-thresh 0.35 --enforce-two-players
```

## Adjustable Parameters

- `--reid-thresh`: Adjust re-identification sensitivity (lower = more permissive matching)
- `--id-frames`: Number of frames to spend identifying players at the start
- `--enforce-two-players`: Forces the system to maintain exactly two player tracks

## Troubleshooting Multiple IDs

If you still encounter multiple IDs for the same player:

1. Increase track history with `--id-frames 100` for better initial identification
2. Try running with `--reid-thresh 0.3` for more permissive matching
3. For videos with very fast movement, lower the detection confidence: `--det-conf 0.4`

## Limitations

- Very rapid camera movement may still cause tracking issues
- Extreme lighting changes can affect appearance consistency
- Players wearing identical uniforms might be confused
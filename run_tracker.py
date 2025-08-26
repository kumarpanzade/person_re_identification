#!/usr/bin/env python3
"""
Squash Player Tracker - Main Script

This script provides a simple command-line interface to run the
squash player tracking system on video files.

Example usage:
    python run_tracker.py --video path/to/squash_video.mp4 --output results.mp4
"""

import os
import sys
import argparse
from squash_player_tracker.squash_tracker import SquashPlayerTracker

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Squash Player Tracking System')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (optional)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display of processed frames')
    parser.add_argument('--id-frames', type=int, default=50,
                        help='Number of frames for identification phase')
    parser.add_argument('--face-conf', type=float, default=0.5,
                        help='Face recognition confidence threshold')
    parser.add_argument('--det-conf', type=float, default=0.5,
                        help='Person detection confidence threshold')
    parser.add_argument('--reid-thresh', type=float, default=0.35,
                        help='Re-ID similarity threshold')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Validate video path
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Create output directory if needed
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    print("Initializing Squash Player Tracker...")
    
    try:
        # Initialize tracker
        tracker = SquashPlayerTracker(
            face_recognition_conf=args.face_conf,
            detection_conf=args.det_conf,
            reid_threshold=args.reid_thresh,
            use_gpu=not args.cpu
        )
        
        # Set identification phase frames
        tracker.set_identification_phase(args.id_frames)
        
        # Process video
        print(f"Processing video: {args.video}")
        tracker.process_video(
            video_path=args.video,
            output_path=args.output,
            display=not args.no_display
        )
        
        print("Processing complete!")
        if args.output:
            print(f"Output saved to: {args.output}")
            
        return 0
    
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
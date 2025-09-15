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
import cv2
import argparse
import tempfile
import datetime
from squash_player_tracker.squash_tracker import SquashPlayerTracker
from result_manager import create_result_directory, get_result_path, copy_input_to_results, save_run_info

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Squash Player Tracking System')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (optional)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display of processed frames')
    parser.add_argument('--det-conf', type=float, default=0.6,
                        help='Person detection confidence threshold')
    parser.add_argument('--reid-thresh', type=float, default=0.35,
                        help='Re-ID similarity threshold')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--max-width', type=int, default=None, 
                        help='Maximum display width (auto-detected if not specified)')
    parser.add_argument('--max-height', type=int, default=None,
                        help='Maximum display height (auto-detected if not specified)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to store results (default: results)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom name for this run (default: timestamp)')
    
    args = parser.parse_args()
    
    # Validate video path
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Create result directory structure
    result_dirs = create_result_directory(args.results_dir, args.run_name)
    print(f"Results will be saved to: {result_dirs['root']}")
    
    # Set up output path if not explicitly provided
    output_path = args.output
    if not output_path:
        output_path = get_result_path(result_dirs, args.video, "videos")
        print(f"Output video will be saved to: {output_path}")
        
    # Save a copy of the input video for reference
    copy_input_to_results(args.video, result_dirs)
    
    # Save run information
    params = vars(args)
    description = f"Player tracking run on video: {args.video}"
    save_run_info(result_dirs, params, description)
    
    print("Initializing Squash Player Tracker...")
    
    try:
        # Initialize tracker
        tracker = SquashPlayerTracker(
            detection_conf=args.det_conf,
            reid_threshold=args.reid_thresh,
            use_gpu=not args.cpu
        )
        
        video_path = args.video
        
        # Process video
        print(f"Processing video: {video_path}")
        tracker.process_video(
            video_path=video_path,
            output_path=output_path,
            display=not args.no_display,
            max_display_width=args.max_width,
            max_display_height=args.max_height
        )
        
        
        print("Processing complete!")
        print(f"Output saved to: {output_path}")
        print(f"All results saved in: {result_dirs['root']}")
            
        return 0
    
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
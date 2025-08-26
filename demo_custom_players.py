#!/usr/bin/env python3
"""
Squash Player Tracker - Demo with Custom Player Registration

This script demonstrates how to register custom players with the tracking system
before processing a video. Use this when you want to pre-register known players.

Example usage:
    python demo_custom_players.py --video path/to/squash_video.mp4 --player1 "John" --player1-img path/to/john.jpg
"""

import os
import sys
import cv2
import argparse
from squash_player_tracker.squash_tracker import SquashPlayerTracker

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Squash Player Tracking Demo with Custom Players')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (optional)')
    
    # Player 1 arguments
    parser.add_argument('--player1', type=str,
                        help='Name for Player 1')
    parser.add_argument('--player1-img', type=str,
                        help='Path to face image for Player 1')
    
    # Player 2 arguments
    parser.add_argument('--player2', type=str,
                        help='Name for Player 2')
    parser.add_argument('--player2-img', type=str,
                        help='Path to face image for Player 2')
    
    # Other arguments
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display of processed frames')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Validate video path
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Validate player image paths
    player_data = []
    
    if args.player1 and args.player1_img:
        if not os.path.exists(args.player1_img):
            print(f"Error: Player 1 image not found: {args.player1_img}")
            return 1
        player_data.append((args.player1, args.player1_img))
    
    if args.player2 and args.player2_img:
        if not os.path.exists(args.player2_img):
            print(f"Error: Player 2 image not found: {args.player2_img}")
            return 1
        player_data.append((args.player2, args.player2_img))
    
    # Initialize tracker
    print("Initializing Squash Player Tracker...")
    tracker = SquashPlayerTracker(use_gpu=not args.cpu)
    
    # Register custom players
    if player_data:
        print("Registering custom players...")
        for name, img_path in player_data:
            # Load player face image
            face_img = cv2.imread(img_path)
            if face_img is None:
                print(f"Error: Could not load image: {img_path}")
                continue
            
            # Register the player
            success = tracker.register_player(name, face_img)
            if success:
                print(f"Registered player: {name}")
            else:
                print(f"Failed to register player: {name}")
    
    # Skip identification phase since we've manually registered players
    if player_data:
        tracker.set_identification_phase(0)
    
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

if __name__ == "__main__":
    sys.exit(main())
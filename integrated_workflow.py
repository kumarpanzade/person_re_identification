#!/usr/bin/env python3
"""
Integrated Person Re-identification Workflow

This script combines person detection/clip extraction with person re-identification
to create a complete workflow:
1. Input video -> Extract person clips
2. Process each clip with person re-identification
3. Generate results and analysis

Example usage:
    python integrated_workflow.py --video path/to/video.mp4 --output results/
"""

import os
import sys
import cv2
import argparse
import tempfile
import datetime
import json
from typing import List, Dict, Any
from pathlib import Path

# Import torchreid compatibility module first to fix import issues
import torchreid_compat

# Import our modules
from extract_person_clips import process_video_for_clips
from squash_player_tracker.squash_tracker import SquashPlayerTracker
from result_manager import create_result_directory, get_result_path, copy_input_to_results
from player_metadata_generator import process_video_with_metadata


class IntegratedPersonReIDWorkflow:
    def __init__(self, 
                 clip_extraction_params=None,
                 reid_params=None,
                 results_dir="integrated_results"):
        """
        Initialize the integrated workflow
        
        Args:
            clip_extraction_params: Parameters for clip extraction
            reid_params: Parameters for person re-identification
            results_dir: Directory to store all results
        """
        self.results_dir = results_dir
        
        # Default clip extraction parameters
        self.clip_extraction_params = clip_extraction_params or {
            'sample_rate': 1,
            'min_segment_duration': 10.0,
            'max_empty_duration': 1.0,
            'create_visualization': False  # Disabled by default to avoid errors
        }
        
        # Default re-identification parameters
        self.reid_params = reid_params or {
            'detection_conf': 0.6,
            'reid_threshold': 0.35,
            'use_gpu': True,
            'max_players': 5  # Increased to support flexible ID management
        }
        
        # Initialize tracker
        self.tracker = None
        self._initialize_tracker()
        
    def _initialize_tracker(self):
        """Initialize the person re-identification tracker"""
        print("Initializing Person Re-identification Tracker...")
        self.tracker = SquashPlayerTracker(
            detection_conf=self.reid_params['detection_conf'],
            reid_threshold=self.reid_params['reid_threshold'],
            use_gpu=self.reid_params['use_gpu'],
            max_players=self.reid_params['max_players']
        )
        print("Tracker initialized successfully!")
        
        # Log performance capabilities
        if hasattr(self.tracker, 'device') and self.tracker.device.type == 'cuda':
            print("✓ CUDA acceleration enabled for maximum performance")
        else:
            print("⚠ Running on CPU - consider enabling CUDA for better performance")
    
    def process_video(self, video_path, output_dir=None, run_name=None):
        """
        Process a video through the complete workflow
        
        Args:
            video_path: Path to input video
            output_dir: Output directory (optional, will create timestamped dir if not provided)
            run_name: Custom name for this run (optional)
            
        Returns:
            dict: Complete processing results
        """
        print(f"\n{'='*60}")
        print("INTEGRATED PERSON RE-IDENTIFICATION WORKFLOW")
        print(f"{'='*60}")
        
        # Create result directory structure
        result_dirs = create_result_directory(self.results_dir, run_name)
        print(f"Results will be saved to: {result_dirs['root']}")
        
        # Save input video for reference
        copy_input_to_results(video_path, result_dirs)
        
        # Save workflow parameters
        workflow_params = {
            'clip_extraction': self.clip_extraction_params,
            'reid': self.reid_params,
            'input_video': video_path
        }
        # save_run_info(result_dirs, workflow_params, "Integrated Person Re-ID Workflow")
        
        # Step 1: Extract person clips
        print(f"\n{'='*40}")
        print("STEP 1: EXTRACTING PERSON CLIPS")
        print(f"{'='*40}")
        
        # Create temporary directory for clips processing
        import tempfile
        clips_output_dir = tempfile.mkdtemp(prefix="clips_")
        clip_results = process_video_for_clips(
            video_path=video_path,
            output_dir=clips_output_dir,
            use_gpu=self.reid_params['use_gpu'],
            **self.clip_extraction_params
        )
        
        if not clip_results['clips']:
            print("No person clips found in video. Workflow complete.")
            return {
                'success': False,
                'error': 'No person clips detected',
                'clip_results': clip_results,
                'reid_results': [],
                'result_dirs': result_dirs
            }
        
        print(f"Successfully extracted {len(clip_results['clips'])} person clips")
        
        # Step 2: Process each clip with person re-identification and metadata generation
        print(f"\n{'='*40}")
        print("STEP 2: PERSON RE-IDENTIFICATION ON CLIPS WITH METADATA")
        print(f"{'='*40}")
        
        reid_results = []
        reid_output_dir = result_dirs['reid_results']
        metadata_output_dir = result_dirs['player_metadata']
        
        for i, clip_info in enumerate(clip_results['clips']):
            print(f"\nProcessing clip {i+1}/{len(clip_results['clips'])}: {clip_info['filename']}")
            print(f"  Duration: {clip_info['duration']:.1f}s ({clip_info['start_time']:.1f}s - {clip_info['end_time']:.1f}s)")
            
            # Create output path for this clip's re-id results
            clip_name = os.path.splitext(clip_info['filename'])[0]
            clip_reid_output = os.path.join(reid_output_dir, f"{clip_name}_reid.mp4")
            
            try:
                # Process clip with person re-identification and metadata generation
                print(f"  Running person re-identification with metadata generation...")
                
                # Generate metadata for this clip
                metadata_result = process_video_with_metadata(
                    video_path=clip_info['path'],
                    tracker=self.tracker,
                    output_dir=metadata_output_dir,
                    segment_name=clip_name
                )
                
                # Also create the re-id output video
                self.tracker.process_video(
                    video_path=clip_info['path'],
                    output_path=clip_reid_output,
                    display=False,  # Disable display for batch processing
                    max_display_width=1280,
                    max_display_height=720
                )
                
                # Collect results
                clip_reid_result = {
                    'clip_info': clip_info,
                    'reid_output_path': clip_reid_output,
                    'metadata': metadata_result,
                    'success': True,
                    'error': None
                }
                
                print(f"  ✓ Re-ID processing complete: {clip_reid_output}")
                print(f"  ✓ Metadata generated: {metadata_result['simple_metadata_path']}")
                
            except Exception as e:
                print(f"  ✗ Error processing clip: {str(e)}")
                clip_reid_result = {
                    'clip_info': clip_info,
                    'reid_output_path': None,
                    'metadata': None,
                    'success': False,
                    'error': str(e)
                }
            
            reid_results.append(clip_reid_result)
        
        # Step 3: Generate summary and analysis
        print(f"\n{'='*40}")
        print("STEP 3: GENERATING SUMMARY AND ANALYSIS")
        print(f"{'='*40}")
        
        # Calculate statistics
        successful_clips = [r for r in reid_results if r['success']]
        failed_clips = [r for r in reid_results if not r['success']]
        
        total_duration = sum(clip['duration'] for clip in clip_results['clips'])
        processed_duration = sum(r['clip_info']['duration'] for r in successful_clips)
        
        # Collect player metadata from all clips
        all_player_metadata = {}
        for result in successful_clips:
            if result['metadata'] and 'simple_metadata' in result['metadata']:
                clip_name = result['clip_info']['filename']
                for player_id, player_data in result['metadata']['simple_metadata'].items():
                    if player_id not in all_player_metadata:
                        all_player_metadata[player_id] = []
                    all_player_metadata[player_id].append({
                        'clip': clip_name,
                        'data': player_data
                    })
        
        # Create summary
        summary = {
            'input_video': video_path,
            'total_clips': len(clip_results['clips']),
            'successful_reid_clips': len(successful_clips),
            'failed_reid_clips': len(failed_clips),
            'total_clip_duration': total_duration,
            'processed_duration': processed_duration,
            'success_rate': len(successful_clips) / len(clip_results['clips']) if clip_results['clips'] else 0,
            'video_info': clip_results['video_info'],
            'workflow_params': workflow_params,
            'player_metadata': all_player_metadata,
            'total_players_detected': len(all_player_metadata)
        }
        
        # Save summary (disabled)
        # summary_path = os.path.join(result_dirs['root'], 'workflow_summary.json')
        # with open(summary_path, 'w') as f:
        #     json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\nWORKFLOW SUMMARY:")
        print(f"  Input video: {os.path.basename(video_path)}")
        print(f"  Total clips extracted: {summary['total_clips']}")
        print(f"  Successfully processed: {summary['successful_reid_clips']}")
        print(f"  Failed processing: {summary['failed_reid_clips']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Total clip duration: {summary['total_clip_duration']:.1f}s")
        print(f"  Processed duration: {summary['processed_duration']:.1f}s")
        print(f"  Total players detected: {summary['total_players_detected']}")
        
        # Print player metadata summary
        if all_player_metadata:
            print(f"\nPLAYER METADATA SUMMARY:")
            for player_id, appearances in all_player_metadata.items():
                total_duration = sum(app['data']['appearance_duration'] for app in appearances)
                total_frames = sum(app['data']['total_frames'] for app in appearances)
                print(f"  {player_id}: {len(appearances)} appearance(s), {total_duration:.1f}s total, {total_frames} frames")
                for app in appearances:
                    print(f"    - {app['clip']}: {app['data']['start_time']} - {app['data']['end_time']} ({app['data']['appearance_duration']:.1f}s)")
        
        if failed_clips:
            print(f"\nFailed clips:")
            for failed in failed_clips:
                print(f"  - {failed['clip_info']['filename']}: {failed['error']}")
        
        # Clean up temporary clips directory
        try:
            import shutil
            shutil.rmtree(clips_output_dir)
            print(f"Temporary clips directory cleaned up: {clips_output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {clips_output_dir}: {e}")
        
        print(f"\nResults saved to: {result_dirs['root']}")
        # print(f"Summary saved to: {summary_path}")
        print(f"Player metadata saved to: {metadata_output_dir}")
        
        return {
            'success': True,
            'clip_results': clip_results,
            'reid_results': reid_results,
            'summary': summary,
            'result_dirs': result_dirs
        }
    
    def process_single_clip(self, clip_path, output_path=None):
        """
        Process a single clip with person re-identification
        
        Args:
            clip_path: Path to the clip video
            output_path: Output path for processed video (optional)
            
        Returns:
            dict: Processing results
        """
        if not os.path.exists(clip_path):
            return {'success': False, 'error': f'Clip not found: {clip_path}'}
        
        if not output_path:
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            output_path = f"{clip_name}_reid.mp4"
        
        try:
            print(f"Processing clip: {os.path.basename(clip_path)}")
            self.tracker.process_video(
                video_path=clip_path,
                output_path=output_path,
                display=False
            )
            return {'success': True, 'output_path': output_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Integrated Person Re-identification Workflow')
    
    # Input/Output arguments
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default='integrated_results',
                        help='Output directory for results')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom name for this run')
    
    # Clip extraction parameters
    parser.add_argument('--sample-rate', type=int, default=1,
                        help='Process every Nth frame for clip extraction')
    parser.add_argument('--min-segment-duration', type=float, default=10.0,
                        help='Minimum clip duration in seconds')
    parser.add_argument('--max-empty-duration', type=float, default=1.0,
                        help='Maximum duration without person before ending segment')
    parser.add_argument('--create-visualization', action='store_true',
                        help='Create visualization video (disabled by default)')
    
    # Re-identification parameters
    parser.add_argument('--det-conf', type=float, default=0.6,
                        help='Person detection confidence threshold')
    parser.add_argument('--reid-thresh', type=float, default=0.35,
                        help='Re-ID similarity threshold')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--max-players', type=int, default=2,
                        help='Maximum number of players to track')
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Set up parameters
    clip_extraction_params = {
        'sample_rate': args.sample_rate,
        'min_segment_duration': args.min_segment_duration,
        'max_empty_duration': args.max_empty_duration,
        'create_visualization': args.create_visualization
    }
    
    reid_params = {
        'detection_conf': args.det_conf,
        'reid_threshold': args.reid_thresh,
        'use_gpu': not args.cpu,
        'max_players': args.max_players
    }
    
    # Initialize and run workflow
    workflow = IntegratedPersonReIDWorkflow(
        clip_extraction_params=clip_extraction_params,
        reid_params=reid_params,
        results_dir=args.output
    )
    
    try:
        results = workflow.process_video(
            video_path=args.video,
            output_dir=args.output,
            run_name=args.run_name
        )
        
        if results['success']:
            print(f"\n✓ Workflow completed successfully!")
            return 0
        else:
            print(f"\n✗ Workflow failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n✗ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

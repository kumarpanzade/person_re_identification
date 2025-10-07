"""
Player Metadata Generator

This module generates detailed metadata for tracked players including
appearance times, durations, and frame counts.
"""

import json
import os
from datetime import timedelta
from typing import Dict, List, Any
import cv2


class PlayerMetadataGenerator:
    def __init__(self):
        """Initialize the metadata generator"""
        self.player_tracking_data = {}  # Store tracking data for each player
        self.current_frame = 0
        self.fps = 30.0  # Will be set from video
        
    def set_fps(self, fps: float):
        """Set the video FPS for accurate time calculations"""
        self.fps = fps
        
    def update_tracking(self, player_info: Dict[str, Any]):
        """
        Update tracking data for current frame
        
        Args:
            player_info: Dictionary containing player information from tracker
        """
        self.current_frame += 1
        
        for player_id, info in player_info.items():
            if player_id not in self.player_tracking_data:
                self.player_tracking_data[player_id] = {
                    'appearances': [],
                    'current_appearance': None,
                    'total_frames': 0,
                    'total_duration': 0.0
                }
            
            # If player is currently being tracked
            if info and 'bbox' in info:
                # Start new appearance if not currently appearing
                if self.player_tracking_data[player_id]['current_appearance'] is None:
                    self.player_tracking_data[player_id]['current_appearance'] = {
                        'start_frame': self.current_frame,
                        'start_time': self.current_frame / self.fps
                    }
                
                # Update total frames
                self.player_tracking_data[player_id]['total_frames'] += 1
                
            else:
                # End current appearance if player is no longer tracked
                if self.player_tracking_data[player_id]['current_appearance'] is not None:
                    appearance = self.player_tracking_data[player_id]['current_appearance']
                    appearance['end_frame'] = self.current_frame - 1
                    appearance['end_time'] = (self.current_frame - 1) / self.fps
                    appearance['duration'] = appearance['end_time'] - appearance['start_time']
                    appearance['frame_count'] = appearance['end_frame'] - appearance['start_frame'] + 1
                    
                    # Add to appearances list
                    self.player_tracking_data[player_id]['appearances'].append(appearance)
                    self.player_tracking_data[player_id]['current_appearance'] = None
                    
                    # Update total duration
                    self.player_tracking_data[player_id]['total_duration'] += appearance['duration']
    
    def finalize_tracking(self):
        """Finalize tracking data by ending any ongoing appearances"""
        for player_id, data in self.player_tracking_data.items():
            if data['current_appearance'] is not None:
                appearance = data['current_appearance']
                appearance['end_frame'] = self.current_frame
                appearance['end_time'] = self.current_frame / self.fps
                appearance['duration'] = appearance['end_time'] - appearance['start_time']
                appearance['frame_count'] = appearance['end_frame'] - appearance['start_frame'] + 1
                
                # Add to appearances list
                data['appearances'].append(appearance)
                data['current_appearance'] = None
                
                # Update total duration
                data['total_duration'] += appearance['duration']
    
    def generate_metadata(self, segment_name: str = "segment") -> Dict[str, Any]:
        """
        Generate metadata in the format similar to the example
        
        Args:
            segment_name: Name of the video segment
            
        Returns:
            Dictionary containing player metadata
        """
        metadata = {}
        
        for player_id, data in self.player_tracking_data.items():
            if data['appearances']:
                # Get the longest appearance for this player
                longest_appearance = max(data['appearances'], key=lambda x: x['duration'])
                
                metadata[player_id] = {
                    'start_time': self._format_time(longest_appearance['start_time']),
                    'end_time': self._format_time(longest_appearance['end_time']),
                    'segment': segment_name,
                    'appearance_duration': round(longest_appearance['duration'], 2),
                    'total_frames': longest_appearance['frame_count']
                }
        
        return metadata
    
    def generate_detailed_metadata(self, segment_name: str = "segment") -> Dict[str, Any]:
        """
        Generate detailed metadata including all appearances
        
        Args:
            segment_name: Name of the video segment
            
        Returns:
            Dictionary containing detailed player metadata
        """
        metadata = {
            'segment': segment_name,
            'total_frames': self.current_frame,
            'fps': self.fps,
            'players': {}
        }
        
        for player_id, data in self.player_tracking_data.items():
            if data['appearances']:
                metadata['players'][player_id] = {
                    'total_appearances': len(data['appearances']),
                    'total_duration': round(data['total_duration'], 2),
                    'total_frames': data['total_frames'],
                    'appearances': []
                }
                
                for i, appearance in enumerate(data['appearances']):
                    metadata['players'][player_id]['appearances'].append({
                        'appearance_id': i + 1,
                        'start_time': self._format_time(appearance['start_time']),
                        'end_time': self._format_time(appearance['end_time']),
                        'duration': round(appearance['duration'], 2),
                        'start_frame': appearance['start_frame'],
                        'end_frame': appearance['end_frame'],
                        'frame_count': appearance['frame_count']
                    })
        
        return metadata
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to HH:MM:SS format"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:01d}:{minutes:02d}:{seconds:02d}"
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str):
        """Save metadata to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Player metadata saved to: {output_path}")
    
    def reset(self):
        """Reset tracking data for new video"""
        self.player_tracking_data = {}
        self.current_frame = 0


def process_video_with_metadata(video_path: str, tracker, output_dir: str, segment_name: str = "segment"):
    """
    Process a video and generate player metadata
    
    Args:
        video_path: Path to input video
        tracker: Person re-identification tracker
        output_dir: Output directory for results
        segment_name: Name of the segment
        
    Returns:
        Dictionary containing metadata and results
    """
    # Initialize metadata generator
    metadata_gen = PlayerMetadataGenerator()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_gen.set_fps(fps)
    
    print(f"Processing video for metadata: {os.path.basename(video_path)}")
    print(f"FPS: {fps}, Total Frames: {total_frames}")
    
    # Process frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with tracker
        processed_frame, player_info = tracker.process_frame(frame)
        
        # Update metadata tracking
        metadata_gen.update_tracking(player_info)
        
        frame_count += 1
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Metadata processing progress: {progress:.1f}%")
    
    cap.release()
    
    # Finalize tracking
    metadata_gen.finalize_tracking()
    
    # Generate metadata
    simple_metadata = metadata_gen.generate_metadata(segment_name)
    detailed_metadata = metadata_gen.generate_detailed_metadata(segment_name)
    
    # Save metadata files
    simple_metadata_path = os.path.join(output_dir, f"{segment_name}_tracking.json")
    detailed_metadata_path = os.path.join(output_dir, f"{segment_name}_detailed_tracking.json")
    
    metadata_gen.save_metadata(simple_metadata, simple_metadata_path)
    metadata_gen.save_metadata(detailed_metadata, detailed_metadata_path)
    
    return {
        'simple_metadata': simple_metadata,
        'detailed_metadata': detailed_metadata,
        'simple_metadata_path': simple_metadata_path,
        'detailed_metadata_path': detailed_metadata_path
    }

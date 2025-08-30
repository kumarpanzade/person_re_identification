import os
import cv2
import numpy as np
import argparse
import time
from collections import defaultdict

# Import core modules
from squash_player_tracker.core.person_detection import PersonDetector
from squash_player_tracker.core.person_reid import PersonReID
from squash_player_tracker.core.deepsort_tracker import DeepSORTTracker
from squash_player_tracker.utils.visualization import draw_player_info, create_summary_image
try:
    from torchreid.utils import FeatureExtractor
except Exception:
    from torchreid.utils.feature_extractor import FeatureExtractor

class SquashPlayerTracker:
    def __init__(self, detection_conf=0.7, reid_threshold=0.6, use_gpu=True):
        """
        Initialize the squash player tracking system.
        
        Args:
            detection_conf: Confidence threshold for person detection
            reid_threshold: Similarity threshold for person re-identification
            use_gpu: Whether to use GPU for inference
        """
        print("Initializing Squash Player Tracker...")
        
        # Initialize components
        print("Loading person detection model...")
        self.person_detector = PersonDetector(conf_threshold=detection_conf)
        
        print("Loading person re-identification model...")
        self.person_reid = PersonReID(use_gpu=use_gpu)
        self.person_reid.similarity_threshold = reid_threshold
        
        print("Initializing tracker...")
        self.tracker = DeepSORTTracker(max_age=60, min_hits=2, iou_threshold=0.15)
        # Modify the max_tracks attribute to limit tracking to 3 players
        self.tracker.max_tracks = 3
        
        # State variables
        self.player_ids = {}  # Mapping from track ID to player name/ID
        self.current_frame = 0
        self.player_features = {}  # Storing player features for re-identification
        
        # Add temporal consistency tracking for glass reflections
        self.last_valid_tracks = []
        self.consistency_window = 3  # Consider last 3 frames for temporal consistency
        self.valid_track_history = []  # Track positions over time
        
        # Track player ID consistency
        self.player_track_history = {}  # Map player ID to their track history
        self.reid_memory_frames = 30  # Remember player associations for 30 frames
        self.player_positions = {}  # Last known positions of players
        
        # Direct mapping of track IDs to player IDs for stronger consistency
        self.track_to_player_map = {}  # Maps track ID -> player ID
        self.player_to_track_map = {}  # Maps player ID -> track ID
        
        print("Initialization complete!")
    
    def process_frame(self, frame):
        """
        Process a video frame
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Processed frame with visualizations
            player_info: Dictionary of player information
        """
        self.current_frame += 1
        frame_copy = frame.copy()
        player_info = {}
        
        # Detect people
        try:
            detections = self.person_detector.detect(frame)
            if not detections:
                print("No person detections found in frame")
            else:
                print(f"Found {len(detections)} person detections")
            
            # Validate detections format before passing to tracker
            valid_detections = []
            for i, detection in enumerate(detections):
                try:
                    bbox, confidence = detection
                    if isinstance(bbox, tuple) and len(bbox) == 4:
                        # Validate bbox coordinates
                        x1, y1, x2, y2 = bbox
                        if x2 > x1 and y2 > y1:
                            valid_detections.append((bbox, confidence))
                        else:
                            print(f"Skipping detection {i} with invalid dimensions: {bbox}")
                    else:
                        print(f"Skipping detection {i} with invalid bbox format: {bbox}")
                except Exception as e:
                    print(f"Error validating detection {i}: {e}")
            
            # Update tracker with validated detections (DeepSORT benefits from the frame)
            active_tracks = self.tracker.update(valid_detections, frame=frame)
        except Exception as e:
            print(f"Error in detection/tracking pipeline: {e}")
            active_tracks = []
        
        # Track ID management and consistency checks
        
        # First check for existing tracks that already have IDs in our mapping
        for track in active_tracks:
            if track.id in self.track_to_player_map:
                player_id = self.track_to_player_map[track.id]
                if track.player_name != player_id:
                    print(f"Restoring mapped ID: {player_id} to track {track.id}")
                    track.set_player_name(player_id)
        
        # Then check for spatial consistency with previously known player positions
        self._match_tracks_with_previous_positions(active_tracks, frame.shape[1], frame.shape[0])
        
        # Then identify remaining unidentified tracks using appearance features
        for track in active_tracks:
            if track.player_name is None:
                # Try to identify using person re-ID
                best_match, similarity = self.person_reid.identify_person(frame, track.bbox)
                if best_match and similarity > 0.4:  # More permissive threshold for initial ID
                    track.set_player_name(best_match)
                    continue
        
        # If we still have unidentified tracks, assign them player IDs
        # For tracking, we should have at most 3 players
        assigned_ids = [t.player_name for t in active_tracks if t.player_name]
        
        # HARD RESET: Force assignment of unique IDs P1, P2, P3 to the top 3 tracks
        # This ensures we never have duplicate IDs
        
        # First collect all tracks (both named and unnamed)
        all_tracks = []
        for track in active_tracks:
            track_center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
            track_area = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
            all_tracks.append({
                'track': track,
                'center': track_center,
                'area': track_area,
                'hits': track.hits
            })
        
        # If we have more than 3 tracks, sort by reliability (more hits = more reliable)
        # and keep only the top 3
        if len(all_tracks) > 3:
            all_tracks.sort(key=lambda x: x['hits'], reverse=True)
            all_tracks = all_tracks[:3]
        
        # Sort tracks by horizontal position (left to right)
        # This ensures consistent assignment: leftmost = P1, middle = P2, rightmost = P3
        all_tracks.sort(key=lambda x: x['center'][0])
        
        # Assign P1, P2, P3 based on position
        player_ids = ["P1", "P2", "P3"]
        
        print(f"RESETTING ALL PLAYER IDS - tracks: {len(all_tracks)}")
        for i, track_data in enumerate(all_tracks):
            if i < len(player_ids):  # Ensure we don't exceed available IDs
                track = track_data['track']
                player_id = player_ids[i]
                
                # Only log if the ID is changing
                if track.player_name != player_id:
                    old_id = track.player_name if track.player_name else "None"
                    print(f"FORCE ASSIGNING: {old_id} -> {player_id} for track {track.id}")
                
                # Force set the ID
                track.set_player_name(player_id)
                
                # Update our direct track-to-player mapping
                self.track_to_player_map[track.id] = player_id
                self.player_to_track_map[player_id] = track.id
                
                # Add features to re-ID model (overwriting any existing)
                self.person_reid.add_person(frame, track.bbox, player_id)
        
        # Update re-ID features for identified tracks
        for track in active_tracks:
            if track.player_name:
                self.person_reid.update_features(track.player_name, frame, track.bbox)
                
                # Update player info dictionary
                player_info[track.player_name] = {
                    'track_id': track.id,
                    'name': track.player_name,
                    'bbox': track.bbox
                }
        
        # Apply temporal consistency filtering to remove false detections
        active_tracks = self._apply_temporal_consistency(active_tracks, frame.shape[:2])
        
        # Update player info with filtered tracks
        player_info = {}
        for track in active_tracks:
            if track.player_name:
                player_info[track.player_name] = {
                    'track_id': track.id,
                    'name': track.player_name,
                    'bbox': track.bbox
                }
        
        # Draw tracked players
        frame_copy = self.tracker.draw_tracks(frame_copy)
        
        # Add player info to the frame
        frame_copy = draw_player_info(frame_copy, player_info)
        
        # Update player positions for next frame
        self._update_player_positions(active_tracks)
        
        return frame_copy, player_info
        
    def _update_player_positions(self, tracks):
        """
        Update player position history for improved tracking
        
        Args:
            tracks: List of active tracks with player names
        """
        # CRITICAL: Check for and prevent duplicate IDs before updating positions
        self._prevent_duplicate_ids(tracks)
        
        # Update player positions for next frame
        for track in tracks:
            if track.player_name:
                center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                
                # Store position info
                self.player_positions[track.player_name] = {
                    'center': center,
                    'bbox': track.bbox,
                    'frame': self.current_frame,
                    'track_id': track.id
                }
                
                # Update player track history
                if track.player_name not in self.player_track_history:
                    self.player_track_history[track.player_name] = []
                    
                self.player_track_history[track.player_name].append({
                    'frame': self.current_frame,
                    'bbox': track.bbox,
                    'track_id': track.id
                })
                
                # Keep history within limit
                if len(self.player_track_history[track.player_name]) > self.reid_memory_frames:
                    self.player_track_history[track.player_name] = \
                        self.player_track_history[track.player_name][-self.reid_memory_frames:]
                        
    def _prevent_duplicate_ids(self, tracks):
        """
        Explicitly check for and fix duplicate player IDs
        
        Args:
            tracks: List of active tracks
        """
        # Check for duplicate IDs
        id_to_tracks = {}
        
        # First, collect all tracks by ID
        for track in tracks:
            if track.player_name:
                if track.player_name not in id_to_tracks:
                    id_to_tracks[track.player_name] = []
                id_to_tracks[track.player_name].append(track)
        
        # Check for duplicates
        for player_id, track_list in id_to_tracks.items():
            if len(track_list) > 1:
                print(f"DUPLICATE ID DETECTED: {player_id} assigned to {len(track_list)} tracks")
                
                # Sort by hits (more reliable tracks first)
                track_list.sort(key=lambda t: t.hits, reverse=True)
                
                # Keep the ID only for the most reliable track
                kept_track = track_list[0]
                
                # For all other tracks with this ID, reset their ID
                for track in track_list[1:]:
                    print(f"  Removing duplicate ID {player_id} from track {track.id}")
                    track.player_name = None
    
    def _match_tracks_with_previous_positions(self, tracks, frame_width, frame_height):
        """
        Match tracks with previously known player positions for ID consistency
        
        Args:
            tracks: List of current tracks
            frame_width: Width of the frame
            frame_height: Height of the frame
        """
        if not self.player_positions:
            return
            
        # Calculate distance threshold based on frame size
        # Players can move, but not teleport across the frame
        distance_threshold = min(frame_width, frame_height) * 0.3
        
        # Try to match tracks with previous player positions
        for track in tracks:
            if track.player_name is not None:
                # Already identified, update position
                center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                self.player_positions[track.player_name] = {
                    'center': center,
                    'bbox': track.bbox,
                    'frame': self.current_frame
                }
                continue
                
            # Find closest previous player
            closest_player = None
            min_distance = float('inf')
            
            track_center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
            
            for player_id, data in self.player_positions.items():
                # Skip if player is already assigned to another track
                if player_id in [t.player_name for t in tracks if t.player_name]:
                    continue
                    
                # Skip if data is too old
                if self.current_frame - data['frame'] > self.reid_memory_frames:
                    continue
                
                # Calculate distance to previous position
                prev_center = data['center']
                distance = np.sqrt((track_center[0] - prev_center[0])**2 + 
                                  (track_center[1] - prev_center[1])**2)
                
                if distance < distance_threshold and distance < min_distance:
                    min_distance = distance
                    closest_player = player_id
            
            # Assign the closest player ID to this track
            if closest_player:
                print(f"Matched track {track.id} with previous player {closest_player} (distance: {min_distance:.1f}px)")
                track.set_player_name(closest_player)
                
                # Update position
                self.player_positions[closest_player] = {
                    'center': track_center,
                    'bbox': track.bbox,
                    'frame': self.current_frame
                }
    
    def _apply_temporal_consistency(self, tracks, frame_shape):
        """
        Apply temporal consistency filtering to remove false detections caused by glass
        
        Args:
            tracks: List of active tracks
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            filtered_tracks: List of temporally consistent tracks
        """
        height, width = frame_shape
        
        # Store current track positions
        current_tracks = []
        for track in tracks:
            current_tracks.append({
                'id': track.id,
                'bbox': track.bbox,
                'name': track.player_name,
                'center': ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
            })
        
        # Add to history (keep only last N frames)
        self.valid_track_history.append(current_tracks)
        if len(self.valid_track_history) > self.consistency_window:
            self.valid_track_history.pop(0)
        
        # If we don't have enough history yet, return all tracks
        if len(self.valid_track_history) < 2:
            return tracks
        
        # Filter tracks based on temporal consistency
        filtered_tracks = []
        for track in tracks:
            track_id = track.id
            track_center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
            
            # Track is consistent if:
            # 1. It appears in multiple consecutive frames
            # 2. Its movement is smooth (no teleporting)
            # 3. Its size doesn't change dramatically between frames
            
            consistency_score = 0
            prev_center = None
            
            # Check track consistency across history
            for frame_idx, frame_tracks in enumerate(self.valid_track_history[:-1]):  # Skip current frame
                for t in frame_tracks:
                    if t['id'] == track_id:
                        consistency_score += 1
                        if prev_center:
                            # Calculate distance between consecutive appearances
                            dist = np.sqrt((t['center'][0] - prev_center[0])**2 + (t['center'][1] - prev_center[1])**2)
                            # Penalize large jumps (possible reflection)
                            if dist > width * 0.3:  # If jump is more than 30% of frame width
                                consistency_score -= 0.5
                        prev_center = t['center']
            
            # Glass often causes duplicate detections that appear and disappear
            # Only keep tracks that have consistency
            if consistency_score >= 1 or track.hits > 5:
                filtered_tracks.append(track)
        
        # If filtering removed all tracks but we had tracks before, keep some previous ones
        if not filtered_tracks and self.last_valid_tracks:
            # Just use previous tracks (with adjusted positions if possible)
            for prev_track in self.last_valid_tracks:
                # Check if any current track is close to this previous track
                for track in tracks:
                    prev_center = ((prev_track.bbox[0] + prev_track.bbox[2]) // 2, 
                                  (prev_track.bbox[1] + prev_track.bbox[3]) // 2)
                    curr_center = ((track.bbox[0] + track.bbox[2]) // 2, 
                                  (track.bbox[1] + track.bbox[3]) // 2)
                    
                    dist = np.sqrt((prev_center[0] - curr_center[0])**2 + 
                                  (prev_center[1] - curr_center[1])**2)
                    
                    if dist < width * 0.2:  # If within 20% of frame width
                        # Update position but keep ID and name
                        prev_track.update(track.bbox)
                        filtered_tracks.append(prev_track)
                        break
        
        # Update last valid tracks
        self.last_valid_tracks = filtered_tracks.copy()
        
        return filtered_tracks
        
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (if None, don't save)
            display: Whether to display the processed frames
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        print(f"Processing video with {frame_count} frames...")
        frame_idx = 0
        processing_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            start_time = time.time()
            processed_frame, player_info = self.process_frame(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Display frame info
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%) - "
                      f"Processing time: {processing_time:.3f}s")
            
            # Display processed frame
            if display:
                cv2.imshow('Player Tracking', processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
            
            # Write frame to output video
            if writer:
                writer.write(processed_frame)
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print processing statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"Average processing time per frame: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")
    


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Player Tracking System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--no-display', action='store_true', help='Disable display of processed frames')
    parser.add_argument('--det-conf', type=float, default=0.5, help='Person detection confidence threshold')
    parser.add_argument('--reid-thresh', type=float, default=0.5, help='Re-ID similarity threshold')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = SquashPlayerTracker(
        detection_conf=args.det_conf,
        reid_threshold=args.reid_thresh,
        use_gpu=not args.cpu
    )
    
    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
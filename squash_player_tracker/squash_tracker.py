import os
import cv2
import numpy as np
import argparse
import time
import ctypes
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Import core modules
from squash_player_tracker.core.person_detection import PersonDetector
from squash_player_tracker.core.person_reid import PersonReID
from squash_player_tracker.core.deepsort_tracker import DeepSORTTracker
from squash_player_tracker.utils.visualization import draw_player_info
try:
    from torchreid.utils import FeatureExtractor
except Exception:
    from torchreid.utils.feature_extractor import FeatureExtractor

def get_screen_resolution():
    """
    Detect the primary screen resolution.
    
    Returns:
        tuple: (width, height) of the primary screen
    """
    try:
        # Try to get screen resolution using ctypes for Windows
        if os.name == 'nt':  # Windows
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        # For Linux/WSL, try using an environment variable or default to common laptop resolution
        else:
            # Default to a common laptop screen resolution if detection fails
            return 1920, 1080
    except Exception:
        # Default fallback resolution
        return 1280, 720

class SquashPlayerTracker:
    def __init__(self, detection_conf=0.6, reid_threshold=0.6, use_gpu=True, reference_embeddings=None, 
                 duplicate_iou_threshold=0.3, duplicate_distance_threshold=30, debug_duplicates=False, 
                 debug_crossing=False, max_players=2):
        """
        Initialize the squash player tracking system.
        
        Args:
            detection_conf: Confidence threshold for person detection
            reid_threshold: Similarity threshold for person re-identification
            use_gpu: Whether to use GPU for inference
            reference_embeddings: Optional dictionary of pre-defined person embeddings
            duplicate_iou_threshold: IoU threshold for detecting duplicate detections
            duplicate_distance_threshold: Distance threshold for detecting duplicate detections
        """
        print("Initializing Squash Player Tracker...")
        
        # Initialize components
        print("Loading person detection model...")
        self.person_detector = PersonDetector(conf_threshold=detection_conf)
        
        print("Loading person re-identification model...")
        self.person_reid = PersonReID(use_gpu=use_gpu)
        self.person_reid.similarity_threshold = reid_threshold
        
        print("Initializing tracker...")
        self.tracker = DeepSORTTracker(max_age=15, min_hits=4, iou_threshold=0.8)
        # No limit on number of players that can be tracked
        self.tracker.max_tracks = None
        
        # State variables
        self.player_ids = {}  # Mapping from track ID to player name/ID
        self.current_frame = 0
        self.player_features = {}  # Storing player features for re-identification
        
        # Add temporal consistency tracking with median filtering for smooth tracking
        self.last_valid_tracks = []
        self.consistency_window = 10  # Number of frames to use for median filtering
        self.valid_track_history = []  # Track positions over time used for median filtering
        
        # Track player ID consistency
        self.player_track_history = {}  # Map player ID to their track history
        self.reid_memory_frames = 10000  # Remember player associations for 30 frames
        self.player_positions = {}  # Last known positions of players
        
        # Direct mapping of track IDs to player IDs for stronger consistency
        self.track_to_player_map = {}  # Maps track ID -> player ID
        self.player_to_track_map = {}  # Maps player ID -> track ID
        self.locked_identities = set()  # Set of player IDs with locked identities
        
        # Flag to indicate if using reference embeddings
        self.using_reference_embeddings = False
        self.strict_id_enforcement = True  # Whether to enforce strict ID assignment
        
        # Track which players have had their features initialized
        self.initialized_features = set()
        
        # Duplicate detection thresholds
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.duplicate_distance_threshold = duplicate_distance_threshold
        self.debug_duplicates = debug_duplicates
        self.debug_crossing = debug_crossing
        self.max_players = max_players
        
        # Crossing detection and ID preservation
        self.player_motion_history = {}  # Store motion vectors for each player
        self.crossing_detected = False
        self.crossing_frames = 0
        self.max_crossing_frames = 60  # Maximum frames to consider as crossing
        self.motion_prediction_weight = 0.8  # Weight for motion prediction vs appearance
        
        # Initialize with reference embeddings if provided
        if reference_embeddings:
            self.set_reference_embeddings(reference_embeddings)
        
        print("Initialization complete!")
        
    def ensure_features_initialized_once(self, player_id, frame=None, bbox=None):
        """
        Ensure a player's features are initialized only once
        
        Args:
            player_id: ID of the player
            frame: Current frame (optional)
            bbox: Bounding box for the player (optional)
            
        Returns:
            bool: True if initialized, False if already initialized
        """
        # If player already has features initialized, do nothing
        if player_id in self.initialized_features:
            return False
            
        # If we have reference embeddings, use those (they are already stored in the person_reid)
        if player_id in self.person_reid.reference_embeddings:
            self.initialized_features.add(player_id)
            return True
            
        # If we have frame and bbox, extract and store features
        if frame is not None and bbox is not None:
            success = self.person_reid.add_person(frame, bbox, player_id)
            if success:
                self.initialized_features.add(player_id)
                return True
                
        return False
        
    def set_reference_embeddings(self, embeddings):
        """
        Set reference embeddings for player identification
        
        Args:
            embeddings: Dictionary mapping player_id to feature vector
            
        Returns:
            bool: True if successful
        """
        if not embeddings:
            return False
            
        # Set each embedding directly in the re-ID model
        for player_id, features in embeddings.items():
            self.person_reid.set_reference_embedding(player_id, features)
            # Lock these identities
            self.locked_identities.add(player_id)
            # Mark features as initialized
            self.initialized_features.add(player_id)
            
        # Set the flag to use reference embeddings only
        self.person_reid.use_reference_only = True
        self.using_reference_embeddings = True
        self.person_reid.strict_id_enforcement = self.strict_id_enforcement
        
        print(f"Set {len(embeddings)} reference embeddings")
        print(f"Strict ID enforcement: {self.strict_id_enforcement}")
        return True
        
    def enforce_id_consistency(self, track, player_id):
        """
        Enforce consistent identity for a track
        
        Args:
            track: Track object
            player_id: Proposed player ID
            
        Returns:
            str: Most consistent player ID for this track
        """
        # If this is a locked identity, prioritize it highly
        if player_id in self.locked_identities:
            # Update our tracking maps
            if track.id in self.track_to_player_map and self.track_to_player_map[track.id] != player_id:
                # This track previously had a different ID - this is a switch!
                old_id = self.track_to_player_map[track.id]
                print(f"WARNING: Track {track.id} changing from {old_id} to locked ID {player_id}")
                
                # Remove old mapping if it exists
                if old_id in self.player_to_track_map and self.player_to_track_map[old_id] == track.id:
                    self.player_to_track_map.pop(old_id)
            
            # Set the new mapping
            self.track_to_player_map[track.id] = player_id
            self.player_to_track_map[player_id] = track.id
            return player_id
            
        # If not using reference embeddings, or not enforcing consistency, return as is
        if not self.using_reference_embeddings or not self.strict_id_enforcement:
            return player_id
            
        # Check if this track has a consistent ID from history
        consistent_id = self.person_reid.get_consistent_id(track.id)
        if consistent_id is not None:
            if consistent_id != player_id:
                print(f"ID consistency enforced: {player_id} -> {consistent_id} for track {track.id}")
            return consistent_id
            
        # Use the provided ID but record it for consistency
        enforced_id = self.person_reid.enforce_id_consistency(track.id, player_id)
        
        # Update our tracking maps with the enforced ID
        self.track_to_player_map[track.id] = enforced_id
        self.player_to_track_map[enforced_id] = track.id
        
        return enforced_id
        
    def initialize_from_reference_frames(self, reference_frames, use_reference_only=True):
        """
        Initialize person embeddings from reference frames
        
        Args:
            reference_frames: Dictionary mapping player_id to (image, bbox)
            use_reference_only: Whether to use only reference embeddings
            
        Returns:
            bool: True if successful
        """
        if not reference_frames:
            return False
            
        # Initialize the re-ID model with reference images
        success = self.person_reid.initialize_from_reference_images(
            reference_frames, 
            use_reference_only=use_reference_only
        )
        
        if success:
            self.using_reference_embeddings = True
            
            # Set up player-to-track mapping for reference players
            for player_id in reference_frames.keys():
                self.player_to_track_map[player_id] = None  # Will be filled in when track is created
                self.initialized_features.add(player_id)  # Mark these player IDs as having initialized features
            
            print(f"Initialized {len(reference_frames)} player embeddings")
        
        return success
    
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
            
            # Apply additional duplicate detection before tracking
            valid_detections = self._remove_duplicate_detections(valid_detections)
            
            # Update tracker with validated detections (DeepSORT benefits from the frame)
            active_tracks = self.tracker.update(valid_detections, frame=frame)
            
            # Apply additional duplicate detection after tracking
            active_tracks = self._remove_duplicate_tracks(active_tracks)
            
            # Update motion history for crossing detection
            self._update_motion_history(active_tracks)
            
            # Detect crossing events
            self.crossing_detected = self._detect_crossing(active_tracks)
            if self.crossing_detected:
                self.crossing_frames += 1
                if self.debug_crossing:
                    print(f"CROSSING DETECTED - Frame {self.crossing_frames}")
            else:
                self.crossing_frames = 0
                
            # Preserve IDs during crossing
            active_tracks = self._preserve_ids_during_crossing(active_tracks)
            
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
        
        # Handle ID assignment differently based on whether we're using reference embeddings
        if self.using_reference_embeddings:
            # When using reference embeddings, we only assign IDs based on appearance similarity
            # We don't force-assign IDs based on position
            
            # First, collect all tracks (both named and unnamed)
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
            
            # Update track-to-player mappings for all identified tracks
            for track_data in all_tracks:
                track = track_data['track']
                if track.player_name:
                    self.track_to_player_map[track.id] = track.player_name
                    self.player_to_track_map[track.player_name] = track.id
            
            # Only update appearance features for already-identified tracks
            # No need to call add_person as we're using fixed reference embeddings
            
            print(f"Using pre-defined embeddings - identified tracks: {len([t for t in active_tracks if t.player_name])}")
        else:
            # ID assignment strategy based on crossing detection
            if self.crossing_detected and self.crossing_frames < self.max_crossing_frames:
                # During crossing: Preserve existing IDs and only assign new ones to truly new tracks
                if self.debug_crossing:
                    print(f"CROSSING MODE: Preserving existing IDs (frame {self.crossing_frames})")
                
                # Only assign IDs to tracks that don't have them
                unnamed_tracks = [t for t in active_tracks if not t.player_name]
                if unnamed_tracks:
                    # Sort by hits and assign IDs to new tracks
                    unnamed_tracks.sort(key=lambda t: t.hits, reverse=True)
                    for i, track in enumerate(unnamed_tracks):
                        if i < self.max_players:  # Support up to max_players
                            player_id = f"P{len([t for t in active_tracks if t.player_name]) + i + 1}"
                            track.set_player_name(player_id)
                            self.track_to_player_map[track.id] = player_id
                            self.player_to_track_map[player_id] = track.id
                            self.ensure_features_initialized_once(player_id, frame, track.bbox)
                            print(f"CROSSING: Assigned new ID {player_id} to track {track.id}")
            else:
                # Normal mode: Use motion prediction and appearance for ID assignment
                if self.debug_crossing:
                    print("NORMAL MODE: Using motion prediction and appearance for ID assignment")
                
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
                
                # Sort tracks by reliability (more hits = more reliable)
                all_tracks.sort(key=lambda x: x['hits'], reverse=True)
                
                # For tracks without IDs, try to match with previous positions using motion prediction
                for track_data in all_tracks:
                    track = track_data['track']
                    if not track.player_name:
                        # Try to match with previous player positions
                        best_match = None
                        best_score = float('inf')
                        
                        for player_id, data in self.player_positions.items():
                            # Skip if player is already assigned to another track
                            if player_id in [t.player_name for t in active_tracks if t.player_name]:
                                continue
                                
                            # Get predicted position
                            predicted_pos = self._predict_motion(player_id, data['center'])
                            
                            # Calculate distance to predicted position
                            track_center = track_data['center']
                            distance = np.sqrt((track_center[0] - predicted_pos[0])**2 + 
                                             (track_center[1] - predicted_pos[1])**2)
                            
                            if distance < best_score and distance < 80:  # Reverted to 2-player threshold
                                best_score = distance
                                best_match = player_id
                        
                        if best_match:
                            track.set_player_name(best_match)
                            self.track_to_player_map[track.id] = best_match
                            self.player_to_track_map[best_match] = track.id
                            print(f"MOTION MATCH: {best_match} -> track {track.id} (distance: {best_score:.1f})")
                
                # Assign remaining IDs to tracks without names
                unnamed_tracks = [t for t in active_tracks if not t.player_name]
                if unnamed_tracks:
                    # Sort by horizontal position for consistent assignment
                    unnamed_tracks.sort(key=lambda t: ((t.bbox[0] + t.bbox[2]) // 2))
                    
                    for i, track in enumerate(unnamed_tracks):
                        if i < self.max_players:  # Support up to max_players
                            player_id = f"P{i+1}"
                            track.set_player_name(player_id)
                            self.track_to_player_map[track.id] = player_id
                            self.player_to_track_map[player_id] = track.id
                            self.ensure_features_initialized_once(player_id, frame, track.bbox)
                            print(f"NEW ASSIGNMENT: {player_id} -> track {track.id}")
        
        # Apply our enhanced ID consistency enforcement to all tracks
        final_tracks = []
        player_id_used = set()  # Track which player IDs have been used
        
        # First pass: handle tracks with consistent IDs
        for track in active_tracks:
            if track.player_name:
                # Apply strict ID consistency enforcement
                consistent_id = self.enforce_id_consistency(track, track.player_name)
                
                # Check if this ID has been used already (prevent duplicates)
                if consistent_id in player_id_used:
                    print(f"WARNING: Duplicate player ID {consistent_id} detected! Removing track {track.id}")
                    # Reset the track's player name to prevent duplicate IDs
                    track.player_name = None
                    continue
                
                # Apply the consistent ID
                track.set_player_name(consistent_id)
                player_id_used.add(consistent_id)
                
                # Don't update features after initialization - we'll use the initial features only
                # Comment out the feature updating code
                # if not self.person_reid.use_reference_only:
                #     self.person_reid.update_features(consistent_id, frame, track.bbox)
                
                # Update player info dictionary
                player_info[consistent_id] = {
                    'track_id': track.id,
                    'name': consistent_id,
                    'bbox': track.bbox
                }
                
                final_tracks.append(track)
            else:
                # Keep unidentified tracks for the second pass
                final_tracks.append(track)
                
        # Ensure all reference identities are maintained if tracks exist
        if self.using_reference_embeddings and len(final_tracks) >= len(self.locked_identities):
            missing_ids = self.locked_identities - player_id_used
            
            if missing_ids:
                print(f"Warning: Missing locked identities: {missing_ids}")
                # Try to assign missing IDs to unidentified tracks
                for missing_id in missing_ids:
                    for track in final_tracks:
                        if not track.player_name:
                            # Try to identify with a lower threshold
                            similarity = self.person_reid.calculate_similarity(
                                frame, track.bbox, missing_id)
                            
                            if similarity > 0.2:  # Very permissive threshold for recovery
                                print(f"Recovering missing ID {missing_id} for track {track.id} with similarity {similarity:.2f}")
                                track.set_player_name(missing_id)
                                player_id_used.add(missing_id)
                                
                                # Update player info
                                player_info[missing_id] = {
                                    'track_id': track.id,
                                    'name': missing_id,
                                    'bbox': track.bbox
                                }
                                break
        
        # Apply temporal consistency filtering to remove false detections and duplicates
        active_tracks = self._apply_temporal_consistency(active_tracks, frame.shape[:2])
        
        # Limit to maximum tracks based on max_players setting
        if len(active_tracks) > self.max_players:
            # Sort by hits and keep only the most reliable tracks
            active_tracks = sorted(active_tracks, key=lambda t: t.hits, reverse=True)[:self.max_players]
            print(f"Limited tracks to {self.max_players} most reliable ones (was {len(active_tracks)})")
        
        # Final aggressive duplicate detection before output
        active_tracks = self._final_duplicate_cleanup(active_tracks)
        
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
        Apply temporal consistency filtering and median filtering for smooth tracking
        
        Args:
            tracks: List of active tracks
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            filtered_tracks: List of temporally consistent tracks with smoothed positions
        """
        height, width = frame_shape
        
        # Store current track positions
        current_tracks = []
        for track in tracks:
            current_tracks.append({
                'id': track.id,
                'bbox': track.bbox,
                'name': track.player_name,
                'center': ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2),
                'width': track.bbox[2] - track.bbox[0],
                'height': track.bbox[3] - track.bbox[1]
            })
        
        # Add to history (keep only last N frames)
        self.valid_track_history.append(current_tracks)
        if len(self.valid_track_history) > self.consistency_window:
            self.valid_track_history.pop(0)
        
        # If we don't have enough history yet, return all tracks
        if len(self.valid_track_history) < 2:
            return tracks
        
        # Filter tracks based on temporal consistency and apply median filtering
        filtered_tracks = []
        smoothed_tracks = {}  # Will store median-filtered track data
        
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
            
            # Collect historical positions for this track for median filtering
            if track_id not in smoothed_tracks:
                smoothed_tracks[track_id] = {
                    'x_positions': [],
                    'y_positions': [],
                    'widths': [],
                    'heights': [],
                    'original_track': track
                }
            
            # Collect positions from track history (including current frame)
            for frame_tracks in self.valid_track_history:
                for t in frame_tracks:
                    if t['id'] == track_id:
                        smoothed_tracks[track_id]['x_positions'].append(t['center'][0])
                        smoothed_tracks[track_id]['y_positions'].append(t['center'][1])
                        smoothed_tracks[track_id]['widths'].append(t['width'])
                        smoothed_tracks[track_id]['heights'].append(t['height'])
            
            # Glass often causes duplicate detections that appear and disappear
            # Only keep tracks that have consistency (more aggressive filtering)
            if consistency_score >= 2 or track.hits > 8:  # Increased thresholds
                # Don't add to filtered_tracks yet, we'll add the smoothed versions later
                pass
            else:
                # Skip this track as it's not consistent enough
                print(f"Removing inconsistent track {track.id} (consistency: {consistency_score}, hits: {track.hits})")
                continue
        
        # Apply median filtering to consistent tracks
        for track_id, track_data in smoothed_tracks.items():
            if len(track_data['x_positions']) >= 2:  # Need at least 2 points for median
                # Apply median filter to positions and size
                med_x = int(np.median(track_data['x_positions']))
                med_y = int(np.median(track_data['y_positions']))
                med_w = int(np.median(track_data['widths']))
                med_h = int(np.median(track_data['heights']))
                
                # Reconstruct bbox from median-filtered values
                x1 = med_x - med_w // 2
                y1 = med_y - med_h // 2
                x2 = x1 + med_w
                y2 = y1 + med_h
                
                # Update the track with median-filtered position
                original_track = track_data['original_track']
                # Use the update method with is_smoothed=True to avoid double counting hits
                original_track.update((x1, y1, x2, y2), is_smoothed=True)
                
                # Add the smoothed track to filtered_tracks
                filtered_tracks.append(original_track)
        
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
                        # Use is_smoothed=False since this is a regular update
                        prev_track.update(track.bbox, is_smoothed=False)
                        filtered_tracks.append(prev_track)
                        break
        
        # Update last valid tracks
        self.last_valid_tracks = filtered_tracks.copy()
        
        return filtered_tracks
    
    def _remove_duplicate_detections(self, detections):
        """
        Remove duplicate detections using IoU and distance thresholds
        
        Args:
            detections: List of (bbox, confidence) tuples
            
        Returns:
            filtered_detections: List of detections with duplicates removed
        """
        if len(detections) <= 1:
            return detections
            
        # Sort by confidence (higher first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        filtered_detections = []
        
        for i, (bbox1, conf1) in enumerate(detections):
            is_duplicate = False
            
            # Check against already filtered detections
            for j, (bbox2, conf2) in enumerate(filtered_detections):
                # Calculate IoU
                iou = self._calculate_iou(bbox1, bbox2)
                
                # Calculate center distance
                center1 = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
                center2 = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Check if this is a duplicate
                if iou > self.duplicate_iou_threshold or distance < self.duplicate_distance_threshold:
                    is_duplicate = True
                    if self.debug_duplicates:
                        print(f"Removing duplicate detection: IoU={iou:.2f}, distance={distance:.1f}")
                    break
            
            if not is_duplicate:
                filtered_detections.append((bbox1, conf1))
        
        print(f"Filtered {len(detections)} detections to {len(filtered_detections)} (removed {len(detections) - len(filtered_detections)} duplicates)")
        return filtered_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        xx1 = max(x1_1, x1_2)
        yy1 = max(y1_1, y1_2)
        xx2 = min(x2_1, x2_2)
        yy2 = min(y2_1, y2_2)
        
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        # Prevent division by zero
        if union == 0:
            return 0
            
        return intersection / union
    
    def _remove_duplicate_tracks(self, tracks):
        """
        Remove duplicate tracks that are tracking the same person
        
        Args:
            tracks: List of active tracks
            
        Returns:
            filtered_tracks: List of tracks with duplicates removed
        """
        if len(tracks) <= 1:
            return tracks
            
        # Sort by hits (more reliable tracks first)
        tracks = sorted(tracks, key=lambda t: t.hits, reverse=True)
        
        filtered_tracks = []
        
        for i, track1 in enumerate(tracks):
            is_duplicate = False
            
            # Check against already filtered tracks
            for j, track2 in enumerate(filtered_tracks):
                # Calculate IoU
                iou = self._calculate_iou(track1.bbox, track2.bbox)
                
                # Calculate center distance
                center1 = ((track1.bbox[0] + track1.bbox[2]) // 2, (track1.bbox[1] + track1.bbox[3]) // 2)
                center2 = ((track2.bbox[0] + track2.bbox[2]) // 2, (track2.bbox[1] + track2.bbox[3]) // 2)
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Check if this is a duplicate track
                if iou > self.duplicate_iou_threshold or distance < self.duplicate_distance_threshold:
                    is_duplicate = True
                    if self.debug_duplicates:
                        print(f"Removing duplicate track {track1.id} (IoU={iou:.2f}, distance={distance:.1f})")
                    
                    # Transfer player name if track1 has one and track2 doesn't
                    if track1.player_name and not track2.player_name:
                        track2.set_player_name(track1.player_name)
                        if self.debug_duplicates:
                            print(f"Transferred player name {track1.player_name} from track {track1.id} to track {track2.id}")
                    break
            
            if not is_duplicate:
                filtered_tracks.append(track1)
        
        print(f"Filtered {len(tracks)} tracks to {len(filtered_tracks)} (removed {len(tracks) - len(filtered_tracks)} duplicates)")
        return filtered_tracks
    
    def _final_duplicate_cleanup(self, tracks):
        """
        Final aggressive cleanup to remove any remaining duplicate tracks
        This is the last line of defense against duplicate bounding boxes
        
        Args:
            tracks: List of active tracks
            
        Returns:
            cleaned_tracks: List of tracks with all duplicates removed
        """
        if len(tracks) <= 1:
            return tracks
            
        # Sort by hits and confidence (most reliable first)
        tracks = sorted(tracks, key=lambda t: (t.hits, getattr(t, 'confidence', 0)), reverse=True)
        
        cleaned_tracks = []
        
        for i, track1 in enumerate(tracks):
            is_duplicate = False
            
            # Check against already cleaned tracks
            for j, track2 in enumerate(cleaned_tracks):
                # Calculate IoU
                iou = self._calculate_iou(track1.bbox, track2.bbox)
                
                # Calculate center distance
                center1 = ((track1.bbox[0] + track1.bbox[2]) // 2, (track1.bbox[1] + track1.bbox[3]) // 2)
                center2 = ((track2.bbox[0] + track2.bbox[2]) // 2, (track2.bbox[1] + track2.bbox[3]) // 2)
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Reverted to 2-player thresholds (more aggressive)
                if iou > 0.3 or distance < 15:  # Reverted to 2-player settings
                    is_duplicate = True
                    if self.debug_duplicates:
                        print(f"FINAL CLEANUP: Removing duplicate track {track1.id} (IoU={iou:.2f}, distance={distance:.1f})")
                    
                    # Transfer player name if track1 has one and track2 doesn't
                    if track1.player_name and not track2.player_name:
                        track2.set_player_name(track1.player_name)
                        if self.debug_duplicates:
                            print(f"Transferred player name {track1.player_name} from track {track1.id} to track {track2.id}")
                    break
            
            if not is_duplicate:
                cleaned_tracks.append(track1)
        
        print(f"FINAL CLEANUP: Filtered {len(tracks)} tracks to {len(cleaned_tracks)} (removed {len(tracks) - len(cleaned_tracks)} duplicates)")
        return cleaned_tracks
    
    def _detect_crossing(self, tracks):
        """
        Detect if players are crossing each other based on motion vectors
        
        Args:
            tracks: List of current tracks
            
        Returns:
            bool: True if crossing is detected
        """
        if len(tracks) < 2:
            return False
            
        # Get current positions
        current_positions = {}
        for track in tracks:
            if track.player_name:
                center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                current_positions[track.player_name] = center
        
        # Check if we have motion history for at least 2 players
        if len(current_positions) < 2 or len(self.player_motion_history) < 2:
            return False
            
        # Calculate motion vectors
        motion_vectors = {}
        for player_id, current_pos in current_positions.items():
            if player_id in self.player_motion_history:
                prev_pos = self.player_motion_history[player_id]['position']
                motion_vectors[player_id] = (
                    current_pos[0] - prev_pos[0],
                    current_pos[1] - prev_pos[1]
                )
        
        if len(motion_vectors) < 2:
            return False
            
        # Check for crossing by analyzing motion vectors
        player_ids = list(motion_vectors.keys())
        for i in range(len(player_ids)):
            for j in range(i + 1, len(player_ids)):
                player1, player2 = player_ids[i], player_ids[j]
                pos1, pos2 = current_positions[player1], current_positions[player2]
                vel1, vel2 = motion_vectors[player1], motion_vectors[player2]
                
                # Calculate distance between players
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Check if players are close and moving towards each other
                if distance < 100:  # Reverted to 2-player threshold
                    # Calculate dot product of velocity vectors
                    dot_product = vel1[0] * vel2[0] + vel1[1] * vel2[1]
                    
                    # If dot product is negative, players are moving towards each other
                    if dot_product < 0:
                        return True
                        
        return False
    
    def _predict_motion(self, player_id, current_pos):
        """
        Predict next position based on motion history
        
        Args:
            player_id: ID of the player
            current_pos: Current position (x, y)
            
        Returns:
            tuple: Predicted position (x, y)
        """
        if player_id not in self.player_motion_history:
            return current_pos
            
        history = self.player_motion_history[player_id]
        if len(history['positions']) < 2:
            return current_pos
            
        # Calculate average velocity
        velocities = []
        for i in range(1, len(history['positions'])):
            prev_pos = history['positions'][i-1]
            curr_pos = history['positions'][i]
            vel = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            velocities.append(vel)
            
        if not velocities:
            return current_pos
            
        # Average velocity
        avg_vel = (
            sum(v[0] for v in velocities) / len(velocities),
            sum(v[1] for v in velocities) / len(velocities)
        )
        
        # Predict next position
        predicted_pos = (
            current_pos[0] + avg_vel[0],
            current_pos[1] + avg_vel[1]
        )
        
        return predicted_pos
    
    def _update_motion_history(self, tracks):
        """
        Update motion history for all tracks
        
        Args:
            tracks: List of current tracks
        """
        for track in tracks:
            if track.player_name:
                center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
                
                if track.player_name not in self.player_motion_history:
                    self.player_motion_history[track.player_name] = {
                        'positions': [],
                        'frames': []
                    }
                
                # Add current position
                self.player_motion_history[track.player_name]['positions'].append(center)
                self.player_motion_history[track.player_name]['frames'].append(self.current_frame)
                
                # Keep only last 10 positions
                if len(self.player_motion_history[track.player_name]['positions']) > 10:
                    self.player_motion_history[track.player_name]['positions'].pop(0)
                    self.player_motion_history[track.player_name]['frames'].pop(0)
                
                # Update position for next frame
                self.player_motion_history[track.player_name]['position'] = center
        
    def _preserve_ids_during_crossing(self, tracks):
        """
        Preserve player IDs during crossing events using motion prediction
        
        Args:
            tracks: List of current tracks
            
        Returns:
            tracks: List of tracks with preserved IDs
        """
        if not self.crossing_detected or len(tracks) < 2:
            return tracks
            
        # Get tracks with and without IDs
        named_tracks = [t for t in tracks if t.player_name]
        unnamed_tracks = [t for t in tracks if not t.player_name]
        
        if len(named_tracks) == 0:
            return tracks
            
        # For each unnamed track, try to match with a named track using motion prediction
        for unnamed_track in unnamed_tracks:
            best_match = None
            best_score = float('inf')
            
            for named_track in named_tracks:
                # Skip if this named track is already assigned
                if any(t.player_name == named_track.player_name for t in tracks if t != unnamed_track):
                    continue
                    
                # Get predicted position for the named track
                predicted_pos = self._predict_motion(named_track.player_name, 
                    ((named_track.bbox[0] + named_track.bbox[2]) // 2, 
                     (named_track.bbox[1] + named_track.bbox[3]) // 2))
                
                # Calculate distance to predicted position
                unnamed_center = ((unnamed_track.bbox[0] + unnamed_track.bbox[2]) // 2,
                                 (unnamed_track.bbox[1] + unnamed_track.bbox[3]) // 2)
                
                distance = np.sqrt((unnamed_center[0] - predicted_pos[0])**2 + 
                                 (unnamed_center[1] - predicted_pos[1])**2)
                
                if distance < best_score:
                    best_score = distance
                    best_match = named_track
            
            # Assign the best match if distance is reasonable
            if best_match and best_score < 50:  # Within 50 pixels
                print(f"CROSSING: Assigning {best_match.player_name} to track {unnamed_track.id} (distance: {best_score:.1f})")
                unnamed_track.set_player_name(best_match.player_name)
                
                # Update mappings
                self.track_to_player_map[unnamed_track.id] = best_match.player_name
                self.player_to_track_map[best_match.player_name] = unnamed_track.id
                
                # Remove the old track from named_tracks to prevent double assignment
                named_tracks = [t for t in named_tracks if t != best_match]
        
        return tracks
        
    def process_video(self, video_path, output_path=None, display=True, max_display_width=None, max_display_height=None):
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (if None, don't save)
            display: Whether to display the processed frames
            max_display_width: Maximum width for display window (auto-detected if None)
            max_display_height: Maximum height for display window (auto-detected if None)
        """
        # Auto-detect screen size if not provided
        if max_display_width is None or max_display_height is None:
            screen_width, screen_height = get_screen_resolution()
            # Use 80% of screen size as maximum display size
            max_display_width = max_display_width or int(screen_width * 0.8)
            max_display_height = max_display_height or int(screen_height * 0.8)
            print(f"Auto-detected display size: {max_display_width}x{max_display_height}")
        
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
                # Resize display frame to fit screen if needed
                display_frame = processed_frame.copy()
                frame_h, frame_w = display_frame.shape[:2]
                
                # Calculate scaling factor to fit within max dimensions while preserving aspect ratio
                scale_w = min(1.0, max_display_width / frame_w)
                scale_h = min(1.0, max_display_height / frame_h)
                scale = min(scale_w, scale_h)  # Use the smaller scale to ensure it fits both dimensions
                
                # Only resize if needed (scale < 1.0)
                if scale < 1.0:
                    new_width = int(frame_w * scale)
                    new_height = int(frame_h * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height), 
                                              interpolation=cv2.INTER_AREA)
                    
                    # Add info about resizing
                    resize_info = f"Display resized: {frame_w}x{frame_h} -> {new_width}x{new_height}"
                    cv2.putText(display_frame, resize_info, (10, 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                # Add keyboard controls info
                controls_info = "ESC: Exit | F: Toggle Fullscreen | +/-: Resize"
                cv2.putText(display_frame, controls_info, (10, display_frame.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Create a named window that can be resized by the user if needed
                cv2.namedWindow('Player Tracking', cv2.WINDOW_NORMAL)
                cv2.imshow('Player Tracking', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('f'):  # 'f' key to toggle fullscreen
                    if cv2.getWindowProperty('Player Tracking', cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                        cv2.setWindowProperty('Player Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    else:
                        cv2.setWindowProperty('Player Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                elif key == ord('+') or key == ord('='):  # Increase display size
                    current_width = int(cv2.getWindowProperty('Player Tracking', cv2.WND_PROP_WIDTH))
                    current_height = int(cv2.getWindowProperty('Player Tracking', cv2.WND_PROP_HEIGHT))
                    cv2.resizeWindow('Player Tracking', int(current_width * 1.1), int(current_height * 1.1))
                elif key == ord('-') or key == ord('_'):  # Decrease display size
                    current_width = int(cv2.getWindowProperty('Player Tracking', cv2.WND_PROP_WIDTH))
                    current_height = int(cv2.getWindowProperty('Player Tracking', cv2.WND_PROP_HEIGHT))
                    cv2.resizeWindow('Player Tracking', int(current_width * 0.9), int(current_height * 0.9))
            
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
    parser.add_argument('--debug-duplicates', action='store_true', help='Enable debug output for duplicate detection')
    parser.add_argument('--debug-crossing', action='store_true', help='Enable debug output for crossing detection')
    parser.add_argument('--max-players', type=int, default=2, help='Maximum number of players to track (default: 2)')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = SquashPlayerTracker(
        detection_conf=args.det_conf,
        reid_threshold=args.reid_thresh,
        use_gpu=not args.cpu,
        debug_duplicates=args.debug_duplicates,
        debug_crossing=args.debug_crossing,
        max_players=args.max_players
    )
    
    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
import numpy as np
import cv2
from collections import deque

class Track:
    """Class to represent a tracked player"""
    
    def __init__(self, track_id, bbox, max_age=30):
        self.id = track_id
        self.bbox = bbox
        self.max_age = max_age
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = deque(maxlen=50)
        self.history.append(bbox)
        self.features = []
        self.player_name = None  # Will be set if face is recognized
        
        # Calculate initial center position
        self.center = self._get_center(bbox)
        
        # Velocity tracking for Kalman-like prediction
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_w = 0
        self.velocity_h = 0
        self.velocity_history = deque(maxlen=5)  # Store recent velocities
        
    def _get_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def update(self, bbox, is_smoothed=False):
        """
        Update track with new detection
        
        Args:
            bbox: New bounding box coordinates (x1, y1, x2, y2)
            is_smoothed: Whether this bbox is already smoothed (e.g., from median filter)
        """
        # Validate bbox format
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            print(f"Invalid bbox format: {bbox}")
            return
            
        try:
            x1, y1, x2, y2 = bbox
            # Ensure all values are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure bbox has positive width and height
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bbox dimensions: {bbox}")
                return
                
            # Create validated bbox
            validated_bbox = (x1, y1, x2, y2)
            
            # Update velocity information before updating position
            if len(self.history) > 0 and not is_smoothed:
                prev_bbox = self.history[-1]
                x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
                w_prev = x2_prev - x1_prev
                h_prev = y2_prev - y1_prev
                
                w = x2 - x1
                h = y2 - y1
                
                # Calculate center velocities instead of corner velocities
                center_x_prev = (x1_prev + x2_prev) / 2
                center_y_prev = (y1_prev + y2_prev) / 2
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate velocities
                velocity_x = center_x - center_x_prev
                velocity_y = center_y - center_y_prev
                velocity_w = w - w_prev
                velocity_h = h - h_prev
                
                # Store velocity in history
                self.velocity_history.appendleft((velocity_x, velocity_y, velocity_w, velocity_h))
                
                # Update track velocity (smoothed)
                alpha = 0.7  # Smoothing factor
                self.velocity_x = alpha * velocity_x + (1 - alpha) * self.velocity_x
                self.velocity_y = alpha * velocity_y + (1 - alpha) * self.velocity_y
                self.velocity_w = alpha * velocity_w + (1 - alpha) * self.velocity_w
                self.velocity_h = alpha * velocity_h + (1 - alpha) * self.velocity_h
            
            # Update track state
            self.bbox = validated_bbox
            self.center = self._get_center(validated_bbox)
            self.history.append(validated_bbox)
            
            # Only increment hits if not smoothed to avoid duplicate counting
            if not is_smoothed:
                self.hits += 1
                
            self.age = 0
            self.time_since_update = 0
        except Exception as e:
            print(f"Error updating track: {e}, bbox: {bbox}")
        
    def predict(self):
        """Predict next position based on enhanced velocity model"""
        if len(self.history) < 2:
            return self.bbox
            
        # Get current position
        curr_bbox = self.history[-1]
        x1, y1, x2, y2 = curr_bbox
        w = x2 - x1
        h = y2 - y1
        
        # Calculate smoothed velocity using multiple history points if available
        if len(self.velocity_history) > 0:
            # Use weighted average of recent velocities (more weight to recent)
            weights = np.array([0.5, 0.3, 0.15, 0.05, 0.0][:len(self.velocity_history)])
            weights = weights / weights.sum()  # Normalize weights
            
            avg_vx = 0
            avg_vy = 0
            avg_vw = 0
            avg_vh = 0
            
            for i, (vx, vy, vw, vh) in enumerate(self.velocity_history):
                avg_vx += vx * weights[i]
                avg_vy += vy * weights[i]
                avg_vw += vw * weights[i]
                avg_vh += vh * weights[i]
                
            # Add acceleration component for fast movements
            if len(self.velocity_history) >= 2:
                last_vx, last_vy, _, _ = self.velocity_history[0]
                prev_vx, prev_vy, _, _ = self.velocity_history[1] if len(self.velocity_history) > 1 else (0, 0, 0, 0)
                
                # Calculate acceleration
                ax = last_vx - prev_vx
                ay = last_vy - prev_vy
                
                # Add acceleration component (weighted)
                avg_vx += ax * 0.5
                avg_vy += ay * 0.5
            
            # Predict new position with smoothed velocity
            pred_x1 = int(x1 + avg_vx)
            pred_y1 = int(y1 + avg_vy)
            pred_w = int(w + avg_vw)
            pred_h = int(h + avg_vh)
            pred_x2 = pred_x1 + pred_w
            pred_y2 = pred_y1 + pred_h
            
        else:  # Fallback to simple prediction if no velocity history
            # Get previous position
            prev_bbox = self.history[-2]
            x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
            
            # Calculate instantaneous velocity
            vx1 = x1 - x1_prev
            vy1 = y1 - y1_prev
            vx2 = x2 - x2_prev
            vy2 = y2 - y2_prev
            
            # Predict with instantaneous velocity
            pred_x1 = int(x1 + vx1 * 1.5)  # Scale up for faster movements
            pred_y1 = int(y1 + vy1 * 1.5)
            pred_x2 = int(x2 + vx2 * 1.5)
            pred_y2 = int(y2 + vy2 * 1.5)
        
        # Ensure bounding box has positive width and height
        if pred_x2 <= pred_x1:
            pred_x2 = pred_x1 + 1
        if pred_y2 <= pred_y1:
            pred_y2 = pred_y1 + 1
            
        return (pred_x1, pred_y1, pred_x2, pred_y2)
    
    def mark_missed(self):
        """Mark this track as missed (not matched to detection)"""
        self.time_since_update += 1
        self.age += 1
        
    def is_deleted(self):
        """Check if track should be deleted"""
        return self.time_since_update > self.max_age
        
    def set_player_name(self, name):
        """Set player name if recognized"""
        self.player_name = name


class ByteTracker:
    """ByteTrack-inspired tracker for squash players"""
    
    def __init__(self, max_age=60, min_hits=2, iou_threshold=0.15):  # Lower IoU threshold for fast movements
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: List of (bbox, confidence) tuples
            
        Returns:
            list: List of active tracks
        """
        # Increment time since update for all tracks (will be reset to 0 for matched ones)
        for t in self.tracks:
            t.time_since_update += 1

        # Get active tracks
        active_tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Sort detections by confidence (higher confidence first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        # Match detections to existing tracks
        matched_track_indices, matched_detection_indices = self._match_detections_to_tracks(
            active_tracks, detections
        )
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_track_indices, matched_detection_indices):
            try:
                bbox, confidence = detections[det_idx]
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    active_tracks[track_idx].update(bbox)
                else:
                    print(f"Skipping invalid bbox format: {bbox}")
            except Exception as e:
                print(f"Error updating track from detection: {e}")
        
        # For squash court specifically, limit the number of tracks
        # Only create new tracks if we have fewer than 2 active tracks
        active_count = sum(1 for t in self.tracks if t.hits >= self.min_hits and not t.is_deleted())
        
        # Create new tracks for unmatched detections, but limit to 2 players max
        if active_count < 2:
            for i, detection in enumerate(detections):
                if i not in matched_detection_indices:
                    try:
                        bbox, confidence = detection
                        if confidence > 0.6 and isinstance(bbox, tuple) and len(bbox) == 4:  # Only high confidence and valid bbox
                            self._create_new_track(bbox)
                        elif confidence > 0.6:
                            print(f"Cannot create track with invalid bbox: {bbox}")
                    except Exception as e:
                        print(f"Error creating new track: {e}")
                    active_count += 1
                    if active_count >= 2:  # Stop after creating 2 tracks
                        break
        
        # Update states of all tracks
        self._update_tracks()
        
        # Return active tracks
        return [t for t in self.tracks if t.hits >= self.min_hits and not t.is_deleted()]
    
    def _create_new_track(self, bbox):
        """Create a new track from detection"""
        try:
            # Validate bbox format
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                print(f"Cannot create track with invalid bbox format: {bbox}")
                return
                
            x1, y1, x2, y2 = bbox
            # Ensure all values are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure bbox has positive width and height
            if x2 <= x1 or y2 <= y1:
                print(f"Cannot create track with invalid bbox dimensions: {bbox}")
                return
                
            # Create validated bbox
            validated_bbox = (x1, y1, x2, y2)
            
            new_track = Track(self.next_id, validated_bbox, self.max_age)
            self.next_id += 1
            self.tracks.append(new_track)
        except Exception as e:
            print(f"Error creating new track: {e}, bbox: {bbox}")
    
    def _match_detections_to_tracks(self, tracks, detections):
        """
        Match detections to existing tracks using IoU and center distance
        
        Args:
            tracks: List of active tracks
            detections: List of (bbox, confidence) tuples
            
        Returns:
            tuple: (matched_track_indices, matched_detection_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        distance_matrix = np.zeros((len(tracks), len(detections)))
        
        # Calculate both IoU and distance
        for t, track in enumerate(tracks):
            # Use track's predicted position if available
            track_bbox = track.predict() if len(track.history) >= 2 else track.bbox
            
            for d, (bbox, _) in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track_bbox, bbox)
                distance_matrix[t, d] = self._calculate_distance(track_bbox, bbox)
        
        # Normalize distance matrix to [0, 1] range (inverted, so closer is higher)
        if distance_matrix.max() > 0:
            distance_matrix = 1 - (distance_matrix / distance_matrix.max())
        
        # Combine IoU and distance with weights (70% IoU, 30% distance)
        combined_matrix = 0.7 * iou_matrix + 0.3 * distance_matrix
        
        # Perform greedy matching
        matched_track_indices = []
        matched_detection_indices = []
        
        # Sort combined scores in descending order
        combined_flat = combined_matrix.flatten()
        order = np.argsort(-combined_flat)
        
        track_matched = set()
        detection_matched = set()
        
        for idx in order:
            # Use IoU threshold on the IoU part, not the combined score
            track_idx = idx // len(detections)
            det_idx = idx % len(detections)
            
            # Adaptive threshold based on time since last update
            adaptive_threshold = max(0.05, self.iou_threshold - (0.02 * tracks[track_idx].time_since_update))
            
            # Only consider matches with IoU above threshold
            if iou_matrix[track_idx, det_idx] < adaptive_threshold:
                # For tracks that haven't been seen for a while, allow matching based on distance
                if tracks[track_idx].time_since_update > 3 and distance_matrix[track_idx, det_idx] > 0.6:
                    pass  # Allow matching based on distance
                else:
                    continue  # Skip this match
            
            if track_idx in track_matched or det_idx in detection_matched:
                continue
                
            matched_track_indices.append(track_idx)
            matched_detection_indices.append(det_idx)
            track_matched.add(track_idx)
            detection_matched.add(det_idx)
        
        return matched_track_indices, matched_detection_indices
    
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
        
    def _calculate_distance(self, bbox1, bbox2):
        """Calculate distance between centers of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate center points
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
        
        # Calculate Euclidean distance
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _update_tracks(self):
        """Update all tracks (mark missed, delete old)"""
        # For unmatched tracks (time_since_update > 0), coast using prediction
        for track in self.tracks:
            if track.time_since_update > 0:
                try:
                    predicted_bbox = track.predict()
                    # Apply predicted bbox without increasing hits, keep time_since_update
                    track.bbox = predicted_bbox
                    track.center = track._get_center(predicted_bbox)
                    track.history.append(predicted_bbox)
                except Exception:
                    pass
                # Increment age of the track while unmatched
                track.age += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
    
    def draw_tracks(self, image, text_scale=0.5):
        """
        Draw tracks on the image
        
        Args:
            image: Image to draw on
            text_scale: Scale of the player ID text
            
        Returns:
            image: Image with drawn tracks
        """
        img = image.copy()
        
        for track in self.tracks:
            if track.hits >= self.min_hits and not track.is_deleted():
                x1, y1, x2, y2 = track.bbox
                
                # Draw bounding box in yellow (highlight color)
                color = (0, 255, 255)  # yellow in BGR
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID and player name
                text = f"ID: {track.id}"
                if track.player_name:
                    text = f"{track.player_name} (ID: {track.id})"
                
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           text_scale, color, 2)
                
                # Draw track history (trail)
                for i in range(1, len(track.history)):
                    if i % 2 == 0:  # Draw every other point to avoid clutter
                        # Get center points
                        prev_bbox = track.history[i-1]
                        curr_bbox = track.history[i]
                        
                        prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, 
                                      (prev_bbox[1] + prev_bbox[3]) // 2)
                        curr_center = ((curr_bbox[0] + curr_bbox[2]) // 2, 
                                      (curr_bbox[1] + curr_bbox[3]) // 2)
                        
                        cv2.line(img, prev_center, curr_center, (0, 255, 255), 2)  # yellow in BGR
        
        return img
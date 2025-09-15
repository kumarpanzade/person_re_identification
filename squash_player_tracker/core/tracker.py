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


import cv2
import numpy as np
from typing import List, Tuple, Dict
from deep_sort_realtime.deepsort_tracker import DeepSort

from squash_player_tracker.core.tracker import Track  # reuse your Track class


class DeepSORTTracker:
    def __init__(
        self,
        max_age: int = 90,  # Increased from 60 to keep tracks alive longer during occlusions
        min_hits: int = 2,
        iou_threshold: float = 0.15,
        embedder: str = "mobilenet",
        half: bool = False,
        n_init: int = 2,
    ):
        # DeepSort ctor parameters map
        # - max_age: tracks are deleted if not updated in this many frames
        # - n_init: number of consecutive detections before confirming a track
        # - max_iou_distance: gating for association
        self.ds = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max(0.01, iou_threshold),  # keep small but >0
            max_cosine_distance=0.4,  # More permissive appearance matching (default is 0.2)
            nn_budget=100,  # Larger appearance descriptor budget
            embedder=embedder,  # 'mobilenet' is lightweight
            half=half,
            bgr=True,  # our frames are BGR (OpenCV)
            embedder_gpu=False,  # set True if you want GPU for embedder
        )

        self.min_hits = min_hits
        self.max_age = max_age
        self.max_tracks = None  # No limit on maximum number of tracks to maintain
        self.nms_distance_threshold = 50  # Pixel distance threshold for NMS between tracks

        # Map DeepSORT track_id -> our Track
        self._id_map: Dict[int, Track] = {}

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]], frame=None) -> List[Track]:
        """
        detections: list of (bbox_xyxy, confidence)
            bbox_xyxy: (x1,y1,x2,y2) ints
        frame: ndarray (BGR) optional, improves DeepSORT embeddings if provided
        returns: list of active our Track objects (confirmed, not deleted)
        """
        # Apply NMS on the detections to prevent duplicate bounding boxes
        detections = self._apply_nms_to_detections(detections)
        
        # Convert to deep sort format: (ltwh, conf, class_id)
        # DeepSort expects [left, top, width, height] format
        ds_dets = []
        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            # Convert from xyxy (x1, y1, x2, y2) to ltwh (left, top, width, height)
            w = x2 - x1
            h = y2 - y1
            ds_dets.append(([x1, y1, w, h], float(conf), 0))

        # Remove bbox_format parameter as it's not supported in version 1.3.2
        ds_tracks = self.ds.update_tracks(ds_dets, frame=frame)

        # Mark all existing as potentially missed; will reset when updated
        for tr in self._id_map.values():
            tr.time_since_update += 1
            tr.age += 1

        active_tracks: List[Track] = []

        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            tid = int(t.track_id)
            x1, y1, x2, y2 = [int(v) for v in t.to_tlbr()]  # tlbr == xyxy

            if tid not in self._id_map:
                self._id_map[tid] = Track(tid, (x1, y1, x2, y2), max_age=self.max_age)
            else:
                self._id_map[tid].update((x1, y1, x2, y2))

            active_tracks.append(self._id_map[tid])

        # Drop old/deleted tracks from map
        to_delete = [tid for tid, tr in self._id_map.items() if tr.is_deleted()]
        for tid in to_delete:
            del self._id_map[tid]

        # Apply non-maximum suppression between tracks to remove duplicates
        active_tracks = self._apply_nms_to_tracks(active_tracks)
        
        # If max_tracks is set, limit the number of tracks
        if self.max_tracks is not None and len(active_tracks) > self.max_tracks:
            # Sort by hits (more hits = more reliable track)
            active_tracks.sort(key=lambda t: t.hits, reverse=True)
            active_tracks = active_tracks[:self.max_tracks]

        # Apply motion-based filtering to handle glass reflections
        active_tracks = self._filter_tracks_by_motion(active_tracks)
        
        # Only return confirmed tracks with enough hits
        return [t for t in active_tracks if t.hits >= self.min_hits and not t.is_deleted()]

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes in xyxy format"""
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
    
    def _calculate_center_distance(self, bbox1, bbox2):
        """Calculate distance between centers of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate center points
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
        
        # Calculate Euclidean distance
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _apply_nms_to_detections(self, detections):
        """Apply non-maximum suppression to detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (higher first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        # NMS
        keep = []
        indices = list(range(len(detections)))
        
        while indices:
            # Take the detection with highest confidence
            current_idx = indices[0]
            current_bbox = detections[current_idx][0]
            keep.append(current_idx)
            indices.pop(0)
            
            # Compare with rest
            i = 0
            while i < len(indices):
                bbox = detections[indices[i]][0]
                iou = self._calculate_iou(current_bbox, bbox)
                
                # If IoU is high or distance is small, they're likely the same object - remove
                distance = self._calculate_center_distance(current_bbox, bbox)
                if iou > 0.45 or distance < 40:  # More aggressive NMS for glass reflections
                    indices.pop(i)
                else:
                    i += 1
        
        return [detections[i] for i in keep]
    
    def _apply_nms_to_tracks(self, tracks):
        """Apply non-maximum suppression to tracks based on IoU and distance"""
        if len(tracks) <= 1:
            return tracks
        
        # Sort by hits (more hits = more reliable track)
        tracks = sorted(tracks, key=lambda t: t.hits, reverse=True)
        
        # NMS
        keep = []
        indices = list(range(len(tracks)))
        
        while indices:
            # Take the track with highest hits
            current_idx = indices[0]
            current_track = tracks[current_idx]
            keep.append(current_idx)
            indices.pop(0)
            
            # Compare with rest
            i = 0
            while i < len(indices):
                track = tracks[indices[i]]
                
                # Calculate both IoU and center distance
                iou = self._calculate_iou(current_track.bbox, track.bbox)
                distance = self._calculate_center_distance(current_track.bbox, track.bbox)
                
                # More aggressive merging for glass reflections
                # Check IoU, distance, and also size similarity (reflections often have similar size)
                w1, h1 = current_track.bbox[2]-current_track.bbox[0], current_track.bbox[3]-current_track.bbox[1]
                w2, h2 = track.bbox[2]-track.bbox[0], track.bbox[3]-track.bbox[1]
                size_ratio = min(w1/w2 if w2 > 0 else 0, w2/w1 if w1 > 0 else 0) * min(h1/h2 if h2 > 0 else 0, h2/h1 if h1 > 0 else 0)
                
                if iou > 0.4 or distance < self.nms_distance_threshold or (distance < self.nms_distance_threshold*2 and size_ratio > 0.7):
                    # The tracks are likely the same person
                    # If the current track has a player name and the other doesn't, transfer it
                    if current_track.player_name and not track.player_name:
                        track.set_player_name(current_track.player_name)
                    # If the other track has a player name and the current doesn't, transfer it
                    elif not current_track.player_name and track.player_name:
                        current_track.set_player_name(track.player_name)
                    
                    indices.pop(i)
                else:
                    i += 1
        
        return [tracks[i] for i in keep]

    def _filter_tracks_by_motion(self, tracks):
        """
        Filter tracks based on motion patterns to handle glass reflections
        
        Glass often creates stationary or erratically moving reflections
        while real players have more consistent motion patterns
        
        Args:
            tracks: List of active tracks
            
        Returns:
            filtered_tracks: List of tracks after motion-based filtering
        """
        # Always perform motion filtering to improve quality of tracking
        # even if we have fewer tracks
            
        # Calculate motion consistency for each track
        track_scores = []
        
        for track in tracks:
            # Extract motion pattern from track history
            if len(track.history) < 3:  # Need at least 3 points for motion analysis
                # New tracks get medium score
                track_scores.append((track, 0.5))
                continue
                
            # Calculate velocity consistency
            velocities = []
            for i in range(1, min(10, len(track.history))):
                prev_bbox = track.history[i-1]
                curr_bbox = track.history[i]
                
                # Get centers
                prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2)
                curr_center = ((curr_bbox[0] + curr_bbox[2]) // 2, (curr_bbox[1] + curr_bbox[3]) // 2)
                
                # Calculate velocity
                vx = curr_center[0] - prev_center[0]
                vy = curr_center[1] - prev_center[1]
                velocities.append((vx, vy))
            
            # Measure consistency of motion
            if len(velocities) >= 2:
                # Calculate variance of velocity (lower is more consistent)
                vx_values = [v[0] for v in velocities]
                vy_values = [v[1] for v in velocities]
                
                # Total velocity magnitude (higher is more movement)
                total_motion = sum(abs(vx) + abs(vy) for vx, vy in velocities)
                
                # Variance of direction changes
                direction_changes = 0
                for i in range(1, len(velocities)):
                    prev_vx, prev_vy = velocities[i-1]
                    curr_vx, curr_vy = velocities[i]
                    
                    # Direction change detected
                    if (prev_vx * curr_vx < 0) or (prev_vy * curr_vy < 0):
                        direction_changes += 1
                
                # Score calculation:
                # - Higher score for more movement (real players move more than reflections)
                # - Higher score for consistent direction (fewer direction changes)
                # - Higher score for longer tracks (more reliable)
                
                motion_score = min(1.0, total_motion / (100 * len(velocities)))  # Normalize motion
                consistency_score = 1.0 - (direction_changes / (len(velocities) - 1))  # 1 = consistent, 0 = erratic
                history_score = min(1.0, len(track.history) / 30)  # Favor longer tracks
                
                # Reflections often have zero or erratic movement
                # Real players have consistent, non-zero movement
                final_score = (0.4 * motion_score + 0.4 * consistency_score + 0.2 * history_score)
                
                # Bonus for named tracks
                if track.player_name:
                    final_score += 0.2
                    
                track_scores.append((track, final_score))
            else:
                # Not enough history for analysis
                track_scores.append((track, 0.3))  # Lower score for tracks with little history
        
        # Sort tracks by score (descending)
        track_scores.sort(key=lambda x: x[1], reverse=True)
        
        # If max_tracks is defined, keep top tracks by motion score
        if self.max_tracks is not None:
            return [t[0] for t in track_scores[:self.max_tracks]]
        # Otherwise return all tracks sorted by score
        return [t[0] for t in track_scores]
    
    def draw_tracks(self, image, text_scale=0.5):
        img = image.copy()
        for tr in self._id_map.values():
            if tr.hits >= self.min_hits and not tr.is_deleted():
                x1, y1, x2, y2 = tr.bbox
                # Use yellow (BGR: 0, 255, 255) for bounding boxes
                color = (0, 255, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                text = f"ID: {tr.id}"
                if tr.player_name:
                    text = f"{tr.player_name} (ID: {tr.id})"
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 2)

                for i in range(1, len(tr.history)):
                    if i % 2 == 0:
                        prev_bbox = tr.history[i - 1]
                        curr_bbox = tr.history[i]
                        prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2)
                        curr_center = ((curr_bbox[0] + curr_bbox[2]) // 2, (curr_bbox[1] + curr_bbox[3]) // 2)
                        cv2.line(img, prev_center, curr_center, color, 2)
        return img
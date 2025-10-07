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
        iou_threshold: float = 0.1,
        embedder: str = "torchreid",
        half: bool = False,
        n_init: int = 4,
    ):
        # DeepSort ctor parameters map
        # - max_age: tracks are deleted if not updated in this many frames
        # - n_init: number of consecutive detections before confirming a track
        # - max_iou_distance: gating for association
        # Use mobilenet embedder to avoid torchreid import issues
        # The deep_sort_realtime library has compatibility issues with newer torchreid versions
        print("Using mobilenet embedder for DeepSORT tracker")
        embedder_to_use = "torchreid"
        embedder_gpu = True  # mobilenet embedder doesn't support GPU in deep_sort_realtime
        
        self.ds = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max(0.1, iou_threshold),  # keep small but >0
            max_cosine_distance=0.8,  # More permissive appearance matching (default is 0.2)
            nn_budget=100,  # Larger appearance descriptor budget
            embedder=embedder_to_use,  # Use available embedder
            half=half,
            bgr=True,  # our frames are BGR (OpenCV)
            embedder_gpu=embedder_gpu,  # set True if you want GPU for embedder
        )

        self.min_hits = min_hits
        self.max_age = max_age
        self.max_tracks = None  # No limit on maximum number of tracks to maintain
        self.nms_distance_threshold = 80  # Pixel distance threshold for NMS between tracks

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
    
    def _calculate_batch_iou(self, bboxes1, bboxes2):
        """Calculate IoU between two sets of bounding boxes efficiently using vectorized operations"""
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))
        
        # Convert to numpy arrays for vectorized operations
        bboxes1 = np.array(bboxes1)
        bboxes2 = np.array(bboxes2)
        
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
        x1_2, y1_2, x2_2, y2_2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]
        
        # Calculate intersection areas using broadcasting
        xx1 = np.maximum(x1_1[:, np.newaxis], x1_2[np.newaxis, :])
        yy1 = np.maximum(y1_1[:, np.newaxis], y1_2[np.newaxis, :])
        xx2 = np.minimum(x2_1[:, np.newaxis], x2_2[np.newaxis, :])
        yy2 = np.minimum(y2_1[:, np.newaxis], y2_2[np.newaxis, :])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union using broadcasting
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        
        # Prevent division by zero
        union = np.maximum(union, 1e-6)
        
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
    
    def _calculate_batch_center_distance(self, bboxes1, bboxes2):
        """Calculate distances between centers of two sets of bounding boxes efficiently"""
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))
        
        # Convert to numpy arrays
        bboxes1 = np.array(bboxes1)
        bboxes2 = np.array(bboxes2)
        
        # Calculate centers
        centers1 = np.column_stack([
            (bboxes1[:, 0] + bboxes1[:, 2]) // 2,
            (bboxes1[:, 1] + bboxes1[:, 3]) // 2
        ])
        centers2 = np.column_stack([
            (bboxes2[:, 0] + bboxes2[:, 2]) // 2,
            (bboxes2[:, 1] + bboxes2[:, 3]) // 2
        ])
        
        # Calculate distances using broadcasting
        diff = centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        return distances
    
    def _apply_nms_to_detections(self, detections):
        """Apply non-maximum suppression to detections using optimized vectorized operations"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (higher first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        # Extract bboxes and confidences
        bboxes = [det[0] for det in detections]
        confidences = [det[1] for det in detections]
        
        # Use vectorized NMS for better performance with more aggressive thresholds
        keep_indices = self._vectorized_nms(bboxes, confidences, iou_threshold=0.3, distance_threshold=30)
        
        return [detections[i] for i in keep_indices]
    
    def _vectorized_nms(self, bboxes, scores, iou_threshold=0.5, distance_threshold=50):
        """
        Vectorized Non-Maximum Suppression for better performance
        
        Args:
            bboxes: List of bounding boxes in xyxy format
            scores: List of confidence scores
            iou_threshold: IoU threshold for suppression
            distance_threshold: Distance threshold for suppression
            
        Returns:
            keep_indices: List of indices to keep after NMS
        """
        if len(bboxes) == 0:
            return []
        
        # Convert to numpy arrays
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Calculate IoU and distance matrices
        iou_matrix = self._calculate_batch_iou(bboxes, bboxes)
        distance_matrix = self._calculate_batch_center_distance(bboxes, bboxes)
        
        keep = []
        suppressed = set()
        
        for i in sorted_indices:
            if i in suppressed:
                continue
                
            keep.append(i)
            
            # Suppress boxes with high IoU or small distance
            iou_mask = iou_matrix[i] > iou_threshold
            distance_mask = distance_matrix[i] < distance_threshold
            suppress_mask = iou_mask | distance_mask
            
            # Add suppressed indices to set
            for j in np.where(suppress_mask)[0]:
                if j != i:  # Don't suppress self
                    suppressed.add(j)
        
        return keep
    
    def _apply_nms_to_tracks(self, tracks):
        """Apply non-maximum suppression to tracks using optimized vectorized operations"""
        if len(tracks) <= 1:
            return tracks
        
        # Sort by hits (more hits = more reliable track)
        tracks = sorted(tracks, key=lambda t: t.hits, reverse=True)
        
        # Extract bboxes and scores (hits)
        bboxes = [track.bbox for track in tracks]
        scores = [track.hits for track in tracks]
        
        # Use vectorized NMS with custom thresholds for tracks (more aggressive)
        keep_indices = self._vectorized_track_nms(tracks, bboxes, scores, 
                                                iou_threshold=0.3, 
                                                distance_threshold=40)
        
        return [tracks[i] for i in keep_indices]
    
    def _vectorized_track_nms(self, tracks, bboxes, scores, iou_threshold=0.4, distance_threshold=80):
        """
        Vectorized NMS specifically for tracks with player name transfer logic
        
        Args:
            tracks: List of Track objects
            bboxes: List of bounding boxes
            scores: List of hit scores
            iou_threshold: IoU threshold for suppression
            distance_threshold: Distance threshold for suppression
            
        Returns:
            keep_indices: List of indices to keep after NMS
        """
        if len(bboxes) == 0:
            return []
        
        # Convert to numpy arrays
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Calculate IoU and distance matrices
        iou_matrix = self._calculate_batch_iou(bboxes, bboxes)
        distance_matrix = self._calculate_batch_center_distance(bboxes, bboxes)
        
        # Calculate size similarity matrix for glass reflection detection
        size_similarity = self._calculate_size_similarity_matrix(bboxes)
        
        keep = []
        suppressed = set()
        
        for i in sorted_indices:
            if i in suppressed:
                continue
                
            keep.append(i)
            current_track = tracks[i]
            
            # Suppress boxes with high IoU, small distance, or similar size
            iou_mask = iou_matrix[i] > iou_threshold
            distance_mask = distance_matrix[i] < distance_threshold
            size_mask = (distance_matrix[i] < distance_threshold * 2) & (size_similarity[i] > 0.7)
            suppress_mask = iou_mask | distance_mask | size_mask
            
            # Add suppressed indices to set and transfer player names
            for j in np.where(suppress_mask)[0]:
                if j != i:  # Don't suppress self
                    suppressed.add(j)
                    track_j = tracks[j]
                    
                    # Transfer player names between merged tracks
                    if current_track.player_name and not track_j.player_name:
                        track_j.set_player_name(current_track.player_name)
                    elif not current_track.player_name and track_j.player_name:
                        current_track.set_player_name(track_j.player_name)
        
        return keep
    
    def _calculate_size_similarity_matrix(self, bboxes):
        """Calculate size similarity matrix for glass reflection detection"""
        if len(bboxes) == 0:
            return np.zeros((0, 0))
        
        # Calculate widths and heights
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        
        # Calculate size ratios using broadcasting
        w_ratios = np.minimum(widths[:, np.newaxis] / (widths[np.newaxis, :] + 1e-6),
                             widths[np.newaxis, :] / (widths[:, np.newaxis] + 1e-6))
        h_ratios = np.minimum(heights[:, np.newaxis] / (heights[np.newaxis, :] + 1e-6),
                             heights[np.newaxis, :] / (heights[:, np.newaxis] + 1e-6))
        
        # Size similarity is the product of width and height ratios
        size_similarity = w_ratios * h_ratios
        
        return size_similarity

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
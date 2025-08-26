import os
import cv2
import numpy as np
import argparse
import time
from collections import defaultdict

# Import core modules
from squash_player_tracker.core.face_recognition import FaceRecognizer
from squash_player_tracker.core.person_detection import PersonDetector
from squash_player_tracker.core.person_reid import PersonReID
from squash_player_tracker.core.tracker import ByteTracker
from squash_player_tracker.utils.visualization import draw_player_info, create_summary_image

class SquashPlayerTracker:
    def __init__(self, face_recognition_conf=0.5, detection_conf=0.5, reid_threshold=0.5, use_gpu=True):
        """
        Initialize the squash player tracking system.
        
        Args:
            face_recognition_conf: Confidence threshold for face recognition
            detection_conf: Confidence threshold for person detection
            reid_threshold: Similarity threshold for person re-identification
            use_gpu: Whether to use GPU for inference
        """
        print("Initializing Squash Player Tracker...")
        
        # Initialize components
        print("Loading face recognition model...")
        self.face_recognizer = FaceRecognizer(gpu_id=0 if use_gpu else -1)
        self.face_recognizer.recognition_threshold = face_recognition_conf
        
        print("Loading person detection model...")
        self.person_detector = PersonDetector(conf_threshold=detection_conf)
        
        print("Loading person re-identification model...")
        self.person_reid = PersonReID(use_gpu=use_gpu)
        self.person_reid.similarity_threshold = reid_threshold
        
        print("Initializing tracker...")
        self.tracker = ByteTracker(max_age=60, min_hits=2, iou_threshold=0.15)  # Lower IoU threshold for better tracking
        
        # State variables
        self.player_ids = {}  # Mapping from track ID to player name/ID
        self.identification_phase = True  # Whether we're in the initial identification phase
        self.id_phase_frames = 50  # Number of frames for identification phase
        self.current_frame = 0
        self.player_face_images = {}  # Storing face images for each player
        
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
        
        # During identification phase, focus on face recognition
        if self.identification_phase:
            # Detect faces
            face_results = self.face_recognizer.recognize_faces(frame)
            
            # Draw face detections on the frame
            if face_results:
                frame_copy = self.face_recognizer.draw_faces(frame_copy, face_results)
                
                # Store recognized faces
                for bbox, identity, confidence in face_results:
                    if identity != "unknown":
                        # Extract face image
                        x1, y1, x2, y2 = bbox
                        face_img = frame[y1:y2, x1:x2]
                        self.player_face_images[identity] = face_img
                        
                        # Add to player info
                        player_info[identity] = {'name': identity, 'confidence': confidence}
            
            # Add text indicating identification phase
            cv2.putText(frame_copy, "Identification Phase", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check if we should exit identification phase
            if self.current_frame >= self.id_phase_frames:
                self.identification_phase = False
                
                # If no faces recognized, use P1/P2 labels
                if not self.player_face_images:
                    print("No faces recognized. Switching to P1/P2 labeling.")
                    self.player_face_images = {"P1": None, "P2": None}
                    
                # No limit on number of player IDs; allow tracking many players
        
        # In tracking phase, detect and track players
        else:
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
                
                # Update tracker with validated detections
                active_tracks = self.tracker.update(valid_detections)
            except Exception as e:
                print(f"Error in detection/tracking pipeline: {e}")
                active_tracks = []
            
            # Track ID management and consistency checks
            
            # First identify tracks using appearance features
            for track in active_tracks:
                if track.player_name is None:
                    x1, y1, x2, y2 = track.bbox
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Check if we can identify using face recognition
                    face_results = self.face_recognizer.recognize_faces(person_crop)
                    if face_results:
                        _, identity, confidence = face_results[0]
                        if identity != "unknown":
                            track.set_player_name(identity)
                            # Add features to re-ID model
                            self.person_reid.add_person(frame, track.bbox, identity)
                            continue
                    
                    # Try to identify using person re-ID
                    best_match, similarity = self.person_reid.identify_person(frame, track.bbox)
                    if best_match and similarity > 0.4:  # More permissive threshold for initial ID
                        track.set_player_name(best_match)
                        continue
            
            # If we still have unidentified tracks, assign them player IDs
            # For squash, we should have exactly 2 players
            assigned_ids = [t.player_name for t in active_tracks if t.player_name]
            available_ids = [pid for pid in self.player_face_images.keys() 
                            if pid not in assigned_ids]
            
            # Count how many players we have named
            if len(assigned_ids) < 2 and available_ids:
                for track in active_tracks:
                    if track.player_name is None:
                        # Assign next available ID
                        next_id = available_ids.pop(0)
                        track.set_player_name(next_id)
                        # Add features to re-ID model
                        self.person_reid.add_person(frame, track.bbox, next_id)
                        if not available_ids:
                            break  # No more IDs to assign
            
            # Force a maximum of 2 players for squash
            # If we have more than 2 tracks with different names, keep only the 2 with most hits
            player_tracks = {}
            for track in active_tracks:
                if track.player_name:
                    if track.player_name not in player_tracks or \
                       track.hits > player_tracks[track.player_name].hits:
                        player_tracks[track.player_name] = track
            
            # Keep only 2 players maximum
            if len(player_tracks) > 2:
                # Sort by hits (most reliable tracks first)
                sorted_players = sorted(player_tracks.items(), 
                                        key=lambda x: x[1].hits, reverse=True)
                # Keep only top 2
                kept_players = [p[0] for p in sorted_players[:2]]
                
                # Reset names for other tracks
                for track in active_tracks:
                    if track.player_name and track.player_name not in kept_players:
                        track.player_name = None
            
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
            
            # Draw tracked players
            frame_copy = self.tracker.draw_tracks(frame_copy)
        
        # Add player info to the frame
        frame_copy = draw_player_info(frame_copy, player_info)
        
        return frame_copy, player_info
    
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
                cv2.imshow('Squash Player Tracking', processed_frame)
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
        
        # Create and save summary image with player information
        if self.player_face_images:
            summary_img = create_summary_image(self.player_face_images, 
                                              {id: {'name': id} for id in self.player_face_images},
                                              "Squash Players Summary")
            if output_path:
                summary_path = os.path.splitext(output_path)[0] + "_summary.jpg"
                cv2.imwrite(summary_path, summary_img)
                print(f"Saved player summary to {summary_path}")
            
            if display:
                cv2.imshow('Player Summary', summary_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    def register_player(self, name, face_image):
        """
        Manually register a player with a face image
        
        Args:
            name: Player name
            face_image: Face image
            
        Returns:
            success: Whether registration was successful
        """
        success = self.face_recognizer.add_face(face_image, name)
        if success:
            self.player_face_images[name] = face_image
            print(f"Successfully registered player: {name}")
        else:
            print(f"Failed to register player: {name} - No face detected")
        return success
    
    def set_identification_phase(self, frames=50):
        """Set the number of frames for the identification phase"""
        self.id_phase_frames = frames
        self.identification_phase = True
        self.current_frame = 0


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Squash Player Tracking System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--no-display', action='store_true', help='Disable display of processed frames')
    parser.add_argument('--id-frames', type=int, default=50, help='Number of frames for identification phase')
    parser.add_argument('--face-conf', type=float, default=0.5, help='Face recognition confidence threshold')
    parser.add_argument('--det-conf', type=float, default=0.5, help='Person detection confidence threshold')
    parser.add_argument('--reid-thresh', type=float, default=0.5, help='Re-ID similarity threshold')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = SquashPlayerTracker(
        face_recognition_conf=args.face_conf,
        detection_conf=args.det_conf,
        reid_threshold=args.reid_thresh,
        use_gpu=not args.cpu
    )
    
    # Set identification phase frames
    tracker.set_identification_phase(args.id_frames)
    
    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
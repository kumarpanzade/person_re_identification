import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import subprocess
from typing import List, Tuple

class VideoPersonDetector:
    def __init__(self, 
                 sample_rate,  # Process every N frames
                 min_segment_duration,  # Minimum duration for a clip in seconds
                 max_empty_duration,  # Maximum duration without person (in seconds)
                 use_gpu=True):  # Whether to use GPU for inference
        """
        Initialize the person detector
        
        Args:
            sample_rate: Process every Nth frame (higher = faster but less accurate)
            min_segment_duration: Minimum duration for a clip in seconds
            max_empty_duration: Maximum duration without person in seconds before ending a segment
            use_gpu: Whether to use GPU for inference
        """
        # Load YOLO model
        self.model = YOLO('yolo11l.pt')  # Using large model for better accuracy
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.model.to(self.device)
            print(f"YOLO model moved to GPU: {self.device}")
        else:
            print(f"YOLO model using CPU: {self.device}")
        
        self.sample_rate = sample_rate
        self.min_segment_duration = min_segment_duration
        self.max_empty_duration = max_empty_duration
        self.max_consecutive_empty_frames = None  # Will be calculated per video
        self.merge_threshold = None  # Will be calculated per video

    def detect_people_with_continuity(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Detect people in every frame and create segments, ending segments when 
        no person is detected for max_consecutive_empty_frames consecutive frames
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of tuples containing (start_frame, end_frame)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate max consecutive empty frames based on FPS and max empty duration
        # For higher framerates, use a larger value to prevent over-segmentation
        fps_scale_factor = max(1.0, fps / 24.0)  # Scale based on typical 24 FPS
        self.max_consecutive_empty_frames = int(self.max_empty_duration * fps / self.sample_rate * fps_scale_factor)
        
        # Set merge threshold (for higher fps, use larger threshold)
        self.merge_threshold = int(1.0 * fps)  # Merge segments that are within 1 second of each other
        
        print(f"Dynamic empty frame threshold: {self.max_consecutive_empty_frames} frames ({self.max_empty_duration} seconds at {fps} FPS)")
        print(f"Merge threshold: {self.merge_threshold} frames (1.0 seconds at {fps} FPS)")
        
        # Initialize variables
        frame_count = 0
        empty_frame_count = 0
        segments = []
        current_segment_start = None
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total Frames: {total_frames}")

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every Nth frame based on sample_rate
            if frame_count % self.sample_rate == 0:
                # Detect people in frame
                results = self.model(frame, classes=[0])  # class 0 is person in COCO
                
                # Check if any person detected
                person_detected = False
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        person_detected = True
                        break
                
                if person_detected:
                    # Reset empty frame counter
                    empty_frame_count = 0
                    
                    # Start a new segment if we don't have one
                    if current_segment_start is None:
                        current_segment_start = frame_count
                else:
                    # Increment empty frame counter
                    empty_frame_count += 1
                    
                    # If we have a segment going and reached max empty frames
                    if current_segment_start is not None and empty_frame_count >= self.max_consecutive_empty_frames:
                        # End current segment
                        segment_duration_frames = frame_count - current_segment_start
                        min_segment_frames = int(self.min_segment_duration * fps)
                        
                        if segment_duration_frames >= min_segment_frames:
                            # Subtract the empty frames from the end of the segment
                            end_frame = frame_count - empty_frame_count
                            segments.append((current_segment_start, end_frame))
                        
                        # Reset segment
                        current_segment_start = None

                # Print progress
                if frame_count % (self.sample_rate * 100) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")

            frame_count += 1

        # Handle last segment if video ends with a person
        if current_segment_start is not None:
            segment_duration_frames = frame_count - current_segment_start - empty_frame_count
            min_segment_frames = int(self.min_segment_duration * fps)
            
            if segment_duration_frames >= min_segment_frames:
                end_frame = frame_count - empty_frame_count
                segments.append((current_segment_start, end_frame))

        cap.release()
        
        # Merge segments that are close to each other (adaptive to FPS)
        merged_segments = []
        
        if not segments:
            return []
            
        # Sort segments by start time
        segments.sort(key=lambda x: x[0])
        
        # Start with the first segment
        current_start, current_end = segments[0]
        
        # Merge segments that are close to each other
        for start, end in segments[1:]:
            # If this segment starts close to the previous segment's end
            if start - current_end <= self.merge_threshold:
                # Extend the current segment
                current_end = max(current_end, end)
            else:
                # Save the current segment and start a new one
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
                
        # Add the last segment
        merged_segments.append((current_start, current_end))
        
        print(f"Original segments: {len(segments)}, Merged segments: {len(merged_segments)}")
        return merged_segments

    def extract_clips(self, video_path: str, output_dir: str, segments: List[Tuple[float, float]]):
        """
        Extract video clips for each segment
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the clips
            segments: List of (start_time, end_time) tuples in seconds
        """
        # Get video filename without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, f"{video_name}_clips")
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"\nSaving clips to: {video_output_dir}")

        for i, (start, end) in enumerate(segments):
            output_filename = f"segment_{i+1}.mp4"
            output_path = os.path.join(video_output_dir, output_filename)

            # Try CUDA acceleration with different settings
            cuda_cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', '0',
                '-i', video_path,
                '-ss', str(start),
                '-t', str(end - start),
                '-c:v', 'hevc_nvenc',  # Try HEVC/H.265 instead of H.264
                '-preset', 'fast',      # Use fast preset
                '-b:v', '5M',
                '-c:a', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',
                output_path
            ]

            try:
                print(f"\nTrying CUDA acceleration for segment {i+1}...")
                process = subprocess.run(cuda_cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    # If HEVC fails, try alternative CUDA approach
                    print("Trying alternative CUDA approach...")
                    cuda_cmd_alt = [
                        'ffmpeg',
                        '-hwaccel', 'cuda',
                        '-hwaccel_device', '0',
                        '-i', video_path,
                        '-ss', str(start),
                        '-t', str(end - start),
                        '-c:v', 'h264_nvenc',
                        '-preset', 'll',  # low latency preset
                        '-b:v', '5M',
                        '-c:a', 'copy',
                        '-avoid_negative_ts', 'make_zero',
                        '-y',
                        output_path
                    ]
                    process = subprocess.run(cuda_cmd_alt, capture_output=True, text=True)
                    
                if process.returncode != 0:
                    print("CUDA Error Details:")
                    print(process.stderr)
                    
                    print("\nFalling back to CPU encoding...")
                    cpu_cmd = [
                        'ffmpeg',
                        '-i', video_path,
                        '-ss', str(start),
                        '-t', str(end - start),
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-c:a', 'aac',
                        '-avoid_negative_ts', 'make_zero',
                        '-y',
                        output_path
                    ]
                    
                    cpu_process = subprocess.run(cpu_cmd, capture_output=True, text=True)
                    if cpu_process.returncode == 0:
                        print(f"Successfully extracted with CPU: {output_filename}")
                    else:
                        print("CPU Encoding Error Details:")
                        print(cpu_process.stderr)
                else:
                    print(f"Successfully extracted with CUDA: {output_filename}")
                    
            except Exception as e:
                print(f"Error processing segment {i+1}: {str(e)}")
    
    def create_visualization(self, video_path: str, output_dir: str, segments: List[Tuple[int, int]]):
        """
        Create a visualization video showing frame numbers and detection segments
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the visualization
            segments: List of (start_frame, end_frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        viz_path = os.path.join(output_dir, f"{video_name}_visualization.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(viz_path, fourcc, fps, (width, height))
        
        # Create lookup for which frames are in segments
        segment_frames = set()
        for start, end in segments:
            segment_frames.update(range(start, end + 1))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add frame information
            cv2.putText(frame, f"Frame: {frame_count}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
            # Add segment indicator
            if frame_count in segment_frames:
                cv2.putText(frame, "PERSON DETECTED", (width - 220, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Add green border
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 255, 0), 3)
            
            out.write(frame)
            frame_count += 1
            
            # Print progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Visualization progress: {progress:.1f}%")
                
        cap.release()
        out.release()
        print(f"Visualization saved to: {viz_path}")


def process_video_for_clips(video_path, output_dir="person_clips", 
                           sample_rate=1, min_segment_duration=10.0, max_empty_duration=1.0,
                           create_visualization=True, use_gpu=True):
    """
    Process a video to extract person clips
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the clips
        sample_rate: Process every Nth frame
        min_segment_duration: Minimum clip duration in seconds
        max_empty_duration: Maximum duration without person before ending segment
        create_visualization: Whether to create visualization video
        use_gpu: Whether to use GPU for inference
        
    Returns:
        dict: Information about extracted clips including paths and metadata
    """
    # Initialize detector
    detector = VideoPersonDetector(
        sample_rate=sample_rate,
        min_segment_duration=min_segment_duration,
        max_empty_duration=max_empty_duration,
        use_gpu=use_gpu
    )

    try:
        # Detect segments with people
        print("Detecting people with continuity awareness...")
        segments = detector.detect_people_with_continuity(video_path)

        if not segments:
            print("No people detected in video.")
            return {
                'clips': [],
                'video_info': {},
                'segments': []
            }

        print(f"Found {len(segments)} segments with people")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        # Convert frame segments to time segments for extraction
        time_segments = [(start/fps, end/fps) for start, end in segments]
        
        # Extract clips for each segment
        print("Extracting clips...")
        detector.extract_clips(video_path, output_dir, time_segments)
        
        # Get video name for clip directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clips_dir = os.path.join(output_dir, f"{video_name}_clips")
        
        # Collect clip information
        clips_info = []
        for i, (start_time, end_time) in enumerate(time_segments):
            clip_filename = f"segment_{i+1}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)
            
            if os.path.exists(clip_path):
                clips_info.append({
                    'filename': clip_filename,
                    'path': clip_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'start_frame': segments[i][0],
                    'end_frame': segments[i][1]
                })
        
        # Optional: Create visualization
        if create_visualization:
            print("Creating visualization...")
            detector.create_visualization(video_path, output_dir, segments)
        
        print("Clip extraction complete!")
        
        return {
            'clips': clips_info,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            },
            'segments': segments,
            'clips_directory': clips_dir
        }
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return {
            'clips': [],
            'video_info': {},
            'segments': [],
            'error': str(e)
        }


def main():
    # Example usage
    video_path = "https://sportsskill-staging-assets.s3.ap-south-1.amazonaws.com/videos/9e51103dd9329163e99b9a123f74ed47.mp4"
    base_output_dir = "person_clips"  # This will be the parent directory

    # Process video for clips
    result = process_video_for_clips(
        video_path=video_path,
        output_dir=base_output_dir,
        sample_rate=1,
        min_segment_duration=10.0,
        max_empty_duration=1.0,
        create_visualization=True
    )
    
    if result['clips']:
        print(f"\nExtracted {len(result['clips'])} clips:")
        for i, clip in enumerate(result['clips']):
            print(f"  {i+1}. {clip['filename']} - {clip['duration']:.1f}s ({clip['start_time']:.1f}s - {clip['end_time']:.1f}s)")
    else:
        print("No clips were extracted.")

if __name__ == "__main__":
    main()
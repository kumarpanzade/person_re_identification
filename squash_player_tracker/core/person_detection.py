import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path=None, conf_threshold=0.6):
        """
        Initialize the person detector using YOLOv8.
        
        Args:
            model_path (str): Path to a custom YOLOv8 model, if None use pretrained
            conf_threshold (float): Confidence threshold for detections
        """
        # Load YOLOv8 model (either custom or pretrained)
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8l.pt")  # Use small model for real-time performance
        
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def detect(self, image):
        """
        Detect people in an image.
        
        Args:
            image: Input image
            
        Returns:
            list: List of (bbox, confidence) tuples for person detections
        """
        if image is None or image.size == 0:
            print("Error: Empty or invalid image for detection")
            return []
            
        try:
            # Run inference with YOLOv8
            results = self.model(image, verbose=False)
            
            person_detections = []
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    try:
                        # Only consider 'person' class (class 0 in COCO)
                        if box.cls.cpu().numpy()[0] == 0:  # COCO class 0 is person
                            confidence = box.conf.cpu().numpy()[0]
                            
                            if confidence >= self.conf_threshold:
                                # Get box coordinates and ensure they're integers
                                coords = box.xyxy.cpu().numpy()[0].astype(int)
                                x1, y1, x2, y2 = coords
                                
                                # Ensure box has positive width and height
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                    
                                # Ensure box is within image boundaries
                                h, w = image.shape[:2]
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(w - 1, x2)
                                y2 = min(h - 1, y2)
                                
                                # Check if box is still valid after adjustments
                                if x2 > x1 and y2 > y1:
                                    person_detections.append(((x1, y1, x2, y2), confidence))
                    except Exception as e:
                        print(f"Error processing detection box: {e}")
                        continue
        except Exception as e:
            print(f"Error in person detection: {e}")
            return []
        
        return person_detections
    
    def draw_detections(self, image, detections):
        """
        Draw person bounding boxes on an image.
        
        Args:
            image: Image to draw on
            detections: List of (bbox, confidence) tuples
            
        Returns:
            image: Image with drawn detections
        """
        img = image.copy()
        for (x1, y1, x2, y2), confidence in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow in BGR
            label = f"Person: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow in BGR
            
        return img
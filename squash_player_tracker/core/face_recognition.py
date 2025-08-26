import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class FaceRecognizer:
    def __init__(self, det_size=(640, 640), gpu_id=0):
        """
        Initialize the face recognition system using InsightFace.
        
        Args:
            det_size (tuple): Detection size for the face detector
            gpu_id (int): GPU ID to use (-1 for CPU)
        """
        # Initialize the face analysis module
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider'] if gpu_id >= 0 else ['CPUExecutionProvider'])
        self.app.prepare(ctx_id=gpu_id, det_size=det_size)
        
        # Dictionary to store face embeddings for known people
        self.known_faces = {}
        # Threshold for face recognition similarity
        self.recognition_threshold = 0.5
    
    def add_face(self, image, person_id):
        """
        Add a face to the known faces database
        
        Args:
            image: Image containing the face
            person_id: ID or name of the person
            
        Returns:
            bool: True if face was added successfully, False otherwise
        """
        # Get face embeddings
        faces = self.app.get(image)
        
        if not faces:
            return False
        
        # Use the largest face found (assuming it's the main person)
        largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
        self.known_faces[person_id] = largest_face.embedding
        return True
    
    def recognize_faces(self, image):
        """
        Recognize faces in an image and return their identities
        
        Args:
            image: Input image
            
        Returns:
            list: List of (bbox, identity, confidence) tuples
        """
        if not self.known_faces:
            return []  # No known faces to compare against
        
        # Get faces from the image
        faces = self.app.get(image)
        results = []
        
        for face in faces:
            # Find the closest matching face
            identity = "unknown"
            highest_similarity = -1
            
            for person_id, embedding in self.known_faces.items():
                # Calculate cosine similarity
                similarity = np.dot(face.embedding, embedding) / (
                    np.linalg.norm(face.embedding) * np.linalg.norm(embedding)
                )
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    identity = person_id
            
            # Only accept if similarity is above threshold
            if highest_similarity > self.recognition_threshold:
                bbox = face.bbox.astype(int)
                results.append((bbox, identity, highest_similarity))
            else:
                bbox = face.bbox.astype(int)
                results.append((bbox, "unknown", highest_similarity))
                
        return results
    
    def detect_faces(self, image):
        """
        Detect faces in an image without recognition
        
        Args:
            image: Input image
            
        Returns:
            list: List of detected face bounding boxes
        """
        faces = self.app.get(image)
        return [face.bbox.astype(int) for face in faces]
    
    def draw_faces(self, image, results):
        """
        Draw face bounding boxes and labels on an image
        
        Args:
            image: Image to draw on
            results: List of (bbox, identity, confidence) tuples
            
        Returns:
            image: Image with drawn faces
        """
        img = image.copy()
        for bbox, identity, confidence in results:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with identity and confidence
            label = f"{identity} ({confidence:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return img
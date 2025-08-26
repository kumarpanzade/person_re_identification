import os
import numpy as np
import torch
import torch.nn as nn
import torchreid
from torchreid.utils import FeatureExtractor

class PersonReID:
    def __init__(self, model_name='osnet_x1_0', model_path=None, use_gpu=True):
        """
        Initialize person re-identification model using OSNet.
        
        Args:
            model_name (str): Name of the model architecture
            model_path (str): Path to custom weights file, if None use pretrained
            use_gpu (bool): Whether to use GPU for inference
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Initialize the feature extractor
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=self.device
        )
        
        # Dictionary to store person feature vectors
        self.person_features = {}
        self.similarity_threshold = 0.35  # Lower threshold for more permissive matching
        self.features_history = {}  # Store multiple feature vectors per person
        self.max_history_size = 5   # Number of feature vectors to store per person
    
    def extract_features(self, image, bbox):
        """
        Extract features from a person crop.
        
        Args:
            image: Full image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            features: Feature vector for the person
        """
        x1, y1, x2, y2 = bbox
        person_crop = image[y1:y2, x1:x2]
        
        # Ensure the crop is not empty or too small
        if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
            return None
        
        # Convert to RGB if grayscale
        if len(person_crop.shape) == 2:
            person_crop = np.stack([person_crop] * 3, axis=2)
        
        # Extract features
        features = self.extractor(person_crop)
        return features.cpu().numpy().flatten()
    
    def add_person(self, image, bbox, person_id):
        """
        Add a person to the database.
        
        Args:
            image: Image containing the person
            bbox: Bounding box of the person
            person_id: ID or name of the person
            
        Returns:
            bool: True if person was added successfully
        """
        features = self.extract_features(image, bbox)
        if features is None:
            return False
        
        # Initialize feature history for this person if not exists
        if person_id not in self.features_history:
            self.features_history[person_id] = []
        
        # Add features to history
        self.features_history[person_id].append(features)
        
        # Keep only the most recent features
        if len(self.features_history[person_id]) > self.max_history_size:
            self.features_history[person_id] = self.features_history[person_id][-self.max_history_size:]
        
        # Compute average feature vector for quick matching
        self.person_features[person_id] = np.mean(self.features_history[person_id], axis=0)
        return True
    
    def identify_person(self, image, bbox):
        """
        Identify a person based on appearance features.
        
        Args:
            image: Image containing the person
            bbox: Bounding box of the person
            
        Returns:
            tuple: (person_id, confidence) or (None, 0) if no match
        """
        if not self.person_features:
            return None, 0
        
        # Extract features from the current detection
        features = self.extract_features(image, bbox)
        if features is None:
            return None, 0
        
        # Find the best match by checking against all historical features
        best_match = None
        highest_similarity = -1
        
        for person_id, feature_list in self.features_history.items():
            # Calculate maximum similarity with any historical feature vector
            similarities = []
            for stored_features in feature_list:
                # Calculate cosine similarity
                similarity = np.dot(features, stored_features) / (
                    np.linalg.norm(features) * np.linalg.norm(stored_features)
                )
                similarities.append(similarity)
            
            # Use the maximum similarity for this person
            if similarities:
                person_max_similarity = max(similarities)
                if person_max_similarity > highest_similarity:
                    highest_similarity = person_max_similarity
                    best_match = person_id
        
        if highest_similarity > self.similarity_threshold:
            return best_match, highest_similarity
        else:
            return None, highest_similarity
    
    def update_features(self, person_id, image, bbox):
        """
        Update features for a person by adding to history.
        
        Args:
            person_id: ID of the person to update
            image: Current image
            bbox: Current bounding box
        """
        if person_id not in self.person_features:
            return
        
        new_features = self.extract_features(image, bbox)
        if new_features is None:
            return
        
        # Only update if similarity with existing features is high enough
        # This prevents corrupting the feature history with misidentifications
        should_update = False
        
        if person_id in self.features_history and self.features_history[person_id]:
            similarities = []
            for stored_features in self.features_history[person_id]:
                similarity = np.dot(new_features, stored_features) / (
                    np.linalg.norm(new_features) * np.linalg.norm(stored_features)
                )
                similarities.append(similarity)
            
            # If similar enough to at least one existing feature, update
            if max(similarities) > 0.7:  # Higher threshold for updates
                should_update = True
        else:
            # If no history yet, always update
            should_update = True
        
        if should_update:
            # Add to feature history
            if person_id not in self.features_history:
                self.features_history[person_id] = []
            
            self.features_history[person_id].append(new_features)
            
            # Keep only the most recent features
            if len(self.features_history[person_id]) > self.max_history_size:
                self.features_history[person_id] = self.features_history[person_id][-self.max_history_size:]
            
            # Update average feature vector
            self.person_features[person_id] = np.mean(self.features_history[person_id], axis=0)
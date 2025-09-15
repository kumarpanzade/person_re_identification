import os
import numpy as np
import torch
import torchreid
from torchreid.utils.feature_extractor import FeatureExtractor


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
        self.similarity_threshold = 0.8  # Lower threshold for more permissive matching
        self.features_history = {}  # Store multiple feature vectors per person
        self.max_history_size = 30   # Number of feature vectors to store per person
        
        # Flag to indicate if re-ID has been initialized with reference embeddings
        self.is_initialized = False
        self.reference_embeddings = {}  # Store reference embeddings from initialization
        self.use_reference_only = False  # Whether to use only reference embeddings
        
        # Enhanced tracking consistency
        self.last_assignments = {}  # Track ID -> Player ID
        self.assignment_history = {}  # Track ID -> list of Player IDs with counts
        self.id_consistency_window = 60  # Number of frames to consider for ID consistency
        self.strict_id_enforcement = True  # Whether to enforce strict ID consistency
    
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
        
        # Debug output for tracking re-ID
        print(f"Stored initial features for {person_id}, total people: {len(self.person_features)}")
        return True
    
    def identify_person(self, image, bbox):
        """
        Identify a person based on appearance features with enhanced accuracy.
        
        Args:
            image: Image containing the person
            bbox: Bounding box of the person
            
        Returns:
            tuple: (person_id, confidence) or (None, 0) if no match
        """
        # Check if we have any features to compare against
        if not self.person_features and not self.reference_embeddings:
            return None, 0
        
        # Extract features from the current detection
        features = self.extract_features(image, bbox)
        if features is None:
            return None, 0
        
        # Find the best match by checking against features
        best_match = None
        highest_similarity = -1
        all_similarities = {}  # Store all similarities for analysis
        
        # First check if this is a reference embedding
        if self.reference_embeddings:
            # Prioritize reference embeddings by using a higher base similarity
            reference_boost = 0.4  # Boost reference similarities to prioritize them
            
            for person_id, ref_features in self.reference_embeddings.items():
                # Calculate cosine similarity
                similarity = np.dot(features, ref_features) / (
                    np.linalg.norm(features) * np.linalg.norm(ref_features)
                )
                
                # Apply boost to reference embeddings
                boosted_similarity = similarity + reference_boost
                
                # Store all similarities for analysis
                all_similarities[person_id] = similarity
                
                if boosted_similarity > highest_similarity:
                    highest_similarity = boosted_similarity
                    best_match = person_id
        
        # If not using reference-only mode, check all historical features too
        if not self.use_reference_only:
            for person_id, feature_list in self.features_history.items():
                # Skip if this was already checked as a reference
                if person_id in all_similarities:
                    continue
                    
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
                    all_similarities[person_id] = person_max_similarity
                    
                    if person_max_similarity > highest_similarity:
                        highest_similarity = person_max_similarity
                        best_match = person_id
        
        # Advanced matching with similarity analysis
        if all_similarities:
            # Look for clear winner vs. ambiguous match
            sorted_similarities = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)
            
            # If we have multiple matches, check if the top one is significantly better
            if len(sorted_similarities) > 1:
                top_id, top_sim = sorted_similarities[0]
                second_id, second_sim = sorted_similarities[1]
                
                # If top two are very close, be more conservative
                if top_sim - second_sim < 0.1:  # Less than 0.1 difference is ambiguous
                    # Require higher threshold for ambiguous matches
                    threshold = self.similarity_threshold + 0.1
                else:
                    # Clear winner, can use normal threshold
                    threshold = self.similarity_threshold
            else:
                threshold = self.similarity_threshold
        else:
            threshold = self.similarity_threshold
        
        # Handle different thresholds based on context
        if self.use_reference_only:
            # More strict threshold when using only reference embeddings
            threshold = max(threshold, self.similarity_threshold + 0.05)
        
        # Decide based on adjusted threshold
        if highest_similarity > threshold:
            if len(all_similarities) > 1:
                # Show top matches for debugging
                top_matches = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
                match_str = ", ".join([f"{id}:{sim:.2f}" for id, sim in top_matches])
                print(f"Re-ID match: {best_match} with similarity {highest_similarity:.2f}, top matches: {match_str}")
            else:
                print(f"Re-ID match: {best_match} with similarity {highest_similarity:.2f}")
            return best_match, highest_similarity
        else:
            if all_similarities:
                top_matches = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
                match_str = ", ".join([f"{id}:{sim:.2f}" for id, sim in top_matches])
                print(f"No re-ID match found, top similarities: {match_str}")
            else:
                print(f"No re-ID match found, no valid similarities")
            return None, highest_similarity
    
    def update_features(self, person_id, image, bbox):
        """
        Update features for a person by adding to history.
        
        Args:
            person_id: ID of the person to update
            image: Current image
            bbox: Current bounding box
        """
        # If we're using reference embeddings only, don't update
        if self.use_reference_only:
            return
            
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
            
    def set_reference_embedding(self, person_id, features):
        """
        Set a reference embedding for a person directly
        
        Args:
            person_id: ID of the person
            features: Feature vector (numpy array)
            
        Returns:
            bool: True if successful
        """
        if features is None or not isinstance(features, np.ndarray):
            return False
            
        # Store as reference embedding
        self.reference_embeddings[person_id] = features
        
        # Also add to features_history and person_features for consistent API
        self.features_history[person_id] = [features]
        self.person_features[person_id] = features
        
        # Mark as initialized
        self.is_initialized = True
        
        print(f"Set reference embedding for {person_id}")
        return True
    
    def initialize_from_reference_images(self, reference_images, use_reference_only=True):
        """
        Initialize with reference images for each person
        
        Args:
            reference_images: Dictionary mapping person_id to (image, bbox)
            use_reference_only: Whether to use only reference embeddings for matching
            
        Returns:
            bool: True if successful
        """
        if not reference_images:
            return False
            
        success = True
        for person_id, (image, bbox) in reference_images.items():
            features = self.extract_features(image, bbox)
            if features is not None:
                self.set_reference_embedding(person_id, features)
            else:
                print(f"Failed to extract features for {person_id}")
                success = False
                
        # Set flag to use only reference embeddings if requested
        self.use_reference_only = use_reference_only
        
        print(f"Initialized {len(reference_images)} reference embeddings")
        print(f"Using reference embeddings only: {use_reference_only}")
        
        return success
        
    def enforce_id_consistency(self, track_id, player_id):
        """
        Enforce consistent identity assignments for tracks
        
        Args:
            track_id: Tracker's track ID
            player_id: Player ID to assign
            
        Returns:
            str: Most consistent player ID for this track
        """
        # Initialize history for this track if needed
        if track_id not in self.assignment_history:
            self.assignment_history[track_id] = {}
        
        # Record this assignment in history
        if player_id not in self.assignment_history[track_id]:
            self.assignment_history[track_id][player_id] = 0
        self.assignment_history[track_id][player_id] += 1
        
        # Store most recent assignment
        self.last_assignments[track_id] = player_id
        
        if not self.strict_id_enforcement:
            return player_id
            
        # Return the most frequent assignment for this track
        if self.assignment_history[track_id]:
            most_common_id = max(self.assignment_history[track_id].items(), 
                               key=lambda x: x[1])[0]
            
            # Only override if there's a strong history
            if self.assignment_history[track_id][most_common_id] > 3:
                if most_common_id != player_id:
                    print(f"ID consistency enforced: {player_id} -> {most_common_id} for track {track_id}")
                return most_common_id
                
        return player_id
    
    def get_consistent_id(self, track_id):
        """
        Get the consistent ID for a track based on history
        
        Args:
            track_id: Track ID to check
            
        Returns:
            str: Most consistent player ID or None if no history
        """
        if track_id not in self.assignment_history:
            return None
            
        if not self.assignment_history[track_id]:
            return None
            
        # Get most common assignment
        most_common_id = max(self.assignment_history[track_id].items(), 
                           key=lambda x: x[1])[0]
        count = self.assignment_history[track_id][most_common_id]
        
        # Only return if we have sufficient history
        if count >= 3:
            return most_common_id
            
        return None
        
    def calculate_similarity(self, image, bbox, player_id):
        """
        Calculate similarity between a detected person and a specific player ID
        
        Args:
            image: Current frame
            bbox: Bounding box of detected person
            player_id: Player ID to compare against
            
        Returns:
            float: Similarity score (0-1) or -1 if calculation fails
        """
        # Check if we have reference embedding for this player
        if self.reference_embeddings and player_id in self.reference_embeddings:
            # Extract features from current detection
            features = self.extract_features(image, bbox)
            if features is None:
                return -1
                
            # Get reference features
            ref_features = self.reference_embeddings[player_id]
            
            # Calculate cosine similarity
            similarity = np.dot(features, ref_features) / (
                np.linalg.norm(features) * np.linalg.norm(ref_features)
            )
            
            return similarity
            
        # Check in features history if no reference embedding
        elif player_id in self.features_history and self.features_history[player_id]:
            # Extract features from current detection
            features = self.extract_features(image, bbox)
            if features is None:
                return -1
                
            # Calculate similarity with all stored features
            similarities = []
            for stored_features in self.features_history[player_id]:
                similarity = np.dot(features, stored_features) / (
                    np.linalg.norm(features) * np.linalg.norm(stored_features)
                )
                similarities.append(similarity)
                
            # Return maximum similarity
            if similarities:
                return max(similarities)
        
        # No features available for this player ID
        return -1
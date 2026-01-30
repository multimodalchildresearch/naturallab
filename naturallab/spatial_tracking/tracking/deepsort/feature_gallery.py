import numpy as np
import logging
from typing import Dict, List, Optional
from scipy.spatial.distance import cdist

class PersonFeatureGallery:
    """Maintain a gallery of person features for re-identification"""
    
    def __init__(self, max_features_per_id: int = 20, max_inactive_time: int = 60,
                 similarity_threshold: float = 0.6):
        """
        Initialize the feature gallery
        
        Args:
            max_features_per_id: Maximum number of feature vectors to store per person
            max_inactive_time: Maximum number of frames to keep a person without updates
            similarity_threshold: Threshold for feature matching similarity
        """
        self.logger = logging.getLogger("motion_tracking.PersonFeatureGallery")
        self.gallery = {}  # track_id -> [features, last_seen]
        self.max_features_per_id = max_features_per_id
        self.max_inactive_time = max_inactive_time
        self.similarity_threshold = similarity_threshold
        self.frame_count = 0
        
        self.logger.info(f"Initialized with max_features_per_id={max_features_per_id}, "
                        f"max_inactive_time={max_inactive_time}, "
                        f"similarity_threshold={similarity_threshold}")
    
    def add_features(self, track_id: str, features: np.ndarray) -> None:
        """Add new features for a person"""
        if features is None:
            return
            
        if track_id not in self.gallery:
            self.gallery[track_id] = {
                'features': [features],
                'last_seen': self.frame_count
            }
            self.logger.debug(f"Created new entry for track {track_id[:6]}")
        else:
            # Update last seen time
            self.gallery[track_id]['last_seen'] = self.frame_count
            
            # Add new features
            feature_list = self.gallery[track_id]['features']
            feature_list.append(features)
            
            # Keep only the most recent features
            if len(feature_list) > self.max_features_per_id:
                self.gallery[track_id]['features'] = feature_list[-self.max_features_per_id:]
    
    def find_matching_id(self, features: np.ndarray) -> Optional[str]:
        """Find the best matching person ID for new features"""
        if features is None or len(self.gallery) == 0:
            return None
            
        best_id = None
        best_similarity = -1
        
        for track_id, data in self.gallery.items():
            # Calculate similarity against all stored features for this ID
            similarities = []
            for feat in data['features']:
                # Compute cosine similarity
                similarity = 1.0 - cdist(
                    features.reshape(1, -1),
                    feat.reshape(1, -1),
                    metric='cosine'
                )[0][0]
                similarities.append(similarity)
            
            # Use the maximum similarity for this ID
            max_similarity = max(similarities) if similarities else 0
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_id = track_id
        
        # Only return a match if similarity is above threshold
        if best_similarity > self.similarity_threshold:
            self.logger.debug(f"Found match for features: track {best_id[:6]} with similarity {best_similarity:.3f}")
            return best_id
        
        return None
    
    def clean_inactive(self) -> None:
        """Remove inactive persons from the gallery"""
        inactive_ids = []
        
        for track_id, data in self.gallery.items():
            if self.frame_count - data['last_seen'] > self.max_inactive_time:
                inactive_ids.append(track_id)
        
        for track_id in inactive_ids:
            self.logger.debug(f"Removed inactive track {track_id[:6]}")
            del self.gallery[track_id]
    
    def update_frame_count(self) -> None:
        """Update the internal frame counter"""
        self.frame_count += 1
        self.clean_inactive()
    
    def get_feature_count(self, track_id: str) -> int:
        """Get the number of features stored for a track ID"""
        if track_id not in self.gallery:
            return 0
        return len(self.gallery[track_id]['features'])
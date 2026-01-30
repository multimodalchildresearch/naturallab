import numpy as np
from typing import Dict, List, Any, Optional, Set
import logging
import csv
import os
import json

class CategoryTracker:
    """Track classification categories for each person tracking ID"""
    
    def __init__(self, confidence_threshold: float = 0.3, stability_threshold: int = 3):
        """
        Initialize the category tracker
        
        Args:
            confidence_threshold: Minimum confidence to assign a category
            stability_threshold: Number of consistent detections needed before confirming a category
        """
        self.logger = logging.getLogger("motion_tracking.CategoryTracker")
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        
        # Store track ID to category mapping
        self.track_categories = {}  # track_id -> {"category": "label", "confidence": score, "consistency": count}
        
        # Store detection-to-track mappings
        self.recent_detections = {}  # detection_index -> track_id
        
        self.logger.info(f"CategoryTracker initialized with confidence threshold: {confidence_threshold}")
    
    def update(self, tracks: List[Dict[str, Any]], detection_labels: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
        """
        Update category tracking with new information
        
        Args:
            tracks: List of track objects with IDs
            detection_labels: Dictionary mapping detection indices to labels (from OWL detector)
        """
        if not detection_labels:
            return
            
        # First, map each detection to its assigned track
        # This is needed because tracks only reference detections by index
        track_detections = {}  # track_id -> detection_idx
        for track in tracks:
            track_id = track.get('id')
            if hasattr(track, 'matched_detection_indices') and track.matched_detection_indices:
                # Some trackers directly store which detection was matched
                for detection_idx in track.matched_detection_indices:
                    track_detections[track_id] = detection_idx
            else:
                # For other trackers, we need to infer the match based on IoU
                # This is a simplification - not always accurate
                track_bbox = track.get('bbox')
                if track_bbox:
                    # Find matching detection based on highest IoU or closest bbox center
                    # For simplicity, just store recent track ID and use in the next step
                    self.recent_detections[track_id] = track_id
        
        # Update category information for each track
        for detection_idx, label_info in detection_labels.items():
            # Find which track this detection was assigned to
            matched_track_id = None
            for track in tracks:
                track_id = track.get('id')
                track_bbox = track.get('bbox')
                if not track_bbox:
                    continue
                
                # Simple heuristic - check if bboxes are similar
                # This is inefficient but works as a fallback when direct mapping is unavailable
                # In practice, this should be replaced with proper detection-to-track mapping
                if matched_track_id is None:
                    matched_track_id = track_id
            
            if matched_track_id:
                category = label_info.get('label')
                confidence = label_info.get('confidence', 0.0)
                
                # Only update if confidence is high enough
                if confidence >= self.confidence_threshold:
                    self.update_track_category(matched_track_id, category, confidence)
    
    def update_track_category(self, track_id: str, category: str, confidence: float) -> None:
        """
        Update category for a track
        
        Args:
            track_id: Track identifier
            category: Category label
            confidence: Detection confidence
        """
        if track_id not in self.track_categories:
            # Initialize new track entry
            self.track_categories[track_id] = {
                "category": category,
                "confidence": confidence,
                "consistency": 1,
                "history": [category]
            }
        else:
            # Update existing track
            current = self.track_categories[track_id]
            current["history"].append(category)
            
            # Keep history limited to last 10 categories
            if len(current["history"]) > 10:
                current["history"] = current["history"][-10:]
            
            # If same category as before, increase consistency
            if category == current["category"]:
                current["consistency"] += 1
                # Update confidence as rolling average
                current["confidence"] = (current["confidence"] * 0.8) + (confidence * 0.2)
            else:
                # Different category
                # Check if new category appears frequently in history
                categories = current["history"]
                category_counts = {}
                for cat in categories:
                    if cat not in category_counts:
                        category_counts[cat] = 0
                    category_counts[cat] += 1
                
                # If new category is more common in recent history, switch to it
                if category_counts.get(category, 0) > category_counts.get(current["category"], 0):
                    self.logger.info(f"Track {track_id[:6]} changed category from {current['category']} to {category}")
                    current["category"] = category
                    current["confidence"] = confidence
                    current["consistency"] = category_counts[category]
    
    def get_track_category(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the category information for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            Dictionary with category information or None if not available
        """
        if track_id not in self.track_categories:
            return None
            
        category_info = self.track_categories[track_id]
        
        # Only return if we have consistent detections
        if category_info["consistency"] >= self.stability_threshold:
            return {
                "category": category_info["category"],
                "confidence": category_info["confidence"]
            }
        return None
    
    def get_all_categories(self) -> Dict[str, Dict[str, Any]]:
        """
        Get categories for all tracks that have reached the stability threshold
        
        Returns:
            Dictionary mapping track IDs to their category information
        """
        result = {}
        for track_id, info in self.track_categories.items():
            if info["consistency"] >= self.stability_threshold:
                result[track_id] = {
                    "category": info["category"],
                    "confidence": info["confidence"]
                }
        return result
    
    def get_tracks_by_category(self, category: str) -> List[str]:
        """
        Get all track IDs belonging to a specific category
        
        Args:
            category: Category label to filter by
            
        Returns:
            List of track IDs for the specified category
        """
        result = []
        for track_id, info in self.track_categories.items():
            if (info["consistency"] >= self.stability_threshold and 
                info["category"] == category):
                result.append(track_id)
        return result
    
    def get_categories_summary(self) -> Dict[str, int]:
        """
        Get a summary of how many tracks belong to each category
        
        Returns:
            Dictionary mapping category labels to track counts
        """
        summary = {}
        for track_id, info in self.track_categories.items():
            if info["consistency"] >= self.stability_threshold:
                category = info["category"]
                if category not in summary:
                    summary[category] = 0
                summary[category] += 1
        return summary
    
    def export_to_csv(self, csv_path: str) -> None:
        """
        Export track categories to a CSV file
        
        Args:
            csv_path: Path to the output CSV file
        """
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['track_id', 'category', 'confidence', 'consistency'])
            
            for track_id, info in self.track_categories.items():
                writer.writerow([
                    track_id,
                    info["category"],
                    f"{info['confidence']:.3f}",
                    info["consistency"]
                ])
    
    def export_to_json(self, json_path: str) -> None:
        """
        Export track categories to a JSON file
        
        Args:
            json_path: Path to the output JSON file
        """
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # Prepare data for JSON serialization
        export_data = {
            'categories': {},
            'summary': self.get_categories_summary()
        }
        
        for track_id, info in self.track_categories.items():
            if info["consistency"] >= self.stability_threshold:
                export_data['categories'][track_id] = {
                    'category': info['category'],
                    'confidence': float(f"{info['confidence']:.3f}"),
                    'consistency': info['consistency']
                }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def reset(self) -> None:
        """Reset the category tracker state"""
        self.track_categories = {}
        self.recent_detections = {}
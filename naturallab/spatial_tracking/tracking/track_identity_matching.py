import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
import clip
from pathlib import Path

class TrackIdentityMatcher:
    """
    Match DeepSORT track IDs to specific identities using CLIP embeddings.
    
    This class uses CLIP to compare visual features from track galleries
    with textual descriptions of people to identify which track belongs to whom.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the track identity matcher with CLIP model
        
        Args:
            device: Device to run CLIP on ('cpu' or 'cuda'). If None, use CUDA if available.
        """
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing CLIP model on {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Store gallery images and embeddings
        self.track_galleries = {}  # track_id -> list of images
        self.track_embeddings = {}  # track_id -> average CLIP embedding
        
    def extract_track_galleries(self, video_path: str, tracks_csv: str, output_dir: str, 
                              frames_per_track: int = 10) -> Dict[str, List[str]]:
        """
        Extract gallery images for each track from video based on tracking data
        
        Args:
            video_path: Path to the input video
            tracks_csv: Path to DeepSORT tracking CSV
            output_dir: Directory to save gallery images
            frames_per_track: Number of frames to extract per track
            
        Returns:
            Dictionary mapping track IDs to lists of image paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read tracking data
        df = pd.read_csv(tracks_csv)
        
        # Get unique track IDs
        track_ids = df['track_id'].unique()
        print(f"Found {len(track_ids)} unique tracks in {tracks_csv}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # For each track, select frames distributed throughout its appearance
        galleries = {}
        
        for track_id in track_ids:
            # Get frames where this track appears
            track_frames = df[df['track_id'] == track_id]
            
            # Skip tracks with too few appearances
            if len(track_frames) < frames_per_track:
                print(f"Track {track_id} has only {len(track_frames)} frames, skipping")
                continue
            
            # Select frames evenly distributed throughout track's appearance
            sample_indices = np.linspace(0, len(track_frames)-1, frames_per_track, dtype=int)
            selected_frames = track_frames.iloc[sample_indices]
            
            # Extract images for each selected frame
            track_gallery = []
            
            for _, row in selected_frames.iterrows():
                frame_idx = int(row['frame'])
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                
                # Set video to this frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Could not read frame {frame_idx}")
                    continue
                
                # Extract person crop
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Save crop
                crop_path = os.path.join(output_dir, f"{track_id}_{frame_idx}.jpg")
                cv2.imwrite(crop_path, crop)
                track_gallery.append(crop_path)
            
            galleries[track_id] = track_gallery
            print(f"Extracted {len(track_gallery)} images for track {track_id}")
        
        cap.release()
        self.track_galleries = galleries
        return galleries
    
    def compute_track_embeddings(self, gallery_dir: Optional[str] = None, 
                               galleries: Optional[Dict[str, List[str]]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute CLIP embeddings for track galleries
        
        Args:
            gallery_dir: Directory containing gallery images (optional)
            galleries: Dictionary mapping track IDs to lists of image paths (optional)
            
        Returns:
            Dictionary mapping track IDs to average CLIP embeddings
        """
        if galleries is None and gallery_dir is not None:
            # Build galleries from directory structure
            galleries = {}
            for path in Path(gallery_dir).glob("*.jpg"):
                track_id = path.stem.split("_")[0]
                if track_id not in galleries:
                    galleries[track_id] = []
                galleries[track_id].append(str(path))
        
        if galleries is None:
            galleries = self.track_galleries
            
        if not galleries:
            raise ValueError("No galleries provided")
        
        # Compute embeddings for each track
        embeddings = {}
        
        for track_id, image_paths in galleries.items():
            if not image_paths:
                continue
                
            # Load and preprocess images
            images = []
            for path in image_paths:
                try:
                    img = Image.open(path)
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            
            if not images:
                continue
                
            # Convert to tensor and compute embeddings
            image_input = torch.stack(images).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute average embedding
            avg_embedding = image_features.mean(dim=0)
            avg_embedding /= avg_embedding.norm()
            
            embeddings[track_id] = avg_embedding
            
        print(f"Computed embeddings for {len(embeddings)} tracks")
        self.track_embeddings = embeddings
        return embeddings
    
    def match_identities(self, identity_descriptions: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Match track IDs to identity descriptions using CLIP
        
        Args:
            identity_descriptions: Dictionary mapping identity names to descriptions
                e.g., {"person1": "person wearing red shirt with glasses", 
                       "person2": "tall person in blue jeans"}
            
        Returns:
            Dictionary mapping identity names to dictionaries of (track_id, similarity) pairs
        """
        if not self.track_embeddings:
            raise ValueError("No track embeddings computed. Run compute_track_embeddings first.")
        
        # Encode text descriptions
        text_inputs = [f"a photo of {desc}" for desc in identity_descriptions.values()]
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
        # Normalize text features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities for each identity
        results = {}
        identity_names = list(identity_descriptions.keys())
        

        # Find best track for each identity
        for i, identity in enumerate(identity_names):
            identity_text_feature = text_features[i]
            
            # Find best track for this identity
            best_score = -1
            best_track = None
            
            for track_id, track_embedding in self.track_embeddings.items():
                similarity = (100.0 * identity_text_feature @ track_embedding).item()
                if similarity > best_score:
                    best_score = similarity
                    best_track = track_id
            
            # Create a dictionary with just the best match
            if best_track:
                results[identity] = {best_track: best_score}
        
        return results
    
    def assign_tracks_to_identities(self, identity_descriptions: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Assign each track to the best-matching identity using CLIP
        
        This method assigns every available track to an identity, unlike match_identities 
        which only finds the best track for each identity.
        
        Args:
            identity_descriptions: Dictionary mapping identity names to descriptions
                e.g., {"Caregiver": "adult person", "Child": "young child"}
            
        Returns:
            Dictionary mapping identity names to dictionaries of (track_id, similarity) pairs
            All tracks will be assigned to their best-matching identity
        """
        if not self.track_embeddings:
            raise ValueError("No track embeddings computed. Run compute_track_embeddings first.")
        
        # Encode text descriptions
        text_inputs = [f"a photo of {desc}" for desc in identity_descriptions.values()]
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
        # Normalize text features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Initialize results for each identity
        results = {identity: {} for identity in identity_descriptions.keys()}
        identity_names = list(identity_descriptions.keys())
        
        # Simple binary classification for Caregiver vs Child
        for track_id, track_embedding in self.track_embeddings.items():
            # Get scores for both identities
            caregiver_score = (100.0 * text_features[0] @ track_embedding).item()
            child_score = (100.0 * text_features[1] @ track_embedding).item()
            
            # Calculate margin and assign to winner
            margin = abs(caregiver_score - child_score)
            confidence = min(100.0, margin * 5)  # Scale margin to confidence
            
            # Assign to higher scoring identity
            if caregiver_score > child_score:
                results["Caregiver"][track_id] = confidence
            else:
                results["Child"][track_id] = confidence
        
        return results
    
    def get_detailed_scoring(self, identity_descriptions: Dict[str, str]) -> Dict[str, Dict[str, Dict]]:
        """
        Simple binary scoring for Caregiver vs Child
        """
        if not self.track_embeddings:
            raise ValueError("No track embeddings computed. Run compute_track_embeddings first.")
        
        # Encode text descriptions
        text_inputs = [f"a photo of {desc}" for desc in identity_descriptions.values()]
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
        # Normalize text features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        detailed_results = {}
        
        # Simple binary classification
        for track_id, track_embedding in self.track_embeddings.items():
            caregiver_score = (100.0 * text_features[0] @ track_embedding).item()
            child_score = (100.0 * text_features[1] @ track_embedding).item()
            
            margin = abs(caregiver_score - child_score)
            confidence = min(100.0, margin * 5)
            
            best_match = "Caregiver" if caregiver_score > child_score else "Child"
            
            detailed_results[track_id] = {
                'caregiver_score': caregiver_score,
                'child_score': child_score,
                'margin': margin,
                'confidence': confidence,
                'best_match': best_match
            }
        
        return detailed_results
    
    def visualize_matches(self, video_path: str, tracks_csv: str, identity_matches: Dict[str, Dict[str, float]],
                         output_path: str, confidence_threshold: float = 70.0):
        """
        Create a visualization video showing the identity matches
        
        Args:
            video_path: Path to the input video
            tracks_csv: Path to DeepSORT tracking CSV
            identity_matches: Results from match_identities
            output_path: Path to save the visualization video
            confidence_threshold: Minimum similarity score to display a match
        """
        # Read tracking data
        df = pd.read_csv(tracks_csv)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create reverse mapping from track_id to most likely identity
        track_to_identity = {}
        for identity, similarities in identity_matches.items():
            best_track_id = next(iter(similarities))
            best_score = similarities[best_track_id]
            
            if best_score >= confidence_threshold:
                if best_track_id not in track_to_identity or best_score > track_to_identity[best_track_id][1]:
                    track_to_identity[best_track_id] = (identity, best_score)
        
        # Process video
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get tracks for this frame
            frame_tracks = df[df['frame'] == frame_idx]
            
            # Draw bounding boxes and labels
            for _, track in frame_tracks.iterrows():
                track_id = track['track_id']
                x1, y1, x2, y2 = int(track['x1']), int(track['y1']), int(track['x2']), int(track['y2'])
                
                # Draw box
                if track_id in track_to_identity:
                    # Green for matched tracks
                    color = (0, 255, 0)
                    identity, score = track_to_identity[track_id]
                    label = f"{identity} ({score:.1f}%)"
                else:
                    # Red for unmatched tracks
                    color = (0, 0, 255)
                    label = f"Track {track_id}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_idx}/{frame_count}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save frame
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"Visualization saved to {output_path}")
        return output_path

# Example usage
if __name__ == "__main__":
    # Initialize matcher
    matcher = TrackIdentityMatcher()
    
    # Extract galleries from tracking results
    galleries = matcher.extract_track_galleries(
        video_path="input/video.mp4",
        tracks_csv="output/data/deepsort_tracks.csv",
        output_dir="output/galleries",
        frames_per_track=10
    )
    
    # Compute embeddings
    embeddings = matcher.compute_track_embeddings()
    
    # Define identity descriptions
    identities = {
        "John": "tall man wearing blue shirt",
        "Sarah": "woman with blonde hair in red dress",
        "Mike": "man with glasses and black t-shirt"
    }
    
    # Match identities to tracks
    matches = matcher.match_identities(identities)
    
    # Print results
    for identity, similarities in matches.items():
        print(f"\nMatches for {identity}:")
        for track_id, score in list(similarities.items())[:3]:  # Top 3 matches
            print(f"  Track {track_id}: {score:.2f}% similarity")
    
    # Create visualization
    matcher.visualize_matches(
        video_path="input/video.mp4",
        tracks_csv="output/data/deepsort_tracks.csv",
        identity_matches=matches,
        output_path="output/visualizations/identity_matches.mp4",
        confidence_threshold=70.0
    )
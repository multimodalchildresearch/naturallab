from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid

class DetectionResult:
    """Standard data structure for detection results"""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int = 0):
        """
        Initialize detection data
        
        Args:
            bbox: Bounding box in format [x1, y1, x2, y2]
            confidence: Detection confidence score
            class_id: Class ID of the detection (0 = person in COCO)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
    
    def to_array(self) -> np.ndarray:
        """Convert to array format [x1, y1, x2, y2, confidence]"""
        return np.array([*self.bbox, self.confidence])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'DetectionResult':
        """Create DetectionResult from array [x1, y1, x2, y2, confidence]"""
        return cls(arr[:4].tolist(), float(arr[4]))


class TrackResult:
    """Standard data structure for track information"""
    
    def __init__(self, track_id: str, bbox: List[float], confidence: float,
                time_since_update: int = 0, hits: int = 0, color: Optional[Tuple[int, int, int]] = None):
        """
        Initialize track data
        
        Args:
            track_id: Unique track identifier
            bbox: Bounding box in format [x1, y1, x2, y2]
            confidence: Detection confidence score
            time_since_update: Frames since last update
            hits: Number of detections for this track
            color: Display color for this track (BGR)
        """
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.time_since_update = time_since_update
        self.hits = hits
        self.color = color
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.track_id,
            'bbox': self.bbox,
            'score': self.confidence,
            'time_since_update': self.time_since_update,
            'hits': self.hits,
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, track_dict: Dict[str, Any]) -> 'TrackResult':
        """Create TrackResult from dictionary"""
        return cls(
            track_id=track_dict['id'],
            bbox=track_dict['bbox'],
            confidence=track_dict['score'],
            time_since_update=track_dict.get('time_since_update', 0),
            hits=track_dict.get('hits', 0),
            color=track_dict.get('color')
        )


class PoseResult:
    """Standard data structure for pose estimation results"""
    
    def __init__(self, track_id: str, landmarks: Any, bbox: List[float], crop_origin: Tuple[int, int] = None):
        """
        Initialize pose data
        
        Args:
            track_id: Track identifier
            landmarks: MediaPipe pose landmarks
            bbox: Bounding box used for pose estimation
            crop_origin: Origin of the crop in original image coordinates
        """
        self.track_id = track_id
        self.landmarks = landmarks
        self.bbox = bbox
        self.crop_origin = crop_origin
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'track_id': self.track_id,
            'landmarks': self.landmarks,
            'bbox': self.bbox,
            'crop_origin': self.crop_origin
        }

    
class MovementResult:
    """Standard data structure for movement analysis results"""
    
    def __init__(self, track_id: str):
        """
        Initialize movement data
        
        Args:
            track_id: Track identifier
        """
        self.track_id = track_id
        self.position_history = []
        self.time_history = []
        self.total_distance = 0
        self.current_speed = 0
        self.is_turning = False
        self.is_leaning = False
        self.last_floor_position = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'track_id': self.track_id,
            'position_history': self.position_history,
            'time_history': self.time_history,
            'total_distance': self.total_distance,
            'current_speed': self.current_speed,
            'is_turning': self.is_turning,
            'is_leaning': self.is_leaning,
            'last_floor_position': self.last_floor_position,
        }


def generate_track_id() -> str:
    """Generate a unique track ID"""
    return str(uuid.uuid4())
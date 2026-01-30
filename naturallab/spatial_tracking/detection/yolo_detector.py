import numpy as np
import time
from typing import Dict, List, Any, Optional

from motion_tracking.base import TrackerModule
from motion_tracking.utils.data_structures import DetectionResult

class YOLODetectorModule(TrackerModule):
    """Person detection module using YOLOv8"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.25):
        """
        Initialize the YOLOv8 detector
        
        Args:
            model_path: Path to the YOLOv8 model file (can also be 'yolov8n.pt', 'yolov8s.pt', etc.)
            confidence: Confidence threshold for detections
        """
        super().__init__(name="YOLODetector")
        
        # Import here for better error handling and to avoid dependency issues
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            self.log_error("Failed to import ultralytics. Install with: pip install ultralytics")
            raise
        except Exception as e:
            self.log_error(f"Failed to load YOLO model from {model_path}: {e}")
            raise
            
        self.confidence = confidence
        self.log_info(f"Loaded YOLO model from {model_path} with confidence threshold {confidence}")
        
        # Person class ID in COCO dataset (used by YOLOv8)
        self.person_class_id = 0
        
        # For tracking performance
        self.last_inference_time = 0
        self.frame_count = 0
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect persons in the frame
        
        Args:
            data: Dictionary containing 'frame' (np.ndarray)
            
        Returns:
            Dictionary with 'detections' containing list of bounding boxes [x1, y1, x2, y2, score]
        """
        frame = data['frame']
        
        # Start timer
        start_time = time.time()
        
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        # Extract person detections
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Only keep person class (class 0 in COCO)
                if box.cls.cpu().numpy()[0] == self.person_class_id:
                    # Get box coordinates in [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    confidence = box.conf.cpu().numpy()[0]
                    
                    # Keep using the old format for compatibility, but we could switch to DetectionResult objects
                    detections.append([x1, y1, x2, y2, confidence])
        
        # Update timing statistics
        self.last_inference_time = time.time() - start_time
        self.frame_count += 1
        
        # Log performance occasionally
        if self.frame_count % 100 == 0:
            self.log_info(f"Frame {self.frame_count}: Detected {len(detections)} persons in {self.last_inference_time*1000:.1f}ms")
        
        # Return the detected bounding boxes
        return {'detections': detections, **data}
    
    def reset(self) -> None:
        """Reset the detector state"""
        self.frame_count = 0
        self.last_inference_time = 0
        self.log_info("Detector reset")
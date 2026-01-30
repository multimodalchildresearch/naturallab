import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
from PIL import Image

from motion_tracking.base import TrackerModule
from motion_tracking.utils.data_structures import DetectionResult

class OWLDetectorModule(TrackerModule):
    """Person detection module using OWLv2 with text prompts"""
    
    def __init__(self, 
                 model_name: str = 'google/owlv2-base-patch16-ensemble',
                 text_prompts: List[str] = None,
                 confidence: float = 0.25,
                 device: str = None):
        """
        Initialize the OWLv2 detector
        
        Args:
            model_name: Name or path to OWLv2 model
            text_prompts: List of text prompts for detection (default: ['a photo of a person'])
            confidence: Confidence threshold for detections
            device: Device to run the model on ('cpu', 'cuda', etc.). If None, use cuda if available
        """
        super().__init__(name="OWLDetector")
        
        # Default text prompts if none provided
        if text_prompts is None:
            text_prompts = ['a photo of a person']
        
        self.text_prompts = text_prompts
        self.confidence = confidence
        
        # Format text prompts as expected by OWLv2 (list of lists)
        self.formatted_prompts = [self.text_prompts]
        
        # Import here for better error handling
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            self.log_info(f"Loading OWLv2 model: {model_name}")
            
            # Set up device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            self.log_info(f"Using device: {self.device}")
            
            # Load model and processor
            self.processor = Owlv2Processor.from_pretrained(model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
            
            self.log_info(f"OWLv2 model loaded successfully with prompts: {text_prompts}")
            self.has_model = True
            
        except ImportError:
            self.log_error("Failed to import transformers. Install with: pip install transformers")
            self.has_model = False
        except Exception as e:
            self.log_error(f"Failed to load OWLv2 model: {e}")
            self.has_model = False
        
        # For tracking performance
        self.last_inference_time = 0
        self.frame_count = 0
    
    def set_text_prompts(self, text_prompts: List[str]) -> None:
        """
        Update the text prompts used for detection
        
        Args:
            text_prompts: List of text prompts for detection
        """
        self.text_prompts = text_prompts
        self.formatted_prompts = [self.text_prompts]
        self.log_info(f"Updated text prompts: {text_prompts}")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect persons in the frame using OWLv2 and text prompts
        
        Args:
            data: Dictionary containing 'frame' (np.ndarray)
            
        Returns:
            Dictionary with 'detections' containing list of bounding boxes [x1, y1, x2, y2, score]
        """
        if not self.has_model:
            self.log_error("OWLv2 model not loaded successfully. Returning empty detections.")
            return {'detections': [], **data}
        
        frame = data['frame']
        
        # Start timer
        start_time = time.time()
        
        # Convert frame to PIL Image (required by OWLv2)
        pil_image = Image.fromarray(frame[..., ::-1])  # BGR to RGB conversion
        
        # Get original image dimensions
        image_height, image_width = frame.shape[:2]
        target_sizes = torch.tensor([(image_height, image_width)]).to(self.device)
        
        # Run OWLv2 inference
        with torch.no_grad():
            # Process inputs
            inputs = self.processor(text=self.formatted_prompts, images=pil_image, return_tensors="pt").to(self.device)
            
            # Get predictions
            outputs = self.model(**inputs)
            
            # Post-process outputs to get boxes and scores
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=self.confidence, 
                text_labels=self.formatted_prompts
            )
        
        # Extract detections for the single image
        detections = []
        detection_labels = {}  # Store labels for each detection
        result = results[0]  # Only one image in our case
        
        if len(result["boxes"]) > 0:
            boxes, scores, labels = result["boxes"], result["scores"], result["text_labels"]
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                # Convert box to [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                confidence = score.cpu().item()
                
                # Add detection if it meets confidence threshold
                if confidence >= self.confidence:
                    # Create a unique ID for this detection
                    detection_id = f"det_{self.frame_count}_{i}"
                    
                    # Store the detection with same format as YOLO for compatibility
                    detections.append([x1, y1, x2, y2, confidence])
                    
                    # Store the label for this detection
                    # We'll use the index in the detections list to associate with the label
                    detection_index = len(detections) - 1
                    detection_labels[detection_index] = {
                        'label': label,
                        'confidence': confidence
                    }
                    
                    # Log high-confidence detections periodically
                    if self.frame_count % 100 == 0 and confidence > 0.5:
                        self.log_debug(f"Detected {label} with confidence {confidence:.3f}")
        
        # Update timing statistics
        self.last_inference_time = time.time() - start_time
        self.frame_count += 1
        
        # Log performance occasionally
        if self.frame_count % 100 == 0:
            self.log_info(f"Frame {self.frame_count}: Detected {len(detections)} objects in {self.last_inference_time*1000:.1f}ms")
        
        # Return the detected bounding boxes and labels
        return {
            'detections': detections,
            'detection_labels': detection_labels,  # Add labels to output
            **data
        }
    
    def reset(self) -> None:
        """Reset the detector state"""
        self.frame_count = 0
        self.last_inference_time = 0
        self.log_info("OWL detector reset")
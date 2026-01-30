import numpy as np
import cv2
import torch
import logging
from typing import Optional

class AppearanceFeatureExtractor:
    """Extract appearance features from person crops using a deep learning model"""
    
    def __init__(self, model_path: str = 'osnet_x0_25_market.pt', device: str = 'cpu'):
        """
        Initialize the feature extractor
        
        Args:
            model_path: Path to a pre-trained ReID model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.logger = logging.getLogger("motion_tracking.AppearanceFeatureExtractor")
        
        # Handle device selection - torchreid expects a string
        if device == 'mps':
            self.logger.warning("MPS device specified, but torchreid may not support it. Using CPU instead.")
            self.device_str = 'cpu'
            self.device = torch.device('cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device_str = 'cuda'
            self.device = torch.device('cuda')
        else:
            self.device_str = 'cpu'
            self.device = torch.device('cpu')
        
        self.logger.info(f"Using device: {self.device_str}")
        
        # Determine the correct model architecture from the filename
        self.model_arch = self._detect_model_architecture(model_path)
        self.logger.info(f"Detected model architecture: {self.model_arch}")
        
        # Load the model
        try:
            import torchreid
            self.logger.info(f"torchreid imported successfully, version: {torchreid.__version__}")
            self.model = self._load_model(model_path)
            
            # Fix data type issues by converting model to float32
            self._fix_model_dtypes()
            
            self.has_model = True
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.has_model = False
        
        # Image preprocessing parameters
        self.img_size = (128, 256)  # h, w 
        
        # Mean and std for normalization (ImageNet values)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def _detect_model_architecture(self, model_path: str) -> str:
        """Detect the model architecture from the filename"""
        model_path = model_path.lower()
        
        # Look for specific architectures in the filename
        if 'osnet_ain_x1_0' in model_path:
            return 'osnet_ain_x1_0'
        elif 'osnet_ain_x0_75' in model_path:
            return 'osnet_ain_x0_75'
        elif 'osnet_ain_x0_5' in model_path:
            return 'osnet_ain_x0_5'
        elif 'osnet_ain_x0_25' in model_path:
            return 'osnet_ain_x0_25'
        elif 'osnet_x1_0' in model_path:
            return 'osnet_x1_0'
        elif 'osnet_x0_75' in model_path:
            return 'osnet_x0_75'
        elif 'osnet_x0_5' in model_path:
            return 'osnet_x0_5'
        elif 'osnet_x0_25' in model_path:
            return 'osnet_x0_25'
        else:
            # Default to a common architecture
            self.logger.warning("Could not detect model architecture from filename. Using osnet_x0_25 as default.")
            return 'osnet_x0_25'
    
    def _fix_model_dtypes(self):
        """Fix model data types by converting all parameters to float32"""
        if hasattr(self.model, 'model'):
            # Convert model parameters to float32
            self.model.model = self.model.model.float()
            self.logger.info("Converted model to float32")
    
    def _load_model(self, model_path: str):
        """Load the feature extraction model"""
        import torchreid
        
        # Load pre-trained OSNet model - use the string device name and detected architecture
        model = torchreid.utils.FeatureExtractor(
            model_name=self.model_arch,  # Use detected architecture
            model_path=model_path,
            device=self.device_str
        )
        
        # Make sure the model is in evaluation mode
        if hasattr(model, 'eval') and callable(model.eval):
            model.eval()
        elif hasattr(model, 'model') and hasattr(model.model, 'eval'):
            model.model.eval()
        
        return model
    
    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image for the model"""
        if image is None or image.size == 0:
            return None
            
        # Resize image
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        image = (image - self.mean) / self.std
        
        # Convert to channel-first format (C, H, W)
        image = image.transpose(2, 0, 1)
        
        return image
    
    def extract_deep_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract deep features using the model"""
        if image is None or image.size == 0:
            return None
            
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(image)
            if preprocessed is None:
                return None
                
            # Convert to tensor and ensure float32 data type
            tensor = torch.from_numpy(preprocessed).float().unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(tensor)
            
            # Return as numpy array
            return features.cpu().numpy()[0]
        except Exception as e:
            self.logger.error(f"Error in deep feature extraction: {e}")
            return self.extract_fallback_features(image)
    
    def extract_fallback_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract color histogram features as a fallback"""
        if image is None or image.size == 0:
            return None
            
        # Resize image for consistent feature size
        image = cv2.resize(image, (64, 128))
        
        # Convert to HSV color space for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Concatenate histograms into a single feature vector
        features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        
        return features
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract appearance features from an image"""
        if image is None or image.size == 0:
            return None
            
        if self.has_model:
            features = self.extract_deep_features(image)
            if features is not None:
                return features
            self.logger.warning("Deep feature extraction failed, falling back to histogram features")
            
        return self.extract_fallback_features(image)
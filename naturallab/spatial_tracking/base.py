from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class TrackerModule(ABC):
    """Base abstract class for all tracker modules"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the module
        
        Args:
            name: Optional name for this module (for logging)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"motion_tracking.{self.name}")
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return processed results
        
        Args:
            data: Dictionary containing input data
            
        Returns:
            Dictionary with processed data (extending the input data)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the module state"""
        pass
    
    def log_debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
        
    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
        
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
        
    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
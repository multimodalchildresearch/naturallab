import time
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

from motion_tracking.base import TrackerModule

class TrackerPipeline:
    """Pipeline to manage the flow of data through tracker modules"""
    
    def __init__(self, modules: List[TrackerModule]):
        """
        Initialize the tracker pipeline with a list of modules
        
        Args:
            modules: List of TrackerModule instances in processing order
        """
        self.logger = logging.getLogger("motion_tracking.TrackerPipeline")
        self.modules = modules
        
        # Performance monitoring
        self.processing_times = {module.name: [] for module in modules}
        self.total_frames = 0
        
        module_names = [module.name for module in modules]
        self.logger.info(f"Initialized pipeline with modules: {module_names}")
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Process a single frame through the pipeline
        
        Args:
            frame: Input video frame
            frame_idx: Current frame index
            
        Returns:
            Tuple of (success, output_frame, data)
        """
        # Start overall timing
        pipeline_start = time.time()
        
        # Initialize data dictionary with the input frame and frame index
        data = {'frame': frame, 'frame_idx': frame_idx}
        
        # Process through each module in sequence
        for module in self.modules:
            try:
                # Time each module's processing
                module_start = time.time()
                
                # Process the data
                data = module.process(data)
                
                # Record processing time
                module_time = time.time() - module_start
                self.processing_times[module.name].append(module_time)
                
                # Keep only recent times for average calculation
                if len(self.processing_times[module.name]) > 100:
                    self.processing_times[module.name] = self.processing_times[module.name][-100:]
                
            except Exception as e:
                self.logger.error(f"Error in module {module.name}: {e}")
                import traceback
                traceback.print_exc()
                return False, frame, data
        
        # Calculate overall pipeline time
        pipeline_time = time.time() - pipeline_start
        
        # Log performance statistics periodically
        self.total_frames += 1
        if self.total_frames % 100 == 0:
            self.log_performance()
        
        # Return the output frame if available
        output_frame = data.get('output_frame', frame)
        return True, output_frame, data
    
    def log_performance(self) -> None:
        """Log performance statistics for each module"""
        total_avg = 0
        self.logger.info(f"Performance after {self.total_frames} frames:")
        
        for module in self.modules:
            name = module.name
            times = self.processing_times.get(name, [])
            
            if times:
                avg_time = sum(times) / len(times)
                total_avg += avg_time
                self.logger.info(f"  {name}: {avg_time*1000:.1f}ms ({1/avg_time:.1f} FPS)")
            else:
                self.logger.info(f"  {name}: No timing data")
        
        if total_avg > 0:
            self.logger.info(f"  Total pipeline: {total_avg*1000:.1f}ms ({1/total_avg:.1f} FPS)")
    
    def reset(self) -> None:
        """Reset all modules in the pipeline"""
        for module in self.modules:
            try:
                module.reset()
            except Exception as e:
                self.logger.error(f"Error resetting module {module.name}: {e}")
        
        self.processing_times = {module.name: [] for module in self.modules}
        self.logger.info("Pipeline reset complete")
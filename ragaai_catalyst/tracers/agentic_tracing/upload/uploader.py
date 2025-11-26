"""
uploader.py - Simple abstraction for trace uploaders
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class AbstractTraceUploader(ABC):
    """
    Abstract base class for trace uploaders.
    
    This interface defines the public API for all trace uploaders.
    Implementations should handle async processing, caching, and error handling.
    """
    
    @abstractmethod
    def submit(self, trace_data: Dict[str, Any]) -> Optional[str]:
        """
        Submit trace data for asynchronous upload.
        
        Args:
            trace_data: Dictionary containing trace information
                - filepath: Path to trace file
                - project_name: Name of the project
                - dataset_name: Name of the dataset
                - base_url: Backend API base URL
                - (and other required fields)
        
        Returns:
            Task ID string if submission successful, None otherwise
        """
        pass
    
    @abstractmethod
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get upload status for a specific task.
        
        Args:
            task_id: Unique identifier for the upload task
        
        Returns:
            Dictionary with task status information
        """
        pass
    
    @abstractmethod
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get overall upload queue status and statistics.
        
        Returns:
            Dictionary with queue metrics (pending, completed, failed, etc.)
        """
        pass
    
    @abstractmethod
    def shutdown(self, timeout: int = 120) -> None:
        """
        Gracefully shutdown the uploader and release resources.
        
        Args:
            timeout: Maximum time in seconds to wait for pending tasks
        """
        pass


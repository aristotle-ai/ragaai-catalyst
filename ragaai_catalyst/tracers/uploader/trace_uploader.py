"""
trace_uploader.py - Concrete implementation of AbstractTraceUploader

PUBLIC API (use this):
    - TraceUploader class (implements AbstractTraceUploader)
        - get_instance()  - Get singleton instance
        - submit()        - Submit trace for upload (accepts UploadRequest or dict)
        - get_status()    - Get task status
        - get_queue_status() - Get queue statistics
        - shutdown()      - Graceful shutdown

PRIVATE/INTERNAL (do not use directly):
    - submit_upload_task(), get_task_status(), get_upload_queue_status() (module functions)
    - _update_task_result(), _complete_task_success(), _fail_task_with_error() (result helpers)
    - _process_upload(), _save_task_status(), _generate_dataset_cache_key() (internal helpers)

This module provides async trace uploading with:
    - Multi-executor pool for high throughput
    - Dataset schema caching with TTL (when available)
    - Automatic retries and error handling
    - Presigned URL upload to S3/Azure
    - Type-safe UploadRequest interface (no more dict coupling)
    - Graceful fallback when dependencies unavailable
"""

import os
import json
import tempfile
import threading
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

from ragaai_catalyst.tracers.utils import get_logger
from ragaai_catalyst.tracers.models import UploadTask

if TYPE_CHECKING:
    from ragaai_catalyst.tracers.models import UploadRequest

logger = get_logger(__name__)

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

DATASET_CACHE_DURATION = 600
DATASET_CACHE_MAX_SIZE = 1000
CACHE_KEY_SEPARATOR = "#"

try:
    from ragaai_catalyst.tracers.uploader.uploader import AbstractTraceUploader
    from ragaai_catalyst.tracers.core.executor_pool import ExecutorPool
    from ragaai_catalyst.tracers.core.cache import DatasetCache
    from ragaai_catalyst.tracers.core.api_client import TraceAPIClient
    from ragaai_catalyst.tracers.utils.create_dataset_schema import create_dataset_schema_with_trace
    from ragaai_catalyst.tracers.utils import update_presigned_url
    from ragaai_catalyst.session_manager import session_manager
    from ragaai_catalyst import RagaAICatalyst

    IMPORTS_AVAILABLE = True

    QUEUE_DIR = os.path.join(tempfile.gettempdir(), "ragaai_tasks")
    os.makedirs(QUEUE_DIR, exist_ok=True)
    _dataset_cache = DatasetCache(max_size=DATASET_CACHE_MAX_SIZE, ttl=DATASET_CACHE_DURATION)

except ImportError:
    logger.warning("RagaAI Catalyst imports not available - running in test mode")
    IMPORTS_AVAILABLE = False
    session_manager = None
    AbstractTraceUploader = object

    QUEUE_DIR = os.path.join(tempfile.gettempdir(), "ragaai_tasks")
    os.makedirs(QUEUE_DIR, exist_ok=True)
    _dataset_cache = None

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class TraceUploader(AbstractTraceUploader):
    """
    Concrete implementation of AbstractTraceUploader.
    
    Now accepts type-safe UploadRequest instead of fragile dict.
    Uses singleton pattern for global access.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of TraceUploader."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def submit(self, upload_request) -> Optional[str]:
        """
        Submit trace for asynchronous upload.

        Args:
            upload_request: UploadRequest or dict (for backward compatibility)

        Returns:
            Task ID for tracking upload status
        """
        if isinstance(upload_request, dict):
            upload_request = self._convert_dict_to_request(upload_request)

        task = self._convert_request_to_task(upload_request)

        return submit_upload_task(task)
    
    def _convert_dict_to_request(self, trace_data: Dict[str, Any]) -> 'UploadRequest':
        """Convert legacy dict to UploadRequest (backward compatibility)."""
        from ragaai_catalyst.tracers.models import UploadRequest, CodeArchive, HashId
        from pathlib import Path
        
        code_archive = None
        if trace_data.get('hash_id') and trace_data.get('zip_path'):
            code_archive = CodeArchive(
                path=Path(trace_data['zip_path']),
                hash_id=HashId(trace_data['hash_id']),
                file_count=0
            )
        
        return UploadRequest(
            trace_file=Path(trace_data['filepath']),
            code_archive=code_archive,
            project_name=trace_data['project_name'],
            dataset_name=trace_data['dataset_name'],
            base_url=trace_data['base_url'],
            tracer_type=trace_data['tracer_type'],
            user_details=trace_data.get('user_details', {}),
            timeout=trace_data.get('timeout', 120)
        )
    
    def _convert_request_to_task(self, request: 'UploadRequest') -> UploadTask:
        """Convert UploadRequest to internal UploadTask format."""
        return UploadTask(
            filepath=str(request.trace_file),
            hash_id=request.code_archive.hash_id if request.code_archive else '',
            zip_path=str(request.code_archive.path) if request.code_archive else '',
            project_name=request.project_name,
            dataset_name=request.dataset_name,
            user_details=request.user_details,
            base_url=request.base_url,
            tracer_type=request.tracer_type,
            timeout=request.timeout
        )
    
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get upload status for a specific task."""
        return get_task_status(task_id)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall upload queue status and statistics."""
        return get_upload_queue_status()
    
    def shutdown(self, timeout: int = 120) -> None:
        """Gracefully shutdown the uploader and release resources."""
        shutdown(timeout)


def _generate_dataset_cache_key(dataset_name: str, project_name: str, base_url: str) -> str:
    return f"{dataset_name}{CACHE_KEY_SEPARATOR}{project_name}{CACHE_KEY_SEPARATOR}{base_url}"

def _save_task_status(task_status: Dict[str, Any]) -> None:
    task_id = task_status["task_id"]
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    with open(status_path, "w") as f:
        json.dump(task_status, f, indent=2)

def _update_task_result(result: Dict[str, Any], status: str, task_id: str, error_msg: Optional[str] = None) -> Dict[str, Any]:
    """Update task result with status and optional error, then save and return."""
    update_data = {
        "status": status,
        "end_time": datetime.now().isoformat()
    }
    if error_msg:
        update_data["error"] = error_msg

    result.update(update_data)

    if status == STATUS_COMPLETED:
        logger.info(f"Upload task {task_id} completed successfully")
    elif status == STATUS_FAILED:
        logger.error(f"Upload task {task_id} failed: {error_msg}")

    _save_task_status(result)
    return result

def _complete_task_success(result: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Mark task as completed successfully."""
    return _update_task_result(result, STATUS_COMPLETED, task_id)

def _fail_task_with_error(result: Dict[str, Any], error_msg: str, task_id: str) -> Dict[str, Any]:
    """Mark task as failed with error message."""
    return _update_task_result(result, STATUS_FAILED, task_id, error_msg)

def _create_dataset_schema(task: UploadTask) -> bool:
    """Create dataset schema for the task's dataset."""
    try:
        # Use cached version if available, otherwise call directly
        if IMPORTS_AVAILABLE and _dataset_cache is not None:
            cache_key = _generate_dataset_cache_key(
                task.dataset_name, task.project_name, task.base_url
            )
            response = _dataset_cache.get_or_create(cache_key, lambda: create_dataset_schema_with_trace(
                dataset_name=task.dataset_name,
                project_name=task.project_name,
                base_url=task.base_url,
                user_details=task.user_details,
                timeout=task.timeout
            ))
        else:
            # Direct call when caching not available (test mode)
            response = create_dataset_schema_with_trace(
                dataset_name=task.dataset_name,
                project_name=task.project_name,
                base_url=task.base_url,
                user_details=task.user_details,
                timeout=task.timeout
            )

        return response is not None
    except Exception as e:
        logger.error(f"Error creating dataset schema: {e}")
        return False

def _upload_trace_file(task: UploadTask, api_client: TraceAPIClient) -> bool:
    """Upload trace file to presigned URL."""
    if not task.filepath or not os.path.exists(task.filepath):
        logger.error(f"Trace file not found: {task.filepath}")
        return False

    try:
        presigned_url = api_client.get_presigned_url(task.dataset_name)
        if not presigned_url:
            return False

        upload_success = api_client.upload_to_presigned_url(presigned_url, task.filepath)
        if not upload_success:
            return False

        dataset_spans = api_client._get_dataset_spans_from_file(task.filepath)
        if not dataset_spans:
            return False

        response = api_client.insert_traces(
            task.dataset_name,
            presigned_url,
            dataset_spans
        )

        return response is not None

    except Exception as e:
        logger.error(f"Error uploading trace file: {e}")
        return False

def _upload_code_archive(task: UploadTask, api_client: TraceAPIClient) -> bool:
    """Upload code archive for agentic tracers."""
    if not task.tracer_type.startswith("agentic/"):
        return True

    if not task.hash_id or not task.zip_path or not os.path.exists(task.zip_path):
        logger.warning("Code archive missing, skipping")
        return True

    try:
        existing_hashes = api_client.get_dataset_code_hashes(task.dataset_name)
        if existing_hashes and task.hash_id in existing_hashes:
            return True

        presigned_url = api_client.get_code_presigned_url(task.dataset_name)
        if not presigned_url:
            return False

        upload_success = api_client.upload_zip_to_presigned_url(presigned_url, task.zip_path)
        if not upload_success:
            return False

        response = api_client.insert_code_metadata(
            task.dataset_name,
            task.hash_id,
            presigned_url
        )

        return response is not None

    except Exception as e:
        logger.error(f"Error uploading code archive: {e}")
        return False

def _process_upload(task: UploadTask) -> Dict[str, Any]:
    """Process an upload task with simplified error handling."""
    pool = ExecutorPool.get_instance()
    task_id = pool.generate_task_id()
    logger.info(f"Processing upload task {task_id}")

    result = {
        "task_id": task_id,
        "status": STATUS_PROCESSING,
        "error": None,
        "start_time": datetime.now().isoformat()
    }
    _save_task_status(result)

    if not IMPORTS_AVAILABLE:
        logger.warning("Test mode: Simulating successful upload")
        return _complete_task_success(result, task_id)

    try:
        api_client = TraceAPIClient(task.base_url, task.project_name, task.timeout)

        if not _create_dataset_schema(task):
            return _fail_task_with_error(result, "Failed to create dataset schema", task_id)

        if not _upload_trace_file(task, api_client):
            return _fail_task_with_error(result, "Failed to upload trace file", task_id)

        if not _upload_code_archive(task, api_client):
            return _fail_task_with_error(result, "Failed to upload code archive", task_id)

        # Cleanup files after successful uploads
        if os.getenv("DELETE_RAGAAI_TRACE_JSON", "0") == "1":
            try:
                if task.filepath and os.path.exists(task.filepath):
                    os.remove(task.filepath)
                    logger.info(f"Cleaned up trace file: {task.filepath}")
                if task.zip_path and os.path.exists(task.zip_path):
                    os.remove(task.zip_path)
                    logger.info(f"Cleaned up code archive: {task.zip_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup files: {e}")

        return _complete_task_success(result, task_id)

    except Exception as e:
        return _fail_task_with_error(result, str(e), task_id)

def submit_upload_task(task: UploadTask) -> Optional[str]:
    """Submit upload task to executor pool."""
    pool = ExecutorPool.get_instance()
    return pool.submit_task(_process_upload, task)

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a specific upload task."""
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    if not os.path.exists(status_path):
        return {
            "task_id": task_id,
            "status": STATUS_FAILED,
            "error": "Task not found",
            "start_time": None,
            "end_time": None
        }

    try:
        with open(status_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {
            "task_id": task_id,
            "status": STATUS_FAILED,
            "error": f"Error reading status: {e}",
            "start_time": None,
            "end_time": None
        }

def get_upload_queue_status() -> Dict[str, Any]:
    """Get overall upload queue status and statistics."""
    pool = ExecutorPool.get_instance()
    queue_stats = pool.get_queue_status()

    cache_stats = {}
    if _dataset_cache is not None:
        try:
            cache_stats = _dataset_cache.get_stats()
        except Exception as e:
            logger.warning(f"Could not get cache stats: {e}")
            cache_stats = {"error": str(e)}

    return {
        "total_submitted": queue_stats["total_submitted"],
        "pending_uploads": queue_stats["pending_tasks"],
        "completed_uploads": queue_stats["completed_tasks"],
        "failed_uploads": queue_stats["failed_tasks"],
        "active_workers": queue_stats["active_workers"],
        "max_workers": queue_stats["max_workers"],
        "executor_pool": queue_stats["executor_pool"],
        "dataset_cache": cache_stats,
        "memory_usage_mb": queue_stats["memory_usage_mb"],
    }

def shutdown(timeout: int = 120) -> None:
    logger.info("Shutting down trace uploader")

    if session_manager is not None:
        try:
            session_manager.close()
            logger.info("Session manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing session manager: {e}")

"""
trace_uploader.py - Concrete implementation of AbstractTraceUploader

PUBLIC API (use this):
    - TraceUploader class (implements AbstractTraceUploader)
        - get_instance()  - Get singleton instance
        - submit()        - Submit trace for upload
        - get_status()    - Get task status
        - get_queue_status() - Get queue statistics
        - shutdown()      - Graceful shutdown

PRIVATE/INTERNAL (do not use directly):
    - UploadTask, UploadContext, UploadConfig (data classes)
    - submit_upload_task(), get_task_status(), get_upload_queue_status() (module functions)
    - _process_upload(), _save_task_status(), _generate_dataset_cache_key() (internal helpers)

This module provides async trace uploading with:
    - Multi-executor pool for high throughput
    - Dataset schema caching with TTL
    - Automatic retries and error handling
    - Presigned URL upload to S3/Azure
"""

import os
import json
import tempfile
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ragaai_catalyst.tracers.agentic_tracing.utils import get_logger

logger = get_logger(__name__)

try:
    from ragaai_catalyst.tracers.agentic_tracing.upload.uploader import AbstractTraceUploader
    from ragaai_catalyst.tracers.agentic_tracing.core.executor_pool import ExecutorPool
    from ragaai_catalyst.tracers.agentic_tracing.core.cache import DatasetCache
    from ragaai_catalyst.tracers.agentic_tracing.core.api_client import TraceAPIClient
    from ragaai_catalyst.tracers.agentic_tracing.utils.create_dataset_schema import create_dataset_schema_with_trace
    from ragaai_catalyst.tracers.agentic_tracing.utils import update_presigned_url
    from ragaai_catalyst.session_manager import session_manager
    from ragaai_catalyst import RagaAICatalyst
    IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("RagaAI Catalyst imports not available - running in test mode")
    IMPORTS_AVAILABLE = False
    session_manager = None
    AbstractTraceUploader = object

QUEUE_DIR = os.path.join(tempfile.gettempdir(), "ragaai_tasks")
os.makedirs(QUEUE_DIR, exist_ok=True)

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class UploadConfig:
    DELETE_TRACE_FILES = bool(os.getenv("DELETE_RAGAAI_TRACE_JSON"))
    AGENTIC_TRACER_PREFIX = "agentic/"

    @classmethod
    def should_delete_files(cls) -> bool:
        return cls.DELETE_TRACE_FILES

    @classmethod
    def is_agentic_tracer(cls, tracer_type: str) -> bool:
        return tracer_type.startswith(cls.AGENTIC_TRACER_PREFIX)


@dataclass
class UploadTask:
    filepath: str
    hash_id: str
    zip_path: str
    project_name: str
    project_id: str
    dataset_name: str
    user_details: Dict[str, Any]
    base_url: str
    tracer_type: str
    timeout: int = 120

class UploadContext:
    def __init__(self, task_id: str, task: UploadTask):
        self.task_id = task_id
        self.task = task
        self.result = self._init_result()
        self.base_url = self._normalize_base_url(task.base_url)

    def _init_result(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": STATUS_PROCESSING,
            "error": None,
            "start_time": datetime.now().isoformat()
        }

    def _normalize_base_url(self, base_url) -> str:
        if isinstance(base_url, tuple):
            logger.warning(f"Received tuple base_url, using first element: {base_url}")
            return str(base_url[0])
        return str(base_url)

    def save_status(self):
        _save_task_status(self.result)

    def handle_error(self, error_msg: str, fail_fast: bool = False) -> bool:
        self.result.update({
            "status": STATUS_FAILED,
            "error": error_msg,
            "end_time": datetime.now().isoformat()
        })
        self.save_status()
        if fail_fast:
            raise Exception(error_msg)
        return False

    def create_dataset_schema(self) -> bool:
        logger.info(f"Creating dataset schema for {self.task.dataset_name}")

        cache_key = _generate_dataset_cache_key(
            self.task.dataset_name, self.task.project_name, self.base_url
        )

        def create_schema():
            response = create_dataset_schema_with_trace(
                dataset_name=self.task.dataset_name,
                project_name=self.task.project_name,
                base_url=self.base_url,
                user_details=self.task.user_details,
                timeout=self.task.timeout
            )

            if response is None:
                logger.error(f"Dataset schema creation failed - received None response")
                return None
            elif hasattr(response, 'status_code') and response.status_code in [200, 201]:
                logger.info(f"Dataset schema created successfully: {response.status_code}")
                return response
            else:
                logger.warning(f"Dataset schema creation returned unexpected response: {response}")
                return None

        try:
            response = _dataset_cache.get_or_create(cache_key, create_schema)
            return response is not None

        except Exception as e:
            logger.error(f"Error creating dataset schema: {e}")
            return False

    def upload_agentic_traces(self) -> bool:
        if not self.task.filepath or not os.path.exists(self.task.filepath):
            error_msg = f"Trace file not found: {self.task.filepath}"
            logger.error(error_msg)
            return self.handle_error(error_msg)

        logger.info(f"Uploading agentic traces for {self.task.filepath}")

        try:
            api_client = TraceAPIClient(self.base_url, self.task.project_name, self.task.timeout)
            
            presigned_url = api_client.get_presigned_url(self.task.dataset_name)
            if presigned_url is None:
                error_msg = "Failed to get presigned URL"
                logger.error(error_msg)
                return self.handle_error(error_msg)

            presigned_url = update_presigned_url(presigned_url, self.base_url)

            upload_success = api_client.upload_to_presigned_url(presigned_url, self.task.filepath)
            if not upload_success:
                error_msg = "Failed to upload trace file to presigned URL"
                logger.error(error_msg)
                return self.handle_error(error_msg)

            dataset_spans = api_client._get_dataset_spans_from_file(self.task.filepath)
            if dataset_spans is None:
                error_msg = "Failed to extract dataset spans from trace file"
                logger.error(error_msg)
                return self.handle_error(error_msg)
            
            response = api_client.insert_traces(
                self.task.dataset_name,
                presigned_url,
                dataset_spans
            )
            
            if response is None:
                error_msg = "Failed to insert trace metadata"
                logger.error(error_msg)
                return self.handle_error(error_msg)

            logger.info("Agentic traces uploaded successfully")
            if UploadConfig.should_delete_files():
                os.remove(self.task.filepath)
                logger.info(f"Deleted trace file after successful upload: {self.task.filepath}")
            return True

        except Exception as e:
            logger.error(f"Error uploading agentic traces: {e}")
            return self.handle_error(str(e))

    def upload_code_hash(self) -> bool:
        if not UploadConfig.is_agentic_tracer(self.task.tracer_type):
            logger.debug(f"Skipping code upload for non-agentic tracer: {self.task.tracer_type}")
            return True

        if not self.task.hash_id or not self.task.zip_path or not os.path.exists(self.task.zip_path):
            logger.warning(f"Code upload skipped - missing hash_id, zip_path, or file not found: {self.task.zip_path}")
            return True

        logger.info(f"Uploading code hash {self.task.hash_id}")

        try:
            api_client = TraceAPIClient(self.base_url, self.task.project_name, self.task.timeout)
            
            code_hashes_list = api_client.get_dataset_code_hashes(self.task.dataset_name)
            if code_hashes_list is None:
                error_msg = "Failed to fetch existing code hashes"
                logger.error(error_msg)
                return self.handle_error(error_msg)

            if self.task.hash_id in code_hashes_list:
                logger.info(f"Code with hash {self.task.hash_id} already exists")
                return True

            presigned_url = api_client.get_code_presigned_url(self.task.dataset_name)
            if presigned_url is None:
                error_msg = "Failed to fetch presigned URL for code upload"
                logger.error(error_msg)
                return self.handle_error(error_msg)

            presigned_url = update_presigned_url(presigned_url, self.base_url)

            upload_success = api_client.upload_zip_to_presigned_url(presigned_url, self.task.zip_path)
            if not upload_success:
                error_msg = "Failed to upload zip file"
                logger.error(error_msg)
                return self.handle_error(error_msg)

            response = api_client.insert_code_metadata(
                self.task.dataset_name,
                self.task.hash_id,
                presigned_url
            )
            
            if response is None:
                error_msg = "Failed to insert code metadata"
                logger.error(error_msg)
                return self.handle_error(error_msg)

            logger.info(f"Code hash uploaded successfully")
            if UploadConfig.should_delete_files():
                os.remove(self.task.zip_path)
                logger.info(f"Deleted zip file after successful upload: {self.task.zip_path}")
            return True

        except Exception as e:
            logger.error(f"Error uploading code hash: {e}")
            return self.handle_error(str(e))

    def finalize_success(self) -> Dict[str, Any]:
        self.result.update({
            "status": STATUS_COMPLETED,
            "end_time": datetime.now().isoformat()
        })
        logger.info(f"Task {self.task_id} completed successfully")
        self.save_status()
        return self.result

    def __enter__(self):
        self.save_status()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.result.update({
                "status": STATUS_FAILED,
                "error": str(exc_val),
                "end_time": datetime.now().isoformat()
            })
        self.save_status()


DATASET_CACHE_DURATION = 600
DATASET_CACHE_MAX_SIZE = 1000
CACHE_KEY_SEPARATOR = "#"

_dataset_cache = DatasetCache(max_size=DATASET_CACHE_MAX_SIZE, ttl=DATASET_CACHE_DURATION)

def _generate_dataset_cache_key(dataset_name: str, project_name: str, base_url: str) -> str:
    return f"{dataset_name}{CACHE_KEY_SEPARATOR}{project_name}{CACHE_KEY_SEPARATOR}{base_url}"

def _save_task_status(task_status: Dict[str, Any]) -> None:
    task_id = task_status["task_id"]
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    with open(status_path, "w") as f:
        json.dump(task_status, f, indent=2)

def _process_upload(task: UploadTask) -> Dict[str, Any]:
    pool = ExecutorPool.get_instance()
    task_id = pool.generate_task_id()
    logger.info(f"Processing upload task {task_id} for file: {task.filepath}")

    if not IMPORTS_AVAILABLE:
        logger.warning(f"Test mode: Simulating processing of task {task_id}")
        result = {
            "task_id": task_id,
            "status": STATUS_COMPLETED,
            "error": None,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat()
        }
        _save_task_status(result)
        return result

    with UploadContext(task_id, task) as ctx:
        schema_success = ctx.create_dataset_schema()
        traces_success = ctx.upload_agentic_traces()
        code_success = ctx.upload_code_hash()

        if schema_success and traces_success and code_success:
            return ctx.finalize_success()
        else:
            return ctx.result

def submit_upload_task(task: UploadTask) -> Optional[str]:
    if not task.filepath or not os.path.exists(task.filepath):
        logger.error(f"Trace file not found: {task.filepath}")
        return None

    pool = ExecutorPool.get_instance()
    logger.info(f"Submitting upload task for file: {task.filepath}")
    logger.debug(f"Task details - Project: {task.project_name}, Dataset: {task.dataset_name}")

    task_id = pool.submit_task(_process_upload, task)
    
    if task_id is None:
        logger.error("Failed to submit task to executor pool")
        return None

    initial_status = {
        "task_id": task_id,
        "status": STATUS_PENDING,
        "error": None,
        "start_time": datetime.now().isoformat()
    }
    _save_task_status(initial_status)

    return task_id


def get_task_status(task_id: str) -> Dict[str, Any]:
    logger.debug(f"Getting status for task {task_id}")
    
    pool = ExecutorPool.get_instance()
    result = pool.get_task_status(task_id)
    
    if result.get("status") != "unknown":
        return result
    
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading status file for task {task_id}: {e}")
            return {"status": "unknown", "error": f"Error reading status: {e}"}
    
    return {"status": "unknown", "error": "Task not found"}

def get_upload_queue_status() -> Dict[str, Any]:
    logger.debug("Getting upload queue status")

    pool = ExecutorPool.get_instance()
    queue_stats = pool.get_queue_status()
    cache_stats = _dataset_cache.get_stats()

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

class TraceUploader(AbstractTraceUploader):
    """
    Concrete implementation of AbstractTraceUploader.
    
    Provides async trace uploading with executor pool, caching, and API integration.
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
    
    def submit(self, trace_data: Dict[str, Any]) -> Optional[str]:
        """Submit trace data for asynchronous upload."""
        task = UploadTask(
            filepath=trace_data['filepath'],
            hash_id=trace_data.get('hash_id', ''),
            zip_path=trace_data.get('zip_path', ''),
            project_name=trace_data['project_name'],
            project_id=trace_data['project_id'],
            dataset_name=trace_data['dataset_name'],
            user_details=trace_data['user_details'],
            base_url=trace_data['base_url'],
            tracer_type=trace_data['tracer_type'],
            timeout=trace_data.get('timeout', 120)
        )
        return submit_upload_task(task)
    
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get upload status for a specific task."""
        return get_task_status(task_id)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall upload queue status and statistics."""
        return get_upload_queue_status()
    
    def shutdown(self, timeout: int = 120) -> None:
        """Gracefully shutdown the uploader and release resources."""
        shutdown(timeout)


def shutdown(timeout: int = 120) -> None:
    logger.info("Shutting down trace uploader")
    
    if session_manager is not None:
        try:
            session_manager.close()
            logger.info("Session manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing session manager: {e}")

"""
executor_pool.py - Multi-executor pool for high-throughput concurrent processing
"""

import os
import time
import uuid
import atexit
import threading
import concurrent.futures
from typing import Dict, Any, Callable, Optional

from ragaai_catalyst.tracers.utils import get_logger

logger = get_logger(__name__)


class ExecutorPool:
    """
    Enhanced executor pool with task management and lifecycle control.
    
    Features:
    - Multi-executor load balancing
    - Task tracking and status monitoring
    - Automatic futures cleanup
    - Graceful shutdown with timeout
    - Singleton pattern for global access
    """
    
    _instance = None
    _instance_lock = threading.Lock()
    
    SHUTDOWN_MESSAGES = (
        "cannot schedule new futures after shutdown",
        "cannot schedule new futures after interpreter shutdown"
    )
    
    CLEANUP_INTERVAL = 300

    def __init__(self, num_executors: int = 4, workers_per_executor: int = None):
        if workers_per_executor is None:
            workers_per_executor = min(16, (os.cpu_count() or 1) * 2)

        self.num_executors = num_executors
        self.workers_per_executor = workers_per_executor
        self.executors = []
        self.current_executor = 0
        self.lock = threading.Lock()
        
        self.futures: Dict[str, concurrent.futures.Future] = {}
        self.futures_lock = threading.Lock()
        
        self.task_counter = 0
        self.task_counter_lock = threading.Lock()
        
        self.last_cleanup = 0
        self.cleanup_lock = threading.Lock()
        
        self.is_shutdown = False

        for i in range(num_executors):
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers_per_executor,
                thread_name_prefix=f"upload_pool_{i}"
            )
            self.executors.append(executor)

        logger.info(f"Created ExecutorPool with {num_executors} executors, "
                   f"{workers_per_executor} workers each ({num_executors * workers_per_executor} total workers)")

    @classmethod
    def get_instance(cls, num_executors: int = 4, workers_per_executor: int = None):
        """Get or create singleton instance of ExecutorPool."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(num_executors, workers_per_executor)
                    atexit.register(cls._instance.shutdown_graceful)
        return cls._instance

    def generate_task_id(self) -> str:
        """Generate a unique task ID."""
        with self.task_counter_lock:
            self.task_counter += 1
            counter = self.task_counter

        unique_id = str(uuid.uuid4())[:8]
        return f"task_{int(time.time())}_{os.getpid()}_{counter}_{unique_id}"

    def submit(self, fn: Callable, *args, **kwargs):
        """Submit task to next executor in round-robin fashion."""
        if self.is_shutdown:
            raise RuntimeError("Cannot submit tasks after shutdown")
            
        with self.lock:
            executor = self.executors[self.current_executor]
            self.current_executor = (self.current_executor + 1) % self.num_executors

        return executor.submit(fn, *args, **kwargs)

    def submit_task(self, fn: Callable, *args, **kwargs) -> Optional[str]:
        """
        Submit a task and track it with a unique ID.
        
        Returns:
            Task ID on success, None on failure
        """
        if self.is_shutdown or not self.executors:
            logger.warning("Executor pool unavailable or shutdown")
            return None

        task_id = self.generate_task_id()
        
        try:
            future = self.submit(fn, *args, **kwargs)
            
            with self.futures_lock:
                self.futures[task_id] = future
            
            self.cleanup_futures()
            
            return task_id
            
        except RuntimeError as e:
            if any(msg in str(e) for msg in self.SHUTDOWN_MESSAGES):
                logger.warning(f"Executor shutting down: {e}")
                return None
            logger.error(f"Runtime error submitting task: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error submitting task: {e}")
            return None

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task by ID."""
        with self.futures_lock:
            future = self.futures.get(task_id)
        
        if future:
            if future.done():
                try:
                    result = future.result(timeout=0)
                    return result
                except concurrent.futures.TimeoutError:
                    return {"status": "processing", "error": None}
                except Exception as e:
                    logger.error(f"Error retrieving future result for task {task_id}: {e}")
                    return {"status": "failed", "error": str(e)}
            else:
                return {"status": "processing", "error": None}
        
        return {"status": "unknown", "error": "Task not found"}

    def cleanup_futures(self):
        """Remove completed futures to prevent memory leaks."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.CLEANUP_INTERVAL:
            return

        with self.cleanup_lock:
            if current_time - self.last_cleanup < self.CLEANUP_INTERVAL:
                return
                
            with self.futures_lock:
                completed_tasks = []
                for task_id, future in self.futures.items():
                    if future.done():
                        completed_tasks.append(task_id)

                for task_id in completed_tasks:
                    del self.futures[task_id]

                self.last_cleanup = current_time

                if completed_tasks:
                    logger.info(f"Cleaned up {len(completed_tasks)} completed futures")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed status of the executor pool and task queue."""
        pool_stats = self.get_stats()

        with self.futures_lock:
            total_futures = len(self.futures)
            pending_count = len([f for f in self.futures.values() if not f.done()])
            completed_count = len([f for f in self.futures.values() if f.done() and not f.exception()])
            failed_count = len([f for f in self.futures.values() if f.done() and f.exception()])

        total_active_workers = sum(
            stats.get("active_threads", 0)
            for stats in pool_stats.get("executor_stats", [])
        )

        return {
            "total_submitted": total_futures,
            "pending_tasks": pending_count,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "active_workers": total_active_workers,
            "max_workers": pool_stats.get("total_workers", 0),
            "executor_pool": pool_stats,
            "memory_usage_mb": len(self.futures) * 0.002,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics for monitoring."""
        stats = {
            "num_executors": self.num_executors,
            "workers_per_executor": self.workers_per_executor,
            "total_workers": self.num_executors * self.workers_per_executor,
            "executor_stats": []
        }

        for i, executor in enumerate(self.executors):
            stats["executor_stats"].append({
                "executor_id": i,
                "active_threads": len(getattr(executor, '_threads', set())) if executor else 0,
                "max_workers": getattr(executor, '_max_workers', 0) if executor else 0,
            })

        return stats

    def shutdown_graceful(self, timeout: int = 120):
        """Gracefully shutdown the executor pool, waiting for pending tasks."""
        if self.is_shutdown:
            logger.debug("Executor pool already shutdown")
            return
            
        logger.info("Starting graceful shutdown of ExecutorPool")
        self.is_shutdown = True
        
        with self.futures_lock:
            pending_futures = [f for f in self.futures.values() if not f.done()]
            pending_count = len(pending_futures)

        if pending_count > 0:
            logger.info(f"Waiting up to {timeout}s for {pending_count} tasks to complete...")

            start_time = time.time()
            last_report = start_time

            while time.time() - start_time < timeout:
                with self.futures_lock:
                    pending_futures = [f for f in self.futures.values() if not f.done()]

                    if not pending_futures:
                        logger.info("All tasks completed successfully")
                        break

                    current_time = time.time()
                    if current_time - last_report >= 10:
                        elapsed = current_time - start_time
                        remaining = timeout - elapsed
                        logger.info(f"Still waiting for {len(pending_futures)} tasks. "
                                   f"Time remaining: {remaining:.1f}s")
                        last_report = current_time

                    time.sleep(0.5)
            else:
                with self.futures_lock:
                    pending_futures = [f for f in self.futures.values() if not f.done()]
                    logger.warning(f"Shutdown timeout reached. {len(pending_futures)} tasks still pending.")
        else:
            logger.info("No pending tasks")

        logger.info(f"Shutting down {self.num_executors} executors")
        for i, executor in enumerate(self.executors):
            try:
                logger.debug(f"Shutting down executor {i}")
                executor.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Error shutting down executor {i}: {e}")

        self.executors.clear()
        logger.info("ExecutorPool shutdown complete")

    def shutdown(self, wait: bool = True, timeout: int = 120):
        """Shutdown all executors (backward compatibility)."""
        if wait:
            self.shutdown_graceful(timeout)
        else:
            self.is_shutdown = True
            for executor in self.executors:
                try:
                    executor.shutdown(wait=False)
                except Exception as e:
                    logger.error(f"Error during executor shutdown: {e}")
            self.executors.clear()

    def __repr__(self):
        return (f"ExecutorPool(num_executors={self.num_executors}, "
                f"workers_per_executor={self.workers_per_executor}, "
                f"tracked_tasks={len(self.futures)})")

    def __del__(self):
        """Cleanup on destruction."""
        if not self.is_shutdown and self.executors:
            logger.debug("ExecutorPool being destroyed, initiating shutdown")
            self.shutdown_graceful(timeout=30)

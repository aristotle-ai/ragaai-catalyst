"""
cache.py - Production-grade caching infrastructure with LRU, TTL, and stampede protection

This module provides a thread-safe, high-performance caching solution suitable for:
- Dataset schema caching
- API response caching
- Expensive computation results
- Any resource that benefits from TTL-based caching with LRU eviction
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, Callable, TypeVar

from ragaai_catalyst.tracers.utils import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class DatasetCache:
    """
    Thread-safe LRU cache with TTL and stampede protection.
    
    Features:
    - LRU eviction when max_size is reached
    - TTL-based expiration for cache entries
    - Stampede protection (prevents duplicate concurrent requests)
    - Background cleanup thread for expired entries
    - Comprehensive statistics tracking
    
    Usage:
        cache = DatasetCache(max_size=1000, ttl=600)
        result = cache.get_or_create("key", lambda: expensive_operation())
    """

    def __init__(self, max_size: int = 1000, ttl: int = 600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
        self.pending_requests = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'stampede_prevented': 0
        }

        self._start_cleanup_thread()
        logger.info(f"Initialized DatasetCache with max_size={max_size}, ttl={ttl}s")

    def get_or_create(self, cache_key: str, create_fn: Callable[[], T]) -> Optional[T]:
        """Get from cache or create with stampede protection."""
        cached_value = self._get_from_cache(cache_key)
        if cached_value is not None:
            with self.lock:
                self.stats['hits'] += 1
            logger.debug(f"Dataset cache hit for key: {cache_key}")
            return cached_value

        with self.lock:
            self.stats['misses'] += 1

            cached_value = self._get_from_cache(cache_key)
            if cached_value is not None:
                return cached_value

            if cache_key in self.pending_requests:
                event = self.pending_requests[cache_key]
                self.stats['stampede_prevented'] += 1
                logger.debug(f"Waiting for pending dataset creation: {cache_key}")

                self.lock.release()
                try:
                    event.wait(timeout=30)
                finally:
                    self.lock.acquire()

                cached_value = self._get_from_cache(cache_key)
                if cached_value is not None:
                    return cached_value

            event = threading.Event()
            self.pending_requests[cache_key] = event

        try:
            logger.debug(f"Creating dataset schema (cache miss): {cache_key}")
            result = create_fn()
            self._set_in_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error creating dataset schema: {e}")
            return None
        finally:
            with self.lock:
                if cache_key in self.pending_requests:
                    del self.pending_requests[cache_key]
                event.set()

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get from cache with TTL check."""
        with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            if time.time() - entry['timestamp'] > self.ttl:
                del self.cache[key]
                return None

            self.cache.move_to_end(key)
            return entry['value']

    def _set_in_cache(self, key: str, value: Any):
        """Set in cache with LRU eviction."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }

            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
                logger.debug(f"Evicted oldest cache entry: {oldest_key}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread for expired entries."""
        def cleanup_loop():
            while True:
                time.sleep(60)
                self._cleanup_expired()

        thread = threading.Thread(target=cleanup_loop, daemon=True, name="dataset_cache_cleanup")
        thread.start()
        logger.debug("Started dataset cache cleanup thread")

    def _cleanup_expired(self):
        """Remove expired entries based on TTL."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry['timestamp'] > self.ttl
            ]

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': f"{hit_rate:.2%}",
                'evictions': self.stats['evictions'],
                'stampede_prevented': self.stats['stampede_prevented'],
                'pending_requests': len(self.pending_requests),
                'ttl_seconds': self.ttl
            }

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            size_before = len(self.cache)
            self.cache.clear()
            logger.info(f"Dataset cache cleared ({size_before} entries removed)")

    def invalidate(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry."""
        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                logger.debug(f"Invalidated cache entry: {cache_key}")
                return True
            return False

    def __repr__(self):
        return f"DatasetCache(size={len(self.cache)}/{self.max_size}, ttl={self.ttl}s)"


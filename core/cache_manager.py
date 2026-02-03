"""
Cache Manager Module
====================

Memory-bounded caching (Issue #S13).
Internal API rate limiting (Issue #S14).

Features:
- LRU cache with memory limits
- TTL-based expiration
- Memory pressure monitoring
- Rate limiting for internal services
"""

from __future__ import annotations

import logging
import time
import threading
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, TypeVar, Generic
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheStats:
    """Statistics for a cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'current_size': self.current_size,
            'max_size': self.max_size,
            'current_memory_bytes': self.current_memory_bytes,
            'max_memory_bytes': self.max_memory_bytes,
            'memory_utilization_pct': self.current_memory_bytes / self.max_memory_bytes * 100 if self.max_memory_bytes > 0 else 0,
        }


@dataclass
class CacheEntry:
    """A single cache entry.

    All timestamps are float values from time.time() (Unix epoch seconds)
    for consistency across the caching system.
    """
    key: str
    value: Any
    created_at: float  # Unix timestamp from time.time()
    expires_at: float | None  # Unix timestamp from time.time(), None for no expiration
    size_bytes: int
    access_count: int = 0
    last_accessed: float = 0.0  # Unix timestamp from time.time()


class BoundedLRUCache(Generic[T]):
    """
    Memory-bounded LRU cache (#S13).

    Features:
    - Maximum item count
    - Maximum memory usage
    - TTL-based expiration
    - LRU eviction
    """

    def __init__(
        self,
        name: str,
        max_items: int = 10000,
        max_memory_bytes: int = 100 * 1024 * 1024,  # 100 MB
        default_ttl_seconds: float | None = 3600.0,  # 1 hour
        on_evict: Callable[[str, T], None] | None = None,
    ):
        self.name = name
        self.max_items = max_items
        self.max_memory_bytes = max_memory_bytes
        self.default_ttl = default_ttl_seconds
        self._on_evict = on_evict

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_items, max_memory_bytes=max_memory_bytes)
        self._current_memory = 0

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            # Check expiration
            now = time.time()
            if entry.expires_at is not None and now > entry.expires_at:
                self._remove(key)
                self._stats.misses += 1
                return default

            # Update access info
            entry.access_count += 1
            entry.last_accessed = now

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            self._stats.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: T,
        ttl_seconds: float | None = None,
    ) -> None:
        """Set value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)

            # Check if we need to evict
            self._ensure_capacity(size_bytes)

            now = time.time()
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            expires_at = now + ttl if ttl is not None else None

            # Remove old entry if exists
            if key in self._cache:
                self._remove(key)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at,
                size_bytes=size_bytes,
                last_accessed=now,
            )

            self._cache[key] = entry
            self._current_memory += size_bytes
            self._stats.current_size = len(self._cache)
            self._stats.current_memory_bytes = self._current_memory

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self._stats.current_size = 0
            self._stats.current_memory_bytes = 0
            logger.info(f"Cache '{self.name}' cleared")

    def _remove(self, key: str) -> None:
        """Remove entry (internal, lock must be held)."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_memory -= entry.size_bytes
            if self._on_evict:
                try:
                    self._on_evict(key, entry.value)
                except Exception as e:
                    logger.error(f"Error in eviction callback: {e}")

    def _ensure_capacity(self, needed_bytes: int) -> None:
        """Ensure we have capacity for new entry (lock must be held)."""
        # Evict expired first
        self._evict_expired()

        # Evict by count
        while len(self._cache) >= self.max_items:
            self._evict_lru()

        # Evict by memory
        while self._current_memory + needed_bytes > self.max_memory_bytes and self._cache:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entry (lock must be held)."""
        if self._cache:
            # Pop first item (least recently used)
            key = next(iter(self._cache))
            self._remove(key)
            self._stats.evictions += 1

    def _evict_expired(self) -> None:
        """Evict all expired entries (lock must be held)."""
        now = time.time()
        expired = [
            key for key, entry in self._cache.items()
            if entry.expires_at is not None and now > entry.expires_at
        ]
        for key in expired:
            self._remove(key)
            self._stats.evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return sys.getsizeof(value)
        except TypeError:
            # Fallback for complex objects
            return 1024  # Assume 1KB

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.current_size = len(self._cache)
            self._stats.current_memory_bytes = self._current_memory
            return CacheStats(**self._stats.__dict__)

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def keys(self) -> list[str]:
        """Get all non-expired keys."""
        with self._lock:
            self._evict_expired()
            return list(self._cache.keys())


class CacheManager:
    """
    Manages multiple named caches (#S13).

    Provides centralized cache management with memory monitoring.
    """

    def __init__(
        self,
        total_max_memory_bytes: int = 500 * 1024 * 1024,  # 500 MB total
        memory_pressure_threshold: float = 0.8,  # 80% triggers cleanup
    ):
        self.total_max_memory = total_max_memory_bytes
        self.memory_pressure_threshold = memory_pressure_threshold

        self._caches: dict[str, BoundedLRUCache] = {}
        self._lock = threading.Lock()

    def create_cache(
        self,
        name: str,
        max_items: int = 10000,
        max_memory_bytes: int = 50 * 1024 * 1024,  # 50 MB
        default_ttl_seconds: float | None = 3600.0,
    ) -> BoundedLRUCache:
        """Create or get a named cache."""
        with self._lock:
            if name not in self._caches:
                # Ensure we don't exceed total memory
                current_allocated = sum(
                    c._stats.max_memory_bytes for c in self._caches.values()
                )
                if current_allocated + max_memory_bytes > self.total_max_memory:
                    # Reduce allocation
                    available = self.total_max_memory - current_allocated
                    max_memory_bytes = min(max_memory_bytes, max(available, 10 * 1024 * 1024))
                    logger.warning(
                        f"Reduced cache '{name}' memory to {max_memory_bytes / 1024 / 1024:.1f}MB "
                        f"due to total limit"
                    )

                self._caches[name] = BoundedLRUCache(
                    name=name,
                    max_items=max_items,
                    max_memory_bytes=max_memory_bytes,
                    default_ttl_seconds=default_ttl_seconds,
                )
                logger.info(f"Created cache '{name}' (max {max_items} items, {max_memory_bytes / 1024 / 1024:.1f}MB)")

            return self._caches[name]

    def get_cache(self, name: str) -> BoundedLRUCache | None:
        """Get cache by name."""
        return self._caches.get(name)

    def delete_cache(self, name: str) -> bool:
        """Delete a cache."""
        with self._lock:
            if name in self._caches:
                self._caches[name].clear()
                del self._caches[name]
                logger.info(f"Deleted cache '{name}'")
                return True
            return False

    def get_memory_usage(self) -> dict:
        """Get memory usage across all caches."""
        total_used = 0
        by_cache = {}

        for name, cache in self._caches.items():
            stats = cache.get_stats()
            used = stats.current_memory_bytes
            total_used += used
            by_cache[name] = {
                'used_bytes': used,
                'max_bytes': stats.max_memory_bytes,
                'utilization_pct': used / stats.max_memory_bytes * 100 if stats.max_memory_bytes > 0 else 0,
            }

        return {
            'total_used_bytes': total_used,
            'total_max_bytes': self.total_max_memory,
            'utilization_pct': total_used / self.total_max_memory * 100,
            'pressure_threshold_pct': self.memory_pressure_threshold * 100,
            'under_pressure': total_used / self.total_max_memory > self.memory_pressure_threshold,
            'by_cache': by_cache,
        }

    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high and trigger cleanup if needed."""
        usage = self.get_memory_usage()

        if usage['under_pressure']:
            logger.warning(f"Memory pressure detected: {usage['utilization_pct']:.1f}%")
            self._cleanup_under_pressure()
            return True

        return False

    def _cleanup_under_pressure(self) -> None:
        """Cleanup caches when under memory pressure."""
        # Evict expired entries from all caches
        for cache in self._caches.values():
            cache._evict_expired()

        # If still under pressure, evict LRU from largest caches
        usage = self.get_memory_usage()
        if usage['under_pressure']:
            # Sort caches by utilization
            sorted_caches = sorted(
                self._caches.items(),
                key=lambda x: x[1].get_stats().current_memory_bytes,
                reverse=True,
            )

            # Evict from top caches
            for name, cache in sorted_caches[:3]:
                for _ in range(min(100, len(cache._cache) // 10)):
                    cache._evict_lru()

                logger.info(f"Forced eviction from cache '{name}' due to memory pressure")

    def get_all_stats(self) -> dict:
        """Get statistics for all caches."""
        return {
            'memory': self.get_memory_usage(),
            'caches': {
                name: cache.get_stats().to_dict()
                for name, cache in self._caches.items()
            },
        }

    def warm_cache(
        self,
        cache_name: str,
        warm_entries: list[tuple[str, Any, float | None]],
    ) -> dict[str, int]:
        """
        P2: Warm a cache with precomputed entries on startup.

        Args:
            cache_name: Name of the cache to warm
            warm_entries: List of (key, value, ttl_seconds) tuples.
                         If ttl_seconds is None, uses the cache's default TTL.

        Returns:
            Dictionary with warming statistics:
                - 'success': Number of entries successfully added
                - 'failed': Number of entries that failed
                - 'cache_name': Name of the cache
        """
        cache = self.get_cache(cache_name)
        if cache is None:
            logger.warning(f"Cache '{cache_name}' not found for warming")
            return {'success': 0, 'failed': len(warm_entries), 'cache_name': cache_name}

        success = 0
        failed = 0

        for entry in warm_entries:
            try:
                if len(entry) == 2:
                    key, value = entry
                    ttl = None
                else:
                    key, value, ttl = entry

                cache.set(key, value, ttl_seconds=ttl)
                success += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache entry '{entry[0]}': {e}")
                failed += 1

        logger.info(
            f"Cache '{cache_name}' warmed: {success} entries added, {failed} failed"
        )
        return {'success': success, 'failed': failed, 'cache_name': cache_name}

    def warm_cache_from_callable(
        self,
        cache_name: str,
        warm_functions: list[tuple[str, Callable[[], Any], float | None]],
    ) -> dict[str, int]:
        """
        P2: Warm a cache by calling functions to compute values.

        Args:
            cache_name: Name of the cache to warm
            warm_functions: List of (key, callable, ttl_seconds) tuples.
                           The callable is invoked to generate the value.
                           If ttl_seconds is None, uses the cache's default TTL.

        Returns:
            Dictionary with warming statistics
        """
        cache = self.get_cache(cache_name)
        if cache is None:
            logger.warning(f"Cache '{cache_name}' not found for warming")
            return {'success': 0, 'failed': len(warm_functions), 'cache_name': cache_name}

        success = 0
        failed = 0

        for entry in warm_functions:
            try:
                if len(entry) == 2:
                    key, compute_fn = entry
                    ttl = None
                else:
                    key, compute_fn, ttl = entry

                value = compute_fn()
                cache.set(key, value, ttl_seconds=ttl)
                success += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache entry '{entry[0]}' via callable: {e}")
                failed += 1

        logger.info(
            f"Cache '{cache_name}' warmed from callables: {success} entries computed, {failed} failed"
        )
        return {'success': success, 'failed': failed, 'cache_name': cache_name}

    def warm_all_caches(
        self,
        warm_specs: dict[str, list[tuple[str, Any, float | None]]],
    ) -> dict[str, dict[str, int]]:
        """
        P2: Warm multiple caches at startup.

        Args:
            warm_specs: Dictionary mapping cache names to lists of
                       (key, value, ttl_seconds) tuples.

        Returns:
            Dictionary mapping cache names to their warming statistics
        """
        results = {}
        total_success = 0
        total_failed = 0

        for cache_name, entries in warm_specs.items():
            result = self.warm_cache(cache_name, entries)
            results[cache_name] = result
            total_success += result['success']
            total_failed += result['failed']

        logger.info(
            f"Cache warming complete: {total_success} total entries, {total_failed} failures"
        )
        results['_summary'] = {'total_success': total_success, 'total_failed': total_failed}
        return results


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float
    burst_size: int = 10
    retry_after_seconds: float = 1.0


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    wait_seconds: float = 0.0
    tokens_remaining: float = 0.0
    limit_config: RateLimitConfig | None = None


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for internal APIs (#S14).

    Features:
    - Smooth rate limiting
    - Burst allowance
    - Multiple rate limit tiers
    """

    def __init__(
        self,
        name: str,
        requests_per_second: float,
        burst_size: int = 10,
    ):
        self.name = name
        self.rps = requests_per_second
        self.burst_size = burst_size

        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = threading.Lock()

        # Stats
        self._total_requests = 0
        self._total_allowed = 0
        self._total_rejected = 0

    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """
        Try to acquire tokens.

        Returns RateLimitResult with allowed status and wait time if rejected.
        """
        with self._lock:
            now = time.time()
            self._total_requests += 1

            # Refill tokens
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.rps
            )
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_allowed += 1
                return RateLimitResult(
                    allowed=True,
                    tokens_remaining=self._tokens,
                    limit_config=RateLimitConfig(
                        requests_per_second=self.rps,
                        burst_size=self.burst_size,
                    ),
                )

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_seconds = tokens_needed / self.rps

            self._total_rejected += 1
            return RateLimitResult(
                allowed=False,
                wait_seconds=wait_seconds,
                tokens_remaining=self._tokens,
                limit_config=RateLimitConfig(
                    requests_per_second=self.rps,
                    burst_size=self.burst_size,
                    retry_after_seconds=wait_seconds,
                ),
            )

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                'name': self.name,
                'requests_per_second': self.rps,
                'burst_size': self.burst_size,
                'current_tokens': self._tokens,
                'total_requests': self._total_requests,
                'total_allowed': self._total_allowed,
                'total_rejected': self._total_rejected,
                'rejection_rate': self._total_rejected / self._total_requests if self._total_requests > 0 else 0,
            }


class RateLimiterManager:
    """
    Manages rate limiters for internal APIs (#S14).

    Provides centralized rate limit management.
    """

    def __init__(self):
        self._limiters: dict[str, TokenBucketRateLimiter] = {}
        self._lock = threading.Lock()

    def create_limiter(
        self,
        name: str,
        requests_per_second: float,
        burst_size: int = 10,
    ) -> TokenBucketRateLimiter:
        """Create or get a rate limiter."""
        with self._lock:
            if name not in self._limiters:
                self._limiters[name] = TokenBucketRateLimiter(
                    name=name,
                    requests_per_second=requests_per_second,
                    burst_size=burst_size,
                )
                logger.info(f"Created rate limiter '{name}' ({requests_per_second} rps, burst {burst_size})")

            return self._limiters[name]

    def get_limiter(self, name: str) -> TokenBucketRateLimiter | None:
        """Get limiter by name."""
        return self._limiters.get(name)

    def check_rate_limit(self, name: str, tokens: int = 1) -> RateLimitResult:
        """Check rate limit for a named limiter."""
        limiter = self._limiters.get(name)
        if limiter is None:
            # No limiter = no limit
            return RateLimitResult(allowed=True)

        return limiter.acquire(tokens)

    def get_all_stats(self) -> dict:
        """Get statistics for all rate limiters."""
        return {
            name: limiter.get_stats()
            for name, limiter in self._limiters.items()
        }


def rate_limited(
    limiter_name: str,
    manager: RateLimiterManager | None = None,
    on_reject: Callable[[RateLimitResult], Any] | None = None,
):
    """
    Decorator for rate-limited functions.

    Args:
        limiter_name: Name of the rate limiter to use
        manager: RateLimiterManager instance (required)
        on_reject: Callback when rate limited (default raises exception)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if manager is None:
                return func(*args, **kwargs)

            result = manager.check_rate_limit(limiter_name)

            if result.allowed:
                return func(*args, **kwargs)

            if on_reject:
                return on_reject(result)

            raise RateLimitExceeded(
                f"Rate limit exceeded for '{limiter_name}'. "
                f"Retry after {result.wait_seconds:.2f} seconds."
            )

        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


# Global instances for convenience
_cache_manager: CacheManager | None = None
_rate_limiter_manager: RateLimiterManager | None = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_rate_limiter_manager() -> RateLimiterManager:
    """Get global rate limiter manager instance."""
    global _rate_limiter_manager
    if _rate_limiter_manager is None:
        _rate_limiter_manager = RateLimiterManager()
    return _rate_limiter_manager

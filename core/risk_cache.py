"""
Risk Metric Caching Module
==========================

Addresses issue #R29: No risk metric caching optimization.

Features:
- Efficient caching of expensive risk calculations
- TTL-based cache invalidation
- Dependency-aware cache management
- Cache warming on startup
- Memory-bounded cache with LRU eviction
"""

from __future__ import annotations

import hashlib
import logging
import time
import threading
import functools
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar, Generic
from collections import OrderedDict
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheInvalidationReason(str, Enum):
    """Reasons for cache invalidation."""
    TTL_EXPIRED = "ttl_expired"
    DEPENDENCY_CHANGED = "dependency_changed"
    MANUAL = "manual"
    MEMORY_PRESSURE = "memory_pressure"
    POSITION_CHANGED = "position_changed"
    MARKET_DATA_UPDATED = "market_data_updated"


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    key: str
    value: T
    created_at: float  # time.time()
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    dependencies: set[str] = field(default_factory=set)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Age of entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_entries: int = 0
    memory_bytes: int = 0
    avg_entry_age_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": self.hit_rate * 100,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "total_entries": self.total_entries,
            "memory_bytes": self.memory_bytes,
            "avg_entry_age_seconds": self.avg_entry_age_seconds,
        }


class RiskMetricCache:
    """
    Specialized cache for risk metrics (#R29).

    Optimized for:
    - VaR calculations
    - Correlation matrices
    - Greeks calculations
    - Stress test results
    """

    # Default TTLs for different metric types (seconds)
    DEFAULT_TTLS = {
        "var": 300,  # 5 minutes
        "cvar": 300,
        "correlation_matrix": 600,  # 10 minutes
        "greeks": 60,  # 1 minute
        "stress_test": 900,  # 15 minutes
        "risk_contribution": 300,
        "position_risk": 60,
        "portfolio_risk": 120,
    }

    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_bytes: int = 100 * 1024 * 1024,  # 100MB
        default_ttl_seconds: float = 300,
    ):
        """
        Initialize risk metric cache.

        Args:
            max_entries: Maximum number of cache entries
            max_memory_bytes: Maximum memory usage
            default_ttl_seconds: Default TTL for entries
        """
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.default_ttl = default_ttl_seconds

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._dependencies: dict[str, set[str]] = {}  # dependency -> dependent keys

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                self._remove_entry(key)
                self._stats.misses += 1
                return None

            # Update access stats and move to end (LRU)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            self._stats.hits += 1

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        metric_type: str | None = None,
        dependencies: set[str] | None = None,
    ) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live (uses type default or global default)
            metric_type: Type of metric for TTL lookup
            dependencies: Keys this entry depends on
        """
        # Determine TTL
        if ttl_seconds is None:
            if metric_type and metric_type in self.DEFAULT_TTLS:
                ttl_seconds = self.DEFAULT_TTLS[metric_type]
            else:
                ttl_seconds = self.default_ttl

        # Estimate size
        size_bytes = self._estimate_size(value)

        with self._lock:
            # Make room if needed
            self._ensure_capacity(size_bytes)

            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + ttl_seconds,
                last_accessed=now,
                dependencies=dependencies or set(),
                size_bytes=size_bytes,
            )

            # Remove old entry if exists
            if key in self._cache:
                self._remove_entry(key)

            self._cache[key] = entry
            self._stats.total_entries = len(self._cache)

            # Register dependencies
            for dep in entry.dependencies:
                if dep not in self._dependencies:
                    self._dependencies[dep] = set()
                self._dependencies[dep].add(key)

    def invalidate(
        self,
        key: str,
        reason: CacheInvalidationReason = CacheInvalidationReason.MANUAL,
    ) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key
            reason: Reason for invalidation

        Returns:
            True if entry was removed
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats.invalidations += 1
                logger.debug(f"Cache invalidated: {key} ({reason.value})")
                return True
            return False

    def invalidate_dependency(
        self,
        dependency_key: str,
    ) -> int:
        """
        Invalidate all entries depending on a key.

        Args:
            dependency_key: The dependency that changed

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            dependents = self._dependencies.get(dependency_key, set())
            count = 0

            for key in list(dependents):
                if self.invalidate(key, CacheInvalidationReason.DEPENDENCY_CHANGED):
                    count += 1

            logger.debug(
                f"Invalidated {count} entries dependent on {dependency_key}"
            )
            return count

    def invalidate_by_type(self, metric_type: str) -> int:
        """
        Invalidate all entries of a specific metric type.

        Args:
            metric_type: Type prefix to match

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if key.startswith(f"{metric_type}:")
            ]

            for key in keys_to_remove:
                self._remove_entry(key)
                self._stats.invalidations += 1

            return len(keys_to_remove)

    def invalidate_all(self) -> int:
        """Invalidate all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._dependencies.clear()
            self._stats.invalidations += count
            self._stats.total_entries = 0
            return count

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl_seconds: float | None = None,
        metric_type: str | None = None,
        dependencies: set[str] | None = None,
    ) -> T:
        """
        Get from cache or compute and cache.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl_seconds: Time-to-live
            metric_type: Type of metric
            dependencies: Keys this entry depends on

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        value = compute_fn()

        # Cache result
        self.set(
            key,
            value,
            ttl_seconds=ttl_seconds,
            metric_type=metric_type,
            dependencies=dependencies,
        )

        return value

    def _remove_entry(self, key: str) -> None:
        """Remove entry and clean up dependencies."""
        if key in self._cache:
            entry = self._cache[key]

            # Remove from dependency tracking
            for dep in entry.dependencies:
                if dep in self._dependencies:
                    self._dependencies[dep].discard(key)
                    if not self._dependencies[dep]:
                        del self._dependencies[dep]

            del self._cache[key]
            self._stats.memory_bytes -= entry.size_bytes

    def _ensure_capacity(self, needed_bytes: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check entry limit
        while len(self._cache) >= self.max_entries:
            self._evict_oldest()

        # Check memory limit
        current_memory = sum(e.size_bytes for e in self._cache.values())
        while current_memory + needed_bytes > self.max_memory_bytes:
            if not self._cache:
                break
            evicted_entry = self._evict_oldest()
            if evicted_entry:
                current_memory -= evicted_entry.size_bytes

    def _evict_oldest(self) -> CacheEntry | None:
        """Evict oldest (LRU) entry."""
        if not self._cache:
            return None

        key = next(iter(self._cache))
        entry = self._cache[key]
        self._remove_entry(key)
        self._stats.evictions += 1
        return entry

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            # Try JSON serialization for estimation
            json_str = json.dumps(value, default=str)
            return len(json_str.encode('utf-8'))
        except Exception:
            # Rough estimate
            return 1024

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.total_entries = len(self._cache)
            self._stats.memory_bytes = sum(e.size_bytes for e in self._cache.values())

            if self._cache:
                ages = [e.age_seconds for e in self._cache.values()]
                self._stats.avg_entry_age_seconds = sum(ages) / len(ages)

        return self._stats

    def warm_cache(
        self,
        warm_functions: list[tuple[str, Callable[[], Any], str | None]],
    ) -> int:
        """
        Warm cache with precomputed values.

        Args:
            warm_functions: List of (key, compute_fn, metric_type) tuples

        Returns:
            Number of entries warmed
        """
        count = 0
        for key, compute_fn, metric_type in warm_functions:
            try:
                value = compute_fn()
                self.set(key, value, metric_type=metric_type)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for {key}: {e}")

        logger.info(f"Cache warmed with {count} entries")
        return count


def cached_risk_metric(
    cache: RiskMetricCache,
    metric_type: str,
    ttl_seconds: float | None = None,
    key_fn: Callable[..., str] | None = None,
    dependencies_fn: Callable[..., set[str]] | None = None,
):
    """
    Decorator for caching risk metric calculations.

    Args:
        cache: RiskMetricCache instance
        metric_type: Type of metric for TTL lookup
        ttl_seconds: Optional explicit TTL
        key_fn: Function to generate cache key from args
        dependencies_fn: Function to get dependencies from args

    Example:
        @cached_risk_metric(cache, "var")
        def calculate_var(portfolio_id, confidence):
            # expensive calculation
            return var_result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = _generate_cache_key(func.__name__, args, kwargs)

            key = f"{metric_type}:{key}"

            # Get dependencies
            dependencies = None
            if dependencies_fn:
                dependencies = dependencies_fn(*args, **kwargs)

            # Try cache
            cached = cache.get(key)
            if cached is not None:
                return cached

            # Compute
            result = func(*args, **kwargs)

            # Cache
            cache.set(
                key,
                result,
                ttl_seconds=ttl_seconds,
                metric_type=metric_type,
                dependencies=dependencies,
            )

            return result

        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments."""
    key_parts = [func_name]

    for arg in args:
        key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    key_str = ":".join(key_parts)

    # Hash if too long
    if len(key_str) > 200:
        return hashlib.md5(key_str.encode()).hexdigest()

    return key_str


class PortfolioRiskCache:
    """
    High-level cache manager for portfolio risk calculations.

    Manages dependencies between different risk calculations.
    """

    def __init__(self):
        self.cache = RiskMetricCache()

        # Track what invalidates what
        self._invalidation_triggers = {
            "positions": ["var", "cvar", "greeks", "risk_contribution", "portfolio_risk"],
            "market_data": ["var", "cvar", "greeks", "correlation_matrix"],
            "volatility": ["var", "cvar", "stress_test"],
            "correlation": ["var", "cvar", "risk_contribution"],
        }

    def on_positions_changed(self, portfolio_id: str) -> None:
        """Handle position change event."""
        for metric_type in self._invalidation_triggers["positions"]:
            self.cache.invalidate_by_type(f"{metric_type}:{portfolio_id}")

    def on_market_data_updated(self, symbols: list[str]) -> None:
        """Handle market data update event."""
        for symbol in symbols:
            self.cache.invalidate_dependency(f"price:{symbol}")

    def on_volatility_updated(self, portfolio_id: str) -> None:
        """Handle volatility update event."""
        for metric_type in self._invalidation_triggers["volatility"]:
            self.cache.invalidate_by_type(f"{metric_type}:{portfolio_id}")

    def get_var(
        self,
        portfolio_id: str,
        confidence: float,
        compute_fn: Callable[[], float],
    ) -> float:
        """Get cached or computed VaR."""
        key = f"var:{portfolio_id}:{confidence}"
        return self.cache.get_or_compute(
            key,
            compute_fn,
            metric_type="var",
            dependencies={f"positions:{portfolio_id}"},
        )

    def get_correlation_matrix(
        self,
        portfolio_id: str,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """Get cached or computed correlation matrix."""
        key = f"correlation_matrix:{portfolio_id}"
        return self.cache.get_or_compute(
            key,
            compute_fn,
            metric_type="correlation_matrix",
        )

    def get_greeks(
        self,
        position_id: str,
        compute_fn: Callable[[], dict],
    ) -> dict:
        """Get cached or computed Greeks."""
        key = f"greeks:{position_id}"
        return self.cache.get_or_compute(
            key,
            compute_fn,
            metric_type="greeks",
        )

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats().to_dict()

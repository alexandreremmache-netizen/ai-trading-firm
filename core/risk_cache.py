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
        except Exception as e:
            # Rough estimate - log for debugging
            logger.debug(f"Failed to estimate cache entry size: {e}")
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


# =============================================================================
# CACHE INVALIDATION STRATEGIES (P3 Enhancement)
# =============================================================================

class CacheInvalidationStrategy(str, Enum):
    """Cache invalidation strategy types."""
    TTL_BASED = "ttl_based"  # Simple time-based expiration
    EVENT_DRIVEN = "event_driven"  # Invalidate on specific events
    WRITE_THROUGH = "write_through"  # Invalidate on any write
    ADAPTIVE = "adaptive"  # Adjust TTL based on access patterns
    HIERARCHICAL = "hierarchical"  # Parent invalidates children


@dataclass
class InvalidationRule:
    """Rule for cache invalidation."""
    trigger_event: str
    affected_patterns: list[str]  # Key patterns to invalidate
    priority: int = 0  # Higher = process first
    cascade: bool = False  # Whether to cascade to dependencies

    def matches(self, key: str) -> bool:
        """Check if key matches any affected pattern."""
        import fnmatch
        return any(fnmatch.fnmatch(key, pattern) for pattern in self.affected_patterns)


class CacheInvalidationManager:
    """
    Advanced cache invalidation strategies (P3 Enhancement).

    Provides multiple invalidation strategies for different use cases.
    """

    def __init__(self, cache: RiskMetricCache):
        """
        Initialize invalidation manager.

        Args:
            cache: RiskMetricCache instance to manage
        """
        self.cache = cache
        self._strategy = CacheInvalidationStrategy.TTL_BASED
        self._rules: list[InvalidationRule] = []
        self._access_patterns: dict[str, list[float]] = defaultdict(list)  # key -> access times
        self._invalidation_history: deque = deque(maxlen=1000)

    def set_strategy(self, strategy: CacheInvalidationStrategy) -> None:
        """Set the invalidation strategy."""
        self._strategy = strategy
        logger.info(f"Cache invalidation strategy set to: {strategy.value}")

    def add_rule(self, rule: InvalidationRule) -> None:
        """Add an invalidation rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)

    def on_event(self, event_type: str, event_data: dict | None = None) -> int:
        """
        Process an event and invalidate matching cache entries.

        Args:
            event_type: Type of event (e.g., "position_changed", "market_data_updated")
            event_data: Additional event context

        Returns:
            Number of entries invalidated
        """
        invalidated = 0

        if self._strategy == CacheInvalidationStrategy.EVENT_DRIVEN:
            for rule in self._rules:
                if rule.trigger_event == event_type:
                    invalidated += self._apply_rule(rule, event_data)

        elif self._strategy == CacheInvalidationStrategy.WRITE_THROUGH:
            # Invalidate everything on any write event
            if event_type in ["position_changed", "market_data_updated", "config_changed"]:
                invalidated = self.cache.invalidate_all()

        elif self._strategy == CacheInvalidationStrategy.ADAPTIVE:
            # Only invalidate if entry is stale based on access pattern
            for rule in self._rules:
                if rule.trigger_event == event_type:
                    invalidated += self._apply_adaptive_invalidation(rule, event_data)

        elif self._strategy == CacheInvalidationStrategy.HIERARCHICAL:
            for rule in self._rules:
                if rule.trigger_event == event_type:
                    invalidated += self._apply_hierarchical_invalidation(rule, event_data)

        # Record in history
        self._invalidation_history.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "strategy": self._strategy.value,
            "invalidated_count": invalidated,
        })

        return invalidated

    def _apply_rule(self, rule: InvalidationRule, event_data: dict | None) -> int:
        """Apply a single invalidation rule."""
        invalidated = 0

        with self.cache._lock:
            keys_to_invalidate = [
                key for key in self.cache._cache
                if rule.matches(key)
            ]

            for key in keys_to_invalidate:
                self.cache.invalidate(key, CacheInvalidationReason.MANUAL)
                invalidated += 1

                if rule.cascade:
                    invalidated += self.cache.invalidate_dependency(key)

        return invalidated

    def _apply_adaptive_invalidation(
        self,
        rule: InvalidationRule,
        event_data: dict | None,
    ) -> int:
        """Apply adaptive invalidation based on access patterns."""
        invalidated = 0
        now = time.time()

        with self.cache._lock:
            for key in list(self.cache._cache.keys()):
                if not rule.matches(key):
                    continue

                entry = self.cache._cache[key]

                # Check if entry is frequently accessed
                access_times = self._access_patterns.get(key, [])
                recent_accesses = [t for t in access_times if now - t < 300]  # Last 5 min

                # If frequently accessed, keep it longer
                if len(recent_accesses) > 5:
                    # Extend TTL by 50%
                    entry.expires_at = max(entry.expires_at, now + 150)
                else:
                    # Invalidate stale entries
                    self.cache.invalidate(key, CacheInvalidationReason.MANUAL)
                    invalidated += 1

        return invalidated

    def _apply_hierarchical_invalidation(
        self,
        rule: InvalidationRule,
        event_data: dict | None,
    ) -> int:
        """Apply hierarchical invalidation (parent invalidates children)."""
        invalidated = 0

        # Determine parent key from event data
        parent_prefix = event_data.get("parent_key", "") if event_data else ""

        with self.cache._lock:
            for key in list(self.cache._cache.keys()):
                if parent_prefix and key.startswith(parent_prefix):
                    self.cache.invalidate(key, CacheInvalidationReason.DEPENDENCY_CHANGED)
                    invalidated += 1
                elif rule.matches(key):
                    self.cache.invalidate(key, CacheInvalidationReason.MANUAL)
                    invalidated += 1

        return invalidated

    def record_access(self, key: str) -> None:
        """Record a cache access for adaptive strategy."""
        self._access_patterns[key].append(time.time())
        # Keep only last 100 accesses per key
        if len(self._access_patterns[key]) > 100:
            self._access_patterns[key] = self._access_patterns[key][-100:]

    def get_invalidation_history(self, last_n: int = 50) -> list[dict]:
        """Get recent invalidation history."""
        return list(self._invalidation_history)[-last_n:]


# =============================================================================
# CACHE SIZE OPTIMIZATION (P3 Enhancement)
# =============================================================================

@dataclass
class CacheSizeRecommendation:
    """Recommendation for cache size optimization."""
    current_size_bytes: int
    recommended_size_bytes: int
    current_entries: int
    recommended_max_entries: int
    memory_efficiency: float  # 0-1, higher is better
    reasons: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current_size_bytes": self.current_size_bytes,
            "recommended_size_bytes": self.recommended_size_bytes,
            "current_entries": self.current_entries,
            "recommended_max_entries": self.recommended_max_entries,
            "memory_efficiency": self.memory_efficiency,
            "reasons": self.reasons,
        }


class CacheSizeOptimizer:
    """
    Cache size optimization (P3 Enhancement).

    Analyzes cache usage patterns and recommends optimal sizes.
    """

    def __init__(self, cache: RiskMetricCache):
        """
        Initialize optimizer.

        Args:
            cache: RiskMetricCache instance to optimize
        """
        self.cache = cache
        self._size_history: deque = deque(maxlen=100)
        self._hit_rate_history: deque = deque(maxlen=100)

    def record_snapshot(self) -> None:
        """Record current cache state for analysis."""
        stats = self.cache.get_stats()

        self._size_history.append({
            "timestamp": time.time(),
            "entries": stats.total_entries,
            "memory_bytes": stats.memory_bytes,
        })

        self._hit_rate_history.append({
            "timestamp": time.time(),
            "hit_rate": stats.hit_rate,
            "hits": stats.hits,
            "misses": stats.misses,
        })

    def analyze(self) -> CacheSizeRecommendation:
        """
        Analyze cache usage and provide size recommendations.

        Returns:
            CacheSizeRecommendation with optimization suggestions
        """
        stats = self.cache.get_stats()
        reasons = []

        current_size = stats.memory_bytes
        current_entries = stats.total_entries

        # Analyze hit rate
        avg_hit_rate = 0.0
        if self._hit_rate_history:
            avg_hit_rate = statistics.mean([h["hit_rate"] for h in self._hit_rate_history])

        # Calculate memory efficiency
        memory_efficiency = avg_hit_rate  # Simple correlation

        # Determine recommendations
        recommended_size = current_size
        recommended_entries = self.cache.max_entries

        # If hit rate is low and we have space, increase size
        if avg_hit_rate < 0.5 and current_entries >= self.cache.max_entries * 0.9:
            recommended_entries = int(self.cache.max_entries * 1.5)
            recommended_size = int(current_size * 1.5)
            reasons.append("Low hit rate with full cache - consider increasing size")

        # If hit rate is high and cache is underutilized, reduce size
        if avg_hit_rate > 0.9 and current_entries < self.cache.max_entries * 0.5:
            recommended_entries = max(100, int(self.cache.max_entries * 0.7))
            recommended_size = int(current_size * 0.8)
            reasons.append("High hit rate with underutilized cache - can reduce size")

        # Check for memory pressure
        if current_size > self.cache.max_memory_bytes * 0.9:
            reasons.append("Near memory limit - consider eviction tuning or size increase")

        # Check eviction rate
        if stats.evictions > stats.hits * 0.2:
            recommended_entries = int(self.cache.max_entries * 1.3)
            reasons.append("High eviction rate - increase cache size")

        if not reasons:
            reasons.append("Cache size is optimal for current workload")

        return CacheSizeRecommendation(
            current_size_bytes=current_size,
            recommended_size_bytes=recommended_size,
            current_entries=current_entries,
            recommended_max_entries=recommended_entries,
            memory_efficiency=memory_efficiency,
            reasons=reasons,
        )

    def get_entry_size_distribution(self) -> dict:
        """Get distribution of entry sizes."""
        with self.cache._lock:
            sizes = [entry.size_bytes for entry in self.cache._cache.values()]

        if not sizes:
            return {"count": 0}

        return {
            "count": len(sizes),
            "min_bytes": min(sizes),
            "max_bytes": max(sizes),
            "avg_bytes": statistics.mean(sizes),
            "median_bytes": statistics.median(sizes),
            "total_bytes": sum(sizes),
        }

    def get_entry_age_distribution(self) -> dict:
        """Get distribution of entry ages."""
        with self.cache._lock:
            ages = [entry.age_seconds for entry in self.cache._cache.values()]

        if not ages:
            return {"count": 0}

        return {
            "count": len(ages),
            "min_seconds": min(ages),
            "max_seconds": max(ages),
            "avg_seconds": statistics.mean(ages),
            "median_seconds": statistics.median(ages),
        }


# =============================================================================
# CACHE PERFORMANCE METRICS (P3 Enhancement)
# =============================================================================

@dataclass
class CachePerformanceMetrics:
    """Comprehensive cache performance metrics."""
    # Basic stats
    total_requests: int
    hits: int
    misses: int
    hit_rate: float

    # Timing metrics
    avg_get_latency_ms: float
    avg_set_latency_ms: float
    p95_get_latency_ms: float
    p99_get_latency_ms: float

    # Memory metrics
    memory_bytes: int
    memory_utilization: float
    avg_entry_size_bytes: float

    # Eviction metrics
    evictions: int
    eviction_rate: float  # Per minute
    invalidations: int

    # Efficiency metrics
    cache_efficiency: float  # (hits - evictions) / total_requests
    working_set_size: int  # Estimated active entries

    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": self.hit_rate * 100,
            "avg_get_latency_ms": self.avg_get_latency_ms,
            "avg_set_latency_ms": self.avg_set_latency_ms,
            "p95_get_latency_ms": self.p95_get_latency_ms,
            "p99_get_latency_ms": self.p99_get_latency_ms,
            "memory_bytes": self.memory_bytes,
            "memory_utilization_pct": self.memory_utilization * 100,
            "avg_entry_size_bytes": self.avg_entry_size_bytes,
            "evictions": self.evictions,
            "eviction_rate_per_min": self.eviction_rate,
            "invalidations": self.invalidations,
            "cache_efficiency_pct": self.cache_efficiency * 100,
            "working_set_size": self.working_set_size,
            "timestamp": self.timestamp.isoformat(),
        }


class CachePerformanceMonitor:
    """
    Comprehensive cache performance monitoring (P3 Enhancement).

    Tracks detailed performance metrics for cache optimization.
    """

    def __init__(self, cache: RiskMetricCache):
        """
        Initialize performance monitor.

        Args:
            cache: RiskMetricCache instance to monitor
        """
        self.cache = cache
        self._get_latencies: deque = deque(maxlen=1000)
        self._set_latencies: deque = deque(maxlen=1000)
        self._request_times: deque = deque(maxlen=1000)
        self._eviction_times: deque = deque(maxlen=100)
        self._start_time = time.time()

    def record_get(self, latency_ms: float, hit: bool) -> None:
        """Record a cache get operation."""
        self._get_latencies.append(latency_ms)
        self._request_times.append(time.time())

    def record_set(self, latency_ms: float) -> None:
        """Record a cache set operation."""
        self._set_latencies.append(latency_ms)

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self._eviction_times.append(time.time())

    def get_metrics(self) -> CachePerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Returns:
            CachePerformanceMetrics with all metrics
        """
        stats = self.cache.get_stats()
        now = time.time()

        # Calculate latency stats
        get_latencies = list(self._get_latencies) or [0]
        set_latencies = list(self._set_latencies) or [0]

        avg_get = statistics.mean(get_latencies)
        avg_set = statistics.mean(set_latencies)

        sorted_get = sorted(get_latencies)
        p95_get = sorted_get[int(len(sorted_get) * 0.95)] if len(sorted_get) > 20 else max(get_latencies)
        p99_get = sorted_get[int(len(sorted_get) * 0.99)] if len(sorted_get) > 100 else max(get_latencies)

        # Calculate eviction rate (per minute)
        recent_evictions = [t for t in self._eviction_times if now - t < 60]
        eviction_rate = len(recent_evictions)

        # Calculate memory utilization
        memory_utilization = stats.memory_bytes / self.cache.max_memory_bytes

        # Calculate average entry size
        avg_entry_size = stats.memory_bytes / stats.total_entries if stats.total_entries > 0 else 0

        # Calculate cache efficiency
        total_requests = stats.hits + stats.misses
        cache_efficiency = 0.0
        if total_requests > 0:
            cache_efficiency = max(0, (stats.hits - stats.evictions) / total_requests)

        # Estimate working set size (entries accessed in last 5 min)
        working_set_size = 0
        with self.cache._lock:
            cutoff = now - 300
            for entry in self.cache._cache.values():
                if entry.last_accessed > cutoff:
                    working_set_size += 1

        return CachePerformanceMetrics(
            total_requests=total_requests,
            hits=stats.hits,
            misses=stats.misses,
            hit_rate=stats.hit_rate,
            avg_get_latency_ms=avg_get,
            avg_set_latency_ms=avg_set,
            p95_get_latency_ms=p95_get,
            p99_get_latency_ms=p99_get,
            memory_bytes=stats.memory_bytes,
            memory_utilization=memory_utilization,
            avg_entry_size_bytes=avg_entry_size,
            evictions=stats.evictions,
            eviction_rate=eviction_rate,
            invalidations=stats.invalidations,
            cache_efficiency=cache_efficiency,
            working_set_size=working_set_size,
            timestamp=datetime.now(timezone.utc),
        )

    def get_health_status(self) -> dict:
        """
        Get cache health status.

        Returns:
            Dict with health indicators
        """
        metrics = self.get_metrics()

        status = "healthy"
        issues = []

        # Check hit rate
        if metrics.hit_rate < 0.5:
            status = "degraded"
            issues.append("Low hit rate - consider cache warming or size increase")

        # Check latency
        if metrics.p95_get_latency_ms > 10:
            status = "degraded"
            issues.append("High get latency - check cache size or contention")

        # Check eviction rate
        if metrics.eviction_rate > 10:
            status = "degraded"
            issues.append("High eviction rate - increase cache size")

        # Check memory
        if metrics.memory_utilization > 0.95:
            status = "warning"
            issues.append("Near memory limit")

        # Check efficiency
        if metrics.cache_efficiency < 0.3:
            status = "degraded"
            issues.append("Low cache efficiency - review invalidation strategy")

        return {
            "status": status,
            "issues": issues,
            "hit_rate_pct": metrics.hit_rate * 100,
            "memory_utilization_pct": metrics.memory_utilization * 100,
            "eviction_rate_per_min": metrics.eviction_rate,
        }

    def get_recommendations(self) -> list[str]:
        """Get performance improvement recommendations."""
        metrics = self.get_metrics()
        recommendations = []

        if metrics.hit_rate < 0.7:
            recommendations.append(
                f"Hit rate is {metrics.hit_rate*100:.1f}%. Consider: "
                "cache warming, longer TTLs, or larger cache size."
            )

        if metrics.memory_utilization > 0.8:
            recommendations.append(
                f"Memory utilization is {metrics.memory_utilization*100:.1f}%. Consider: "
                "increasing max_memory_bytes or more aggressive eviction."
            )

        if metrics.eviction_rate > 5:
            recommendations.append(
                f"Eviction rate is {metrics.eviction_rate}/min. Consider: "
                "increasing max_entries or reviewing entry size distribution."
            )

        if metrics.p95_get_latency_ms > 5:
            recommendations.append(
                f"P95 get latency is {metrics.p95_get_latency_ms:.2f}ms. Consider: "
                "reducing cache size or optimizing key generation."
            )

        if metrics.working_set_size < metrics.total_requests * 0.1:
            recommendations.append(
                "Working set is small relative to traffic. Consider: "
                "implementing cache warming for hot keys."
            )

        if not recommendations:
            recommendations.append("Cache performance is optimal.")

        return recommendations

# risk_cache

**Path**: `C:\Users\Alexa\ai-trading-firm\core\risk_cache.py`

## Overview

Risk Metric Caching Module
==========================

Addresses issue #R29: No risk metric caching optimization.

Features:
- Efficient caching of expensive risk calculations
- TTL-based cache invalidation
- Dependency-aware cache management
- Cache warming on startup
- Memory-bounded cache with LRU eviction

## Classes

### CacheInvalidationReason

**Inherits from**: str, Enum

Reasons for cache invalidation.

### CacheEntry

**Inherits from**: Generic[...]

A single cache entry with metadata.

#### Methods

##### `def is_expired(self) -> bool`

Check if entry has expired.

##### `def age_seconds(self) -> float`

Age of entry in seconds.

### CacheStats

Cache statistics.

#### Methods

##### `def hit_rate(self) -> float`

Cache hit rate.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### RiskMetricCache

Specialized cache for risk metrics (#R29).

Optimized for:
- VaR calculations
- Correlation matrices
- Greeks calculations
- Stress test results

#### Methods

##### `def __init__(self, max_entries: int, max_memory_bytes: int, default_ttl_seconds: float)`

Initialize risk metric cache.

Args:
    max_entries: Maximum number of cache entries
    max_memory_bytes: Maximum memory usage
    default_ttl_seconds: Default TTL for entries

##### `def get(self, key: str)`

Get value from cache.

Args:
    key: Cache key

Returns:
    Cached value or None if not found/expired

##### `def set(self, key: str, value: Any, ttl_seconds: , metric_type: , dependencies: ) -> None`

Store value in cache.

Args:
    key: Cache key
    value: Value to cache
    ttl_seconds: Time-to-live (uses type default or global default)
    metric_type: Type of metric for TTL lookup
    dependencies: Keys this entry depends on

##### `def invalidate(self, key: str, reason: CacheInvalidationReason) -> bool`

Invalidate a cache entry.

Args:
    key: Cache key
    reason: Reason for invalidation

Returns:
    True if entry was removed

##### `def invalidate_dependency(self, dependency_key: str) -> int`

Invalidate all entries depending on a key.

Args:
    dependency_key: The dependency that changed

Returns:
    Number of entries invalidated

##### `def invalidate_by_type(self, metric_type: str) -> int`

Invalidate all entries of a specific metric type.

Args:
    metric_type: Type prefix to match

Returns:
    Number of entries invalidated

##### `def invalidate_all(self) -> int`

Invalidate all entries.

##### `def get_or_compute(self, key: str, compute_fn: Callable[, T], ttl_seconds: , metric_type: , dependencies: ) -> T`

Get from cache or compute and cache.

Args:
    key: Cache key
    compute_fn: Function to compute value if not cached
    ttl_seconds: Time-to-live
    metric_type: Type of metric
    dependencies: Keys this entry depends on

Returns:
    Cached or computed value

##### `def get_stats(self) -> CacheStats`

Get cache statistics.

##### `def warm_cache(self, warm_functions: list[tuple[str, Callable[, Any], ]]) -> int`

Warm cache with precomputed values.

Args:
    warm_functions: List of (key, compute_fn, metric_type) tuples

Returns:
    Number of entries warmed

### PortfolioRiskCache

High-level cache manager for portfolio risk calculations.

Manages dependencies between different risk calculations.

#### Methods

##### `def __init__(self)`

##### `def on_positions_changed(self, portfolio_id: str) -> None`

Handle position change event.

##### `def on_market_data_updated(self, symbols: list[str]) -> None`

Handle market data update event.

##### `def on_volatility_updated(self, portfolio_id: str) -> None`

Handle volatility update event.

##### `def get_var(self, portfolio_id: str, confidence: float, compute_fn: Callable[, float]) -> float`

Get cached or computed VaR.

##### `def get_correlation_matrix(self, portfolio_id: str, compute_fn: Callable[, Any]) -> Any`

Get cached or computed correlation matrix.

##### `def get_greeks(self, position_id: str, compute_fn: Callable[, dict]) -> dict`

Get cached or computed Greeks.

##### `def get_stats(self) -> dict`

Get cache statistics.

## Functions

### `def cached_risk_metric(cache: RiskMetricCache, metric_type: str, ttl_seconds: , key_fn: , dependencies_fn: )`

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

## Constants

- `T`

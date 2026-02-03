# cache_manager

**Path**: `C:\Users\Alexa\ai-trading-firm\core\cache_manager.py`

## Overview

Cache Manager Module
====================

Memory-bounded caching (Issue #S13).
Internal API rate limiting (Issue #S14).

Features:
- LRU cache with memory limits
- TTL-based expiration
- Memory pressure monitoring
- Rate limiting for internal services

## Classes

### CacheStats

Statistics for a cache.

#### Methods

##### `def hit_rate(self) -> float`

##### `def to_dict(self) -> dict`

### CacheEntry

A single cache entry.

All timestamps are float values from time.time() (Unix epoch seconds)
for consistency across the caching system.

### BoundedLRUCache

**Inherits from**: Generic[...]

Memory-bounded LRU cache (#S13).

Features:
- Maximum item count
- Maximum memory usage
- TTL-based expiration
- LRU eviction

#### Methods

##### `def __init__(self, name: str, max_items: int, max_memory_bytes: int, default_ttl_seconds: , on_evict: )`

##### `def get(self, key: str, default: )`

Get value from cache.

##### `def set(self, key: str, value: T, ttl_seconds: ) -> None`

Set value in cache.

##### `def delete(self, key: str) -> bool`

Delete entry from cache.

##### `def clear(self) -> None`

Clear all entries.

##### `def get_stats(self) -> CacheStats`

Get cache statistics.

##### `def contains(self, key: str) -> bool`

Check if key exists and is not expired.

##### `def keys(self) -> list[str]`

Get all non-expired keys.

### CacheManager

Manages multiple named caches (#S13).

Provides centralized cache management with memory monitoring.

#### Methods

##### `def __init__(self, total_max_memory_bytes: int, memory_pressure_threshold: float)`

##### `def create_cache(self, name: str, max_items: int, max_memory_bytes: int, default_ttl_seconds: ) -> BoundedLRUCache`

Create or get a named cache.

##### `def get_cache(self, name: str)`

Get cache by name.

##### `def delete_cache(self, name: str) -> bool`

Delete a cache.

##### `def get_memory_usage(self) -> dict`

Get memory usage across all caches.

##### `def check_memory_pressure(self) -> bool`

Check if memory pressure is high and trigger cleanup if needed.

##### `def get_all_stats(self) -> dict`

Get statistics for all caches.

### RateLimitConfig

Rate limit configuration.

### RateLimitResult

Result of rate limit check.

### TokenBucketRateLimiter

Token bucket rate limiter for internal APIs (#S14).

Features:
- Smooth rate limiting
- Burst allowance
- Multiple rate limit tiers

#### Methods

##### `def __init__(self, name: str, requests_per_second: float, burst_size: int)`

##### `def acquire(self, tokens: int) -> RateLimitResult`

Try to acquire tokens.

Returns RateLimitResult with allowed status and wait time if rejected.

##### `def get_stats(self) -> dict`

Get rate limiter statistics.

### RateLimiterManager

Manages rate limiters for internal APIs (#S14).

Provides centralized rate limit management.

#### Methods

##### `def __init__(self)`

##### `def create_limiter(self, name: str, requests_per_second: float, burst_size: int) -> TokenBucketRateLimiter`

Create or get a rate limiter.

##### `def get_limiter(self, name: str)`

Get limiter by name.

##### `def check_rate_limit(self, name: str, tokens: int) -> RateLimitResult`

Check rate limit for a named limiter.

##### `def get_all_stats(self) -> dict`

Get statistics for all rate limiters.

### RateLimitExceeded

**Inherits from**: Exception

Exception raised when rate limit is exceeded.

## Functions

### `def rate_limited(limiter_name: str, manager: , on_reject: )`

Decorator for rate-limited functions.

Args:
    limiter_name: Name of the rate limiter to use
    manager: RateLimiterManager instance (required)
    on_reject: Callback when rate limited (default raises exception)

### `def get_cache_manager() -> CacheManager`

Get global cache manager instance.

### `def get_rate_limiter_manager() -> RateLimiterManager`

Get global rate limiter manager instance.

## Constants

- `T`

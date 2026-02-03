# order_throttling

**Path**: `C:\Users\Alexa\ai-trading-firm\core\order_throttling.py`

## Overview

Order Throttling Module
=======================

Per-venue order throttling to prevent excessive order submission (Issue #E25).

Features:
- Per-venue rate limits
- Sliding window throttling
- Adaptive throttling based on rejections
- Burst allowance with recovery
- Exchange-specific configurations

## Classes

### ThrottleAction

**Inherits from**: str, Enum

Throttle decision.

### VenueConfig

Rate limit configuration for a venue.

### ThrottleState

Current throttle state for a venue.

#### Methods

##### `def __post_init__(self)`

### ThrottleResult

Result of throttle check.

### OrderThrottler

Per-venue order throttling implementation (#E25).

Prevents excessive order submission to protect against:
- Exchange rate limit violations
- Runaway algorithms
- Unintended order spam

#### Methods

##### `def __init__(self, custom_configs: , global_max_per_second: float, adaptive_throttling: bool)`

##### `def get_config(self, venue_id: str) -> VenueConfig`

Get configuration for venue.

##### `def get_state(self, venue_id: str) -> ThrottleState`

Get or create state for venue.

##### `def check_throttle(self, venue_id: str) -> ThrottleResult`

Check if order should be throttled for a venue.

Returns ThrottleResult with action (ALLOW, DELAY, or REJECT).

##### `def record_order(self, venue_id: str) -> None`

Record that an order was submitted.

##### `def record_rejection(self, venue_id: str, reason: str) -> None`

Record that an order was rejected by the venue.

##### `def record_success(self, venue_id: str) -> None`

Record that an order was accepted by the venue.

##### `def get_venue_stats(self, venue_id: str) -> dict`

Get throttle statistics for a venue.

##### `def get_all_stats(self) -> dict[str, dict]`

Get statistics for all venues.

##### `def reset_venue(self, venue_id: str) -> None`

Reset throttle state for a venue.

### ThrottledOrderExecutor

Wrapper that applies throttling before order execution.

Use this to wrap your order submission logic.

#### Methods

##### `def __init__(self, throttler: OrderThrottler, max_delay_ms: float)`

##### `async def execute_with_throttle(self, venue_id: str, order_func)`

Execute order function with throttling.

Args:
    venue_id: Target venue
    order_func: Async function to execute order
    *args, **kwargs: Arguments for order_func

Returns:
    Result of order_func

Raises:
    ThrottleRejectedException if order is rejected

### ThrottleRejectedException

**Inherits from**: Exception

Exception raised when order is rejected due to throttling.

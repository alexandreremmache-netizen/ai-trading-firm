# event_bus

**Path**: `C:\Users\Alexa\ai-trading-firm\core\event_bus.py`

## Overview

Event Bus
=========

Central event bus for inter-agent communication.
Implements fan-out for signal agents and fan-in synchronization for CIO.

Features:
- Bounded queues with configurable backpressure
- Warning/critical thresholds for queue depth
- Metrics tracking for monitoring
- Priority support for critical events
- Event persistence for crash recovery (#S4)
- Health check with automatic recovery (#SOLID refactoring)

## Classes

### EventBusHealthStatus

**Inherits from**: Enum

Health status levels for EventBus.

### HealthCheckConfig

Configuration for EventBus health checks.

### HealthCheckResult

Result of a health check.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### BackpressureLevel

**Inherits from**: Enum

Backpressure severity levels.

### BackpressureConfig

Configuration for backpressure handling.

### BackpressureMetrics

Metrics for backpressure monitoring.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary for logging/monitoring.

### SignalBarrier

Synchronization barrier for signal aggregation.

Collects signals from all strategy agents before CIO decision.
Implements fan-in pattern with timeout.

Thread-safety: Uses internal lock to prevent race conditions during
rapid signal arrival (fixes #S2).

#### Methods

##### `async def add_signal(self, agent_name: str, signal: SignalEvent) -> bool`

Add a signal from an agent (thread-safe).

Returns True if all expected signals received.
Returns False if barrier is closed (late signal).

##### `async def is_complete(self) -> bool`

Check if all expected agents have reported (thread-safe).

##### `async def wait(self) -> dict[str, SignalEvent]`

Wait for all signals or timeout (thread-safe).

Returns collected signals (may be partial on timeout).
Marks barrier as closed to reject late signals.

##### `async def get_signals_copy(self) -> dict[str, SignalEvent]`

Get a copy of current signals (thread-safe).

##### `async def get_received_count(self) -> int`

Get count of received signals (thread-safe).

##### `def is_closed(self) -> bool`

Check if barrier is closed.

### EventBus

Central event bus for the trading system.

Responsibilities:
- Route events to subscribed handlers
- Implement signal synchronization barrier
- Provide audit trail for all events
- Handle backpressure with bounded queues and rate limiting

Backpressure Features:
- Tiered warning/critical thresholds
- Rate limiting when queue fills
- Priority support for critical events
- Metrics for monitoring

#### Methods

##### `def __init__(self, max_queue_size: int, signal_timeout: float, barrier_timeout: float, backpressure_config: , enable_persistence: bool, persistence_config: PersistenceConfig | None, health_check_config: )`

##### `def register_signal_agent(self, agent_name: str) -> None`

Register an agent as a signal producer.

##### `def subscribe(self, event_type: EventType, handler: Callable[, Coroutine[Any, Any, None]]) -> None`

Subscribe a handler to an event type.

##### `def unsubscribe(self, event_type: EventType, handler: Callable[, Coroutine[Any, Any, None]]) -> None`

Unsubscribe a handler from an event type.

##### `def cleanup_dead_handlers(self, max_idle_seconds: float) -> int`

Remove handlers that haven't been called in a long time (P0-5 memory leak fix).

This prevents memory leaks from handlers that were subscribed but whose
owning objects were garbage collected, leaving orphaned handler references.

Args:
    max_idle_seconds: Remove handlers idle longer than this (default 10 min)

Returns:
    Number of handlers removed

##### `def get_handler_stats(self) -> dict`

Get statistics about registered handlers (for monitoring).

##### `async def publish(self, event: Event, priority: bool) -> bool`

Publish an event to the bus with backpressure handling.

Args:
    event: Event to publish
    priority: If True, attempt to publish even at critical levels

Returns:
    True if event was published, False if dropped

Backpressure behavior:
- NORMAL: Events queued immediately
- WARNING: Events queued with logging
- HIGH: Rate limiting applied, delays may occur
- CRITICAL: Low-priority events dropped, high-priority queued with delay

##### `def on_backpressure_change(self, callback: Callable[, None]) -> None`

Register callback for backpressure level changes.

##### `async def publish_signal(self, signal: SignalEvent) -> None`

Publish a signal event with barrier synchronization (race-condition safe).

Signals are collected until barrier is complete or timeout.

##### `async def wait_for_signals(self) -> dict[str, SignalEvent]`

Wait for signal barrier to complete (fan-in, race-condition safe).

Called by CIO agent before making decisions.
The barrier is atomically consumed and reset to prevent race conditions
where late signals go to the wrong barrier.

##### `async def get_barrier_status(self) -> dict[str, Any]`

Get current barrier status for debugging/monitoring.

##### `async def recover_persisted_events(self) -> int`

Recover and replay persisted events from the last session (#S4).

Should be called before start() to ensure unprocessed events are handled.

Returns:
    Number of events recovered and re-queued

##### `async def start(self) -> None`

Start the event bus processing loop.

##### `async def stop(self) -> None`

Stop the event bus gracefully.

##### `def get_event_history(self, event_type: , limit: int) -> list[Event]`

Get recent event history for audit.

##### `def queue_size(self) -> int`

Current queue size.

##### `def is_running(self) -> bool`

Check if event bus is running.

##### `def backpressure_level(self) -> BackpressureLevel`

Current backpressure level.

##### `def metrics(self) -> BackpressureMetrics`

Get current backpressure metrics.

##### `def get_status(self) -> dict`

Get comprehensive event bus status for monitoring.

##### `def persistence(self) -> EventPersistence | None`

Get the persistence layer (if enabled).

##### `def reset_metrics(self) -> None`

Reset backpressure metrics (useful for testing or periodic reset).

##### `async def check_health(self) -> HealthCheckResult`

Perform a health check on the EventBus.

Checks:
1. Processing latency is within acceptable bounds
2. Queue is not stalled (events are being processed)
3. Consecutive error count is below threshold

Returns:
    HealthCheckResult with status and diagnostics

##### `def on_health_change(self, callback: Callable[, None]) -> None`

Register callback for health status changes.

Args:
    callback: Function to call with HealthCheckResult on each check

##### `def health_status(self) -> EventBusHealthStatus`

Get current health status.

##### `def is_healthy(self) -> bool`

Check if EventBus is in healthy state.

## Constants

- `HIGH_PRIORITY_EVENT_TYPES`

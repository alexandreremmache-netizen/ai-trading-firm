# circuit_breaker

**Path**: `C:\Users\Alexa\ai-trading-firm\core\circuit_breaker.py`

## Overview

Circuit Breaker Pattern
=======================

Implements the circuit breaker pattern for fault tolerance.
Addresses issue #S6: No circuit breaker pattern for broker connection.

The circuit breaker prevents cascading failures by:
1. Monitoring failure rates
2. Opening the circuit when failures exceed threshold
3. Failing fast when circuit is open (no wasted resources)
4. Periodically testing if service has recovered (half-open state)
5. Closing the circuit when service is healthy again

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is down, requests fail immediately
- HALF_OPEN: Testing if service recovered

Reference: Martin Fowler's Circuit Breaker pattern

## Classes

### CircuitState

**Inherits from**: Enum

Circuit breaker states.

### CircuitBreakerConfig

Configuration for circuit breaker.

### CircuitStats

Statistics for circuit breaker monitoring.

#### Methods

##### `def update_time(self) -> None`

Update time in current state.

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary for monitoring.

### CircuitBreakerError

**Inherits from**: Exception

Base exception for circuit breaker errors.

### CircuitOpenError

**Inherits from**: CircuitBreakerError

Raised when circuit is open and call is rejected.

#### Methods

##### `def __init__(self, circuit_name: str, retry_after_seconds: float)`

### CallRecord

Record of a single call for sliding window tracking.

#### Methods

##### `def __init__(self, timestamp: datetime, success: bool)`

### CircuitBreaker

Circuit breaker implementation with sliding window failure tracking.

Usage:
    breaker = CircuitBreaker("broker", config)

    # As decorator
    @breaker.protect
    async def call_broker():
        ...

    # Or manually
    async with breaker:
        await call_broker()

    # Or
    result = await breaker.call(call_broker)

#### Methods

##### `def __init__(self, name: str, config: )`

##### `def name(self) -> str`

Get circuit breaker name.

##### `def state(self) -> CircuitState`

Get current circuit state.

##### `def is_closed(self) -> bool`

Check if circuit is closed (normal operation).

##### `def is_open(self) -> bool`

Check if circuit is open (failing fast).

##### `def is_half_open(self) -> bool`

Check if circuit is half-open (testing recovery).

##### `def on_state_change(self, callback: Callable[, None]) -> None`

Register callback for state changes.

##### `def on_failure(self, callback: Callable[, None]) -> None`

Register callback for failures.

##### `def get_stats(self) -> CircuitStats`

Get current statistics.

##### `async def call(self, func: Callable[Ellipsis, Any]) -> Any`

Execute a function through the circuit breaker.

Args:
    func: Async function to call
    *args: Positional arguments
    **kwargs: Keyword arguments

Returns:
    Result of the function

Raises:
    CircuitOpenError: If circuit is open
    Exception: Any exception from the wrapped function

##### `def protect(self, func: )`

Decorator to protect a function with the circuit breaker.

Args:
    func: Function to protect
    fallback: Optional fallback function when circuit is open

Usage:
    @breaker.protect
    async def my_func():
        ...

    @breaker.protect(fallback=my_fallback)
    async def my_func():
        ...

##### `async def __aenter__(self)`

Context manager entry.

##### `async def __aexit__(self, exc_type, exc_val, exc_tb)`

Context manager exit.

##### `def force_open(self) -> None`

Force the circuit to open (for testing or manual intervention).

##### `def force_close(self) -> None`

Force the circuit to close (for testing or manual intervention).

##### `def reset(self) -> None`

Reset the circuit breaker to initial state.

### CircuitBreakerRegistry

Registry for managing multiple circuit breakers.

Provides centralized access and monitoring for all circuit breakers.

#### Methods

##### `def __init__(self)`

##### `def register(self, name: str, config: ) -> CircuitBreaker`

Register a new circuit breaker.

##### `def get(self, name: str)`

Get a circuit breaker by name.

##### `def get_or_create(self, name: str, config: ) -> CircuitBreaker`

Get existing or create new circuit breaker.

##### `def get_all_stats(self) -> dict[str, dict[str, Any]]`

Get stats for all circuit breakers.

##### `def get_open_circuits(self) -> list[str]`

Get names of all open circuits.

##### `def reset_all(self) -> None`

Reset all circuit breakers.

## Functions

### `def get_circuit_registry() -> CircuitBreakerRegistry`

Get or create the global circuit breaker registry.

### `def get_circuit_breaker(name: str, config: ) -> CircuitBreaker`

Get or create a circuit breaker from the global registry.

## Constants

- `T`

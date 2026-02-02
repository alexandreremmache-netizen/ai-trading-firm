"""
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
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Any, TypeVar, Generic
from functools import wraps
import time


logger = logging.getLogger(__name__)


T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, service considered down
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Failure threshold to open circuit
    failure_threshold: int = 5  # Open circuit after 5 consecutive failures
    failure_rate_threshold: float = 0.5  # Or if 50% of calls fail
    min_calls_for_rate: int = 10  # Minimum calls before rate is considered

    # Recovery settings
    reset_timeout_seconds: float = 30.0  # Wait before trying recovery
    half_open_max_calls: int = 3  # Max calls in half-open state
    success_threshold: int = 2  # Successes needed to close circuit

    # Sliding window for rate calculation
    window_size_seconds: float = 60.0  # Time window for failure rate

    # Call timeout
    call_timeout_seconds: float = 10.0  # Timeout for individual calls

    # Exceptions to track as failures
    failure_exceptions: tuple[type, ...] = (Exception,)

    # Exceptions to ignore (not counted as failures)
    ignore_exceptions: tuple[type, ...] = ()


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    state: CircuitState = CircuitState.CLOSED
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected due to open circuit
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    time_in_current_state_seconds: float = 0.0

    def update_time(self) -> None:
        """Update time in current state."""
        self.time_in_current_state_seconds = (
            datetime.now(timezone.utc) - self.last_state_change
        ).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for monitoring."""
        self.update_time()
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "failure_rate": (
                self.failed_calls / self.total_calls if self.total_calls > 0 else 0.0
            ),
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "last_state_change": self.last_state_change.isoformat(),
            "time_in_current_state_seconds": round(self.time_in_current_state_seconds, 2),
        }


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and call is rejected."""
    def __init__(self, circuit_name: str, retry_after_seconds: float):
        self.circuit_name = circuit_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. Retry after {retry_after_seconds:.1f}s"
        )


class CallRecord:
    """Record of a single call for sliding window tracking."""
    __slots__ = ("timestamp", "success")

    def __init__(self, timestamp: datetime, success: bool):
        self.timestamp = timestamp
        self.success = success


class CircuitBreaker:
    """
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
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()

        # Sliding window for failure rate calculation
        self._call_records: list[CallRecord] = []

        # Half-open state tracking
        self._half_open_calls = 0
        self._half_open_successes = 0

        # Time tracking
        self._last_failure_time: datetime | None = None
        self._state_changed_at = datetime.now(timezone.utc)

        # Callbacks
        self._state_change_callbacks: list[Callable[[CircuitState, CircuitState], None]] = []
        self._failure_callbacks: list[Callable[[Exception], None]] = []

    @property
    def name(self) -> str:
        """Get circuit breaker name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def on_state_change(
        self, callback: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)

    def on_failure(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for failures."""
        self._failure_callbacks.append(callback)

    def get_stats(self) -> CircuitStats:
        """Get current statistics."""
        self._stats.state = self._state
        self._stats.update_time()
        return self._stats

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._state_changed_at = datetime.now(timezone.utc)
        self._stats.last_state_change = self._state_changed_at

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0

        logger.info(
            f"Circuit '{self._name}' state changed: {old_state.value} -> {new_state.value}"
        )

        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def _cleanup_old_records(self) -> None:
        """Remove records outside the sliding window."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self._config.window_size_seconds
        )
        self._call_records = [r for r in self._call_records if r.timestamp >= cutoff]

    def _calculate_failure_rate(self) -> tuple[float, int]:
        """Calculate failure rate from sliding window."""
        self._cleanup_old_records()

        total = len(self._call_records)
        if total == 0:
            return 0.0, 0

        failures = sum(1 for r in self._call_records if not r.success)
        return failures / total, total

    async def _should_allow_call(self) -> bool:
        """Determine if a call should be allowed."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if enough time has passed to try recovery
            if self._last_failure_time:
                elapsed = (
                    datetime.now(timezone.utc) - self._last_failure_time
                ).total_seconds()
                if elapsed >= self._config.reset_timeout_seconds:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self._half_open_calls < self._config.half_open_max_calls

        return False

    async def _record_success(self) -> None:
        """Record a successful call."""
        now = datetime.now(timezone.utc)
        self._call_records.append(CallRecord(now, True))

        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success_time = now

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._config.success_threshold:
                await self._transition_to(CircuitState.CLOSED)
                logger.info(f"Circuit '{self._name}' recovered - closing circuit")

    async def _record_failure(self, exc: Exception) -> None:
        """Record a failed call."""
        now = datetime.now(timezone.utc)
        self._call_records.append(CallRecord(now, False))
        self._last_failure_time = now

        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure_time = now

        # Notify failure callbacks
        for callback in self._failure_callbacks:
            try:
                callback(exc)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")

        if self._state == CircuitState.CLOSED:
            # Check if we should open the circuit
            should_open = False

            # Check consecutive failures
            if self._stats.consecutive_failures >= self._config.failure_threshold:
                should_open = True
                logger.warning(
                    f"Circuit '{self._name}': {self._stats.consecutive_failures} "
                    f"consecutive failures (threshold: {self._config.failure_threshold})"
                )

            # Check failure rate
            failure_rate, total_calls = self._calculate_failure_rate()
            if (
                total_calls >= self._config.min_calls_for_rate
                and failure_rate >= self._config.failure_rate_threshold
            ):
                should_open = True
                logger.warning(
                    f"Circuit '{self._name}': failure rate {failure_rate*100:.1f}% "
                    f"(threshold: {self._config.failure_rate_threshold*100:.1f}%)"
                )

            if should_open:
                await self._transition_to(CircuitState.OPEN)

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open state re-opens the circuit
            logger.warning(
                f"Circuit '{self._name}': failure in HALF_OPEN state - reopening"
            )
            await self._transition_to(CircuitState.OPEN)

    async def call(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        """
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
        """
        async with self._lock:
            if not await self._should_allow_call():
                self._stats.rejected_calls += 1
                retry_after = self._config.reset_timeout_seconds
                if self._last_failure_time:
                    elapsed = (
                        datetime.now(timezone.utc) - self._last_failure_time
                    ).total_seconds()
                    retry_after = max(0, self._config.reset_timeout_seconds - elapsed)
                raise CircuitOpenError(self._name, retry_after)

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self._config.call_timeout_seconds,
                )
            else:
                # Wrap sync function
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: func(*args, **kwargs)
                    ),
                    timeout=self._config.call_timeout_seconds,
                )

            async with self._lock:
                await self._record_success()

            return result

        except asyncio.TimeoutError as e:
            async with self._lock:
                await self._record_failure(e)
            logger.error(f"Circuit '{self._name}': call timed out")
            raise

        except self._config.ignore_exceptions:
            # Don't count ignored exceptions as failures
            raise

        except self._config.failure_exceptions as e:
            async with self._lock:
                await self._record_failure(e)
            raise

    def protect(
        self,
        func: Callable[..., Any] | None = None,
        *,
        fallback: Callable[..., Any] | None = None,
    ):
        """
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
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            async def wrapper(*args, **kwargs):
                try:
                    return await self.call(fn, *args, **kwargs)
                except CircuitOpenError:
                    if fallback:
                        return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
                    raise

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    async def __aenter__(self):
        """Context manager entry."""
        async with self._lock:
            if not await self._should_allow_call():
                self._stats.rejected_calls += 1
                retry_after = self._config.reset_timeout_seconds
                if self._last_failure_time:
                    elapsed = (
                        datetime.now(timezone.utc) - self._last_failure_time
                    ).total_seconds()
                    retry_after = max(0, self._config.reset_timeout_seconds - elapsed)
                raise CircuitOpenError(self._name, retry_after)

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        async with self._lock:
            if exc_type is None:
                await self._record_success()
            elif exc_type not in self._config.ignore_exceptions:
                if issubclass(exc_type, self._config.failure_exceptions):
                    await self._record_failure(exc_val)

        return False  # Don't suppress exceptions

    def force_open(self) -> None:
        """Force the circuit to open (for testing or manual intervention)."""
        logger.warning(f"Circuit '{self._name}' forced OPEN")
        asyncio.create_task(self._transition_to(CircuitState.OPEN))
        self._last_failure_time = datetime.now(timezone.utc)

    def force_close(self) -> None:
        """Force the circuit to close (for testing or manual intervention)."""
        logger.info(f"Circuit '{self._name}' forced CLOSED")
        asyncio.create_task(self._transition_to(CircuitState.CLOSED))
        self._stats.consecutive_failures = 0

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        logger.info(f"Circuit '{self._name}' reset")
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._call_records.clear()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._last_failure_time = None
        self._state_changed_at = datetime.now(timezone.utc)


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access and monitoring for all circuit breakers.
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        if name in self._breakers:
            logger.warning(f"Circuit breaker '{name}' already registered")
            return self._breakers[name]

        breaker = CircuitBreaker(name, config)
        self._breakers[name] = breaker
        logger.info(f"Registered circuit breaker: {name}")
        return breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            return self.register(name, config)
        return self._breakers[name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {
            name: breaker.get_stats().to_dict()
            for name, breaker in self._breakers.items()
        }

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
_circuit_registry: CircuitBreakerRegistry | None = None


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get or create the global circuit breaker registry."""
    global _circuit_registry
    if _circuit_registry is None:
        _circuit_registry = CircuitBreakerRegistry()
    return _circuit_registry


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return get_circuit_registry().get_or_create(name, config)

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
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Any, TypeVar
from functools import wraps
import time


logger = logging.getLogger(__name__)


# =============================================================================
# P3: Circuit Breaker Metrics for State Change Tracking
# =============================================================================


@dataclass
class CircuitStateChangeEvent:
    """Record of a circuit state change for metrics tracking."""
    timestamp: datetime
    circuit_name: str
    old_state: str
    new_state: str
    reason: str = ""
    consecutive_failures: int = 0
    failure_rate: float = 0.0


@dataclass
class CircuitBreakerMetrics:
    """
    Comprehensive metrics for circuit breaker monitoring (P3 fix).

    Tracks state changes, timing, and performance for observability.
    """
    # State change history
    state_changes: list[CircuitStateChangeEvent] = field(default_factory=list)
    max_history_size: int = 100

    # Timing metrics
    total_time_open_seconds: float = 0.0
    total_time_half_open_seconds: float = 0.0
    total_time_closed_seconds: float = 0.0
    last_open_timestamp: datetime | None = None
    last_close_timestamp: datetime | None = None

    # Count metrics
    open_count: int = 0
    half_open_count: int = 0
    recovery_count: int = 0  # Successful transitions from half_open to closed

    # Performance metrics
    avg_time_to_recover_seconds: float = 0.0
    min_time_to_recover_seconds: float = float('inf')
    max_time_to_recover_seconds: float = 0.0
    _recovery_times: list[float] = field(default_factory=list)

    def record_state_change(
        self,
        circuit_name: str,
        old_state: str,
        new_state: str,
        reason: str = "",
        consecutive_failures: int = 0,
        failure_rate: float = 0.0,
    ) -> None:
        """Record a state change event with full context."""
        now = datetime.now(timezone.utc)

        event = CircuitStateChangeEvent(
            timestamp=now,
            circuit_name=circuit_name,
            old_state=old_state,
            new_state=new_state,
            reason=reason,
            consecutive_failures=consecutive_failures,
            failure_rate=failure_rate,
        )

        self.state_changes.append(event)

        # Trim history if needed
        if len(self.state_changes) > self.max_history_size:
            self.state_changes = self.state_changes[-self.max_history_size:]

        # Update timing metrics
        if new_state == "open":
            self.open_count += 1
            self.last_open_timestamp = now

        elif new_state == "half_open":
            self.half_open_count += 1

        elif new_state == "closed" and old_state in ("open", "half_open"):
            self.last_close_timestamp = now

            # Calculate recovery time if we have open timestamp
            if self.last_open_timestamp:
                recovery_time = (now - self.last_open_timestamp).total_seconds()
                self._recovery_times.append(recovery_time)
                self.recovery_count += 1

                # Update recovery stats
                self.min_time_to_recover_seconds = min(
                    self.min_time_to_recover_seconds, recovery_time
                )
                self.max_time_to_recover_seconds = max(
                    self.max_time_to_recover_seconds, recovery_time
                )
                self.avg_time_to_recover_seconds = (
                    sum(self._recovery_times) / len(self._recovery_times)
                )

    def update_time_in_state(self, current_state: str, state_entered_at: datetime) -> None:
        """Update cumulative time spent in current state."""
        elapsed = (datetime.now(timezone.utc) - state_entered_at).total_seconds()

        if current_state == "open":
            self.total_time_open_seconds += elapsed
        elif current_state == "half_open":
            self.total_time_half_open_seconds += elapsed
        elif current_state == "closed":
            self.total_time_closed_seconds += elapsed

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for monitoring/export."""
        return {
            "state_change_count": len(self.state_changes),
            "open_count": self.open_count,
            "half_open_count": self.half_open_count,
            "recovery_count": self.recovery_count,
            "total_time_open_seconds": round(self.total_time_open_seconds, 2),
            "total_time_half_open_seconds": round(self.total_time_half_open_seconds, 2),
            "total_time_closed_seconds": round(self.total_time_closed_seconds, 2),
            "avg_time_to_recover_seconds": round(self.avg_time_to_recover_seconds, 2),
            "min_time_to_recover_seconds": (
                round(self.min_time_to_recover_seconds, 2)
                if self.min_time_to_recover_seconds != float('inf') else None
            ),
            "max_time_to_recover_seconds": round(self.max_time_to_recover_seconds, 2),
            "last_open": self.last_open_timestamp.isoformat() if self.last_open_timestamp else None,
            "last_close": self.last_close_timestamp.isoformat() if self.last_close_timestamp else None,
            "recent_state_changes": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "old_state": e.old_state,
                    "new_state": e.new_state,
                    "reason": e.reason,
                }
                for e in self.state_changes[-5:]  # Last 5 changes
            ],
        }


# =============================================================================
# P3: Graceful Degradation Modes
# =============================================================================


class DegradationMode(Enum):
    """
    Graceful degradation modes for circuit breaker (P3 fix).

    When circuit is open, instead of just failing, we can provide
    degraded service based on the configured mode.
    """
    FAIL_FAST = "fail_fast"          # Immediately raise CircuitOpenError
    CACHED_RESPONSE = "cached"        # Return last known good response
    FALLBACK_VALUE = "fallback"       # Return configured fallback value
    REDUCED_FUNCTIONALITY = "reduced" # Allow subset of operations
    QUEUE_FOR_RETRY = "queue"         # Queue request for later retry


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation behavior."""
    mode: DegradationMode = DegradationMode.FAIL_FAST
    fallback_value: Any = None
    cache_ttl_seconds: float = 60.0  # How long cached responses are valid
    max_queued_requests: int = 100   # Max requests to queue for retry
    allowed_operations_when_degraded: set[str] = field(default_factory=set)


# =============================================================================
# P2: Circuit Breaker State Persistence
# =============================================================================


@dataclass
class PersistenceConfig:
    """
    Configuration for circuit breaker state persistence (P2 fix).

    Enables circuit breakers to persist their state to disk and recover
    it after application restart. This prevents circuits from starting
    in closed state when they were open before a crash/restart.
    """
    enabled: bool = False
    persistence_dir: str = ".circuit_breaker_state"
    save_interval_seconds: float = 30.0  # How often to auto-save
    restore_on_init: bool = True  # Restore state when circuit is created


@dataclass
class PersistedState:
    """Persisted state of a circuit breaker."""
    circuit_name: str
    state: str  # "closed", "open", "half_open"
    consecutive_failures: int
    consecutive_successes: int
    last_failure_time: str | None  # ISO format
    last_success_time: str | None  # ISO format
    total_calls: int
    failed_calls: int
    successful_calls: int
    saved_at: str  # ISO format

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "circuit_name": self.circuit_name,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "successful_calls": self.successful_calls,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersistedState":
        """Create from dictionary."""
        return cls(
            circuit_name=data["circuit_name"],
            state=data["state"],
            consecutive_failures=data["consecutive_failures"],
            consecutive_successes=data["consecutive_successes"],
            last_failure_time=data.get("last_failure_time"),
            last_success_time=data.get("last_success_time"),
            total_calls=data.get("total_calls", 0),
            failed_calls=data.get("failed_calls", 0),
            successful_calls=data.get("successful_calls", 0),
            saved_at=data["saved_at"],
        )


class CircuitStatePersistence:
    """
    Handles persistence of circuit breaker state to disk (P2 fix).

    This allows circuit breakers to maintain their state across application
    restarts, preventing services from being hammered by requests when
    a circuit was open before restart.
    """

    def __init__(self, config: PersistenceConfig):
        self._config = config
        self._persistence_dir = Path(config.persistence_dir)
        self._save_tasks: dict[str, asyncio.Task] = {}

        if config.enabled:
            self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure persistence directory exists."""
        try:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create persistence directory: {e}")

    def _get_state_file(self, circuit_name: str) -> Path:
        """Get the file path for a circuit's state."""
        safe_name = circuit_name.replace("/", "_").replace("\\", "_")
        return self._persistence_dir / f"{safe_name}.json"

    def save_state(self, circuit_name: str, state: PersistedState) -> bool:
        """
        Save circuit state to disk.

        Args:
            circuit_name: Name of the circuit
            state: State to persist

        Returns:
            True if save succeeded
        """
        if not self._config.enabled:
            return False

        try:
            file_path = self._get_state_file(circuit_name)
            temp_path = file_path.with_suffix(".tmp")

            # Write to temp file first (atomic write pattern)
            with open(temp_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Rename temp to final (atomic on most systems)
            temp_path.replace(file_path)

            logger.debug(f"Saved circuit state for '{circuit_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to save circuit state for '{circuit_name}': {e}")
            return False

    def load_state(self, circuit_name: str) -> PersistedState | None:
        """
        Load circuit state from disk.

        Args:
            circuit_name: Name of the circuit

        Returns:
            PersistedState if found and valid, None otherwise
        """
        if not self._config.enabled:
            return None

        try:
            file_path = self._get_state_file(circuit_name)

            if not file_path.exists():
                return None

            with open(file_path, "r") as f:
                data = json.load(f)

            state = PersistedState.from_dict(data)
            logger.debug(f"Loaded circuit state for '{circuit_name}': {state.state}")
            return state

        except Exception as e:
            logger.error(f"Failed to load circuit state for '{circuit_name}': {e}")
            return None

    def delete_state(self, circuit_name: str) -> bool:
        """
        Delete persisted state for a circuit.

        Args:
            circuit_name: Name of the circuit

        Returns:
            True if deleted, False otherwise
        """
        if not self._config.enabled:
            return False

        try:
            file_path = self._get_state_file(circuit_name)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted circuit state for '{circuit_name}'")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete circuit state for '{circuit_name}': {e}")
            return False

    def get_all_persisted_circuits(self) -> list[str]:
        """Get names of all circuits with persisted state."""
        if not self._config.enabled:
            return []

        try:
            circuits = []
            for file_path in self._persistence_dir.glob("*.json"):
                circuits.append(file_path.stem)
            return circuits

        except Exception as e:
            logger.error(f"Failed to list persisted circuits: {e}")
            return []


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

    # P3: Graceful degradation configuration
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    # P3: Enable metrics collection (small overhead)
    collect_metrics: bool = True

    # P2: State persistence configuration
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)


# =============================================================================
# P3: Per-Operation Configurable Thresholds
# =============================================================================


@dataclass
class OperationConfig:
    """
    Per-operation configuration for circuit breaker (P3 fix).

    Different operations may need different thresholds. For example,
    a health check operation might tolerate more failures than order placement.
    """
    failure_threshold: int | None = None      # Override global threshold
    call_timeout_seconds: float | None = None  # Override global timeout
    failure_rate_threshold: float | None = None  # Override global rate threshold
    enabled: bool = True  # Can disable specific operations


class OperationThresholds:
    """
    Manages per-operation thresholds for circuit breakers (P3 fix).

    Allows different operations to have different failure tolerance.

    Usage:
        thresholds = OperationThresholds()
        thresholds.set_operation_config("health_check", OperationConfig(
            failure_threshold=10,  # More tolerant
            call_timeout_seconds=5.0,
        ))
        thresholds.set_operation_config("place_order", OperationConfig(
            failure_threshold=2,  # Less tolerant
            call_timeout_seconds=30.0,
        ))
    """

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        self._default = default_config or CircuitBreakerConfig()
        self._operation_configs: dict[str, OperationConfig] = {}

    def set_operation_config(self, operation: str, config: OperationConfig) -> None:
        """Set configuration for a specific operation."""
        self._operation_configs[operation] = config
        logger.debug(f"Set operation config for '{operation}': {config}")

    def get_failure_threshold(self, operation: str | None = None) -> int:
        """Get failure threshold for operation (or default)."""
        if operation and operation in self._operation_configs:
            op_config = self._operation_configs[operation]
            if op_config.failure_threshold is not None:
                return op_config.failure_threshold
        return self._default.failure_threshold

    def get_call_timeout(self, operation: str | None = None) -> float:
        """Get call timeout for operation (or default)."""
        if operation and operation in self._operation_configs:
            op_config = self._operation_configs[operation]
            if op_config.call_timeout_seconds is not None:
                return op_config.call_timeout_seconds
        return self._default.call_timeout_seconds

    def get_failure_rate_threshold(self, operation: str | None = None) -> float:
        """Get failure rate threshold for operation (or default)."""
        if operation and operation in self._operation_configs:
            op_config = self._operation_configs[operation]
            if op_config.failure_rate_threshold is not None:
                return op_config.failure_rate_threshold
        return self._default.failure_rate_threshold

    def is_operation_enabled(self, operation: str) -> bool:
        """Check if operation is enabled."""
        if operation in self._operation_configs:
            return self._operation_configs[operation].enabled
        return True

    def get_all_configs(self) -> dict[str, OperationConfig]:
        """Get all operation configurations."""
        return self._operation_configs.copy()


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

    P3 Improvements:
    - Metrics for circuit state changes
    - Configurable thresholds per operation
    - Graceful degradation modes

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

        # With operation-specific thresholds (P3)
        result = await breaker.call(call_broker, operation="health_check")
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

        # P3: Metrics tracking
        self._metrics = CircuitBreakerMetrics() if self._config.collect_metrics else None

        # P3: Per-operation thresholds
        self._operation_thresholds = OperationThresholds(self._config)

        # P3: Graceful degradation - cached responses
        self._cached_responses: dict[str, tuple[Any, datetime]] = {}  # key -> (value, timestamp)

        # P3: Graceful degradation - queued requests for retry
        self._queued_requests: list[tuple[Callable, tuple, dict, str]] = []  # (func, args, kwargs, operation)

        # P2: State persistence
        self._persistence: CircuitStatePersistence | None = None
        if self._config.persistence.enabled:
            self._persistence = CircuitStatePersistence(self._config.persistence)
            if self._config.persistence.restore_on_init:
                self._restore_state()
            # Start auto-save task
            self._auto_save_task: asyncio.Task | None = None

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

    def get_metrics(self) -> CircuitBreakerMetrics | None:
        """Get detailed metrics (P3). Returns None if metrics collection disabled."""
        return self._metrics

    def set_operation_config(self, operation: str, config: OperationConfig) -> None:
        """Set per-operation configuration (P3)."""
        self._operation_thresholds.set_operation_config(operation, config)

    async def _transition_to(
        self,
        new_state: CircuitState,
        reason: str = "",
    ) -> None:
        """
        Transition to a new state with metrics tracking (P3 enhanced).

        Args:
            new_state: Target state
            reason: Reason for transition (for metrics/logging)
        """
        if new_state == self._state:
            return

        old_state = self._state
        now = datetime.now(timezone.utc)

        # P3: Update time spent in previous state before transition
        if self._metrics:
            self._metrics.update_time_in_state(
                old_state.value, self._state_changed_at
            )

        self._state = new_state
        self._state_changed_at = now
        self._stats.last_state_change = now

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0

        # P3: Record state change in metrics
        if self._metrics:
            failure_rate, _ = self._calculate_failure_rate()
            self._metrics.record_state_change(
                circuit_name=self._name,
                old_state=old_state.value,
                new_state=new_state.value,
                reason=reason,
                consecutive_failures=self._stats.consecutive_failures,
                failure_rate=failure_rate,
            )

        logger.info(
            f"Circuit '{self._name}' state changed: {old_state.value} -> {new_state.value}"
            + (f" (reason: {reason})" if reason else "")
        )

        # P3: Process queued requests when circuit closes
        if new_state == CircuitState.CLOSED and self._queued_requests:
            asyncio.create_task(self._process_queued_requests())

        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.exception(f"State change callback error for circuit '{self._name}'")

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
                logger.exception(f"Failure callback error for circuit '{self._name}'")

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
                await self._transition_to(
                    CircuitState.OPEN,
                    reason=f"failures={self._stats.consecutive_failures}, rate={failure_rate*100:.1f}%"
                )

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open state re-opens the circuit
            logger.warning(
                f"Circuit '{self._name}': failure in HALF_OPEN state - reopening"
            )
            await self._transition_to(CircuitState.OPEN, reason="half_open_failure")

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
        # P3: Clear caches and queues
        self._cached_responses.clear()
        self._queued_requests.clear()
        # P2: Clear persisted state
        self.delete_persisted_state()

    # =========================================================================
    # P3: Graceful Degradation Methods
    # =========================================================================

    def cache_response(self, cache_key: str, value: Any) -> None:
        """
        Cache a successful response for degraded mode fallback (P3).

        Args:
            cache_key: Key to identify this response (e.g., "get_portfolio")
            value: The response value to cache
        """
        self._cached_responses[cache_key] = (value, datetime.now(timezone.utc))

    def get_cached_response(self, cache_key: str) -> tuple[Any, bool]:
        """
        Get a cached response if available and not expired (P3).

        Args:
            cache_key: Key to look up

        Returns:
            Tuple of (cached_value, is_valid). If not found or expired, returns (None, False).
        """
        if cache_key not in self._cached_responses:
            return None, False

        value, cached_at = self._cached_responses[cache_key]
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()

        if age > self._config.degradation.cache_ttl_seconds:
            # Expired
            del self._cached_responses[cache_key]
            return None, False

        return value, True

    async def call_with_degradation(
        self,
        func: Callable[..., Any],
        *args,
        cache_key: str | None = None,
        operation: str | None = None,
        **kwargs,
    ) -> tuple[Any, bool]:
        """
        Execute a function with graceful degradation support (P3).

        When the circuit is open, instead of failing immediately, this method
        applies the configured degradation strategy.

        Args:
            func: Async function to call
            *args: Positional arguments
            cache_key: Optional key for caching (required for CACHED_RESPONSE mode)
            operation: Optional operation name for per-operation thresholds
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, is_degraded). is_degraded=True if fallback was used.
        """
        # Check if operation is enabled
        if operation and not self._operation_thresholds.is_operation_enabled(operation):
            logger.warning(f"Operation '{operation}' is disabled for circuit '{self._name}'")
            return self._config.degradation.fallback_value, True

        try:
            # Try normal execution
            result = await self.call(func, *args, **kwargs)

            # Cache successful response if key provided
            if cache_key:
                self.cache_response(cache_key, result)

            return result, False

        except CircuitOpenError:
            # Circuit is open - apply degradation strategy
            mode = self._config.degradation.mode

            if mode == DegradationMode.FAIL_FAST:
                raise

            elif mode == DegradationMode.CACHED_RESPONSE:
                if cache_key:
                    cached, is_valid = self.get_cached_response(cache_key)
                    if is_valid:
                        logger.info(
                            f"Circuit '{self._name}' open - using cached response for '{cache_key}'"
                        )
                        return cached, True
                # No valid cache - fall through to fail
                raise

            elif mode == DegradationMode.FALLBACK_VALUE:
                logger.info(
                    f"Circuit '{self._name}' open - returning fallback value"
                )
                return self._config.degradation.fallback_value, True

            elif mode == DegradationMode.QUEUE_FOR_RETRY:
                if len(self._queued_requests) < self._config.degradation.max_queued_requests:
                    self._queued_requests.append((func, args, kwargs, operation or ""))
                    logger.info(
                        f"Circuit '{self._name}' open - queued request for retry "
                        f"({len(self._queued_requests)} in queue)"
                    )
                    return self._config.degradation.fallback_value, True
                else:
                    logger.warning(
                        f"Circuit '{self._name}' queue full - cannot queue more requests"
                    )
                    raise

            elif mode == DegradationMode.REDUCED_FUNCTIONALITY:
                # Check if this operation is allowed when degraded
                if operation and operation in self._config.degradation.allowed_operations_when_degraded:
                    # Allow the call even though circuit is open
                    # This is useful for read-only or non-critical operations
                    logger.info(
                        f"Circuit '{self._name}' open - allowing reduced functionality "
                        f"for operation '{operation}'"
                    )
                    # Still try the call but don't count against circuit
                    try:
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                        return result, True
                    except Exception as e:
                        logger.warning(
                            f"Reduced functionality call failed: {e}"
                        )
                        raise
                else:
                    raise

            # Unknown mode - fail fast
            raise

    async def _process_queued_requests(self) -> None:
        """
        Process queued requests when circuit closes (P3).

        Called automatically when circuit transitions to CLOSED state.
        """
        if not self._queued_requests:
            return

        logger.info(
            f"Circuit '{self._name}' closed - processing {len(self._queued_requests)} "
            f"queued requests"
        )

        processed = 0
        failed = 0

        while self._queued_requests and self.is_closed:
            func, args, kwargs, operation = self._queued_requests.pop(0)

            try:
                await self.call(func, *args, **kwargs)
                processed += 1
            except Exception as e:
                failed += 1
                logger.warning(
                    f"Queued request failed: {e}"
                )
                # If circuit reopens, stop processing
                if not self.is_closed:
                    break

        logger.info(
            f"Circuit '{self._name}' processed queued requests: "
            f"{processed} succeeded, {failed} failed, "
            f"{len(self._queued_requests)} remaining"
        )

    def get_queue_size(self) -> int:
        """Get number of requests waiting in the retry queue (P3)."""
        return len(self._queued_requests)

    def clear_queue(self) -> int:
        """Clear all queued requests and return how many were cleared (P3)."""
        count = len(self._queued_requests)
        self._queued_requests.clear()
        return count

    # =========================================================================
    # P2: State Persistence Methods
    # =========================================================================

    def _get_persisted_state(self) -> PersistedState:
        """Create a PersistedState from current circuit state."""
        now = datetime.now(timezone.utc)
        return PersistedState(
            circuit_name=self._name,
            state=self._state.value,
            consecutive_failures=self._stats.consecutive_failures,
            consecutive_successes=self._stats.consecutive_successes,
            last_failure_time=(
                self._stats.last_failure_time.isoformat()
                if self._stats.last_failure_time else None
            ),
            last_success_time=(
                self._stats.last_success_time.isoformat()
                if self._stats.last_success_time else None
            ),
            total_calls=self._stats.total_calls,
            failed_calls=self._stats.failed_calls,
            successful_calls=self._stats.successful_calls,
            saved_at=now.isoformat(),
        )

    def _restore_state(self) -> bool:
        """
        Restore circuit state from persistence (P2).

        Called on initialization if persistence is enabled and restore_on_init is True.

        Returns:
            True if state was restored, False otherwise
        """
        if not self._persistence:
            return False

        try:
            persisted = self._persistence.load_state(self._name)
            if not persisted:
                return False

            # Restore state
            state_map = {
                "closed": CircuitState.CLOSED,
                "open": CircuitState.OPEN,
                "half_open": CircuitState.HALF_OPEN,
            }
            if persisted.state in state_map:
                self._state = state_map[persisted.state]

            # Restore stats
            self._stats.consecutive_failures = persisted.consecutive_failures
            self._stats.consecutive_successes = persisted.consecutive_successes
            self._stats.total_calls = persisted.total_calls
            self._stats.failed_calls = persisted.failed_calls
            self._stats.successful_calls = persisted.successful_calls

            if persisted.last_failure_time:
                self._stats.last_failure_time = datetime.fromisoformat(
                    persisted.last_failure_time
                )
                self._last_failure_time = self._stats.last_failure_time

            if persisted.last_success_time:
                self._stats.last_success_time = datetime.fromisoformat(
                    persisted.last_success_time
                )

            logger.info(
                f"Restored circuit '{self._name}' state from persistence: "
                f"state={persisted.state}, failures={persisted.consecutive_failures}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to restore circuit state for '{self._name}': {e}")
            return False

    def save_state(self) -> bool:
        """
        Manually save circuit state to persistence (P2).

        Returns:
            True if save succeeded
        """
        if not self._persistence:
            return False

        state = self._get_persisted_state()
        return self._persistence.save_state(self._name, state)

    async def start_auto_save(self) -> None:
        """
        Start automatic state persistence (P2).

        Saves state periodically based on persistence config.
        """
        if not self._persistence:
            return

        if hasattr(self, '_auto_save_task') and self._auto_save_task:
            return  # Already running

        async def auto_save_loop():
            while True:
                try:
                    await asyncio.sleep(self._config.persistence.save_interval_seconds)
                    self.save_state()
                except asyncio.CancelledError:
                    # Save one final time before stopping
                    self.save_state()
                    break
                except Exception as e:
                    logger.error(f"Auto-save error for circuit '{self._name}': {e}")

        self._auto_save_task = asyncio.create_task(auto_save_loop())
        logger.debug(f"Started auto-save for circuit '{self._name}'")

    async def stop_auto_save(self) -> None:
        """Stop automatic state persistence (P2)."""
        if hasattr(self, '_auto_save_task') and self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
            self._auto_save_task = None
            logger.debug(f"Stopped auto-save for circuit '{self._name}'")

    def delete_persisted_state(self) -> bool:
        """
        Delete persisted state for this circuit (P2).

        Useful when resetting a circuit.

        Returns:
            True if deleted
        """
        if not self._persistence:
            return False
        return self._persistence.delete_state(self._name)


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

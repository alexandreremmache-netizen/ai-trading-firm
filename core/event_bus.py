"""
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
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Coroutine, Any, TYPE_CHECKING

from core.events import Event, EventType, SignalEvent

if TYPE_CHECKING:
    from core.event_persistence import EventPersistence, PersistenceConfig


logger = logging.getLogger(__name__)


class BackpressureLevel(Enum):
    """Backpressure severity levels."""
    NORMAL = "normal"       # < 50% capacity
    WARNING = "warning"     # 50-75% capacity
    HIGH = "high"           # 75-90% capacity
    CRITICAL = "critical"   # > 90% capacity


@dataclass
class BackpressureConfig:
    """Configuration for backpressure handling."""
    max_queue_size: int = 10000
    warning_threshold_pct: float = 50.0  # 50%
    high_threshold_pct: float = 75.0     # 75%
    critical_threshold_pct: float = 90.0  # 90%
    drop_low_priority_at_critical: bool = True
    enable_rate_limiting: bool = True
    rate_limit_events_per_second: int = 1000
    cooldown_seconds: float = 1.0  # Slow down publishing when overloaded


@dataclass
class BackpressureMetrics:
    """Metrics for backpressure monitoring."""
    total_events_published: int = 0
    total_events_processed: int = 0
    total_events_dropped: int = 0
    current_queue_size: int = 0
    max_queue_size_reached: int = 0
    backpressure_level: BackpressureLevel = BackpressureLevel.NORMAL
    last_warning_time: datetime | None = None
    events_per_second: float = 0.0
    processing_latency_ms: float = 0.0
    rate_limited_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/monitoring."""
        return {
            "total_published": self.total_events_published,
            "total_processed": self.total_events_processed,
            "total_dropped": self.total_events_dropped,
            "queue_size": self.current_queue_size,
            "max_queue_reached": self.max_queue_size_reached,
            "backpressure_level": self.backpressure_level.value,
            "events_per_second": round(self.events_per_second, 2),
            "processing_latency_ms": round(self.processing_latency_ms, 2),
            "rate_limited_count": self.rate_limited_count,
        }


# High-priority event types that should never be dropped
HIGH_PRIORITY_EVENT_TYPES = {
    EventType.FILL,
    EventType.VALIDATED_DECISION,
    EventType.RISK_ALERT,
    EventType.KILL_SWITCH,
}


@dataclass
class SignalBarrier:
    """
    Synchronization barrier for signal aggregation.

    Collects signals from all strategy agents before CIO decision.
    Implements fan-in pattern with timeout.

    Thread-safety: Uses internal lock to prevent race conditions during
    rapid signal arrival (fixes #S2).
    """
    expected_agents: set[str]
    timeout_seconds: float
    barrier_id: int = 0  # Unique ID to track barrier versions
    signals: dict[str, SignalEvent] = field(default_factory=dict)
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _is_closed: bool = False  # Prevent late signals after barrier is consumed

    async def add_signal(self, agent_name: str, signal: SignalEvent) -> bool:
        """
        Add a signal from an agent (thread-safe).

        Returns True if all expected signals received.
        Returns False if barrier is closed (late signal).
        """
        async with self._lock:
            if self._is_closed:
                logger.warning(
                    f"Late signal from {agent_name} for barrier {self.barrier_id} - "
                    f"barrier already closed"
                )
                return False

            self.signals[agent_name] = signal

            if self._is_complete_unsafe():
                self._event.set()
                return True

        return False

    def _is_complete_unsafe(self) -> bool:
        """Check if all expected agents have reported (no lock, internal use)."""
        return set(self.signals.keys()) >= self.expected_agents

    async def is_complete(self) -> bool:
        """Check if all expected agents have reported (thread-safe)."""
        async with self._lock:
            return self._is_complete_unsafe()

    async def wait(self) -> dict[str, SignalEvent]:
        """
        Wait for all signals or timeout (thread-safe).

        Returns collected signals (may be partial on timeout).
        Marks barrier as closed to reject late signals.
        """
        try:
            await asyncio.wait_for(
                self._event.wait(),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            async with self._lock:
                missing = self.expected_agents - set(self.signals.keys())
            logger.warning(
                f"Signal barrier {self.barrier_id} timeout. "
                f"Received {len(self.signals)}/{len(self.expected_agents)} signals. "
                f"Missing: {missing}"
            )

        # Close barrier to prevent late signals
        async with self._lock:
            self._is_closed = True
            return dict(self.signals)  # Return a copy

    async def get_signals_copy(self) -> dict[str, SignalEvent]:
        """Get a copy of current signals (thread-safe)."""
        async with self._lock:
            return dict(self.signals)

    async def get_received_count(self) -> int:
        """Get count of received signals (thread-safe)."""
        async with self._lock:
            return len(self.signals)

    def is_closed(self) -> bool:
        """Check if barrier is closed."""
        return self._is_closed


class EventBus:
    """
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
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        signal_timeout: float = 5.0,
        barrier_timeout: float = 10.0,
        backpressure_config: BackpressureConfig | None = None,
        enable_persistence: bool = False,
        persistence_config: "PersistenceConfig | None" = None,
    ):
        self._backpressure = backpressure_config or BackpressureConfig(max_queue_size=max_queue_size)
        self._subscribers: dict[EventType, list[Callable[[Event], Coroutine[Any, Any, None]]]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._backpressure.max_queue_size)
        self._signal_timeout = signal_timeout
        self._barrier_timeout = barrier_timeout
        self._running = False
        self._current_barrier: SignalBarrier | None = None
        self._signal_agents: set[str] = set()
        self._event_history: list[Event] = []
        self._max_history = 10000
        self._lock = asyncio.Lock()

        # Backpressure state
        self._metrics = BackpressureMetrics()
        self._last_rate_limit_time: datetime | None = None
        self._events_in_window: list[datetime] = []
        self._rate_window_seconds = 1.0

        # Backpressure callbacks for external monitoring
        self._backpressure_callbacks: list[Callable[[BackpressureLevel, BackpressureMetrics], None]] = []

        # Event persistence for crash recovery (#S4)
        self._enable_persistence = enable_persistence
        self._persistence: "EventPersistence | None" = None
        if enable_persistence:
            from core.event_persistence import EventPersistence
            self._persistence = EventPersistence(persistence_config)
            self._persistence.initialize()
            logger.info("Event persistence enabled")

        # Message deduplication for replay protection (#S7)
        self._enable_deduplication = True
        self._processed_event_ids: set[str] = set()
        self._processed_event_timestamps: dict[str, datetime] = {}
        self._dedup_window_seconds = 300.0  # 5 minute window
        self._dedup_max_ids = 100000  # Max IDs to track
        self._dedup_cleanup_interval = 60  # Cleanup every 60 seconds
        self._last_dedup_cleanup = datetime.now(timezone.utc)
        self._dedup_stats = {
            "duplicates_blocked": 0,
            "ids_tracked": 0,
        }

    def register_signal_agent(self, agent_name: str) -> None:
        """Register an agent as a signal producer."""
        self._signal_agents.add(agent_name)
        logger.info(f"Registered signal agent: {agent_name}")

    def _is_duplicate(self, event_id: str) -> bool:
        """
        Check if an event is a duplicate (#S7).

        Returns True if event ID has been seen within the dedup window.
        """
        if not self._enable_deduplication:
            return False

        # Cleanup old IDs periodically
        now = datetime.now(timezone.utc)
        if (now - self._last_dedup_cleanup).total_seconds() > self._dedup_cleanup_interval:
            self._cleanup_dedup_ids()
            self._last_dedup_cleanup = now

        return event_id in self._processed_event_ids

    def _mark_processed(self, event_id: str) -> None:
        """Mark an event as processed for deduplication (#S7)."""
        if not self._enable_deduplication:
            return

        self._processed_event_ids.add(event_id)
        self._processed_event_timestamps[event_id] = datetime.now(timezone.utc)
        self._dedup_stats["ids_tracked"] = len(self._processed_event_ids)

        # Emergency cleanup if we hit max IDs
        if len(self._processed_event_ids) > self._dedup_max_ids:
            self._cleanup_dedup_ids(force_cleanup_pct=50)

    def _cleanup_dedup_ids(self, force_cleanup_pct: int = 0) -> None:
        """
        Clean up old event IDs from deduplication tracking (#S7).

        Args:
            force_cleanup_pct: If > 0, force removal of this percentage of oldest IDs
        """
        if force_cleanup_pct > 0:
            # Force removal of oldest IDs
            sorted_ids = sorted(
                self._processed_event_timestamps.items(),
                key=lambda x: x[1]
            )
            remove_count = int(len(sorted_ids) * force_cleanup_pct / 100)
            for event_id, _ in sorted_ids[:remove_count]:
                self._processed_event_ids.discard(event_id)
                self._processed_event_timestamps.pop(event_id, None)
            logger.warning(
                f"Deduplication: force-cleaned {remove_count} oldest event IDs"
            )
        else:
            # Normal cleanup - remove IDs outside the window
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._dedup_window_seconds)
            expired_ids = [
                eid for eid, ts in self._processed_event_timestamps.items()
                if ts < cutoff
            ]
            for eid in expired_ids:
                self._processed_event_ids.discard(eid)
                self._processed_event_timestamps.pop(eid, None)

        self._dedup_stats["ids_tracked"] = len(self._processed_event_ids)

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe a handler to an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.value}")

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine[Any, Any, None]],
    ) -> None:
        """Unsubscribe a handler from an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: Event, priority: bool = False) -> bool:
        """
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
        """
        # Check for duplicates (#S7)
        if self._is_duplicate(event.event_id):
            self._dedup_stats["duplicates_blocked"] += 1
            logger.debug(f"Duplicate event blocked: {event.event_id} ({event.event_type.value})")
            return False

        # Update metrics
        self._metrics.total_events_published += 1
        self._metrics.current_queue_size = self._queue.qsize()

        # Check if this is a high-priority event
        is_high_priority = priority or event.event_type in HIGH_PRIORITY_EVENT_TYPES

        # Calculate current backpressure level
        backpressure_level = self._calculate_backpressure_level()
        previous_level = self._metrics.backpressure_level
        self._metrics.backpressure_level = backpressure_level

        # Notify on level change
        if backpressure_level != previous_level:
            await self._notify_backpressure_change(backpressure_level)

        # Handle based on backpressure level
        if backpressure_level == BackpressureLevel.CRITICAL:
            if not is_high_priority and self._backpressure.drop_low_priority_at_critical:
                self._metrics.total_events_dropped += 1
                logger.warning(
                    f"BACKPRESSURE CRITICAL: Dropping low-priority event {event.event_type.value} "
                    f"(queue: {self._queue.qsize()}/{self._backpressure.max_queue_size})"
                )
                return False

            # For high-priority events, wait with timeout
            try:
                await asyncio.wait_for(
                    self._queue.put(event),
                    timeout=self._backpressure.cooldown_seconds
                )
            except asyncio.TimeoutError:
                self._metrics.total_events_dropped += 1
                logger.error(f"CRITICAL: Failed to queue high-priority event {event.event_id}")
                return False

        elif backpressure_level == BackpressureLevel.HIGH:
            # Apply rate limiting
            if self._backpressure.enable_rate_limiting:
                await self._apply_rate_limiting()

            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Try with short wait
                try:
                    await asyncio.wait_for(
                        self._queue.put(event),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    self._metrics.total_events_dropped += 1
                    logger.warning(f"HIGH backpressure: Dropping event {event.event_id}")
                    return False

        elif backpressure_level == BackpressureLevel.WARNING:
            # Log warning but queue normally
            now = datetime.now(timezone.utc)
            if (self._metrics.last_warning_time is None or
                (now - self._metrics.last_warning_time).total_seconds() > 5.0):
                logger.warning(
                    f"BACKPRESSURE WARNING: Queue at {self._queue.qsize()}/{self._backpressure.max_queue_size} "
                    f"({self._queue.qsize() / self._backpressure.max_queue_size * 100:.1f}%)"
                )
                self._metrics.last_warning_time = now

            self._queue.put_nowait(event)

        else:  # NORMAL
            self._queue.put_nowait(event)

        # Track max queue size
        if self._queue.qsize() > self._metrics.max_queue_size_reached:
            self._metrics.max_queue_size_reached = self._queue.qsize()

        # Track event history for audit
        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Persist event for crash recovery (#S4)
        if self._persistence:
            await self._persistence.persist_event_async(event, priority=is_high_priority)

        return True

    def _calculate_backpressure_level(self) -> BackpressureLevel:
        """Calculate current backpressure level based on queue depth."""
        queue_pct = (self._queue.qsize() / self._backpressure.max_queue_size) * 100

        if queue_pct >= self._backpressure.critical_threshold_pct:
            return BackpressureLevel.CRITICAL
        elif queue_pct >= self._backpressure.high_threshold_pct:
            return BackpressureLevel.HIGH
        elif queue_pct >= self._backpressure.warning_threshold_pct:
            return BackpressureLevel.WARNING
        else:
            return BackpressureLevel.NORMAL

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting when queue is filling up."""
        now = datetime.now(timezone.utc)

        # Track events in current window
        cutoff = now - timedelta(seconds=self._rate_window_seconds)
        self._events_in_window = [t for t in self._events_in_window if t > cutoff]
        self._events_in_window.append(now)

        # Calculate current rate
        events_per_second = len(self._events_in_window) / self._rate_window_seconds
        self._metrics.events_per_second = events_per_second

        # If exceeding rate limit, sleep briefly
        if events_per_second > self._backpressure.rate_limit_events_per_second:
            sleep_time = 1.0 / self._backpressure.rate_limit_events_per_second
            await asyncio.sleep(sleep_time)
            self._metrics.rate_limited_count += 1

    async def _notify_backpressure_change(self, new_level: BackpressureLevel) -> None:
        """Notify registered callbacks of backpressure level change."""
        for callback in self._backpressure_callbacks:
            try:
                callback(new_level, self._metrics)
            except Exception as e:
                logger.error(f"Backpressure callback error: {e}")

        # Log level change
        if new_level == BackpressureLevel.CRITICAL:
            logger.critical(
                f"BACKPRESSURE CRITICAL: Queue at {self._queue.qsize()}/{self._backpressure.max_queue_size}"
            )
        elif new_level == BackpressureLevel.HIGH:
            logger.warning(
                f"BACKPRESSURE HIGH: Queue at {self._queue.qsize()}/{self._backpressure.max_queue_size}"
            )
        elif new_level == BackpressureLevel.NORMAL:
            logger.info("Backpressure returned to NORMAL")

    def on_backpressure_change(
        self,
        callback: Callable[[BackpressureLevel, BackpressureMetrics], None]
    ) -> None:
        """Register callback for backpressure level changes."""
        self._backpressure_callbacks.append(callback)

    async def publish_signal(self, signal: SignalEvent) -> None:
        """
        Publish a signal event with barrier synchronization (race-condition safe).

        Signals are collected until barrier is complete or timeout.
        """
        await self.publish(signal)

        async with self._lock:
            # If barrier is closed or doesn't exist, create new one
            if self._current_barrier is None or self._current_barrier.is_closed():
                self._barrier_id_counter = getattr(self, '_barrier_id_counter', 0) + 1
                self._current_barrier = SignalBarrier(
                    expected_agents=self._signal_agents.copy(),
                    timeout_seconds=self._barrier_timeout,
                    barrier_id=self._barrier_id_counter,
                )
                logger.debug(f"Created new signal barrier {self._barrier_id_counter}")

            barrier = self._current_barrier

        # Add signal outside the event bus lock (barrier has its own lock)
        await barrier.add_signal(signal.source_agent, signal)

    async def wait_for_signals(self) -> dict[str, SignalEvent]:
        """
        Wait for signal barrier to complete (fan-in, race-condition safe).

        Called by CIO agent before making decisions.
        The barrier is atomically consumed and reset to prevent race conditions
        where late signals go to the wrong barrier.
        """
        async with self._lock:
            if self._current_barrier is None:
                return {}

            # Take ownership of the barrier - no one else can use it
            barrier = self._current_barrier
            barrier_id = barrier.barrier_id

            # Don't reset the barrier yet - late signals need to know this barrier is done

        logger.debug(f"CIO waiting for signal barrier {barrier_id}")

        # Wait outside the lock so signals can still be added
        signals = await barrier.wait()

        # Now atomically reset the barrier
        async with self._lock:
            # Only reset if it's still the same barrier (prevents double-reset)
            if self._current_barrier is barrier:
                self._current_barrier = None
                logger.debug(f"Signal barrier {barrier_id} consumed with {len(signals)} signals")

        return signals

    async def get_barrier_status(self) -> dict[str, Any]:
        """
        Get current barrier status for debugging/monitoring.
        """
        async with self._lock:
            if self._current_barrier is None:
                return {
                    "active": False,
                    "barrier_id": None,
                    "signals_received": 0,
                    "signals_expected": len(self._signal_agents),
                }

            received = await self._current_barrier.get_received_count()
            return {
                "active": True,
                "barrier_id": self._current_barrier.barrier_id,
                "is_closed": self._current_barrier.is_closed(),
                "signals_received": received,
                "signals_expected": len(self._current_barrier.expected_agents),
                "created_at": self._current_barrier._created_at.isoformat(),
            }

    async def recover_persisted_events(self) -> int:
        """
        Recover and replay persisted events from the last session (#S4).

        Should be called before start() to ensure unprocessed events are handled.

        Returns:
            Number of events recovered and re-queued
        """
        if not self._persistence:
            logger.debug("Persistence not enabled, skipping recovery")
            return 0

        # Reset any events stuck in 'processing' state (from crash)
        self._persistence.reset_stale_processing(timeout_minutes=5)

        # Get pending events
        pending_events = self._persistence.get_pending_events(
            limit=1000,
            include_failed=True
        )

        if not pending_events:
            logger.info("No persisted events to recover")
            return 0

        recovered = 0
        for persisted in pending_events:
            # Mark as processing
            if not self._persistence.mark_processing(persisted.event_id):
                continue

            # Reconstruct the event
            event = self._persistence.reconstruct_event(persisted)
            if event is None:
                self._persistence.mark_failed(
                    persisted.event_id,
                    "Failed to reconstruct event"
                )
                continue

            # Re-queue the event (without re-persisting)
            try:
                self._queue.put_nowait(event)
                recovered += 1
                logger.debug(f"Recovered event {persisted.event_id} ({persisted.event_type})")
            except asyncio.QueueFull:
                # Reset back to pending for next attempt
                logger.warning(f"Queue full during recovery, event {persisted.event_id} will retry later")
                break

        logger.info(f"Recovered {recovered} persisted events for processing")
        return recovered

    async def start(self) -> None:
        """Start the event bus processing loop."""
        self._running = True

        # Recover persisted events before starting (#S4)
        if self._persistence:
            await self.recover_persisted_events()

        logger.info("Event bus started")

        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                await self._dispatch(event)
                self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        self._running = False

        # Process remaining events
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Clean up persistence (#S4)
        if self._persistence:
            # Cleanup old completed events
            self._persistence.cleanup_completed_events()
            self._persistence.close()

        logger.info("Event bus stopped")

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all subscribed handlers with latency tracking."""
        start_time = datetime.now(timezone.utc)
        dispatch_success = True

        handlers = self._subscribers.get(event.event_type, [])

        if not handlers:
            logger.debug(f"No handlers for event type: {event.event_type.value}")
            self._metrics.total_events_processed += 1
            # Mark as completed even if no handlers (it was "processed")
            if self._persistence:
                await self._persistence.mark_completed_async(event.event_id)
            return

        # Execute handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Handler error for {event.event_type.value}: {result}")
                dispatch_success = False

        # Update metrics
        self._metrics.total_events_processed += 1
        self._metrics.current_queue_size = self._queue.qsize()

        # Calculate processing latency (exponential moving average)
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        if self._metrics.processing_latency_ms == 0:
            self._metrics.processing_latency_ms = latency_ms
        else:
            alpha = 0.1  # Smoothing factor
            self._metrics.processing_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self._metrics.processing_latency_ms
            )

        # Mark event as completed in persistence layer (#S4)
        if self._persistence:
            if dispatch_success:
                await self._persistence.mark_completed_async(event.event_id)
            else:
                # Mark as failed but don't block - it can be retried
                self._persistence.mark_failed(
                    event.event_id,
                    f"Handler error during dispatch"
                )

        # Mark event as processed for deduplication (#S7)
        if dispatch_success:
            self._mark_processed(event.event_id)

    def get_event_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get recent event history for audit."""
        history = self._event_history

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        return history[-limit:]

    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._running

    @property
    def backpressure_level(self) -> BackpressureLevel:
        """Current backpressure level."""
        return self._metrics.backpressure_level

    @property
    def metrics(self) -> BackpressureMetrics:
        """Get current backpressure metrics."""
        self._metrics.current_queue_size = self._queue.qsize()
        return self._metrics

    def get_status(self) -> dict:
        """Get comprehensive event bus status for monitoring."""
        barrier_info = {
            "active": self._current_barrier is not None,
            "barrier_id": self._current_barrier.barrier_id if self._current_barrier else None,
            "is_closed": self._current_barrier.is_closed() if self._current_barrier else None,
        }

        # Get persistence statistics if enabled (#S4)
        persistence_info = None
        if self._persistence:
            persistence_info = self._persistence.get_statistics()

        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self._backpressure.max_queue_size,
            "queue_utilization_pct": (self._queue.qsize() / self._backpressure.max_queue_size) * 100,
            "backpressure_level": self._metrics.backpressure_level.value,
            "metrics": self._metrics.to_dict(),
            "signal_agents": list(self._signal_agents),
            "subscriber_count": {
                event_type.value: len(handlers)
                for event_type, handlers in self._subscribers.items()
            },
            "barrier": barrier_info,
            "event_history_size": len(self._event_history),
            "persistence_enabled": self._enable_persistence,
            "persistence": persistence_info,
            "deduplication": {
                "enabled": self._enable_deduplication,
                "ids_tracked": self._dedup_stats["ids_tracked"],
                "duplicates_blocked": self._dedup_stats["duplicates_blocked"],
                "window_seconds": self._dedup_window_seconds,
            },
        }

    @property
    def persistence(self) -> "EventPersistence | None":
        """Get the persistence layer (if enabled)."""
        return self._persistence

    def reset_metrics(self) -> None:
        """Reset backpressure metrics (useful for testing or periodic reset)."""
        self._metrics = BackpressureMetrics()
        self._events_in_window = []
        logger.info("Event bus metrics reset")

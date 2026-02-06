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
- Health check with automatic recovery (#SOLID refactoring)
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Coroutine, Any, TYPE_CHECKING

from core.events import Event, EventType, SignalEvent

if TYPE_CHECKING:
    from core.event_persistence import EventPersistence, PersistenceConfig
    from core.immutable_ledger import ImmutableAuditLedger


logger = logging.getLogger(__name__)


# ============================================================================
# Agent Criticality and Quorum Configuration
# ============================================================================

class AgentCriticality(Enum):
    """
    Agent criticality levels for quorum-based barrier.

    CRITICAL agents MUST respond before barrier release.
    HIGH agents are preferred but not strictly required.
    NORMAL agents contribute to quorum but can be skipped.
    """
    CRITICAL = "critical"  # Must respond: MacroAgent, RiskAgent
    HIGH = "high"          # MomentumAgent, StatArbAgent, MACDvAgent
    NORMAL = "normal"      # Rest


@dataclass
class QuorumConfig:
    """Configuration for quorum-based signal barrier."""
    threshold: float = 0.8  # 80% of agents must respond for quorum
    fast_path_timeout_ms: int = 500  # Fast path - release if quorum met (was 100ms - too short)
    full_timeout_seconds: float = 10.0  # Full barrier timeout
    require_critical: bool = True  # All CRITICAL agents must respond
    confidence_decay_per_second: float = 0.05  # Decay rate for fallback signals


@dataclass
class AgentTimeoutConfig:
    """Per-agent timeout configuration."""
    default_timeout_ms: int = 1000  # 1 second default (was 200ms - too short for HMM/Kalman)
    agent_timeouts_ms: dict[str, int] = field(default_factory=dict)

    def get_timeout_ms(self, agent_name: str) -> int:
        """Get timeout for specific agent."""
        return self.agent_timeouts_ms.get(agent_name, self.default_timeout_ms)


@dataclass
class BarrierResult:
    """
    Result of a signal barrier wait (Phase 12 fix).

    Provides barrier validity info alongside signals so the CIO can make
    an informed decision about whether to proceed.

    Per Architecture Invariant #2:
    'Barrier MUST NOT release if ANY CRITICAL agent is missing'
    """
    signals: dict[str, SignalEvent]
    is_valid: bool = True  # False if CRITICAL agents are missing
    missing_critical: list[str] = field(default_factory=list)
    quorum_met: bool = True  # Whether quorum threshold was reached
    total_expected: int = 0  # Total agents expected
    total_received: int = 0  # Total agents that responded
    missing_agents: list[str] = field(default_factory=list)  # All missing agents


# Default agent criticality mappings
DEFAULT_AGENT_CRITICALITY: dict[str, AgentCriticality] = {
    "MacroAgent": AgentCriticality.HIGH,
    "RiskAgent": AgentCriticality.CRITICAL,
    "MomentumAgent": AgentCriticality.HIGH,
    "StatArbAgent": AgentCriticality.HIGH,
    "MACDvAgent": AgentCriticality.HIGH,
    "TTMSqueezeAgent": AgentCriticality.HIGH,
    "IndexSpreadAgent": AgentCriticality.NORMAL,
    "SessionAgent": AgentCriticality.NORMAL,
    "EventDrivenAgent": AgentCriticality.NORMAL,
    "MeanReversionAgent": AgentCriticality.NORMAL,
    "MarketMakingAgent": AgentCriticality.NORMAL,
    "SentimentAgent": AgentCriticality.NORMAL,
    "ChartAnalysisAgent": AgentCriticality.NORMAL,
    "ForecastingAgent": AgentCriticality.NORMAL,
}


# ============================================================================
# Health Check Configuration and Status
# ============================================================================

class EventBusHealthStatus(Enum):
    """Health status levels for EventBus."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass
class HealthCheckConfig:
    """Configuration for EventBus health checks."""
    enabled: bool = True
    check_interval_seconds: float = 10.0
    max_processing_latency_ms: float = 1000.0  # Max acceptable latency
    max_queue_stall_seconds: float = 30.0  # Max time without processing
    max_consecutive_errors: int = 5
    recovery_attempts: int = 3
    recovery_delay_seconds: float = 1.0


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: EventBusHealthStatus
    timestamp: datetime
    latency_ms: float
    queue_size: int
    last_event_processed: datetime | None
    consecutive_errors: int
    message: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": round(self.latency_ms, 2),
            "queue_size": self.queue_size,
            "last_event_processed": self.last_event_processed.isoformat() if self.last_event_processed else None,
            "consecutive_errors": self.consecutive_errors,
            "message": self.message,
        }


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


# ============================================================================
# P2: Dead Letter Queue for Failed Events
# ============================================================================

@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue for failed events."""
    event: Event
    failure_time: datetime
    failure_reason: str
    retry_count: int = 0
    max_retries: int = 3
    last_retry_time: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event.event_id,
            "event_type": self.event.event_type.value,
            "failure_time": self.failure_time.isoformat(),
            "failure_reason": self.failure_reason,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_retry_time": self.last_retry_time.isoformat() if self.last_retry_time else None,
        }


@dataclass
class DeadLetterQueueConfig:
    """Configuration for dead letter queue."""
    max_size: int = 1000  # Maximum entries to keep
    max_retries: int = 3  # Max retry attempts before permanent failure
    retry_delay_seconds: float = 60.0  # Delay between retries
    cleanup_after_hours: int = 24  # Remove entries older than this


# ============================================================================
# P2: Event Compression for Large Payloads
# ============================================================================

@dataclass
class CompressionConfig:
    """Configuration for event payload compression."""
    enabled: bool = True
    min_size_bytes: int = 1024  # Only compress payloads larger than this
    compression_level: int = 6  # gzip compression level (1-9)


@dataclass
class CompressionStats:
    """Statistics for event compression."""
    events_compressed: int = 0
    events_uncompressed: int = 0
    bytes_before_compression: int = 0
    bytes_after_compression: int = 0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.bytes_before_compression == 0:
            return 0.0
        return 1 - (self.bytes_after_compression / self.bytes_before_compression)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "events_compressed": self.events_compressed,
            "events_uncompressed": self.events_uncompressed,
            "bytes_before_compression": self.bytes_before_compression,
            "bytes_after_compression": self.bytes_after_compression,
            "compression_ratio": round(self.compression_ratio, 3),
            "bytes_saved": self.bytes_before_compression - self.bytes_after_compression,
        }


# ============================================================================
# P2: Event Replay Configuration
# ============================================================================

@dataclass
class ReplayConfig:
    """Configuration for event replay capability."""
    enabled: bool = True
    max_replay_events: int = 10000  # Maximum events to keep for replay
    replay_speed_multiplier: float = 1.0  # 1.0 = real-time, 2.0 = double speed


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
    Synchronization barrier for signal aggregation with quorum support.

    Collects signals from all strategy agents before CIO decision.
    Implements fan-in pattern with timeout and quorum-based early release.

    Features:
    - Quorum-based fast path (100ms): Release when 80% of agents respond
    - Agent criticality: CRITICAL agents must always respond
    - Per-agent timeouts: Different timeouts for different agent types
    - Fallback signals: Use previous signals with decayed confidence

    Thread-safety: Uses internal lock to prevent race conditions during
    rapid signal arrival (fixes #S2).
    """
    expected_agents: set[str]
    timeout_seconds: float
    barrier_id: int = 0  # Unique ID to track barrier versions
    signals: dict[str, SignalEvent] = field(default_factory=dict)
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _quorum_event: asyncio.Event = field(default_factory=asyncio.Event)
    _created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _is_closed: bool = False  # Prevent late signals after barrier is consumed

    # Quorum configuration
    quorum_config: QuorumConfig = field(default_factory=QuorumConfig)
    agent_criticality: dict[str, AgentCriticality] = field(default_factory=dict)

    # Fallback signals cache (last known good signals)
    _last_known_signals: dict[str, SignalEvent] = field(default_factory=dict)

    # Track missing critical agents after barrier close
    _missing_critical_agents: list[str] = field(default_factory=list)

    # Phase 12: Full barrier result with validity info
    _barrier_result: BarrierResult | None = field(default=None)

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
            # Also store as last known for future fallback
            self._last_known_signals[agent_name] = signal

            if self._is_complete_unsafe():
                self._event.set()
                self._quorum_event.set()
                return True

            # Check if quorum is met (triggers fast path)
            if self._is_quorum_met_unsafe():
                self._quorum_event.set()

        return False

    def _is_complete_unsafe(self) -> bool:
        """Check if all expected agents have reported (no lock, internal use)."""
        return set(self.signals.keys()) >= self.expected_agents

    def _is_quorum_met_unsafe(self) -> bool:
        """
        Check if quorum is met (no lock, internal use).

        Quorum conditions:
        1. All CRITICAL agents have responded
        2. At least quorum_threshold (80%) of agents have responded
        """
        received = set(self.signals.keys())
        total_expected = len(self.expected_agents)

        if total_expected == 0:
            return True

        # Check if all CRITICAL agents have responded
        if self.quorum_config.require_critical:
            for agent_name in self.expected_agents:
                criticality = self.agent_criticality.get(
                    agent_name,
                    DEFAULT_AGENT_CRITICALITY.get(agent_name, AgentCriticality.NORMAL)
                )
                if criticality == AgentCriticality.CRITICAL and agent_name not in received:
                    return False

        # Check quorum threshold
        quorum_count = int(total_expected * self.quorum_config.threshold)
        return len(received) >= quorum_count

    async def is_complete(self) -> bool:
        """Check if all expected agents have reported (thread-safe)."""
        async with self._lock:
            return self._is_complete_unsafe()

    async def is_quorum_met(self) -> bool:
        """Check if quorum threshold is met (thread-safe)."""
        async with self._lock:
            return self._is_quorum_met_unsafe()

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

    async def wait_with_quorum(
        self,
        fallback_signals: dict[str, SignalEvent] | None = None,
    ) -> dict[str, SignalEvent]:
        """
        Wait with quorum-based fast path (thread-safe).

        Two-phase waiting:
        1. Fast path (100ms): Wait for quorum (80% + all CRITICAL)
        2. Full timeout: Wait for remaining agents or timeout

        Missing agents get fallback signals with decayed confidence.

        Args:
            fallback_signals: Previous cycle's signals for timeout fallback

        Returns:
            Complete signal dict (may include fallback signals)
        """
        fast_path_timeout = self.quorum_config.fast_path_timeout_ms / 1000.0

        # Phase 1: Fast path - wait for quorum
        try:
            await asyncio.wait_for(
                self._quorum_event.wait(),
                timeout=fast_path_timeout
            )
            # Quorum met on fast path
            async with self._lock:
                if self._is_complete_unsafe():
                    # All agents responded
                    logger.debug(
                        f"Barrier {self.barrier_id}: All {len(self.signals)} agents "
                        f"responded within fast path"
                    )
                else:
                    # Quorum met but not complete - log and continue with partial
                    missing = self.expected_agents - set(self.signals.keys())
                    logger.debug(
                        f"Barrier {self.barrier_id}: Quorum met in fast path "
                        f"({len(self.signals)}/{len(self.expected_agents)}). "
                        f"Missing: {missing}"
                    )
        except asyncio.TimeoutError:
            # Fast path failed - continue to full timeout
            logger.debug(
                f"Barrier {self.barrier_id}: Fast path timeout, "
                f"waiting full {self.timeout_seconds}s"
            )

            # Phase 2: Wait for remaining time
            remaining_timeout = self.timeout_seconds - fast_path_timeout
            if remaining_timeout > 0:
                try:
                    await asyncio.wait_for(
                        self._event.wait(),
                        timeout=remaining_timeout
                    )
                except asyncio.TimeoutError:
                    async with self._lock:
                        missing = self.expected_agents - set(self.signals.keys())
                    logger.warning(
                        f"Signal barrier {self.barrier_id} full timeout. "
                        f"Received {len(self.signals)}/{len(self.expected_agents)} signals. "
                        f"Missing: {missing}"
                    )

        # Close barrier and prepare final signals
        async with self._lock:
            self._is_closed = True
            final_signals = dict(self.signals)

            # Check for missing CRITICAL agents - NO FALLBACK ALLOWED
            # INVARIANT: If a CRITICAL agent is missing, barrier is INVALID
            missing_agents = self.expected_agents - set(final_signals.keys())
            missing_critical = []

            for agent_name in missing_agents:
                criticality = self.agent_criticality.get(
                    agent_name,
                    DEFAULT_AGENT_CRITICALITY.get(agent_name, AgentCriticality.NORMAL)
                )

                if criticality == AgentCriticality.CRITICAL:
                    # CRITICAL agent missing - NO FALLBACK, mark as barrier failure
                    missing_critical.append(agent_name)
                    logger.error(
                        f"BARRIER INVALID: CRITICAL agent {agent_name} did not respond. "
                        f"No fallback allowed for CRITICAL agents."
                    )
                else:
                    # Non-critical agent - fallback allowed with decayed confidence
                    fallback = self._get_fallback_signal(
                        agent_name,
                        fallback_signals or self._last_known_signals
                    )
                    if fallback:
                        final_signals[agent_name] = fallback
                        logger.debug(
                            f"Using fallback signal for {agent_name} "
                            f"(confidence: {fallback.confidence:.2f})"
                        )

            # Store missing critical for caller to check
            self._missing_critical_agents = missing_critical

            if missing_critical:
                logger.critical(
                    f"BARRIER FAILURE: {len(missing_critical)} CRITICAL agents missing: "
                    f"{missing_critical}. Decision should NOT proceed."
                )

            # Build BarrierResult with full validity info
            quorum_met = self._is_quorum_met_unsafe()
            self._barrier_result = BarrierResult(
                signals=final_signals,
                is_valid=len(missing_critical) == 0,
                missing_critical=missing_critical,
                quorum_met=quorum_met,
                total_expected=len(self.expected_agents),
                total_received=len(self.signals),  # Only real signals, not fallbacks
                missing_agents=list(missing_agents),
            )

            return final_signals

    def _get_fallback_signal(
        self,
        agent_name: str,
        fallback_cache: dict[str, SignalEvent],
    ) -> SignalEvent | None:
        """
        Get a fallback signal for a missing agent.

        Returns the previous signal with decayed confidence based on age.
        """
        if agent_name not in fallback_cache:
            return None

        original = fallback_cache[agent_name]
        age_seconds = (datetime.now(timezone.utc) - original.timestamp).total_seconds()

        # Decay confidence based on age
        decay_factor = max(
            0.1,  # Minimum 10% of original confidence
            1.0 - (age_seconds * self.quorum_config.confidence_decay_per_second)
        )
        decayed_confidence = original.confidence * decay_factor

        # Create a copy with decayed confidence
        # Note: We create a modified copy rather than modifying the original
        from dataclasses import replace
        try:
            fallback = replace(
                original,
                confidence=decayed_confidence,
                # Mark as fallback in metadata if possible
            )
            return fallback
        except Exception:
            # If replace fails, return original (shouldn't happen)
            return original

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

    def is_valid(self) -> bool:
        """
        Check if barrier result is valid for decision-making.

        INVARIANT: A barrier is INVALID if any CRITICAL agent is missing.
        The CIO must NOT make a decision based on an invalid barrier.

        Returns:
            True if all CRITICAL agents responded, False otherwise
        """
        return len(self._missing_critical_agents) == 0

    def get_missing_critical_agents(self) -> list[str]:
        """Get list of missing CRITICAL agents (empty if barrier is valid)."""
        return list(self._missing_critical_agents)

    def get_barrier_result(self) -> BarrierResult:
        """
        Get full barrier result with validity info (Phase 12).

        Returns BarrierResult with signals, validity, missing agents, quorum status.
        Must be called AFTER wait_with_quorum() or wait() completes.
        """
        if self._barrier_result is not None:
            return self._barrier_result
        # Fallback if wait_with_quorum wasn't used
        return BarrierResult(
            signals=dict(self.signals),
            is_valid=self.is_valid(),
            missing_critical=list(self._missing_critical_agents),
            quorum_met=True,
            total_expected=len(self.expected_agents),
            total_received=len(self.signals),
        )

    def get_quorum_status(self) -> dict[str, Any]:
        """Get current quorum status for monitoring."""
        received = set(self.signals.keys())
        total = len(self.expected_agents)
        quorum_count = int(total * self.quorum_config.threshold)

        # Check critical agents
        missing_critical = []
        for agent_name in self.expected_agents:
            criticality = self.agent_criticality.get(
                agent_name,
                DEFAULT_AGENT_CRITICALITY.get(agent_name, AgentCriticality.NORMAL)
            )
            if criticality == AgentCriticality.CRITICAL and agent_name not in received:
                missing_critical.append(agent_name)

        return {
            "barrier_id": self.barrier_id,
            "received": len(received),
            "expected": total,
            "quorum_threshold": self.quorum_config.threshold,
            "quorum_count_needed": quorum_count,
            "quorum_met": len(received) >= quorum_count and not missing_critical,
            "missing_critical": missing_critical,
            "missing_all": list(self.expected_agents - received),
        }


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
        health_check_config: HealthCheckConfig | None = None,
        dead_letter_config: DeadLetterQueueConfig | None = None,
        compression_config: CompressionConfig | None = None,
        replay_config: ReplayConfig | None = None,
        quorum_config: QuorumConfig | None = None,
        agent_timeout_config: AgentTimeoutConfig | None = None,
        agent_criticality: dict[str, AgentCriticality] | None = None,
        audit_ledger: "ImmutableAuditLedger | None" = None,
    ):
        self._backpressure = backpressure_config or BackpressureConfig(max_queue_size=max_queue_size)

        # Quorum and per-agent timeout configuration
        self._quorum_config = quorum_config or QuorumConfig()
        self._agent_timeout_config = agent_timeout_config or AgentTimeoutConfig()
        self._agent_criticality = agent_criticality or dict(DEFAULT_AGENT_CRITICALITY)

        # Last known signals for fallback (per agent)
        self._last_known_signals: dict[str, SignalEvent] = {}
        self._subscribers: dict[EventType, list[Callable[[Event], Coroutine[Any, Any, None]]]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._backpressure.max_queue_size)
        self._signal_timeout = signal_timeout
        self._barrier_timeout = barrier_timeout
        self._running = False
        self._current_barrier: SignalBarrier | None = None
        self._signal_agents: set[str] = set()
        # Event history for audit trail - uses deque with maxlen for O(1) append
        # and automatic eviction of oldest events when limit is reached.
        # Memory usage is approximately: max_history * avg_event_size (typically ~1KB each)
        # At 10000 events, this is roughly 10MB of memory for the history buffer.
        self._max_history = 10000
        self._event_history: deque[Event] = deque(maxlen=self._max_history)
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

        # P0-5: Barrier ID counter - initialized properly (fixes race condition)
        self._barrier_id_counter: int = 0

        # P0-5: Handler cleanup tracking - detect dead handlers (fixes memory leak)
        self._handler_call_count: dict[int, int] = {}  # handler_id -> call count
        self._handler_last_call: dict[int, datetime] = {}  # handler_id -> last call time
        self._handler_cleanup_interval = 300.0  # 5 minutes
        self._last_handler_cleanup = datetime.now(timezone.utc)

        # Health check state (#SOLID refactoring - addresses SPOF)
        self._health_config = health_check_config or HealthCheckConfig()
        self._health_status = EventBusHealthStatus.HEALTHY
        self._last_event_processed: datetime | None = None
        self._consecutive_errors = 0
        self._health_check_task: asyncio.Task | None = None
        self._recovery_in_progress = False
        self._health_callbacks: list[Callable[[HealthCheckResult], None]] = []

        # P2: Dead letter queue for failed events
        self._dlq_config = dead_letter_config or DeadLetterQueueConfig()
        self._dead_letter_queue: deque[DeadLetterEntry] = deque(maxlen=self._dlq_config.max_size)
        self._dlq_callbacks: list[Callable[[DeadLetterEntry], None]] = []

        # P2: Event compression for large payloads
        self._compression_config = compression_config or CompressionConfig()
        self._compression_stats = CompressionStats()

        # P2: Event replay capability for debugging
        self._replay_config = replay_config or ReplayConfig()
        self._replay_buffer: deque[tuple[datetime, Event]] = deque(
            maxlen=self._replay_config.max_replay_events
        )
        self._replay_in_progress = False
        self._replay_task: asyncio.Task | None = None

        # MiFID II Compliance: Immutable audit ledger integration
        self._audit_ledger = audit_ledger
        if audit_ledger:
            logger.info("Audit ledger connected - all events will be recorded for compliance")

    def register_signal_agent(self, agent_name: str) -> None:
        """Register an agent as a signal producer."""
        self._signal_agents.add(agent_name)
        logger.info(f"Registered signal agent: {agent_name}")

    def set_audit_ledger(self, ledger: "ImmutableAuditLedger") -> None:
        """
        Set the audit ledger for MiFID II compliance (dependency injection).

        Can be called after construction to wire up the ledger.
        """
        self._audit_ledger = ledger
        if ledger:
            logger.info("Audit ledger connected - all events will be recorded for compliance")

    def get_audit_ledger(self) -> "ImmutableAuditLedger | None":
        """Get the audit ledger if configured."""
        return self._audit_ledger

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
            # P0-5: Clean up handler tracking when unsubscribed
            handler_id = id(handler)
            self._handler_call_count.pop(handler_id, None)
            self._handler_last_call.pop(handler_id, None)

    def cleanup_dead_handlers(self, max_idle_seconds: float = 600.0) -> int:
        """
        Remove handlers that haven't been called in a long time (P0-5 memory leak fix).

        This prevents memory leaks from handlers that were subscribed but whose
        owning objects were garbage collected, leaving orphaned handler references.

        Args:
            max_idle_seconds: Remove handlers idle longer than this (default 10 min)

        Returns:
            Number of handlers removed
        """
        now = datetime.now(timezone.utc)
        removed_count = 0
        cutoff = now - timedelta(seconds=max_idle_seconds)

        # Find handlers to remove (can't modify dict while iterating)
        handlers_to_check = list(self._handler_last_call.items())

        for handler_id, last_call in handlers_to_check:
            if last_call < cutoff:
                # Find and remove from all event type subscriptions
                for event_type, handlers in self._subscribers.items():
                    for handler in list(handlers):
                        if id(handler) == handler_id:
                            handlers.remove(handler)
                            removed_count += 1
                            logger.debug(
                                f"Removed idle handler {handler_id} from {event_type.value}"
                            )

                # Clean up tracking
                self._handler_call_count.pop(handler_id, None)
                self._handler_last_call.pop(handler_id, None)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} idle handlers")

        return removed_count

    def get_handler_stats(self) -> dict:
        """Get statistics about registered handlers (for monitoring)."""
        total_handlers = sum(len(handlers) for handlers in self._subscribers.values())
        return {
            "total_handlers": total_handlers,
            "handlers_by_type": {
                event_type.value: len(handlers)
                for event_type, handlers in self._subscribers.items()
                if handlers
            },
            "tracked_handlers": len(self._handler_call_count),
        }

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
            async with self._lock:
                self._dedup_stats["duplicates_blocked"] += 1
            logger.debug(f"Duplicate event blocked: {event.event_id} ({event.event_type.value})")
            return False

        # Update metrics (protected by lock to prevent race conditions - CONC-002)
        async with self._lock:
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

        # Track event history for audit and update max queue size (protected by lock)
        # deque with maxlen handles automatic eviction - O(1) append
        async with self._lock:
            # Track max queue size (atomic check-and-set to prevent race condition - CONC-002)
            current_size = self._queue.qsize()
            if current_size > self._metrics.max_queue_size_reached:
                self._metrics.max_queue_size_reached = current_size
            self._event_history.append(event)

        # MiFID II Compliance: Log to immutable audit ledger
        if self._audit_ledger:
            try:
                # Convert event to dict for ledger storage
                event_data = {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now(timezone.utc).isoformat(),
                    "symbol": getattr(event, 'symbol', None),
                }
                # Add type-specific fields
                if hasattr(event, 'direction'):
                    event_data["direction"] = event.direction.value if hasattr(event.direction, 'value') else str(event.direction)
                if hasattr(event, 'confidence'):
                    event_data["confidence"] = event.confidence
                if hasattr(event, 'quantity'):
                    event_data["quantity"] = event.quantity
                if hasattr(event, 'price'):
                    event_data["price"] = event.price

                source_agent = getattr(event, 'source_agent', 'EventBus')
                self._audit_ledger.append(
                    event_type=event.event_type.value,
                    source_agent=source_agent,
                    event_data=event_data,
                )
            except Exception as e:
                # Never block event processing for audit logging failure
                logger.warning(f"Audit ledger logging failed: {e}")

        # P2: Record for replay capability
        self._record_for_replay(event)

        # Persist event for crash recovery (#S4)
        if self._persistence:
            try:
                await self._persistence.persist_event_async(event, priority=is_high_priority)
            except (IOError, OSError) as e:
                # Storage I/O errors - log with context but don't block
                logger.warning(f"Event persistence I/O error for {event.event_type}: {e}")
            except Exception as e:
                # Log error with full traceback but don't block event publishing
                logger.exception(f"Event persistence failed unexpectedly for {event.event_type}")

        return True

    def _calculate_backpressure_level(self) -> BackpressureLevel:
        """Calculate current backpressure level based on queue depth."""
        # Guard against division by zero
        if self._backpressure.max_queue_size == 0:
            return BackpressureLevel.NORMAL
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
            except TypeError as e:
                logger.error(f"Backpressure callback signature error: {e}")
            except Exception as e:
                logger.exception("Backpressure callback failed unexpectedly")

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
        Now supports quorum-based early release.
        """
        await self.publish(signal)

        # Store as last known signal for fallback
        self._last_known_signals[signal.source_agent] = signal

        async with self._lock:
            # If barrier is closed or doesn't exist, create new one
            if self._current_barrier is None or self._current_barrier.is_closed():
                # P0-5: Counter is now properly initialized in __init__ (no getattr hack)
                self._barrier_id_counter += 1
                self._current_barrier = SignalBarrier(
                    expected_agents=self._signal_agents.copy(),
                    timeout_seconds=self._barrier_timeout,
                    barrier_id=self._barrier_id_counter,
                    quorum_config=self._quorum_config,
                    agent_criticality=self._agent_criticality,
                    _last_known_signals=dict(self._last_known_signals),
                )
                logger.debug(f"Created new signal barrier {self._barrier_id_counter}")

            barrier = self._current_barrier

        # Add signal outside the event bus lock (barrier has its own lock)
        await barrier.add_signal(signal.source_agent, signal)

    async def wait_for_signals(self, use_quorum: bool = True) -> BarrierResult:
        """
        Wait for signal barrier to complete (fan-in, race-condition safe).

        Called by CIO agent before making decisions.
        The barrier is atomically consumed and reset to prevent race conditions
        where late signals go to the wrong barrier.

        Phase 12 fix: Returns BarrierResult with validity info instead of just signals.
        The CIO MUST check result.is_valid before making decisions
        (Architecture Invariant #2).

        Args:
            use_quorum: If True, use quorum-based fast path (default: True)

        Returns:
            BarrierResult with signals, validity status, and missing agent info.
            Empty BarrierResult if no barrier is active.
        """
        async with self._lock:
            if self._current_barrier is None:
                return BarrierResult(signals={})

            # Take ownership of the barrier - no one else can use it
            barrier = self._current_barrier
            barrier_id = barrier.barrier_id

            # Don't reset the barrier yet - late signals need to know this barrier is done

        logger.debug(f"CIO waiting for signal barrier {barrier_id}")

        # Wait outside the lock so signals can still be added
        if use_quorum:
            signals = await barrier.wait_with_quorum(
                fallback_signals=self._last_known_signals
            )
        else:
            signals = await barrier.wait()

        # Get full barrier result with validity info
        result = barrier.get_barrier_result()

        # Update last known signals cache
        for agent_name, signal in signals.items():
            self._last_known_signals[agent_name] = signal

        # Phase 12: Emit RiskAlertEvent if barrier is invalid (critical agents missing)
        if not result.is_valid:
            await self._emit_barrier_failure_alert(result)

        # Now atomically reset the barrier
        async with self._lock:
            # Only reset if it's still the same barrier (prevents double-reset)
            if self._current_barrier is barrier:
                self._current_barrier = None
                logger.debug(f"Signal barrier {barrier_id} consumed with {len(signals)} signals")

        return result

    async def _emit_barrier_failure_alert(self, result: BarrierResult) -> None:
        """
        Emit RiskAlertEvent when barrier fails due to missing CRITICAL agents.

        Phase 12: Makes barrier failures visible in the dashboard instead of
        silently logging them.
        """
        from core.events import RiskAlertEvent, RiskAlertSeverity

        alert = RiskAlertEvent(
            source_agent="EventBus",
            severity=RiskAlertSeverity.CRITICAL,
            alert_type="barrier_failure",
            message=(
                f"Signal barrier INVALID: {len(result.missing_critical)} CRITICAL agent(s) "
                f"did not respond: {', '.join(result.missing_critical)}. "
                f"Received {result.total_received}/{result.total_expected} signals. "
                f"CIO decisions BLOCKED until agents recover."
            ),
            current_value=float(result.total_received),
            threshold_value=float(result.total_expected),
            halt_trading=False,  # CIO handles this, not kill-switch
        )

        # Publish to all subscribers (including dashboard)
        await self.publish(alert)
        logger.warning(
            f"BARRIER FAILURE ALERT emitted: missing {result.missing_critical}"
        )

    def get_quorum_config(self) -> QuorumConfig:
        """Get current quorum configuration."""
        return self._quorum_config

    def set_quorum_config(self, config: QuorumConfig) -> None:
        """Update quorum configuration."""
        self._quorum_config = config
        logger.info(
            f"Quorum config updated: threshold={config.threshold}, "
            f"fast_path={config.fast_path_timeout_ms}ms"
        )

    def set_agent_criticality(self, agent_name: str, criticality: AgentCriticality) -> None:
        """Set criticality level for an agent."""
        self._agent_criticality[agent_name] = criticality
        logger.debug(f"Agent {agent_name} criticality set to {criticality.value}")

    def get_agent_criticality(self, agent_name: str) -> AgentCriticality:
        """Get criticality level for an agent."""
        return self._agent_criticality.get(
            agent_name,
            DEFAULT_AGENT_CRITICALITY.get(agent_name, AgentCriticality.NORMAL)
        )

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
        self._health_status = EventBusHealthStatus.HEALTHY
        self._consecutive_errors = 0

        # Recover persisted events before starting (#S4)
        if self._persistence:
            await self.recover_persisted_events()

        # Start health check task if enabled
        if self._health_config.enabled:
            self._health_check_task = asyncio.create_task(self._run_health_check_loop())
            logger.info("Event bus health check enabled")

        logger.info("Event bus started")

        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                await self._dispatch(event)
                self._queue.task_done()

                # Reset error count on successful processing
                self._consecutive_errors = 0
                self._last_event_processed = datetime.now(timezone.utc)

                # P0-5: Periodic cleanup of idle handlers (memory leak prevention)
                now = datetime.now(timezone.utc)
                if (now - self._last_handler_cleanup).total_seconds() > self._handler_cleanup_interval:
                    self.cleanup_dead_handlers()
                    self._last_handler_cleanup = now

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._consecutive_errors += 1
                logger.exception(f"Error processing event: {e}")  # Full stack trace for debugging

    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        self._running = False

        # Stop health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

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

        # P0-5: Track handler calls for cleanup (memory leak prevention)
        now = datetime.now(timezone.utc)
        for handler in handlers:
            handler_id = id(handler)
            self._handler_call_count[handler_id] = self._handler_call_count.get(handler_id, 0) + 1
            self._handler_last_call[handler_id] = now

        # Execute handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Handler error for {event.event_type.value}: {result}")
                dispatch_success = False
                # P2: Add to dead letter queue for retry
                self._add_to_dead_letter_queue(event, f"Handler error: {result}")

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
        # Convert deque to list for filtering and slicing
        history = list(self._event_history)

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
            "health": {
                "status": self._health_status.value,
                "enabled": self._health_config.enabled,
                "consecutive_errors": self._consecutive_errors,
                "last_event_processed": self._last_event_processed.isoformat() if self._last_event_processed else None,
                "recovery_in_progress": self._recovery_in_progress,
            },
            # P2: Dead letter queue stats
            "dead_letter_queue": self.get_dead_letter_queue_stats(),
            # P2: Compression stats
            "compression": self.get_compression_stats(),
            # P2: Replay buffer stats
            "replay": self.get_replay_buffer_stats(),
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

    # ========================================================================
    # P2: Dead Letter Queue Methods
    # ========================================================================

    def _add_to_dead_letter_queue(self, event: Event, reason: str) -> None:
        """
        Add a failed event to the dead letter queue.

        Args:
            event: The event that failed processing
            reason: Description of why the event failed
        """
        entry = DeadLetterEntry(
            event=event,
            failure_time=datetime.now(timezone.utc),
            failure_reason=reason,
            max_retries=self._dlq_config.max_retries,
        )
        self._dead_letter_queue.append(entry)

        logger.warning(
            f"Event {event.event_id} ({event.event_type.value}) added to dead letter queue: {reason}"
        )

        # Notify callbacks
        for callback in self._dlq_callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"DLQ callback error: {e}")

    async def retry_dead_letter_events(self, max_events: int = 100) -> dict[str, int]:
        """
        Retry events from the dead letter queue.

        Args:
            max_events: Maximum number of events to retry

        Returns:
            Dict with counts of successful, failed, and skipped retries
        """
        results = {"retried": 0, "succeeded": 0, "failed": 0, "skipped": 0}
        now = datetime.now(timezone.utc)

        entries_to_retry = []
        for entry in list(self._dead_letter_queue):
            if entry.retry_count >= entry.max_retries:
                results["skipped"] += 1
                continue

            # Check retry delay
            if entry.last_retry_time:
                elapsed = (now - entry.last_retry_time).total_seconds()
                if elapsed < self._dlq_config.retry_delay_seconds:
                    continue

            entries_to_retry.append(entry)
            if len(entries_to_retry) >= max_events:
                break

        for entry in entries_to_retry:
            entry.retry_count += 1
            entry.last_retry_time = now
            results["retried"] += 1

            try:
                # Re-queue the event
                self._queue.put_nowait(entry.event)
                # Remove from DLQ on successful queue
                self._dead_letter_queue.remove(entry)
                results["succeeded"] += 1
                logger.info(f"Retried event {entry.event.event_id} from DLQ (attempt {entry.retry_count})")
            except asyncio.QueueFull:
                results["failed"] += 1
                logger.warning(f"Failed to retry event {entry.event.event_id}: queue full")
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Failed to retry event {entry.event.event_id}: {e}")

        return results

    def get_dead_letter_queue_stats(self) -> dict:
        """Get statistics about the dead letter queue."""
        now = datetime.now(timezone.utc)
        return {
            "total_entries": len(self._dead_letter_queue),
            "max_size": self._dlq_config.max_size,
            "retryable": sum(
                1 for e in self._dead_letter_queue
                if e.retry_count < e.max_retries
            ),
            "permanently_failed": sum(
                1 for e in self._dead_letter_queue
                if e.retry_count >= e.max_retries
            ),
            "event_types": dict(
                sorted(
                    ((t, sum(1 for e in self._dead_letter_queue if e.event.event_type == t))
                     for t in set(e.event.event_type for e in self._dead_letter_queue)),
                    key=lambda x: -x[1]
                )
            ) if self._dead_letter_queue else {},
        }

    def on_dead_letter(self, callback: Callable[[DeadLetterEntry], None]) -> None:
        """Register callback for dead letter events."""
        self._dlq_callbacks.append(callback)

    def cleanup_dead_letter_queue(self) -> int:
        """
        Clean up old entries from dead letter queue.

        Returns:
            Number of entries removed
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self._dlq_config.cleanup_after_hours)
        initial_size = len(self._dead_letter_queue)

        # Remove entries older than cutoff
        self._dead_letter_queue = deque(
            (e for e in self._dead_letter_queue if e.failure_time >= cutoff),
            maxlen=self._dlq_config.max_size
        )

        removed = initial_size - len(self._dead_letter_queue)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old entries from dead letter queue")
        return removed

    # ========================================================================
    # P2: Event Compression Methods
    # ========================================================================

    def _compress_event_payload(self, event: Event) -> tuple[bytes, bool]:
        """
        Compress event payload if it exceeds the threshold.

        Args:
            event: Event to potentially compress

        Returns:
            Tuple of (compressed/original bytes, was_compressed)
        """
        if not self._compression_config.enabled:
            return b"", False

        try:
            # Serialize event to JSON
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "source_agent": event.source_agent,
            }

            # Add event-specific fields
            for attr in dir(event):
                if not attr.startswith("_") and attr not in event_data:
                    try:
                        value = getattr(event, attr)
                        if not callable(value):
                            # Handle datetime
                            if isinstance(value, datetime):
                                event_data[attr] = value.isoformat()
                            elif isinstance(value, Enum):
                                event_data[attr] = value.value
                            else:
                                event_data[attr] = value
                    except Exception:
                        pass

            json_bytes = json.dumps(event_data).encode("utf-8")
            original_size = len(json_bytes)

            # Only compress if above threshold
            if original_size < self._compression_config.min_size_bytes:
                self._compression_stats.events_uncompressed += 1
                return json_bytes, False

            # Compress
            compressed = gzip.compress(
                json_bytes,
                compresslevel=self._compression_config.compression_level
            )
            compressed_size = len(compressed)

            # Update stats
            self._compression_stats.events_compressed += 1
            self._compression_stats.bytes_before_compression += original_size
            self._compression_stats.bytes_after_compression += compressed_size

            logger.debug(
                f"Compressed event {event.event_id}: {original_size} -> {compressed_size} bytes "
                f"({(1 - compressed_size/original_size)*100:.1f}% reduction)"
            )

            return compressed, True

        except Exception as e:
            logger.error(f"Failed to compress event {event.event_id}: {e}")
            return b"", False

    def _decompress_event_payload(self, data: bytes, compressed: bool) -> dict | None:
        """
        Decompress event payload if it was compressed.

        Args:
            data: Compressed or uncompressed bytes
            compressed: Whether the data is compressed

        Returns:
            Event data dict or None on failure
        """
        try:
            if compressed:
                decompressed = gzip.decompress(data)
                return json.loads(decompressed.decode("utf-8"))
            else:
                return json.loads(data.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to decompress event payload: {e}")
            return None

    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        return self._compression_stats.to_dict()

    # ========================================================================
    # P2: Event Replay Methods for Debugging
    # ========================================================================

    def _record_for_replay(self, event: Event) -> None:
        """
        Record an event for potential replay.

        Args:
            event: Event to record
        """
        if not self._replay_config.enabled:
            return

        self._replay_buffer.append((datetime.now(timezone.utc), event))

    async def replay_events(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[EventType] | None = None,
        speed_multiplier: float | None = None,
        handler: Callable[[Event], Coroutine[Any, Any, None]] | None = None,
    ) -> dict[str, int]:
        """
        Replay events from the replay buffer for debugging.

        Args:
            start_time: Start of replay window (default: beginning of buffer)
            end_time: End of replay window (default: end of buffer)
            event_types: Filter to specific event types (default: all)
            speed_multiplier: Replay speed (default: from config)
            handler: Custom handler for replayed events (default: dispatch normally)

        Returns:
            Dict with replay statistics
        """
        if self._replay_in_progress:
            logger.warning("Replay already in progress")
            return {"error": "replay_in_progress", "replayed": 0}

        self._replay_in_progress = True
        speed = speed_multiplier or self._replay_config.replay_speed_multiplier

        results = {"replayed": 0, "skipped": 0, "errors": 0}
        prev_time: datetime | None = None

        try:
            # Filter events
            events_to_replay = []
            for recorded_time, event in self._replay_buffer:
                if start_time and recorded_time < start_time:
                    continue
                if end_time and recorded_time > end_time:
                    continue
                if event_types and event.event_type not in event_types:
                    results["skipped"] += 1
                    continue
                events_to_replay.append((recorded_time, event))

            logger.info(
                f"Starting event replay: {len(events_to_replay)} events at {speed}x speed"
            )

            for recorded_time, event in events_to_replay:
                # Simulate timing if speed > 0
                if prev_time and speed > 0:
                    delay = (recorded_time - prev_time).total_seconds() / speed
                    if delay > 0:
                        await asyncio.sleep(delay)

                prev_time = recorded_time

                try:
                    if handler:
                        await handler(event)
                    else:
                        await self._dispatch(event)
                    results["replayed"] += 1
                except Exception as e:
                    logger.error(f"Replay error for event {event.event_id}: {e}")
                    results["errors"] += 1

            logger.info(
                f"Event replay complete: {results['replayed']} replayed, "
                f"{results['skipped']} skipped, {results['errors']} errors"
            )

        finally:
            self._replay_in_progress = False

        return results

    def get_replay_buffer_stats(self) -> dict:
        """Get statistics about the replay buffer."""
        if not self._replay_buffer:
            return {
                "enabled": self._replay_config.enabled,
                "total_events": 0,
                "max_events": self._replay_config.max_replay_events,
                "time_range_seconds": 0,
            }

        first_time = self._replay_buffer[0][0]
        last_time = self._replay_buffer[-1][0]

        return {
            "enabled": self._replay_config.enabled,
            "total_events": len(self._replay_buffer),
            "max_events": self._replay_config.max_replay_events,
            "time_range_seconds": (last_time - first_time).total_seconds(),
            "first_event_time": first_time.isoformat(),
            "last_event_time": last_time.isoformat(),
            "event_types": dict(
                sorted(
                    ((t.value, sum(1 for _, e in self._replay_buffer if e.event_type == t))
                     for t in set(e.event_type for _, e in self._replay_buffer)),
                    key=lambda x: -x[1]
                )
            ),
            "replay_in_progress": self._replay_in_progress,
        }

    def get_events_for_replay(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[EventType] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Get events from replay buffer without replaying them.

        Useful for inspection and debugging.

        Args:
            start_time: Start of time window
            end_time: End of time window
            event_types: Filter to specific event types
            limit: Maximum events to return

        Returns:
            List of events matching criteria
        """
        events = []
        for recorded_time, event in self._replay_buffer:
            if start_time and recorded_time < start_time:
                continue
            if end_time and recorded_time > end_time:
                continue
            if event_types and event.event_type not in event_types:
                continue
            events.append(event)
            if len(events) >= limit:
                break
        return events

    def clear_replay_buffer(self) -> int:
        """
        Clear the replay buffer.

        Returns:
            Number of events cleared
        """
        count = len(self._replay_buffer)
        self._replay_buffer.clear()
        logger.info(f"Cleared {count} events from replay buffer")
        return count

    # ========================================================================
    # Health Check Methods (#SOLID refactoring - addresses SPOF)
    # ========================================================================

    async def _run_health_check_loop(self) -> None:
        """
        Run periodic health checks in background.

        This loop monitors the EventBus health and triggers recovery
        if the bus becomes unresponsive or unhealthy.
        """
        while self._running:
            try:
                await asyncio.sleep(self._health_config.check_interval_seconds)

                if not self._running:
                    break

                result = await self.check_health()

                # Notify callbacks
                for callback in self._health_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")

                # Handle unhealthy state
                if result.status == EventBusHealthStatus.UNHEALTHY:
                    logger.warning(
                        f"EventBus health check UNHEALTHY: {result.message}"
                    )
                    if not self._recovery_in_progress:
                        await self._attempt_recovery()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def check_health(self) -> HealthCheckResult:
        """
        Perform a health check on the EventBus.

        Checks:
        1. Processing latency is within acceptable bounds
        2. Queue is not stalled (events are being processed)
        3. Consecutive error count is below threshold

        Returns:
            HealthCheckResult with status and diagnostics
        """
        now = datetime.now(timezone.utc)
        latency_ms = self._metrics.processing_latency_ms
        queue_size = self._queue.qsize()

        # Check for stalled queue
        stalled = False
        if self._last_event_processed and queue_size > 0:
            seconds_since_last = (now - self._last_event_processed).total_seconds()
            if seconds_since_last > self._health_config.max_queue_stall_seconds:
                stalled = True

        # Determine status
        status = EventBusHealthStatus.HEALTHY
        message = "OK"

        if self._recovery_in_progress:
            status = EventBusHealthStatus.RECOVERING
            message = "Recovery in progress"

        elif self._consecutive_errors >= self._health_config.max_consecutive_errors:
            status = EventBusHealthStatus.UNHEALTHY
            message = f"Too many consecutive errors: {self._consecutive_errors}"

        elif stalled:
            status = EventBusHealthStatus.UNHEALTHY
            message = f"Queue stalled with {queue_size} events pending"

        elif latency_ms > self._health_config.max_processing_latency_ms:
            status = EventBusHealthStatus.DEGRADED
            message = f"High latency: {latency_ms:.1f}ms"

        elif self._metrics.backpressure_level in (BackpressureLevel.HIGH, BackpressureLevel.CRITICAL):
            status = EventBusHealthStatus.DEGRADED
            message = f"Backpressure: {self._metrics.backpressure_level.value}"

        self._health_status = status

        return HealthCheckResult(
            status=status,
            timestamp=now,
            latency_ms=latency_ms,
            queue_size=queue_size,
            last_event_processed=self._last_event_processed,
            consecutive_errors=self._consecutive_errors,
            message=message,
        )

    async def _attempt_recovery(self) -> bool:
        """
        Attempt to recover an unhealthy EventBus.

        Recovery steps:
        1. Clear any stuck events from the queue (preserving high-priority)
        2. Reset error counters
        3. Optionally restart processing

        Returns:
            True if recovery succeeded, False otherwise
        """
        if self._recovery_in_progress:
            return False

        self._recovery_in_progress = True
        self._health_status = EventBusHealthStatus.RECOVERING
        logger.warning("EventBus recovery initiated")

        try:
            for attempt in range(self._health_config.recovery_attempts):
                logger.info(f"Recovery attempt {attempt + 1}/{self._health_config.recovery_attempts}")

                # Step 1: Process high-priority events first
                high_priority_events = []
                low_priority_events = []

                # Drain queue and categorize
                while not self._queue.empty():
                    try:
                        event = self._queue.get_nowait()
                        if event.event_type in HIGH_PRIORITY_EVENT_TYPES:
                            high_priority_events.append(event)
                        else:
                            low_priority_events.append(event)
                        self._queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                # Re-queue high-priority events
                for event in high_priority_events:
                    try:
                        self._queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.error(f"Cannot re-queue high-priority event {event.event_id}")

                # Re-queue some low-priority events (drop oldest if too many)
                max_requeue = self._backpressure.max_queue_size // 2
                for event in low_priority_events[-max_requeue:]:
                    try:
                        self._queue.put_nowait(event)
                    except asyncio.QueueFull:
                        break

                dropped_count = max(0, len(low_priority_events) - max_requeue)
                if dropped_count > 0:
                    logger.warning(f"Dropped {dropped_count} low-priority events during recovery")

                # Step 2: Reset error counters
                self._consecutive_errors = 0

                # Step 3: Wait and check if healthy
                await asyncio.sleep(self._health_config.recovery_delay_seconds)

                result = await self.check_health()
                if result.status in (EventBusHealthStatus.HEALTHY, EventBusHealthStatus.DEGRADED):
                    logger.info("EventBus recovery successful")
                    self._recovery_in_progress = False
                    self._health_status = result.status
                    return True

            logger.error("EventBus recovery failed after all attempts")
            self._recovery_in_progress = False
            return False

        except Exception as e:
            logger.error(f"Recovery error: {e}")
            self._recovery_in_progress = False
            return False

    def on_health_change(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """
        Register callback for health status changes.

        Args:
            callback: Function to call with HealthCheckResult on each check
        """
        self._health_callbacks.append(callback)

    @property
    def health_status(self) -> EventBusHealthStatus:
        """Get current health status."""
        return self._health_status

    @property
    def is_healthy(self) -> bool:
        """Check if EventBus is in healthy state."""
        return self._health_status in (
            EventBusHealthStatus.HEALTHY,
            EventBusHealthStatus.DEGRADED
        )


# ============================================================================
# EventMultiplexer - Filter-based subscriptions and source multiplexing
# ============================================================================

EventFilter = Callable[[Event], bool]
EventTransformer = Callable[[Event], Event]


@dataclass
class FilteredSubscription:
    """A subscription with an optional filter."""
    handler: Callable[[Event], Coroutine[Any, Any, None]]
    event_filter: EventFilter | None = None
    transformer: EventTransformer | None = None
    subscription_id: str = field(default_factory=lambda: f"sub_{datetime.now(timezone.utc).timestamp()}")


@dataclass
class EventSource:
    """Configuration for an event source."""
    name: str
    source_id: str
    priority: int = 0  # Higher = more preferred
    enabled: bool = True
    last_event_time: datetime | None = None
    event_count: int = 0


class EventMultiplexer:
    """
    Event Multiplexer for advanced event routing.

    Inspired by basana's event dispatcher pattern.

    Features:
    - Filter-based subscriptions (subscribe with predicate)
    - Multiple event source multiplexing (combine feeds)
    - Event transformation pipeline
    - Symbol-based routing shortcuts

    Usage:
        multiplexer = EventMultiplexer(event_bus)

        # Subscribe with filter
        multiplexer.subscribe_with_filter(
            EventType.SIGNAL,
            my_handler,
            filter_fn=lambda e: e.symbol == "AAPL"
        )

        # Subscribe to specific symbol
        multiplexer.subscribe_symbol("AAPL", my_handler)

        # Add event sources
        multiplexer.register_source("ib_primary", priority=10)
        multiplexer.register_source("ib_backup", priority=5)
    """

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._filtered_subscriptions: dict[EventType, list[FilteredSubscription]] = defaultdict(list)
        self._symbol_handlers: dict[str, list[Callable[[Event], Coroutine[Any, Any, None]]]] = defaultdict(list)
        self._sources: dict[str, EventSource] = {}
        self._transformers: list[EventTransformer] = []
        self._lock = asyncio.Lock()

        # Metrics
        self._events_routed = 0
        self._events_filtered = 0
        self._events_transformed = 0

        # Internal handler registration
        self._registered_event_types: set[EventType] = set()

    def subscribe_with_filter(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine[Any, Any, None]],
        filter_fn: EventFilter | None = None,
        transformer: EventTransformer | None = None,
    ) -> str:
        """
        Subscribe with an optional filter function.

        Args:
            event_type: Type of events to receive
            handler: Async handler function
            filter_fn: Optional predicate - handler only called if filter returns True
            transformer: Optional transformer - transform event before delivery

        Returns:
            Subscription ID for unsubscribe

        Example:
            # Only receive SIGNAL events for AAPL with high confidence
            sub_id = multiplexer.subscribe_with_filter(
                EventType.SIGNAL,
                my_handler,
                filter_fn=lambda e: e.symbol == "AAPL" and e.confidence > 0.8
            )
        """
        subscription = FilteredSubscription(
            handler=handler,
            event_filter=filter_fn,
            transformer=transformer,
        )
        self._filtered_subscriptions[event_type].append(subscription)

        # Register internal dispatcher with EventBus if not already done
        if event_type not in self._registered_event_types:
            self._event_bus.subscribe(event_type, self._create_dispatcher(event_type))
            self._registered_event_types.add(event_type)

        logger.debug(f"Added filtered subscription {subscription.subscription_id} for {event_type.value}")
        return subscription.subscription_id

    def subscribe_symbol(
        self,
        symbol: str,
        handler: Callable[[Event], Coroutine[Any, Any, None]],
        event_types: list[EventType] | None = None,
    ) -> list[str]:
        """
        Subscribe to all events for a specific symbol.

        Args:
            symbol: Symbol to filter for (e.g., "AAPL", "ES")
            handler: Async handler function
            event_types: Optional list of event types (defaults to SIGNAL, MARKET_DATA)

        Returns:
            List of subscription IDs

        Example:
            # Receive all AAPL signals and market data
            multiplexer.subscribe_symbol("AAPL", my_handler)
        """
        if event_types is None:
            event_types = [EventType.SIGNAL, EventType.MARKET_DATA]

        def symbol_filter(event: Event) -> bool:
            return hasattr(event, 'symbol') and event.symbol == symbol

        subscription_ids = []
        for event_type in event_types:
            sub_id = self.subscribe_with_filter(event_type, handler, filter_fn=symbol_filter)
            subscription_ids.append(sub_id)

        self._symbol_handlers[symbol].append(handler)
        logger.info(f"Subscribed to symbol {symbol} for events: {[et.value for et in event_types]}")
        return subscription_ids

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe by subscription ID.

        Args:
            subscription_id: ID returned from subscribe_with_filter

        Returns:
            True if found and removed, False otherwise
        """
        for event_type, subs in self._filtered_subscriptions.items():
            for sub in subs:
                if sub.subscription_id == subscription_id:
                    subs.remove(sub)
                    logger.debug(f"Removed subscription {subscription_id}")
                    return True
        return False

    def _create_dispatcher(self, event_type: EventType) -> Callable[[Event], Coroutine[Any, Any, None]]:
        """Create an internal dispatcher for an event type."""

        async def dispatch(event: Event) -> None:
            subscriptions = self._filtered_subscriptions.get(event_type, [])
            if not subscriptions:
                return

            # Apply global transformers
            transformed_event = event
            for transformer in self._transformers:
                try:
                    transformed_event = transformer(transformed_event)
                    self._events_transformed += 1
                except Exception as e:
                    logger.exception(f"Transformer error: {e}")

            # Dispatch to filtered subscriptions
            tasks = []
            for sub in subscriptions:
                # Check filter
                if sub.event_filter is not None:
                    try:
                        if not sub.event_filter(transformed_event):
                            self._events_filtered += 1
                            continue
                    except Exception as e:
                        logger.exception(f"Filter error for {sub.subscription_id}: {e}")
                        continue

                # Apply subscription-specific transformer
                final_event = transformed_event
                if sub.transformer is not None:
                    try:
                        final_event = sub.transformer(final_event)
                    except Exception as e:
                        logger.exception(f"Subscription transformer error: {e}")
                        continue

                tasks.append(sub.handler(final_event))

            # Execute handlers concurrently
            if tasks:
                self._events_routed += len(tasks)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.exception(f"Handler error: {result}")

        return dispatch

    def add_global_transformer(self, transformer: EventTransformer) -> None:
        """
        Add a global event transformer applied to all events.

        Args:
            transformer: Function that transforms Event -> Event

        Example:
            # Add timestamp normalization
            multiplexer.add_global_transformer(normalize_timestamp)
        """
        self._transformers.append(transformer)

    def register_source(
        self,
        name: str,
        source_id: str | None = None,
        priority: int = 0,
    ) -> EventSource:
        """
        Register an event source for multiplexing.

        Args:
            name: Human-readable name (e.g., "IB Primary")
            source_id: Unique ID (defaults to name)
            priority: Higher priority sources preferred when duplicates exist

        Returns:
            EventSource object

        Example:
            primary = multiplexer.register_source("IB Primary", priority=10)
            backup = multiplexer.register_source("IB Backup", priority=5)
        """
        source_id = source_id or name.lower().replace(" ", "_")
        source = EventSource(name=name, source_id=source_id, priority=priority)
        self._sources[source_id] = source
        logger.info(f"Registered event source: {name} (priority={priority})")
        return source

    def record_source_event(self, source_id: str) -> None:
        """Record that an event was received from a source."""
        if source_id in self._sources:
            self._sources[source_id].last_event_time = datetime.now(timezone.utc)
            self._sources[source_id].event_count += 1

    def get_active_source(self) -> EventSource | None:
        """
        Get the highest-priority active source.

        An active source has received events within the last 60 seconds.

        Returns:
            Highest priority active source, or None if no active sources
        """
        now = datetime.now(timezone.utc)
        active_sources = []

        for source in self._sources.values():
            if not source.enabled:
                continue
            if source.last_event_time is None:
                continue
            if (now - source.last_event_time).total_seconds() > 60.0:
                continue
            active_sources.append(source)

        if not active_sources:
            return None

        return max(active_sources, key=lambda s: s.priority)

    def get_metrics(self) -> dict:
        """Get multiplexer metrics."""
        return {
            "events_routed": self._events_routed,
            "events_filtered": self._events_filtered,
            "events_transformed": self._events_transformed,
            "subscriptions": sum(len(subs) for subs in self._filtered_subscriptions.values()),
            "registered_event_types": [et.value for et in self._registered_event_types],
            "sources": {
                sid: {
                    "name": s.name,
                    "priority": s.priority,
                    "enabled": s.enabled,
                    "event_count": s.event_count,
                    "last_event": s.last_event_time.isoformat() if s.last_event_time else None,
                }
                for sid, s in self._sources.items()
            },
        }


# ============================================================================
# Convenience filters for common use cases
# ============================================================================

def symbol_filter(symbol: str) -> EventFilter:
    """Create a filter for a specific symbol."""
    def _filter(event: Event) -> bool:
        return hasattr(event, 'symbol') and event.symbol == symbol
    return _filter


def confidence_filter(min_confidence: float) -> EventFilter:
    """Create a filter for minimum confidence."""
    def _filter(event: Event) -> bool:
        return hasattr(event, 'confidence') and event.confidence >= min_confidence
    return _filter


def direction_filter(direction: str) -> EventFilter:
    """Create a filter for signal direction (LONG, SHORT, FLAT)."""
    def _filter(event: Event) -> bool:
        if not hasattr(event, 'direction'):
            return False
        return event.direction.value == direction or str(event.direction) == direction
    return _filter


def combined_filter(*filters: EventFilter) -> EventFilter:
    """Combine multiple filters with AND logic."""
    def _filter(event: Event) -> bool:
        return all(f(event) for f in filters)
    return _filter


def any_filter(*filters: EventFilter) -> EventFilter:
    """Combine multiple filters with OR logic."""
    def _filter(event: Event) -> bool:
        return any(f(event) for f in filters)
    return _filter

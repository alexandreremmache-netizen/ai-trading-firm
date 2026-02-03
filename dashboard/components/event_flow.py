"""
Event Flow Tracker
==================

Track and visualize event flow through the trading system.

Monitors the event chain:
MarketData -> Signals -> CIO Decision -> Risk -> Compliance -> Execution -> Fill

Features:
- Circular buffer for last 1000 events
- Flow statistics (events/minute, latency, counts)
- Agent connection tracking for visualization
- WebSocket-ready export to dict
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from core.events import Event, EventType


logger = logging.getLogger(__name__)


class EventFlowStatus(Enum):
    """Status of an event in the flow."""
    PENDING = "pending"      # Event received, not yet processed
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Successfully processed
    FAILED = "failed"        # Processing failed
    DROPPED = "dropped"      # Dropped due to backpressure


# Define the canonical event flow chain for the trading system
EVENT_FLOW_CHAIN = [
    EventType.MARKET_DATA,
    EventType.SIGNAL,
    EventType.DECISION,
    EventType.VALIDATED_DECISION,
    EventType.ORDER,
    EventType.FILL,
]

# Map event types to their position in the flow chain (for ordering)
EVENT_FLOW_ORDER = {event_type: idx for idx, event_type in enumerate(EVENT_FLOW_CHAIN)}

# Map event types to their expected source and target agents
EVENT_AGENT_MAPPING = {
    EventType.MARKET_DATA: ("MarketData", "SignalAgents"),
    EventType.SIGNAL: ("SignalAgents", "CIOAgent"),
    EventType.DECISION: ("CIOAgent", "RiskAgent"),
    EventType.VALIDATED_DECISION: ("RiskAgent/ComplianceAgent", "ExecutionAgent"),
    EventType.ORDER: ("ExecutionAgent", "Broker"),
    EventType.FILL: ("Broker", "System"),
    EventType.RISK_ALERT: ("RiskAgent", "System"),
    EventType.KILL_SWITCH: ("RiskAgent", "System"),
    EventType.SURVEILLANCE_ALERT: ("SurveillanceAgent", "ComplianceAgent"),
    EventType.TRANSACTION_REPORT: ("TransactionReportingAgent", "Regulator"),
}


@dataclass
class EventFlowRecord:
    """
    Record of a single event in the system flow.

    Captures event metadata for tracking and visualization.
    """
    event_id: str
    event_type: EventType
    source_agent: str
    target_agent: str
    timestamp: datetime
    payload_summary: dict[str, Any]
    status: EventFlowStatus = EventFlowStatus.PENDING
    processing_time_ms: float | None = None
    parent_event_id: str | None = None  # For tracking event chains
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "timestamp": self.timestamp.isoformat(),
            "payload_summary": self.payload_summary,
            "status": self.status.value,
            "processing_time_ms": self.processing_time_ms,
            "parent_event_id": self.parent_event_id,
            "error_message": self.error_message,
        }


@dataclass
class FlowStatistics:
    """
    Statistics about event flow through the system.

    Provides metrics for monitoring system health and performance.
    """
    events_per_minute: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    event_type_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)
    agent_event_counts: dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Time-based metrics
    events_last_minute: int = 0
    events_last_5_minutes: int = 0
    events_last_hour: int = 0
    # Flow chain metrics
    complete_chains: int = 0  # MarketData -> Fill complete cycles
    incomplete_chains: int = 0
    avg_chain_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "events_per_minute": round(self.events_per_minute, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "event_type_counts": self.event_type_counts,
            "status_counts": self.status_counts,
            "agent_event_counts": self.agent_event_counts,
            "last_updated": self.last_updated.isoformat(),
            "events_last_minute": self.events_last_minute,
            "events_last_5_minutes": self.events_last_5_minutes,
            "events_last_hour": self.events_last_hour,
            "complete_chains": self.complete_chains,
            "incomplete_chains": self.incomplete_chains,
            "avg_chain_time_ms": round(self.avg_chain_time_ms, 2),
        }


@dataclass
class AgentConnection:
    """
    Represents a connection between two agents in the event flow.

    Used for visualizing agent relationships and event flow paths.
    """
    source_agent: str
    target_agent: str
    event_types: list[str] = field(default_factory=list)
    total_events: int = 0
    events_last_minute: int = 0
    avg_latency_ms: float = 0.0
    last_event_time: datetime | None = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "event_types": self.event_types,
            "total_events": self.total_events,
            "events_last_minute": self.events_last_minute,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "is_active": self.is_active,
        }

    @property
    def connection_id(self) -> str:
        """Unique identifier for this connection."""
        return f"{self.source_agent}->{self.target_agent}"


class EventFlowTracker:
    """
    Tracks and visualizes event flow through the trading system.

    Maintains a circular buffer of recent events and computes
    flow statistics for monitoring and visualization.

    Usage:
        tracker = EventFlowTracker()

        # Record events as they flow through the system
        tracker.record_event(event)

        # Get recent events for display
        recent = tracker.get_recent_events(limit=50)

        # Get flow statistics
        stats = tracker.get_flow_statistics()

        # Get agent connections for visualization
        connections = tracker.get_agent_connections()

        # Export for WebSocket streaming
        data = tracker.to_dict()
    """

    # Maximum number of events to keep in the circular buffer
    MAX_EVENTS = 1000

    # Time window for "recent" events (seconds)
    RECENT_WINDOW_SECONDS = 60

    def __init__(self, max_events: int = MAX_EVENTS):
        """
        Initialize the event flow tracker.

        Args:
            max_events: Maximum events to keep in circular buffer (default 1000)
        """
        self._max_events = max_events
        self._events: deque[EventFlowRecord] = deque(maxlen=max_events)
        self._connections: dict[str, AgentConnection] = {}
        self._event_chains: dict[str, list[str]] = {}  # symbol -> chain of event IDs
        self._latencies: deque[float] = deque(maxlen=1000)
        self._chain_times: deque[float] = deque(maxlen=100)  # Complete chain times

        # Statistics tracking
        self._total_events = 0
        self._event_type_counts: dict[str, int] = {}
        self._status_counts: dict[str, int] = {s.value: 0 for s in EventFlowStatus}
        self._agent_event_counts: dict[str, int] = {}

        logger.info(f"EventFlowTracker initialized with max_events={max_events}")

    def record_event(
        self,
        event: Event,
        status: EventFlowStatus = EventFlowStatus.COMPLETED,
        processing_time_ms: float | None = None,
        parent_event_id: str | None = None,
        error_message: str | None = None,
    ) -> EventFlowRecord:
        """
        Record an event in the flow tracker.

        Args:
            event: The event to record
            status: Current status of the event
            processing_time_ms: Time taken to process the event
            parent_event_id: ID of the parent event in the chain
            error_message: Error message if event failed

        Returns:
            EventFlowRecord for the recorded event
        """
        # Determine source and target agents
        source_agent = event.source_agent
        target_agent = self._get_target_agent(event)

        # Create payload summary (avoid storing full event data)
        payload_summary = self._create_payload_summary(event)

        # Create the record
        record = EventFlowRecord(
            event_id=event.event_id,
            event_type=event.event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            timestamp=event.timestamp,
            payload_summary=payload_summary,
            status=status,
            processing_time_ms=processing_time_ms,
            parent_event_id=parent_event_id,
            error_message=error_message,
        )

        # Add to circular buffer
        self._events.append(record)
        self._total_events += 1

        # Update statistics
        self._update_statistics(record)

        # Update agent connection
        self._update_connection(record)

        # Track event chain
        self._track_chain(event, record)

        # Track latency
        if processing_time_ms is not None:
            self._latencies.append(processing_time_ms)

        logger.debug(
            f"Recorded event {event.event_id[:8]} ({event.event_type.value}) "
            f"from {source_agent} to {target_agent}"
        )

        return record

    def _get_target_agent(self, event: Event) -> str:
        """Determine the target agent for an event based on its type."""
        mapping = EVENT_AGENT_MAPPING.get(event.event_type)
        if mapping:
            return mapping[1]

        # Default to "System" for unknown event types
        return "System"

    def _create_payload_summary(self, event: Event) -> dict[str, Any]:
        """Create a summary of the event payload for storage."""
        summary: dict[str, Any] = {
            "event_type": event.event_type.value,
        }

        # Add common fields if present
        if hasattr(event, "symbol") and event.symbol:
            summary["symbol"] = event.symbol

        if hasattr(event, "direction"):
            summary["direction"] = event.direction.value if hasattr(event.direction, "value") else str(event.direction)

        if hasattr(event, "quantity"):
            summary["quantity"] = event.quantity

        if hasattr(event, "confidence"):
            summary["confidence"] = round(event.confidence, 3) if event.confidence else None

        if hasattr(event, "strength"):
            summary["strength"] = round(event.strength, 3) if event.strength else None

        if hasattr(event, "side"):
            summary["side"] = event.side.value if hasattr(event.side, "value") else str(event.side)

        if hasattr(event, "approved"):
            summary["approved"] = event.approved

        if hasattr(event, "fill_price"):
            summary["fill_price"] = event.fill_price

        if hasattr(event, "filled_quantity"):
            summary["filled_quantity"] = event.filled_quantity

        if hasattr(event, "severity"):
            summary["severity"] = event.severity.value if hasattr(event.severity, "value") else str(event.severity)

        if hasattr(event, "rationale") and event.rationale:
            # Truncate long rationales
            summary["rationale"] = event.rationale[:100] + "..." if len(event.rationale) > 100 else event.rationale

        return summary

    def _update_statistics(self, record: EventFlowRecord) -> None:
        """Update internal statistics with a new record."""
        # Event type counts
        event_type_key = record.event_type.value
        self._event_type_counts[event_type_key] = self._event_type_counts.get(event_type_key, 0) + 1

        # Status counts
        status_key = record.status.value
        self._status_counts[status_key] = self._status_counts.get(status_key, 0) + 1

        # Agent event counts
        agent_key = record.source_agent
        self._agent_event_counts[agent_key] = self._agent_event_counts.get(agent_key, 0) + 1

    def _update_connection(self, record: EventFlowRecord) -> None:
        """Update or create agent connection tracking."""
        connection_id = f"{record.source_agent}->{record.target_agent}"

        if connection_id not in self._connections:
            self._connections[connection_id] = AgentConnection(
                source_agent=record.source_agent,
                target_agent=record.target_agent,
                event_types=[],
            )

        conn = self._connections[connection_id]
        conn.total_events += 1
        conn.last_event_time = record.timestamp

        # Track event type if not already tracked
        if record.event_type.value not in conn.event_types:
            conn.event_types.append(record.event_type.value)

        # Update latency (exponential moving average)
        if record.processing_time_ms is not None:
            if conn.avg_latency_ms == 0:
                conn.avg_latency_ms = record.processing_time_ms
            else:
                alpha = 0.1  # Smoothing factor
                conn.avg_latency_ms = alpha * record.processing_time_ms + (1 - alpha) * conn.avg_latency_ms

    def _track_chain(self, event: Event, record: EventFlowRecord) -> None:
        """Track event chains for complete flow analysis."""
        symbol = getattr(event, "symbol", None)
        if not symbol:
            return

        # Initialize chain for symbol if needed
        if symbol not in self._event_chains:
            self._event_chains[symbol] = []

        chain = self._event_chains[symbol]

        # Check if this is the start of a new chain (MarketData)
        if event.event_type == EventType.MARKET_DATA:
            # Start a new chain
            self._event_chains[symbol] = [event.event_id]
        else:
            # Add to existing chain
            chain.append(event.event_id)

            # Check if chain is complete (reached Fill)
            if event.event_type == EventType.FILL and len(chain) >= 2:
                # Calculate chain time from first MarketData to Fill
                first_event = self._find_event_by_id(chain[0])
                if first_event:
                    chain_time = (record.timestamp - first_event.timestamp).total_seconds() * 1000
                    self._chain_times.append(chain_time)

                # Reset chain for next cycle
                self._event_chains[symbol] = []

    def _find_event_by_id(self, event_id: str) -> EventFlowRecord | None:
        """Find an event record by ID."""
        for record in self._events:
            if record.event_id == event_id:
                return record
        return None

    def get_recent_events(self, limit: int = 100) -> list[EventFlowRecord]:
        """
        Get the most recent events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of EventFlowRecord, most recent first
        """
        # Convert deque to list and reverse to get most recent first
        events = list(self._events)
        events.reverse()
        return events[:limit]

    def get_events_by_type(
        self,
        event_type: EventType,
        limit: int = 100,
    ) -> list[EventFlowRecord]:
        """
        Get recent events of a specific type.

        Args:
            event_type: Type of events to retrieve
            limit: Maximum number of events to return

        Returns:
            List of matching EventFlowRecord
        """
        matching = [e for e in self._events if e.event_type == event_type]
        matching.reverse()
        return matching[:limit]

    def get_events_by_agent(
        self,
        agent_name: str,
        limit: int = 100,
    ) -> list[EventFlowRecord]:
        """
        Get recent events from a specific agent.

        Args:
            agent_name: Name of the source agent
            limit: Maximum number of events to return

        Returns:
            List of matching EventFlowRecord
        """
        matching = [e for e in self._events if e.source_agent == agent_name]
        matching.reverse()
        return matching[:limit]

    def get_events_by_status(
        self,
        status: EventFlowStatus,
        limit: int = 100,
    ) -> list[EventFlowRecord]:
        """
        Get recent events with a specific status.

        Args:
            status: Status to filter by
            limit: Maximum number of events to return

        Returns:
            List of matching EventFlowRecord
        """
        matching = [e for e in self._events if e.status == status]
        matching.reverse()
        return matching[:limit]

    def get_flow_statistics(self) -> FlowStatistics:
        """
        Calculate and return current flow statistics.

        Returns:
            FlowStatistics with current metrics
        """
        now = datetime.now(timezone.utc)

        # Count events in time windows
        events_last_minute = 0
        events_last_5_minutes = 0
        events_last_hour = 0

        one_minute_ago = now - timedelta(minutes=1)
        five_minutes_ago = now - timedelta(minutes=5)
        one_hour_ago = now - timedelta(hours=1)

        for record in self._events:
            if record.timestamp >= one_minute_ago:
                events_last_minute += 1
                events_last_5_minutes += 1
                events_last_hour += 1
            elif record.timestamp >= five_minutes_ago:
                events_last_5_minutes += 1
                events_last_hour += 1
            elif record.timestamp >= one_hour_ago:
                events_last_hour += 1

        # Calculate events per minute
        events_per_minute = events_last_minute

        # Calculate latency statistics
        latencies = list(self._latencies)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        min_latency = min(latencies) if latencies else 0.0

        # Calculate chain statistics
        chain_times = list(self._chain_times)
        avg_chain_time = sum(chain_times) / len(chain_times) if chain_times else 0.0
        complete_chains = len(chain_times)

        # Count incomplete chains (symbols with non-empty chains)
        incomplete_chains = sum(1 for chain in self._event_chains.values() if chain)

        return FlowStatistics(
            events_per_minute=events_per_minute,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            event_type_counts=dict(self._event_type_counts),
            status_counts=dict(self._status_counts),
            agent_event_counts=dict(self._agent_event_counts),
            last_updated=now,
            events_last_minute=events_last_minute,
            events_last_5_minutes=events_last_5_minutes,
            events_last_hour=events_last_hour,
            complete_chains=complete_chains,
            incomplete_chains=incomplete_chains,
            avg_chain_time_ms=avg_chain_time,
        )

    def get_agent_connections(self) -> list[AgentConnection]:
        """
        Get all agent connections for visualization.

        Returns:
            List of AgentConnection objects
        """
        now = datetime.now(timezone.utc)
        active_threshold = now - timedelta(seconds=self.RECENT_WINDOW_SECONDS)

        connections = []
        for conn in self._connections.values():
            # Update active status
            if conn.last_event_time:
                conn.is_active = conn.last_event_time >= active_threshold
            else:
                conn.is_active = False

            # Count events in last minute for this connection
            conn.events_last_minute = sum(
                1 for e in self._events
                if e.source_agent == conn.source_agent
                and e.target_agent == conn.target_agent
                and e.timestamp >= active_threshold
            )

            connections.append(conn)

        return connections

    def get_event_chain(self, symbol: str) -> list[EventFlowRecord]:
        """
        Get the current event chain for a symbol.

        Args:
            symbol: Symbol to get chain for

        Returns:
            List of EventFlowRecord in the chain
        """
        event_ids = self._event_chains.get(symbol, [])
        chain = []
        for event_id in event_ids:
            record = self._find_event_by_id(event_id)
            if record:
                chain.append(record)
        return chain

    def get_flow_diagram_data(self) -> dict[str, Any]:
        """
        Get data for rendering a flow diagram visualization.

        Returns:
            Dict with nodes (agents) and edges (connections)
        """
        # Collect unique agents
        agents = set()
        for conn in self._connections.values():
            agents.add(conn.source_agent)
            agents.add(conn.target_agent)

        # Create nodes with event counts
        nodes = []
        for agent in agents:
            count = self._agent_event_counts.get(agent, 0)
            nodes.append({
                "id": agent,
                "label": agent,
                "event_count": count,
            })

        # Create edges from connections
        edges = []
        for conn in self._connections.values():
            edges.append({
                "source": conn.source_agent,
                "target": conn.target_agent,
                "event_types": conn.event_types,
                "total_events": conn.total_events,
                "is_active": conn.is_active,
                "avg_latency_ms": round(conn.avg_latency_ms, 2),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def clear(self) -> None:
        """Clear all tracked events and reset statistics."""
        self._events.clear()
        self._connections.clear()
        self._event_chains.clear()
        self._latencies.clear()
        self._chain_times.clear()
        self._total_events = 0
        self._event_type_counts.clear()
        self._status_counts = {s.value: 0 for s in EventFlowStatus}
        self._agent_event_counts.clear()
        logger.info("EventFlowTracker cleared")

    def to_dict(self) -> dict[str, Any]:
        """
        Export tracker state to dictionary for WebSocket streaming.

        Returns:
            Complete tracker state as dict
        """
        stats = self.get_flow_statistics()
        connections = self.get_agent_connections()
        recent_events = self.get_recent_events(limit=50)

        return {
            "statistics": stats.to_dict(),
            "connections": [c.to_dict() for c in connections],
            "recent_events": [e.to_dict() for e in recent_events],
            "flow_diagram": self.get_flow_diagram_data(),
            "total_events_tracked": self._total_events,
            "buffer_size": len(self._events),
            "buffer_max_size": self._max_events,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def total_events(self) -> int:
        """Total number of events recorded (including those evicted from buffer)."""
        return self._total_events

    @property
    def buffer_size(self) -> int:
        """Current number of events in the buffer."""
        return len(self._events)

    @property
    def is_buffer_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._events) >= self._max_events

"""
Dashboard Server
================

FastAPI-based real-time trading dashboard server with WebSocket support.

Features:
- Real-time event streaming via WebSocket
- REST API endpoints for agents, positions, signals, decisions, metrics, risk
- Static file serving for templates
- EventBus integration for system-wide event subscription
- Background tasks for periodic updates

Per CLAUDE.md:
- Observable and auditable system
- No polling loops - event-driven updates
- Integration with existing EventBus infrastructure
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import AgentStatusTracker for comprehensive agent monitoring
from dashboard.components.agent_status import (
    AgentStatusTracker,
    AgentStatus as TrackerAgentStatus,
    AgentType,
    AGENT_TYPE_MAPPING,
)

# Import SignalAggregator for comprehensive signal tracking and consensus
from dashboard.components.signal_aggregation import SignalAggregator

# Import contract specs for proper price display (divide by multiplier)
from core.contract_specs import CONTRACT_SPECS

# Import Advanced Analytics components (Phase 8)
from dashboard.components.advanced_analytics import (
    RollingMetricsCalculator,
    RollingPeriod,
    SessionPerformanceTracker,
    StrategyComparisonTracker,
    RiskHeatmapGenerator,
    TradeJournal,
    SignalConsensusTracker,
    create_all_analytics_components,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.events import Event

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def get_contract_multiplier(symbol: str) -> float:
    """
    Get contract multiplier for a symbol.

    IB returns avg_cost as notional per contract (price × multiplier).
    We need to divide by multiplier to show actual price.

    For stocks, multiplier is 1.0.
    For futures like CL, multiplier is 1000 (1 contract = 1000 barrels).

    Args:
        symbol: Contract symbol (e.g., "CL", "ES", "AAPL")

    Returns:
        Contract multiplier (1.0 for stocks, varies for futures)
    """
    # Extract base symbol (handle "ESZ4" -> "ES", "CLF25" -> "CL")
    base_symbol = symbol.upper().strip()
    for length in [2, 3]:
        if len(base_symbol) >= length:
            candidate = base_symbol[:length]
            if candidate in CONTRACT_SPECS:
                base_symbol = candidate
                break

    spec = CONTRACT_SPECS.get(base_symbol)
    if spec:
        return spec.multiplier

    # Default to 1.0 for stocks and unknown symbols
    return 1.0


# =============================================================================
# Data Models
# =============================================================================

class AgentStatus(Enum):
    """Agent operational status."""
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentInfo:
    """Agent status information for dashboard display."""
    name: str
    type: str
    status: AgentStatus = AgentStatus.IDLE
    last_event_time: datetime | None = None
    event_count: int = 0
    latency_ms: float = 0.0
    error_message: str | None = None
    error_count: int = 0
    uptime_seconds: float = 0.0
    health_score: float = 100.0  # 0-100 scale for health display

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Calculate health score based on status and errors
        health = self.health_score
        if self.status == AgentStatus.ERROR:
            health = max(0, 50 - self.error_count * 5)
        elif self.status == AgentStatus.STOPPED:
            health = 0
        elif self.status == AgentStatus.IDLE:
            health = max(50, health - 10)

        return {
            "name": self.name,
            "type": self.type,
            "status": self.status.value,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "event_count": self.event_count,
            "latency_ms": round(self.latency_ms, 2),
            "error_message": self.error_message,
            "error_count": self.error_count,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "health_score": round(health, 1),
        }


@dataclass
class PositionInfo:
    """Position information for dashboard display."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    market_value: float = 0.0
    entry_time: datetime | None = None  # When position was opened
    conviction: float = 0.0  # Conviction at entry
    signal: str = ""  # Signal direction at entry (LONG/SHORT/FLAT)
    regime: str = ""  # Market regime at entry (risk_on, risk_off, volatile, etc.)
    session: str = ""  # Trading session at entry (Asian, London, NY, Overlap)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "market_value": round(self.market_value, 2),
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "conviction": round(self.conviction, 2),
            "signal": self.signal,
            "regime": self.regime,
            "session": self.session,
        }


@dataclass
class SignalInfo:
    """Signal information for dashboard display."""
    agent: str
    symbol: str
    direction: str
    confidence: float
    strength: float
    timestamp: datetime
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent": self.agent,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "strength": round(self.strength, 3),
            "timestamp": self.timestamp.isoformat(),
            "rationale": self.rationale,
        }


@dataclass
class DecisionInfo:
    """Decision information for dashboard display."""
    decision_id: str
    symbol: str
    direction: str
    quantity: int
    conviction: float
    timestamp: datetime
    rationale: str
    pnl: float = 0.0
    status: str = "pending"
    rejection_reason: str = ""  # Reason for rejection if status is REJECTED
    estimated_value: float = 0.0  # Estimated trade value (price * quantity)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "conviction": round(self.conviction, 3),
            "timestamp": self.timestamp.isoformat(),
            "rationale": self.rationale,
            "pnl": round(self.pnl, 2),
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "estimated_value": round(self.estimated_value, 2),
        }


@dataclass
class RiskLimit:
    """Risk limit information for dashboard display."""
    name: str
    current: float
    limit: float
    usage: float = 0.0
    status: str = "ok"  # ok, warning, breach

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "current": round(self.current, 2),
            "limit": round(self.limit, 2),
            "usage": round(self.usage, 1),
            "status": self.status,
        }


@dataclass
class Metrics:
    """Portfolio metrics for dashboard display."""
    total_pnl: float = 0.0          # Total account P&L (realized + unrealized)
    today_pnl: float = 0.0          # Today's P&L (unrealized + today's realized)
    unrealized_pnl: float = 0.0     # Unrealized P&L from open positions
    realized_pnl: float = 0.0       # Realized P&L from ALL closed positions
    today_realized_pnl: float = 0.0 # Realized P&L from positions closed TODAY only
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    drawdown: float = 0.0
    position_count: int = 0
    total_trades: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_pnl": round(self.total_pnl, 2),              # Total = realized + unrealized
            "today_pnl": round(self.today_pnl, 2),              # Today's running P&L
            "unrealized_pnl": round(self.unrealized_pnl, 2),    # Open positions P&L
            "realized_pnl": round(self.realized_pnl, 2),        # Closed positions P&L (all time)
            "today_realized_pnl": round(self.today_realized_pnl, 2),  # Closed positions P&L (today)
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "win_rate": round(self.win_rate, 3),
            "drawdown": round(self.drawdown, 4),
            "position_count": self.position_count,
            "total_trades": self.total_trades,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


# =============================================================================
# Session Detection Helper
# =============================================================================

def get_current_trading_session(ts: datetime | None = None) -> str:
    """
    Determine the current trading session based on UTC time.

    Sessions (all times UTC):
    - Asian: 00:00-08:00 UTC (Tokyo/Sydney)
    - London: 08:00-12:00 UTC (European open)
    - Overlap: 12:00-17:00 UTC (London/NY overlap)
    - NY: 17:00-21:00 UTC (US afternoon)
    - After Hours: 21:00-00:00 UTC

    Args:
        ts: Timestamp to check, defaults to current UTC time

    Returns:
        Session name string
    """
    if ts is None:
        ts = datetime.now(timezone.utc)

    # Ensure we're working with UTC
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    hour = ts.hour

    if 0 <= hour < 8:
        return "Asian"
    elif 8 <= hour < 12:
        return "London"
    elif 12 <= hour < 17:
        return "Overlap"
    elif 17 <= hour < 21:
        return "NY"
    else:
        return "After Hours"


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Thread-safe connection management with broadcast capability.
    """

    def __init__(self):
        self._active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._message_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10000)
        self._broadcast_task: asyncio.Task | None = None
        self._running = False

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self._active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self._active_connections:
                self._active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self._active_connections)}")

    async def broadcast(self, message: dict) -> None:
        """
        Broadcast message to all connected clients.

        Uses a queue to decouple event processing from I/O.
        """
        try:
            self._message_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Broadcast queue full, dropping message")

    async def start_broadcaster(self) -> None:
        """Start the background broadcast task."""
        if self._running:
            return
        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("WebSocket broadcaster started")

    async def stop_broadcaster(self) -> None:
        """Stop the background broadcast task."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket broadcaster stopped")

    async def _broadcast_loop(self) -> None:
        """Background loop that broadcasts queued messages."""
        while self._running:
            try:
                # Wait for message with timeout
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Broadcast to all connections
                async with self._lock:
                    connections = list(self._active_connections)

                if not connections:
                    continue

                message_json = json.dumps(message, default=str)

                # Send to all connections, removing dead ones
                dead_connections = []
                for connection in connections:
                    try:
                        await connection.send_text(message_json)
                    except Exception as e:
                        logger.debug(f"Failed to send to WebSocket: {e}")
                        dead_connections.append(connection)

                # Clean up dead connections
                if dead_connections:
                    async with self._lock:
                        for conn in dead_connections:
                            if conn in self._active_connections:
                                self._active_connections.remove(conn)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Broadcast loop error: {e}")
                await asyncio.sleep(0.1)

    @property
    def connection_count(self) -> int:
        """Get current number of active connections."""
        return len(self._active_connections)


# =============================================================================
# Dashboard State Manager
# =============================================================================

class DashboardState:
    """
    Manages the state of the trading dashboard.

    Collects and aggregates data from the trading system for display.
    Subscribes to EventBus for real-time updates.
    """

    def __init__(self, event_bus: "EventBus | None" = None):
        self._event_bus = event_bus

        # Agent tracking - pre-populate with all expected agents (including disabled ones)
        self._agents: dict[str, AgentInfo] = {}

        # Define all expected agents (15 signal + 6 core = 21 total)
        all_expected_agents = [
            # Core agents
            ("CIOAgent", "Decision"),
            ("RiskAgent", "Validation"),
            ("ComplianceAgent", "Validation"),
            ("ExecutionAgent", "Execution"),
            ("SurveillanceAgent", "Surveillance"),
            ("TransactionReportingAgent", "Reporting"),
            # Signal agents - Core (always enabled)
            ("MacroAgent", "Signal"),
            ("StatArbAgent", "Signal"),
            ("MomentumAgent", "Signal"),
            ("MarketMakingAgent", "Signal"),
            # Signal agents - LLM (may be disabled)
            ("SentimentAgent", "Signal"),
            ("ChartAnalysisAgent", "Signal"),
            ("ForecastingAgent", "Signal"),
            # Signal agents - Phase 6
            ("SessionAgent", "Signal"),
            ("IndexSpreadAgent", "Signal"),
            ("TTMSqueezeAgent", "Signal"),
            ("EventDrivenAgent", "Signal"),
            ("MeanReversionAgent", "Signal"),
            ("MACDvAgent", "Signal"),
        ]

        # Pre-populate all agents with STOPPED status (will be updated when they start)
        for agent_name, agent_type in all_expected_agents:
            self._agents[agent_name] = AgentInfo(
                name=agent_name,
                type=agent_type,
                status=AgentStatus.STOPPED,
                event_count=0,
                error_count=0,
            )

        # Position tracking
        self._positions: dict[str, PositionInfo] = {}

        # Signal tracking (most recent per agent)
        self._signals: dict[str, SignalInfo] = {}

        # Decision history (bounded)
        self._decisions: deque[DecisionInfo] = deque(maxlen=100)

        # Event stream (bounded)
        self._events: deque[dict] = deque(maxlen=500)

        # Risk limits
        self._risk_limits: dict[str, RiskLimit] = {}

        # Portfolio metrics
        self._metrics = Metrics()

        # Equity curve data (stores ISO timestamp, value, unix_ms)
        self._equity_curve: deque[tuple[str, float, int]] = deque(maxlen=10000)
        # Use absolute path for equity history file
        project_root = Path(__file__).parent.parent
        self._equity_file_path = project_root / "logs" / "equity_history.json"
        self._equity_save_counter = 0
        self._load_equity_history()

        # Kill switch state
        self._kill_switch_active = False

        # Agent enabled/disabled state (runtime toggle)
        # Key: agent_name, Value: enabled (True/False)
        self._agent_enabled_state: dict[str, bool] = {}

        # LLM agents list (marked for cost awareness)
        self._llm_agents = {"SentimentAgent", "ChartAnalysisAgent", "ForecastingAgent"}

        # Essential agents that cannot be disabled (critical for system operation)
        self._essential_agents = {
            "CIOAgent",
            "RiskAgent",
            "ComplianceAgent",
            "ExecutionAgent",
            "ReconciliationAgent",
            "SurveillanceAgent",
        }

        # Mapping from agent names to config keys for reading enabled state
        self._agent_config_key_map = {
            "MacroAgent": "macro",
            "StatArbAgent": "stat_arb",
            "MomentumAgent": "momentum",
            "MarketMakingAgent": "market_making",
            "SentimentAgent": "sentiment",
            "ChartAnalysisAgent": "chart_analysis",
            "ForecastingAgent": "forecasting",
            "SessionAgent": "session",
            "IndexSpreadAgent": "index_spread",
            "TTMSqueezeAgent": "ttm_squeeze",
            "EventDrivenAgent": "event_driven",
            "MeanReversionAgent": "mean_reversion",
            "MACDvAgent": "macdv",
            "SurveillanceAgent": "surveillance",
            "TransactionReportingAgent": "transaction_reporting",
        }

        # Alerts
        self._alerts: deque[dict] = deque(maxlen=100)

        # Closed positions history (increased capacity for full history)
        self._closed_positions: deque[dict] = deque(maxlen=1000)

        # Track broker positions for closed position detection
        self._broker_positions: dict[str, dict] = {}

        # Historical position metadata from audit logs (entry_time, conviction, signal)
        self._position_history: dict[str, dict] = {}
        self._load_position_history()

        # Load historical closed positions from trades log
        self._load_closed_positions_history()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Latest event record for broadcasting (set by _handle_event)
        self._latest_event_record: dict | None = None

        # Advanced Analytics Components (Phase 8)
        analytics = create_all_analytics_components()
        self._rolling_metrics: RollingMetricsCalculator = analytics["rolling_metrics"]
        self._session_performance: SessionPerformanceTracker = analytics["session_performance"]
        self._strategy_comparison: StrategyComparisonTracker = analytics["strategy_comparison"]
        self._risk_heatmap: RiskHeatmapGenerator = analytics["risk_heatmap"]
        self._trade_journal: TradeJournal = analytics["trade_journal"]
        self._signal_consensus: SignalConsensusTracker = analytics["signal_consensus"]

        # Populate analytics from closed positions history
        self._populate_analytics_from_history()

        # Callback for broadcasting events to WebSocket clients
        self._broadcast_callback: Callable[[dict], None] | None = None

        # References to trading system components (set via set_orchestrator)
        self._broker: Any = None
        self._orchestrator: Any = None
        self._risk_agent: Any = None
        self._cio_agent: Any = None
        self._signal_agents: list[Any] = []
        self._execution_agent: Any = None
        self._compliance_agent: Any = None

        # Current market regime (updated from CIO agent events)
        self._current_regime: str = "neutral"

        # Comprehensive agent status tracker for advanced health monitoring
        self._agent_tracker = AgentStatusTracker(idle_threshold_seconds=30.0)

        # Signal aggregator for comprehensive signal tracking and consensus
        self._signal_aggregator = SignalAggregator()

        # WebSocket broadcast callback for real-time signal updates
        self._ws_broadcast: Callable[[dict], Coroutine[Any, Any, None]] | None = None

    def set_ws_broadcast(self, broadcast_fn: Callable[[dict], Coroutine[Any, Any, None]]) -> None:
        """Set the WebSocket broadcast callback for real-time updates."""
        self._ws_broadcast = broadcast_fn

    def set_orchestrator(self, orchestrator: Any) -> None:
        """
        Set the orchestrator reference for pulling real-time data.

        Args:
            orchestrator: TradingFirmOrchestrator instance
        """
        self._orchestrator = orchestrator
        # Extract component references from orchestrator
        if hasattr(orchestrator, '_broker'):
            self._broker = orchestrator._broker
        if hasattr(orchestrator, '_risk_agent'):
            self._risk_agent = orchestrator._risk_agent
        if hasattr(orchestrator, '_cio_agent'):
            self._cio_agent = orchestrator._cio_agent
        if hasattr(orchestrator, '_signal_agents'):
            self._signal_agents = orchestrator._signal_agents
        if hasattr(orchestrator, '_execution_agent'):
            self._execution_agent = orchestrator._execution_agent
        if hasattr(orchestrator, '_compliance_agent'):
            self._compliance_agent = orchestrator._compliance_agent

        # Pre-register agents if available
        self._register_agents_from_orchestrator()
        logger.info("Dashboard connected to orchestrator")

    def _register_agents_from_orchestrator(self) -> None:
        """Pre-register all agents from the orchestrator."""
        # Register signal agents
        for agent in self._signal_agents:
            if hasattr(agent, 'name'):
                self._agents[agent.name] = AgentInfo(
                    name=agent.name,
                    type="Signal",
                    status=AgentStatus.IDLE,
                )

        # Register core agents
        if self._cio_agent and hasattr(self._cio_agent, 'name'):
            self._agents[self._cio_agent.name] = AgentInfo(
                name=self._cio_agent.name,
                type="Decision",
                status=AgentStatus.IDLE,
            )

        if self._risk_agent and hasattr(self._risk_agent, 'name'):
            self._agents[self._risk_agent.name] = AgentInfo(
                name=self._risk_agent.name,
                type="Validation",
                status=AgentStatus.IDLE,
            )

        if self._compliance_agent and hasattr(self._compliance_agent, 'name'):
            self._agents[self._compliance_agent.name] = AgentInfo(
                name=self._compliance_agent.name,
                type="Validation",
                status=AgentStatus.IDLE,
            )

        if self._execution_agent and hasattr(self._execution_agent, 'name'):
            self._agents[self._execution_agent.name] = AgentInfo(
                name=self._execution_agent.name,
                type="Execution",
                status=AgentStatus.IDLE,
            )

    async def initialize(self) -> None:
        """Initialize state and subscribe to events."""
        if self._event_bus:
            await self._subscribe_to_events()

        # Add startup alert
        self._alerts.appendleft({
            "title": "System Started",
            "message": "AI Trading Firm dashboard connected and monitoring",
            "severity": "info",
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        })

        # Historical closed positions are loaded from trades.jsonl in __init__
        # Real-time closes are detected via broker position changes
        logger.info(f"[CLOSED_POSITIONS] At initialize(): deque has {len(self._closed_positions)} positions")

        logger.info("Dashboard state initialized")

    async def _subscribe_to_events(self) -> None:
        """Subscribe to all event types from the EventBus."""
        from core.events import EventType

        # Subscribe to all event types for comprehensive monitoring
        event_types = [
            EventType.MARKET_DATA,
            EventType.SIGNAL,
            EventType.DECISION,
            EventType.VALIDATED_DECISION,
            EventType.ORDER,
            EventType.FILL,
            EventType.RISK_ALERT,
            EventType.SYSTEM,
            EventType.KILL_SWITCH,
        ]

        for event_type in event_types:
            self._event_bus.subscribe(event_type, self._handle_event)

        logger.info(f"Subscribed to {len(event_types)} event types")

    async def _handle_event(self, event: "Event") -> None:
        """
        Handle incoming events from the EventBus.

        Routes events to appropriate handlers based on type.
        """
        from core.events import (
            EventType, SignalEvent, DecisionEvent, ValidatedDecisionEvent,
            FillEvent, RiskAlertEvent, MarketDataEvent, KillSwitchEvent,
        )

        try:
            async with self._lock:
                # Record event in stream
                event_record = {
                    "type": event.event_type.value.upper(),
                    "source": event.source_agent,
                    "time": event.timestamp.strftime("%H:%M:%S"),
                    "summary": self._get_event_summary(event),
                }
                self._events.appendleft(event_record)

                # Update agent status
                if event.source_agent and event.source_agent != "unknown":
                    self._update_agent_status(event.source_agent, event)

                # Handle specific event types
                if event.event_type == EventType.SIGNAL:
                    self._handle_signal_event(event)
                elif event.event_type == EventType.DECISION:
                    self._handle_decision_event(event)
                elif event.event_type == EventType.VALIDATED_DECISION:
                    self._handle_validated_decision_event(event)
                elif event.event_type == EventType.FILL:
                    self._handle_fill_event(event)
                elif event.event_type == EventType.RISK_ALERT:
                    self._handle_risk_alert_event(event)
                elif event.event_type == EventType.KILL_SWITCH:
                    self._handle_kill_switch_event(event)
                elif event.event_type == EventType.MARKET_DATA:
                    self._handle_market_data_event(event)

            # Store event record for broadcasting (outside lock)
            self._latest_event_record = event_record

        except Exception as e:
            logger.exception(f"Error handling event: {e}")

    def _get_event_summary(self, event: "Event") -> str:
        """Generate a human-readable summary of an event."""
        from core.events import EventType

        audit_dict = event.to_audit_dict()

        if event.event_type == EventType.SIGNAL:
            return f"{audit_dict.get('symbol', '?')} {audit_dict.get('direction', '?')} ({audit_dict.get('confidence', 0):.0%})"
        elif event.event_type == EventType.DECISION:
            return f"{audit_dict.get('symbol', '?')} {audit_dict.get('action', '?')} x{audit_dict.get('quantity', 0)}"
        elif event.event_type == EventType.VALIDATED_DECISION:
            status = "APPROVED" if audit_dict.get('approved', False) else "REJECTED"
            return f"{audit_dict.get('symbol', '?')} {status} by {event.source_agent}"
        elif event.event_type == EventType.FILL:
            return f"{audit_dict.get('symbol', '?')} filled {audit_dict.get('filled_quantity', 0)} @ {audit_dict.get('fill_price', 0):.2f}"
        elif event.event_type == EventType.RISK_ALERT:
            return f"{audit_dict.get('alert_type', '?')}: {audit_dict.get('message', '')[:50]}"
        elif event.event_type == EventType.MARKET_DATA:
            return f"{audit_dict.get('symbol', '?')} last={audit_dict.get('last', 0):.2f}"
        else:
            return event.event_type.value

    def _update_agent_status(self, agent_name: str, event: "Event") -> None:
        """Update agent status based on received event."""
        if agent_name not in self._agents:
            # Determine agent type from name
            agent_type = "Unknown"
            if "macro" in agent_name.lower():
                agent_type = "Signal"
            elif "stat_arb" in agent_name.lower() or "statarb" in agent_name.lower():
                agent_type = "Signal"
            elif "momentum" in agent_name.lower():
                agent_type = "Signal"
            elif "market_making" in agent_name.lower():
                agent_type = "Signal"
            elif "options" in agent_name.lower() or "vol" in agent_name.lower():
                agent_type = "Signal"
            elif "cio" in agent_name.lower():
                agent_type = "Decision"
            elif "risk" in agent_name.lower():
                agent_type = "Validation"
            elif "compliance" in agent_name.lower():
                agent_type = "Validation"
            elif "execution" in agent_name.lower():
                agent_type = "Execution"
            elif "surveillance" in agent_name.lower():
                agent_type = "Compliance"
            elif "reporting" in agent_name.lower():
                agent_type = "Compliance"

            self._agents[agent_name] = AgentInfo(
                name=agent_name,
                type=agent_type,
            )

        agent = self._agents[agent_name]
        agent.status = AgentStatus.ACTIVE
        agent.last_event_time = event.timestamp
        agent.event_count += 1

        # Calculate latency (time from event creation to now)
        if event.timestamp:
            now = datetime.now(timezone.utc)
            event_time = event.timestamp
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            latency = (now - event_time).total_seconds() * 1000  # ms
            # Rolling average of latency (weighted toward recent)
            if agent.latency_ms > 0:
                agent.latency_ms = agent.latency_ms * 0.8 + latency * 0.2
            else:
                agent.latency_ms = latency

    def _handle_signal_event(self, event: "Event") -> None:
        """Handle signal event."""
        audit_dict = event.to_audit_dict()
        signal = SignalInfo(
            agent=event.source_agent,
            symbol=audit_dict.get("symbol", ""),
            direction=audit_dict.get("direction", "FLAT").upper(),
            confidence=audit_dict.get("confidence", 0.0),
            strength=audit_dict.get("strength", 0.0),
            timestamp=event.timestamp,
            rationale=audit_dict.get("rationale", ""),
        )
        self._signals[event.source_agent] = signal

    def _handle_decision_event(self, event: "Event") -> None:
        """Handle decision event."""
        audit_dict = event.to_audit_dict()
        quantity = audit_dict.get("quantity", 0)
        target_price = audit_dict.get("target_price") or audit_dict.get("price") or 0.0
        estimated_value = abs(quantity) * target_price if target_price else 0.0

        decision = DecisionInfo(
            decision_id=event.event_id,
            symbol=audit_dict.get("symbol", ""),
            direction=audit_dict.get("action", "").upper() if audit_dict.get("action") else "HOLD",
            quantity=quantity,
            conviction=audit_dict.get("conviction_score", 0.0),
            timestamp=event.timestamp,
            rationale=audit_dict.get("rationale", ""),
            estimated_value=estimated_value,
        )
        self._decisions.appendleft(decision)

    def _handle_validated_decision_event(self, event: "Event") -> None:
        """Handle validated decision event - update decision status."""
        audit_dict = event.to_audit_dict()
        decision_id = audit_dict.get("original_decision_id", "")
        approved = audit_dict.get("approved", False)
        rejection_reason = audit_dict.get("rejection_reason", "")

        # Update the status of the matching decision
        for dec in self._decisions:
            if dec.decision_id == decision_id:
                dec.status = "APPROVED" if approved else "REJECTED"
                if not approved and rejection_reason:
                    dec.rejection_reason = rejection_reason
                break

    def _handle_fill_event(self, event: "Event") -> None:
        """Handle fill event - update positions and track closed positions."""
        audit_dict = event.to_audit_dict()
        symbol = audit_dict.get("symbol", "")
        if not symbol:
            return

        fill_qty = audit_dict.get("filled_quantity", 0)
        fill_price = audit_dict.get("fill_price", 0.0)
        side = audit_dict.get("side", "buy")

        # Update or create position
        if symbol in self._positions:
            pos = self._positions[symbol]
            old_qty = pos.quantity

            if side == "buy":
                # Add to position
                total_cost = (pos.entry_price * abs(pos.quantity)) + (fill_price * fill_qty)
                pos.quantity += fill_qty
                if pos.quantity != 0:
                    pos.entry_price = total_cost / abs(pos.quantity)
            else:
                # Reduce position
                pos.quantity -= fill_qty

            pos.current_price = fill_price

            # Check if position was closed
            if pos.quantity == 0 or (old_qty > 0 and pos.quantity < 0) or (old_qty < 0 and pos.quantity > 0):
                # Position closed - calculate realized P&L
                realized_pnl = (fill_price - pos.entry_price) * abs(old_qty)
                if old_qty < 0:  # Short position
                    realized_pnl = -realized_pnl

                self._closed_positions.appendleft({
                    "symbol": symbol,
                    "quantity": old_qty,
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(fill_price, 2),
                    "pnl": round(realized_pnl, 2),
                    "pnl_pct": round((realized_pnl / (pos.entry_price * abs(old_qty))) * 100, 2) if pos.entry_price != 0 else 0,
                    "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                    "closed_at": event.timestamp.isoformat(),
                    "side": "LONG" if old_qty > 0 else "SHORT",
                    "conviction": round(pos.conviction, 2) if pos.conviction else 0,
                    "signal": pos.signal or "",
                    # Regime/Session attribution for performance analysis
                    "entry_regime": pos.regime or "unknown",
                    "exit_regime": self._get_current_regime(),
                    "entry_session": pos.session or "unknown",
                    "exit_session": get_current_trading_session(event.timestamp),
                })

                # Remove fully closed position
                if pos.quantity == 0:
                    del self._positions[symbol]
        else:
            # Try to get conviction and signal from recent decisions
            entry_conviction = 0.0
            entry_signal = "LONG" if side == "buy" else "SHORT"
            for dec in self._decisions:
                if dec.symbol == symbol:
                    entry_conviction = dec.conviction
                    entry_signal = dec.direction
                    break

            self._positions[symbol] = PositionInfo(
                symbol=symbol,
                quantity=fill_qty if side == "buy" else -fill_qty,
                entry_price=fill_price,
                current_price=fill_price,
                entry_time=event.timestamp,
                conviction=entry_conviction,
                signal=entry_signal,
                regime=self._get_current_regime(),
                session=get_current_trading_session(event.timestamp),
            )

        # Update metrics
        self._metrics.total_trades += 1
        self._metrics.position_count = len(self._positions)

    def _handle_risk_alert_event(self, event: "Event") -> None:
        """Handle risk alert event."""
        audit_dict = event.to_audit_dict()

        # Update risk limit if applicable
        alert_type = audit_dict.get("alert_type", "")
        current_value = audit_dict.get("current_value", 0.0)
        threshold_value = audit_dict.get("threshold_value", 0.0)

        if alert_type and threshold_value > 0:
            usage = (current_value / threshold_value) * 100 if threshold_value > 0 else 0
            status = "ok"
            if audit_dict.get("halt_trading", False):
                status = "breach"
            elif usage >= 75:
                status = "warning"

            self._risk_limits[alert_type] = RiskLimit(
                name=alert_type,
                current=current_value,
                limit=threshold_value,
                usage=usage,
                status=status,
            )

        # Add alert
        severity = audit_dict.get("severity", "info")
        self._alerts.appendleft({
            "title": alert_type,
            "message": audit_dict.get("message", ""),
            "severity": severity,
            "time": event.timestamp.strftime("%H:%M:%S"),
        })

    def _handle_kill_switch_event(self, event: "Event") -> None:
        """Handle kill switch event."""
        audit_dict = event.to_audit_dict()
        self._kill_switch_active = audit_dict.get("activated", False)

        # Add alert
        self._alerts.appendleft({
            "title": "Kill Switch",
            "message": audit_dict.get("reason", "Kill switch state changed"),
            "severity": "critical" if self._kill_switch_active else "info",
            "time": event.timestamp.strftime("%H:%M:%S"),
        })

    def _handle_market_data_event(self, event: "Event") -> None:
        """Handle market data event - update position prices."""
        audit_dict = event.to_audit_dict()
        symbol = audit_dict.get("symbol", "")
        last_price = audit_dict.get("last", 0.0)

        if symbol in self._positions and last_price > 0:
            pos = self._positions[symbol]
            pos.current_price = last_price
            pos.market_value = pos.quantity * last_price

            # Calculate P&L
            if pos.entry_price > 0:
                pos.pnl = (last_price - pos.entry_price) * pos.quantity
                pos.pnl_pct = ((last_price - pos.entry_price) / pos.entry_price) * 100

    def register_agent(self, name: str, agent_type: str) -> None:
        """Register an agent for tracking."""
        self._agents[name] = AgentInfo(name=name, type=agent_type)

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        """Update portfolio metrics."""
        self._metrics.total_pnl = metrics.get("total_pnl", self._metrics.total_pnl)
        self._metrics.today_pnl = metrics.get("today_pnl", self._metrics.today_pnl)
        self._metrics.sharpe_ratio = metrics.get("sharpe_ratio", self._metrics.sharpe_ratio)
        self._metrics.win_rate = metrics.get("win_rate", self._metrics.win_rate)
        self._metrics.drawdown = metrics.get("drawdown", self._metrics.drawdown)
        self._metrics.position_count = len(self._positions)

    def update_equity_curve(self, timestamp: str, value: float) -> None:
        """Add a point to the equity curve with full timestamp."""
        # Convert to full ISO timestamp if only time provided
        now = datetime.now(timezone.utc)
        if 'T' not in timestamp and len(timestamp) <= 8:  # HH:MM:SS format
            iso_timestamp = now.strftime("%Y-%m-%dT") + timestamp + "Z"
        else:
            iso_timestamp = timestamp

        # Calculate unix milliseconds
        try:
            if iso_timestamp.endswith('Z'):
                dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(iso_timestamp)
            unix_ms = int(dt.timestamp() * 1000)
        except ValueError:
            unix_ms = int(now.timestamp() * 1000)

        self._equity_curve.append((iso_timestamp, value, unix_ms))

        # Save to file periodically (every 10 updates) - non-blocking
        self._equity_save_counter += 1
        if self._equity_save_counter >= 10:
            self._schedule_equity_save()
            self._equity_save_counter = 0

    def _schedule_equity_save(self) -> None:
        """Schedule equity history save to run in executor (non-blocking)."""
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._save_equity_history_sync)
        except RuntimeError:
            # No running loop (e.g., during shutdown), save synchronously
            self._save_equity_history_sync()

    def _save_equity_history_sync(self) -> None:
        """Synchronous save method to run in executor."""
        try:
            self._equity_file_path.parent.mkdir(parents=True, exist_ok=True)
            history = [
                {"timestamp": ts, "value": val, "unix_ms": unix_ms}
                for ts, val, unix_ms in self._equity_curve
            ]
            with open(self._equity_file_path, 'w') as f:
                json.dump({"history": history, "updated": datetime.now(timezone.utc).isoformat()}, f)
        except Exception as e:
            logger.warning(f"Could not save equity history: {e}")

    def _load_equity_history(self) -> None:
        """Load equity history from file on startup."""
        try:
            if self._equity_file_path.exists():
                with open(self._equity_file_path, 'r') as f:
                    data = json.load(f)
                    for item in data.get("history", []):
                        ts = item.get("timestamp", "")
                        val = item.get("value", 0.0)
                        unix_ms = item.get("unix_ms", 0)
                        self._equity_curve.append((ts, val, unix_ms))
                logger.info(f"Loaded {len(self._equity_curve)} equity history points")
        except Exception as e:
            logger.warning(f"Could not load equity history: {e}")

    def _load_position_history(self) -> None:
        """Load position entry metadata from audit logs (decisions.jsonl, trades.jsonl).

        This allows displaying entry_time, conviction, and signal direction for
        positions that were opened before the current dashboard session.
        """
        try:
            # Use absolute path based on project root (parent of dashboard/)
            project_root = Path(__file__).parent.parent
            logs_dir = project_root / "logs"

            # Load from decisions.jsonl - contains conviction scores
            decisions_file = logs_dir / "decisions.jsonl"
            if decisions_file.exists():
                with open(decisions_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("entry_type") == "decision":
                                details = entry.get("details", {})
                                symbol = details.get("symbol", "")
                                action = details.get("action", "")
                                conviction = details.get("conviction_score", 0.0)
                                timestamp = entry.get("timestamp", "")

                                if symbol:
                                    # Determine signal direction from action
                                    signal = "LONG" if action == "buy" else "SHORT" if action == "sell" else "FLAT"

                                    # Keep most recent decision per symbol
                                    self._position_history[symbol] = {
                                        "entry_time": timestamp,
                                        "conviction": conviction,
                                        "signal": signal,
                                        "action": action,
                                        "target_price": None,
                                        "stop_loss": None,
                                    }
                        except json.JSONDecodeError:
                            continue
                logger.info(f"Loaded position history for {len(self._position_history)} symbols from decisions.jsonl")

            # Also check trades.jsonl for actual fill times
            trades_file = logs_dir / "trades.jsonl"
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("entry_type") == "trade":
                                details = entry.get("details", {})
                                symbol = details.get("symbol", "")
                                side = details.get("side", "")
                                timestamp = entry.get("timestamp", "")

                                if symbol and symbol in self._position_history:
                                    # Update entry_time with actual fill time if available
                                    self._position_history[symbol]["entry_time"] = timestamp
                                elif symbol:
                                    signal = "LONG" if side == "buy" else "SHORT" if side == "sell" else "FLAT"
                                    self._position_history[symbol] = {
                                        "entry_time": timestamp,
                                        "conviction": 0.0,
                                        "signal": signal,
                                        "action": side,
                                        "target_price": None,
                                        "stop_loss": None,
                                    }
                        except json.JSONDecodeError:
                            continue
                logger.info(f"Updated position history with trade timestamps")

            # Load TP/SL and source_agent from audit.jsonl (signal events)
            audit_file = logs_dir / "audit.jsonl"
            if audit_file.exists():
                with open(audit_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("entry_type") == "event":
                                details = entry.get("details", {})
                                event_type = details.get("event_type", "")
                                if event_type == "signal":
                                    symbol = details.get("symbol", "")
                                    target_price = details.get("target_price")
                                    stop_loss = details.get("stop_loss")
                                    source_agent = details.get("source_agent", "")
                                    direction = details.get("direction", "")

                                    if symbol and symbol in self._position_history:
                                        # Get the position's direction from stored signal
                                        pos_signal = self._position_history[symbol].get("signal", "")
                                        pos_action = self._position_history[symbol].get("action", "")

                                        # Determine position direction (LONG or SHORT)
                                        pos_direction = ""
                                        if "LONG" in pos_signal.upper() or pos_action == "buy":
                                            pos_direction = "LONG"
                                        elif "SHORT" in pos_signal.upper() or pos_action == "sell":
                                            pos_direction = "SHORT"

                                        # Only update TP/SL if signal direction MATCHES position direction
                                        # This prevents LONG positions from getting SHORT signal TP/SL
                                        signal_direction = direction.upper() if direction else ""
                                        directions_match = (
                                            (pos_direction == "LONG" and signal_direction == "LONG") or
                                            (pos_direction == "SHORT" and signal_direction == "SHORT")
                                        )

                                        if directions_match:
                                            if target_price is not None:
                                                self._position_history[symbol]["target_price"] = target_price
                                            if stop_loss is not None:
                                                self._position_history[symbol]["stop_loss"] = stop_loss
                                            # Update signal with agent name and direction
                                            if source_agent and direction:
                                                # Format: "AgentName: DIRECTION"
                                                agent_short = source_agent.replace("Agent", "")
                                                self._position_history[symbol]["signal"] = f"{agent_short}: {direction.upper()}"
                                                self._position_history[symbol]["source_agent"] = source_agent
                        except json.JSONDecodeError:
                            continue
                logger.info(f"Updated position history with TP/SL and agent info from signals")
        except Exception as e:
            logger.warning(f"Could not load position history: {e}")

    def _load_closed_positions_history(self) -> None:
        """Load historical closed positions from trades.jsonl.

        Analyzes trade history to detect position closes by tracking
        running positions per symbol and identifying when they go to zero
        or flip direction.
        """
        try:
            # Use absolute path based on project root (parent of dashboard/)
            project_root = Path(__file__).parent.parent
            trades_file = project_root / "logs" / "trades.jsonl"
            decisions_file = project_root / "logs" / "decisions.jsonl"
            logger.info(f"[CLOSED_POSITIONS] Loading from: {trades_file}")
            logger.info(f"[CLOSED_POSITIONS] Current deque size before load: {len(self._closed_positions)}")

            if not trades_file.exists():
                logger.warning(f"[CLOSED_POSITIONS] trades.jsonl not found at {trades_file}")
                return

            logger.info(f"[CLOSED_POSITIONS] File found, starting parse...")

            # First, load decisions keyed by event_id to get agent info
            # Key: decision_id (event_id) -> {agent, conviction, action}
            import re
            decision_lookup: dict[str, dict] = {}
            if decisions_file.exists():
                with open(decisions_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("entry_type") != "decision":
                                continue
                            details = entry.get("details", {})
                            decision_id = entry.get("event_id", "")
                            conviction = details.get("conviction_score", 0.0)
                            rationale = details.get("rationale", "")
                            action = details.get("action", "")

                            # Extract primary agent from rationale (e.g., "MomentumAgent (20%): short")
                            agent_match = re.search(r'-\s*(\w+Agent)\s*\(\d+%\)', rationale)
                            primary_agent = agent_match.group(1) if agent_match else ""

                            if decision_id:
                                decision_lookup[decision_id] = {
                                    "agent": primary_agent,
                                    "conviction": conviction,
                                    "action": action,
                                    "rationale": rationale,
                                }
                        except (json.JSONDecodeError, Exception):
                            continue
                logger.info(f"[CLOSED_POSITIONS] Loaded {len(decision_lookup)} decisions")

            # Track running positions per symbol: {symbol: {"qty": int, "avg_cost": float, "decision_id": str}}
            running_positions: dict[str, dict] = {}
            closed_positions_raw: list[dict] = []

            def get_decision_info(decision_id: str) -> dict:
                """Get agent and conviction from decision lookup."""
                info = decision_lookup.get(decision_id, {"agent": "", "conviction": 0.0, "action": "", "rationale": ""}).copy()
                # If no agent found, check if it's a special decision type
                if not info.get("agent"):
                    rationale = info.get("rationale", "")
                    if "EMERGENCY DELEVERAGING" in rationale:
                        info["agent"] = "Risk"
                        info["conviction"] = 0.0  # No signal conviction for risk decisions
                    elif "STOP LOSS" in rationale.upper():
                        info["agent"] = "StopLoss"
                        info["conviction"] = 0.0
                return info

            with open(trades_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("entry_type") != "trade":
                            continue

                        details = entry.get("details", {})
                        symbol = details.get("symbol", "")
                        side = details.get("side", "")
                        quantity = details.get("quantity", 0)
                        price = details.get("price", 0)
                        timestamp = entry.get("timestamp", "")

                        decision_id = details.get("decision_id", "")

                        if not symbol or not side or quantity <= 0:
                            continue

                        # Initialize tracking for new symbol
                        if symbol not in running_positions:
                            running_positions[symbol] = {
                                "qty": 0,
                                "avg_cost": 0.0,
                                "entry_time": None,
                                "side": None,
                                "decision_id": "",
                            }

                        pos = running_positions[symbol]
                        old_qty = pos["qty"]

                        # Update position
                        if side == "buy":
                            if pos["qty"] >= 0:
                                # Adding to long or opening long
                                total_cost = pos["avg_cost"] * pos["qty"] + price * quantity
                                pos["qty"] += quantity
                                pos["avg_cost"] = total_cost / pos["qty"] if pos["qty"] > 0 else 0
                                if pos["entry_time"] is None:
                                    pos["entry_time"] = timestamp
                                    pos["side"] = "LONG"
                                    pos["decision_id"] = decision_id
                            else:
                                # Covering short
                                pos["qty"] += quantity
                                if pos["qty"] >= 0:
                                    # Position closed or flipped
                                    pnl = (pos["avg_cost"] - price) * abs(old_qty)
                                    pnl_pct = (pnl / (pos["avg_cost"] * abs(old_qty))) * 100 if pos["avg_cost"] > 0 else 0
                                    # Get agent info from decision
                                    dec_info = get_decision_info(pos["decision_id"])
                                    agent_name = dec_info["agent"].replace("Agent", "") if dec_info["agent"] else ""
                                    signal_str = f"{agent_name}: SHORT" if agent_name else "SHORT"
                                    closed_positions_raw.append({
                                        "symbol": symbol,
                                        "side": "SHORT",
                                        "signal": signal_str,
                                        "entry_price": round(pos["avg_cost"], 2),
                                        "exit_price": round(price, 2),
                                        "quantity": -abs(old_qty),  # Negative for shorts
                                        "pnl": round(pnl, 2),
                                        "pnl_pct": round(pnl_pct, 2),
                                        "entry_time": pos["entry_time"],
                                        "closed_at": timestamp,
                                        "conviction": dec_info["conviction"],
                                        "target_price": None,
                                        "stop_loss": None,
                                    })
                                    if pos["qty"] > 0:
                                        # Flipped to long
                                        pos["avg_cost"] = price
                                        pos["entry_time"] = timestamp
                                        pos["side"] = "LONG"
                                        pos["decision_id"] = decision_id
                                    else:
                                        # Fully closed
                                        pos["entry_time"] = None
                                        pos["side"] = None
                                        pos["decision_id"] = ""
                        else:  # sell
                            if pos["qty"] <= 0:
                                # Adding to short or opening short
                                total_cost = abs(pos["avg_cost"] * pos["qty"]) + price * quantity
                                pos["qty"] -= quantity
                                pos["avg_cost"] = total_cost / abs(pos["qty"]) if pos["qty"] != 0 else 0
                                if pos["entry_time"] is None:
                                    pos["entry_time"] = timestamp
                                    pos["side"] = "SHORT"
                                    pos["decision_id"] = decision_id
                            else:
                                # Selling long
                                pos["qty"] -= quantity
                                if pos["qty"] <= 0:
                                    # Position closed or flipped
                                    pnl = (price - pos["avg_cost"]) * old_qty
                                    pnl_pct = (pnl / (pos["avg_cost"] * old_qty)) * 100 if pos["avg_cost"] > 0 else 0
                                    # Get agent info from decision
                                    dec_info = get_decision_info(pos["decision_id"])
                                    agent_name = dec_info["agent"].replace("Agent", "") if dec_info["agent"] else ""
                                    signal_str = f"{agent_name}: LONG" if agent_name else "LONG"
                                    closed_positions_raw.append({
                                        "symbol": symbol,
                                        "side": "LONG",
                                        "signal": signal_str,
                                        "entry_price": round(pos["avg_cost"], 2),
                                        "exit_price": round(price, 2),
                                        "quantity": old_qty,
                                        "pnl": round(pnl, 2),
                                        "pnl_pct": round(pnl_pct, 2),
                                        "entry_time": pos["entry_time"],
                                        "closed_at": timestamp,
                                        "conviction": dec_info["conviction"],
                                        "target_price": None,
                                        "stop_loss": None,
                                    })
                                    if pos["qty"] < 0:
                                        # Flipped to short
                                        pos["avg_cost"] = price
                                        pos["entry_time"] = timestamp
                                        pos["side"] = "SHORT"
                                        pos["decision_id"] = decision_id
                                    else:
                                        # Fully closed
                                        pos["entry_time"] = None
                                        pos["side"] = None
                                        pos["decision_id"] = ""

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Error processing trade: {e}")
                        continue

            # Sort by close time (most recent first) and add to closed_positions
            closed_positions_raw.sort(key=lambda x: x.get("closed_at", ""), reverse=True)

            logger.info(f"[CLOSED_POSITIONS] Found {len(closed_positions_raw)} closed positions to load")

            for pos in closed_positions_raw:
                self._closed_positions.append(pos)

            logger.info(f"[CLOSED_POSITIONS] Deque size after load: {len(self._closed_positions)}")

            # Log first few positions for verification
            if self._closed_positions:
                sample = list(self._closed_positions)[:3]
                for i, pos in enumerate(sample):
                    logger.info(f"[CLOSED_POSITIONS] Sample {i+1}: {pos.get('symbol')} {pos.get('side')} closed_at={pos.get('closed_at')}")

        except Exception as e:
            logger.exception(f"[CLOSED_POSITIONS] Failed to load history: {e}")

    def _populate_analytics_from_history(self) -> None:
        """Populate analytics components from closed positions history.

        This feeds historical trades to:
        - RollingMetricsCalculator (for Sharpe, Sortino, drawdown)
        - SessionPerformanceTracker (for win rate by session)
        - StrategyComparisonTracker (for strategy performance)
        """
        try:
            if not self._closed_positions:
                logger.info("[ANALYTICS] No closed positions to populate analytics from")
                return

            # Sort positions by close time (oldest first) for proper equity curve building
            sorted_positions = sorted(
                self._closed_positions,
                key=lambda x: x.get("closed_at", ""),
                reverse=False  # Oldest first
            )

            # Build equity curve from closed positions
            # Start with a base equity value (100k as reference)
            base_equity = 100000.0
            running_equity = base_equity
            equity_points_added = 0

            populated_count = 0
            for pos in sorted_positions:
                try:
                    symbol = pos.get("symbol", "")
                    pnl = pos.get("pnl", 0.0)

                    # Parse entry and exit times
                    entry_time_str = pos.get("entry_time", "")
                    exit_time_str = pos.get("closed_at", "")

                    if not entry_time_str or not exit_time_str:
                        continue

                    # Parse timestamps
                    entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))

                    # Ensure timezone awareness
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    if exit_time.tzinfo is None:
                        exit_time = exit_time.replace(tzinfo=timezone.utc)

                    # Update running equity and add equity point
                    running_equity += pnl
                    self._rolling_metrics.add_equity_point(exit_time, running_equity)
                    equity_points_added += 1

                    # Extract strategy/agent from signal field (e.g., "Momentum: LONG")
                    signal = pos.get("signal", "")
                    strategy = signal.split(":")[0].strip() if ":" in signal else "Unknown"
                    if strategy and not strategy.endswith("Agent"):
                        strategy = f"{strategy}Agent"

                    # Record in rolling metrics
                    self._rolling_metrics.add_trade({
                        "pnl": pnl,
                        "close_time": exit_time,
                        "open_time": entry_time,
                        "symbol": symbol,
                    })

                    # Record in session performance
                    self._session_performance.record_trade(
                        open_time=entry_time,
                        close_time=exit_time,
                        pnl=pnl,
                        symbol=symbol,
                    )

                    # Calculate holding hours
                    holding_hours = (exit_time - entry_time).total_seconds() / 3600.0

                    # Record in strategy comparison
                    self._strategy_comparison.record_trade(
                        strategy=strategy,
                        symbol=symbol,
                        pnl=pnl,
                        holding_hours=holding_hours,
                        timestamp=exit_time,
                    )

                    populated_count += 1

                except (ValueError, TypeError) as e:
                    logger.debug(f"[ANALYTICS] Error parsing position for analytics: {e}")
                    continue

            logger.info(f"[ANALYTICS] Populated {populated_count} trades into analytics components")
            logger.info(f"[ANALYTICS] Added {equity_points_added} equity points for Sharpe calculation")

        except Exception as e:
            logger.exception(f"[ANALYTICS] Failed to populate analytics from history: {e}")

    def get_position_metadata(self, symbol: str) -> dict:
        """Get historical metadata for a position (entry_time, conviction, signal, TP, SL, agent)."""
        return self._position_history.get(symbol, {
            "entry_time": None,
            "conviction": 0.0,
            "signal": "",
            "source_agent": "",
            "target_price": None,
            "stop_loss": None,
        })

    def get_agents(self) -> list[dict]:
        """
        Get all agent statuses from the orchestrator with comprehensive health data.

        Returns real-time agent status including:
        - Running state (active/idle/stopped/error)
        - Events processed count
        - Error count
        - Last heartbeat time
        - Uptime
        - Health score
        """
        # Try to get live status from orchestrator
        if self._orchestrator:
            try:
                orchestrator_status = self._orchestrator.get_status()
                agents_data = orchestrator_status.get("agents", {})

                # Helper function to update agent from status dict
                def update_agent_from_status(agent: AgentInfo, agent_status: dict) -> None:
                    """Update AgentInfo fields from orchestrator status dict."""
                    running = agent_status.get("running", False)
                    errors = agent_status.get("errors", 0)

                    # Set status based on running state and errors
                    if errors > 0 and not running:
                        agent.status = AgentStatus.ERROR
                        agent.error_message = f"{errors} errors occurred"
                    elif running:
                        agent.status = AgentStatus.ACTIVE
                        agent.error_message = None
                    else:
                        agent.status = AgentStatus.STOPPED

                    # Use correct field names from BaseAgent.get_status()
                    agent.event_count = agent_status.get("events_processed", 0)
                    agent.error_count = errors
                    agent.uptime_seconds = agent_status.get("uptime_seconds", 0.0)

                    # Parse last_heartbeat for display
                    last_hb = agent_status.get("last_heartbeat")
                    if last_hb:
                        try:
                            agent.last_event_time = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            pass

                # Update our tracked agents with live status from signal agents
                for agent_status in agents_data.get("signal", []):
                    if agent_status is None:
                        continue
                    name = agent_status.get("name")
                    if name:
                        if name not in self._agents:
                            self._agents[name] = AgentInfo(name=name, type="Signal")
                        update_agent_from_status(self._agents[name], agent_status)

                # Update core agents with comprehensive status
                for agent_type, type_label in [
                    ("cio", "Decision"),
                    ("risk", "Validation"),
                    ("compliance", "Validation"),
                    ("execution", "Execution"),
                    ("surveillance", "Surveillance"),
                    ("transaction_reporting", "Reporting"),
                ]:
                    agent_status = agents_data.get(agent_type)
                    if agent_status:
                        name = agent_status.get("name")
                        if name:
                            if name not in self._agents:
                                self._agents[name] = AgentInfo(name=name, type=type_label)
                            update_agent_from_status(self._agents[name], agent_status)

            except Exception as e:
                logger.warning(f"Error getting agent status from orchestrator: {e}")

        # Sort agents by type for consistent display order
        type_order = {"Decision": 0, "Signal": 1, "Validation": 2, "Execution": 3, "Surveillance": 4, "Reporting": 5}
        sorted_agents = sorted(
            self._agents.values(),
            key=lambda a: (type_order.get(a.type, 99), a.name)
        )

        # Add enabled state, LLM flag, and essential flag to each agent
        result = []
        for agent in sorted_agents:
            agent_dict = agent.to_dict()
            agent_name = agent.name

            # Check if this is an essential agent (cannot be disabled)
            is_essential = agent_name in self._essential_agents

            # Default enabled state: check if it's set, otherwise use orchestrator state
            if agent_name not in self._agent_enabled_state:
                # Essential agents are always enabled
                if is_essential:
                    self._agent_enabled_state[agent_name] = True
                # Initialize from orchestrator config if available
                elif self._orchestrator:
                    try:
                        agent_config = self._orchestrator._config.get("agents", {})
                        # Use config key mapping to find the right config section
                        config_key = self._agent_config_key_map.get(agent_name)
                        if config_key and config_key in agent_config:
                            cfg = agent_config[config_key]
                            if isinstance(cfg, dict):
                                self._agent_enabled_state[agent_name] = cfg.get("enabled", True)
                            else:
                                self._agent_enabled_state[agent_name] = True
                        else:
                            # Fallback: default to True for unknown agents
                            self._agent_enabled_state[agent_name] = True
                    except Exception:
                        self._agent_enabled_state[agent_name] = True
                else:
                    self._agent_enabled_state[agent_name] = True

            agent_dict["enabled"] = self._agent_enabled_state.get(agent_name, True)
            agent_dict["uses_llm"] = agent_name in self._llm_agents
            agent_dict["is_essential"] = is_essential
            result.append(agent_dict)

        return result

    async def toggle_agent(self, agent_name: str, enabled: bool) -> dict:
        """
        Toggle an agent's enabled state at runtime.

        Args:
            agent_name: Name of the agent to toggle
            enabled: Whether to enable (True) or disable (False) the agent

        Returns:
            Dict with the new state and any error message
        """
        # Check if this is an essential agent - cannot be disabled
        if agent_name in self._essential_agents and not enabled:
            logger.warning(f"Attempted to disable essential agent {agent_name} - rejected")
            return {
                "agent_name": agent_name,
                "enabled": True,  # Essential agents stay enabled
                "success": False,
                "message": f"Cannot disable essential agent {agent_name} - it is required for system operation",
                "uses_llm": agent_name in self._llm_agents,
                "is_essential": True,
            }

        # Update the state tracking
        self._agent_enabled_state[agent_name] = enabled

        # Try to actually enable/disable the agent in the orchestrator
        success = False
        message = ""

        if self._orchestrator:
            try:
                # Find the agent in signal_agents
                for agent in self._signal_agents:
                    if hasattr(agent, 'name') and agent.name == agent_name:
                        if enabled:
                            # Re-enable: start the agent
                            if hasattr(agent, '_enabled'):
                                agent._enabled = True
                            if hasattr(agent, 'start') and not agent._running:
                                await agent.start()
                            success = True
                            message = f"Agent {agent_name} enabled"
                        else:
                            # Disable: stop the agent
                            if hasattr(agent, '_enabled'):
                                agent._enabled = False
                            if hasattr(agent, 'stop') and agent._running:
                                await agent.stop()
                            success = True
                            message = f"Agent {agent_name} disabled"
                        break

                # Check core agents
                for core_agent in [self._cio_agent, self._risk_agent, self._execution_agent, self._compliance_agent]:
                    if core_agent and hasattr(core_agent, 'name') and core_agent.name == agent_name:
                        if enabled:
                            if hasattr(core_agent, '_enabled'):
                                core_agent._enabled = True
                            success = True
                            message = f"Agent {agent_name} enabled"
                        else:
                            if hasattr(core_agent, '_enabled'):
                                core_agent._enabled = False
                            success = True
                            message = f"Agent {agent_name} disabled (core agents cannot be fully stopped)"
                        break

                if not success:
                    message = f"Agent {agent_name} state tracked but not found in orchestrator"
                    success = True  # Still mark as success since we tracked the state

            except Exception as e:
                logger.exception(f"Error toggling agent {agent_name}: {e}")
                message = f"Error: {str(e)}"
        else:
            message = "Orchestrator not connected, state tracked locally"
            success = True

        logger.info(f"Agent toggle: {agent_name} -> enabled={enabled}, success={success}, message={message}")

        return {
            "agent_name": agent_name,
            "enabled": enabled,
            "success": success,
            "message": message,
            "uses_llm": agent_name in self._llm_agents,
            "is_essential": agent_name in self._essential_agents,
        }

    def get_agent_states(self) -> dict[str, bool]:
        """Get all agent enabled states."""
        return dict(self._agent_enabled_state)

    async def get_positions_async(self) -> list[dict]:
        """Get all positions from broker asynchronously."""
        positions = []
        current_symbols = set()

        # Try to get positions from broker
        if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected:
            try:
                portfolio_state = await self._broker.get_portfolio_state()
                for symbol, pos in portfolio_state.positions.items():
                    current_symbols.add(symbol)
                    pnl_pct = 0.0
                    if pos.avg_cost > 0:
                        pnl_pct = ((pos.market_value / (pos.avg_cost * pos.quantity)) - 1) * 100 if pos.quantity != 0 else 0

                    # IB returns avg_cost and market_value as notional (price × multiplier)
                    # Divide by multiplier to get actual per-unit price
                    multiplier = get_contract_multiplier(symbol)
                    notional_per_contract = pos.market_value / pos.quantity if pos.quantity != 0 else 0
                    entry_price_actual = pos.avg_cost / multiplier
                    current_price_actual = notional_per_contract / multiplier

                    # Get historical metadata from audit logs
                    history = self.get_position_metadata(symbol)

                    # Get TP/SL from history, or calculate defaults based on position direction
                    target_price = history.get("target_price")
                    stop_loss = history.get("stop_loss")

                    # VALIDATION: Check if existing TP/SL makes sense for position direction
                    # If values are inverted (from old signals with wrong direction), reset them
                    if entry_price_actual > 0 and target_price is not None and stop_loss is not None:
                        if pos.quantity > 0:  # LONG position
                            # For LONG: TP must be > entry, SL must be < entry
                            if target_price <= entry_price_actual or stop_loss >= entry_price_actual:
                                # Inverted values - recalculate
                                target_price = None
                                stop_loss = None
                        elif pos.quantity < 0:  # SHORT position
                            # For SHORT: TP must be < entry, SL must be > entry
                            if target_price >= entry_price_actual or stop_loss <= entry_price_actual:
                                # Inverted values - recalculate
                                target_price = None
                                stop_loss = None

                    # If TP/SL missing, calculate defaults based on position direction
                    # LONG (qty > 0): TP = +4%, SL = -2%
                    # SHORT (qty < 0): TP = -4%, SL = +2%
                    if target_price is None and entry_price_actual > 0:
                        if pos.quantity > 0:  # LONG
                            target_price = entry_price_actual * 1.04
                        elif pos.quantity < 0:  # SHORT
                            target_price = entry_price_actual * 0.96

                    if stop_loss is None and entry_price_actual > 0:
                        if pos.quantity > 0:  # LONG
                            stop_loss = entry_price_actual * 0.98
                        elif pos.quantity < 0:  # SHORT
                            stop_loss = entry_price_actual * 1.02

                    positions.append({
                        "symbol": symbol,
                        "quantity": pos.quantity,
                        "entry_price": entry_price_actual,
                        "current_price": current_price_actual,
                        "pnl": round(pos.unrealized_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "market_value": round(pos.market_value, 2),
                        "entry_time": history.get("entry_time"),
                        "conviction": history.get("conviction", 0),  # Raw 0-1 value, frontend converts to %
                        "signal": history.get("signal", ""),
                        "target_price": target_price,
                        "stop_loss": stop_loss,
                    })

                    # Track position for closed position detection (include historical metadata)
                    self._broker_positions[symbol] = {
                        "quantity": pos.quantity,
                        "entry_price": entry_price_actual,
                        "current_price": current_price_actual,
                        "pnl": round(pos.unrealized_pnl, 2),
                        "entry_time": history.get("entry_time"),
                        "conviction": history.get("conviction", 0),  # Raw 0-1 value
                        "signal": history.get("signal", ""),
                        "target_price": target_price,  # Use calculated value
                        "stop_loss": stop_loss,  # Use calculated value
                    }

                # Detect closed positions (symbols that were in previous update but not in current)
                closed_symbols = set(self._broker_positions.keys()) - current_symbols
                for symbol in closed_symbols:
                    prev_pos = self._broker_positions.pop(symbol)
                    # Add to closed positions
                    realized_pnl = prev_pos.get("pnl", 0)
                    entry_price = prev_pos.get("entry_price", 0)
                    qty = prev_pos.get("quantity", 0)

                    self._closed_positions.appendleft({
                        "symbol": symbol,
                        "quantity": qty,
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(prev_pos.get("current_price", entry_price), 2),
                        "pnl": round(realized_pnl, 2),
                        "pnl_pct": round((realized_pnl / (entry_price * abs(qty))) * 100, 2) if entry_price and qty else 0,
                        "entry_time": prev_pos.get("entry_time"),
                        "closed_at": datetime.now(timezone.utc).isoformat(),
                        "side": "LONG" if qty > 0 else "SHORT",
                        "conviction": prev_pos.get("conviction", 0),
                        "signal": prev_pos.get("signal", ""),
                        "target_price": prev_pos.get("target_price"),
                        "stop_loss": prev_pos.get("stop_loss"),
                        # Regime/Session attribution for performance analysis
                        "entry_regime": prev_pos.get("regime", "unknown"),
                        "exit_regime": self._get_current_regime(),
                        "entry_session": prev_pos.get("session", "unknown"),
                        "exit_session": get_current_trading_session(),
                    })
                    logger.info(f"Position closed: {symbol} qty={qty} P&L={realized_pnl}")

            except Exception as e:
                logger.warning(f"Error getting positions from broker: {e}")

        # Fall back to tracked positions if broker not available
        if not positions:
            positions = [pos.to_dict() for pos in self._positions.values()]

        return positions

    def get_positions(self) -> list[dict]:
        """Get all positions (sync wrapper)."""
        # Return tracked positions - async method is called by endpoint
        return [pos.to_dict() for pos in self._positions.values()]

    def get_signals(self) -> list[dict]:
        """Get current signals from all agents."""
        return [signal.to_dict() for signal in self._signals.values()]

    def get_decisions(self) -> list[dict]:
        """Get recent decisions."""
        return [dec.to_dict() for dec in self._decisions]

    async def get_metrics_async(self, positions: list[dict] | None = None) -> dict:
        """Get portfolio metrics from broker asynchronously.

        Args:
            positions: Optional pre-fetched positions to calculate P&L from (for consistency)
        """
        metrics = self._metrics.to_dict()

        # Calculate realized P&L from ALL closed positions
        realized_pnl = sum(pos.get("pnl", 0) for pos in self._closed_positions)
        metrics["realized_pnl"] = round(realized_pnl, 2)

        # Calculate TODAY's realized P&L (only positions closed today)
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_realized_pnl = 0.0
        for pos in self._closed_positions:
            closed_at_str = pos.get("closed_at", "")
            if closed_at_str:
                try:
                    closed_at = datetime.fromisoformat(closed_at_str.replace("Z", "+00:00"))
                    if closed_at >= today_start:
                        today_realized_pnl += pos.get("pnl", 0)
                except (ValueError, TypeError):
                    pass

        # If positions provided, calculate unrealized P&L from them
        unrealized_pnl = 0.0
        if positions:
            unrealized_pnl = sum(pos.get("pnl", 0) for pos in positions)
            metrics["unrealized_pnl"] = round(unrealized_pnl, 2)
            metrics["total_pnl"] = round(unrealized_pnl + realized_pnl, 2)  # Total = open + closed
            metrics["position_count"] = len(positions)

        # Today's P&L = unrealized (current open) + realized from today's closed positions
        metrics["today_pnl"] = round(unrealized_pnl + today_realized_pnl, 2)
        metrics["today_realized_pnl"] = round(today_realized_pnl, 2)

        # Try to get real metrics from broker
        if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected:
            try:
                portfolio_state = await self._broker.get_portfolio_state()

                # Only use broker P&L if we didn't get positions passed in
                if not positions:
                    metrics["position_count"] = len(portfolio_state.positions)
                    # Calculate unrealized P&L from open positions
                    unrealized_pnl = sum(pos.unrealized_pnl for pos in portfolio_state.positions.values())
                    metrics["unrealized_pnl"] = round(unrealized_pnl, 2)
                    # Total P&L = unrealized (open) + realized (closed)
                    metrics["total_pnl"] = round(unrealized_pnl + realized_pnl, 2)
                    # Today's P&L = unrealized (current) + today's realized
                    metrics["today_pnl"] = round(unrealized_pnl + today_realized_pnl, 2)

            except Exception as e:
                logger.warning(f"Error getting metrics from broker: {e}")

        # Try to get risk metrics from risk agent
        if self._risk_agent and hasattr(self._risk_agent, 'get_status'):
            try:
                risk_status = self._risk_agent.get_status()
                risk_state = risk_status.get("risk_state", {})
                if "max_drawdown_pct" in risk_state:
                    metrics["drawdown"] = round(risk_state.get("max_drawdown_pct", 0), 4)
            except Exception as e:
                logger.warning(f"Error getting metrics from risk agent: {e}")

        # Calculate win rate and other metrics from closed positions
        if self._closed_positions:
            closed_list = list(self._closed_positions)
            metrics["total_trades"] = len(closed_list)

            # Win rate calculation
            winning_trades = [pos for pos in closed_list if pos.get("pnl", 0) > 0]
            metrics["win_rate"] = round(len(winning_trades) / len(closed_list), 3) if closed_list else 0.0

        # Get Sharpe ratio and drawdown from rolling metrics calculator (which has been populated from history)
        try:
            rolling_metrics = self._rolling_metrics.calculate_metrics(RollingPeriod.MONTH_1)
            if rolling_metrics.sharpe_ratio is not None:
                metrics["sharpe_ratio"] = round(rolling_metrics.sharpe_ratio, 3)
            # Use drawdown from rolling metrics if not available from risk agent
            if metrics.get("drawdown", 0) == 0 and rolling_metrics.max_drawdown_pct > 0:
                metrics["drawdown"] = round(rolling_metrics.max_drawdown_pct / 100, 4)  # Convert % to decimal
        except Exception as e:
            logger.debug(f"Could not get metrics from rolling metrics calculator: {e}")

        return metrics

    def get_metrics(self) -> dict:
        """Get portfolio metrics (sync wrapper)."""
        return self._metrics.to_dict()

    def get_risk_limits(self) -> list[dict]:
        """Get risk limit statuses from the risk agent."""
        # Try to get real limits from risk agent
        if self._risk_agent and hasattr(self._risk_agent, 'get_status'):
            try:
                risk_status = self._risk_agent.get_status()
                risk_state = risk_status.get("risk_state", {})
                limits = risk_status.get("limits", {})

                # Build risk limits from actual agent state
                result = []

                # Position size limit
                max_pos_pct = limits.get("max_position_size_pct", 5.0)
                current_pos_pct = risk_state.get("largest_position_pct", 0.0)
                usage = (current_pos_pct / max_pos_pct * 100) if max_pos_pct > 0 else 0
                result.append({
                    "name": "Position Size",
                    "current": round(current_pos_pct, 2),
                    "limit": round(max_pos_pct, 2),
                    "usage": round(usage, 1),
                    "status": "breach" if usage >= 100 else "warning" if usage >= 75 else "ok",
                })

                # Leverage limit
                max_leverage = limits.get("max_leverage", 2.0)
                current_leverage = risk_state.get("leverage", 0.0)
                usage = (current_leverage / max_leverage * 100) if max_leverage > 0 else 0
                result.append({
                    "name": "Leverage",
                    "current": round(current_leverage, 2),
                    "limit": round(max_leverage, 2),
                    "usage": round(usage, 1),
                    "status": "breach" if usage >= 100 else "warning" if usage >= 75 else "ok",
                })

                # Daily loss limit
                max_daily_loss = limits.get("max_daily_loss_pct", 3.0)
                current_daily_loss = abs(risk_state.get("daily_pnl_pct", 0.0))
                usage = (current_daily_loss / max_daily_loss * 100) if max_daily_loss > 0 else 0
                result.append({
                    "name": "Daily Loss",
                    "current": round(current_daily_loss, 2),
                    "limit": round(max_daily_loss, 2),
                    "usage": round(usage, 1),
                    "status": "breach" if usage >= 100 else "warning" if usage >= 75 else "ok",
                })

                # Drawdown limit
                max_drawdown = limits.get("max_drawdown_pct", 10.0)
                current_drawdown = risk_state.get("max_drawdown_pct", 0.0)
                usage = (current_drawdown / max_drawdown * 100) if max_drawdown > 0 else 0
                result.append({
                    "name": "Drawdown",
                    "current": round(current_drawdown, 2),
                    "limit": round(max_drawdown, 2),
                    "usage": round(usage, 1),
                    "status": "breach" if usage >= 100 else "warning" if usage >= 75 else "ok",
                })

                # VaR limit
                max_var = limits.get("max_portfolio_var_pct", 2.0)
                current_var = risk_state.get("var_95", 0.0)
                usage = (current_var / max_var * 100) if max_var > 0 else 0
                result.append({
                    "name": "VaR 95%",
                    "current": round(current_var, 2),
                    "limit": round(max_var, 2),
                    "usage": round(usage, 1),
                    "status": "breach" if usage >= 100 else "warning" if usage >= 75 else "ok",
                })

                return result

            except Exception as e:
                logger.warning(f"Error getting risk limits from agent: {e}")

        # Fall back to event-tracked limits
        return [limit.to_dict() for limit in self._risk_limits.values()]

    def get_equity_curve(self) -> dict:
        """Get equity curve data with timestamps for filtering."""
        labels = []
        values = []
        timestamps = []
        for item in self._equity_curve:
            # Handle both old 2-tuple and new 3-tuple format
            if len(item) == 3:
                ts, val, unix_ms = item
                # Format label for display (convert UTC to local time)
                try:
                    if 'T' in ts:
                        dt_utc = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        # Convert to local time for display
                        dt_local = dt_utc.astimezone()
                        labels.append(dt_local.strftime("%H:%M"))
                    else:
                        labels.append(ts[:5] if len(ts) >= 5 else ts)
                except ValueError:
                    labels.append(ts[:5] if len(ts) >= 5 else ts)
                values.append(val)
                timestamps.append(unix_ms)
            else:
                ts, val = item
                labels.append(ts[:5] if len(ts) >= 5 else ts)
                values.append(val)
                # Fallback timestamp calculation
                try:
                    if 'T' in ts:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    else:
                        today = datetime.now(timezone.utc).date()
                        time_parts = ts.split(':')
                        dt = datetime(today.year, today.month, today.day,
                                      int(time_parts[0]), int(time_parts[1]),
                                      int(time_parts[2]) if len(time_parts) > 2 else 0,
                                      tzinfo=timezone.utc)
                    timestamps.append(int(dt.timestamp() * 1000))
                except (ValueError, IndexError):
                    timestamps.append(int(datetime.now(timezone.utc).timestamp() * 1000))
        return {"labels": labels, "values": values, "timestamps": timestamps}

    def get_events(self) -> list[dict]:
        """Get recent events."""
        return list(self._events)

    def get_alerts(self) -> list[dict]:
        """Get recent alerts."""
        return list(self._alerts)

    def _get_current_regime(self) -> str:
        """
        Get the current market regime from CIO agent or default.

        Returns:
            Current regime string (e.g., 'risk_on', 'risk_off', 'volatile', etc.)
        """
        # Try to get regime from CIO agent if available
        if self._cio_agent is not None:
            try:
                if hasattr(self._cio_agent, '_current_regime'):
                    regime = self._cio_agent._current_regime
                    if hasattr(regime, 'value'):
                        return regime.value
                    return str(regime)
            except Exception:
                pass

        return self._current_regime

    def update_regime(self, regime: str) -> None:
        """Update the current market regime (called by orchestrator or CIO agent)."""
        self._current_regime = regime

    def get_closed_positions(self) -> list[dict]:
        """Get recently closed positions with realized P&L."""
        result = list(self._closed_positions)
        logger.debug(f"[CLOSED_POSITIONS] get_closed_positions() returning {len(result)} positions")
        return result

    # =========================================================================
    # Advanced Analytics Getters (Phase 8)
    # =========================================================================

    def get_rolling_metrics(self) -> dict:
        """Get rolling Sharpe/Sortino ratios across time periods."""
        return self._rolling_metrics.get_all_periods()

    def get_session_performance(self) -> dict:
        """Get performance breakdown by trading session."""
        return self._session_performance.get_all_sessions()

    def get_strategy_comparison(self) -> dict:
        """Get all strategy performance data."""
        return self._strategy_comparison.get_all_strategies()

    def get_strategy_ranking(self) -> list[dict]:
        """Get strategies ranked by specified metric."""
        ranking = self._strategy_comparison.get_ranking(metric="total_pnl")
        return ranking[:10]  # Limit to top 10

    def get_risk_heatmap(self) -> list[dict]:
        """Get risk heatmap data for all positions."""
        # Build positions list from current positions
        positions = []
        for symbol, pos in self._positions.items():
            if pos.get("quantity", 0) != 0:
                entry_price = pos.get("entry_price", pos.get("avg_price", 100))
                quantity = abs(pos.get("quantity", 0))
                current_price = pos.get("current_price", entry_price)
                positions.append({
                    "symbol": symbol,
                    "value": abs(quantity * current_price),
                    "var_contribution": pos.get("var_contribution", 5.0),
                    "entry_date": pos.get("entry_time", datetime.now(timezone.utc) - timedelta(days=1)),
                })

        # Get portfolio value from latest equity point or default
        portfolio_value = 100000
        if self._equity_curve:
            _, latest_equity, _ = self._equity_curve[-1]
            portfolio_value = latest_equity if latest_equity > 0 else 100000
        return self._risk_heatmap.get_heatmap_data(positions, portfolio_value)

    def get_signal_consensus(self) -> dict:
        """Get signal consensus data for all symbols."""
        result = {}
        for symbol in self._signal_consensus.get_all_symbols():
            consensus = self._signal_consensus.get_consensus(symbol)
            if consensus:
                result[symbol] = consensus
        return result

    def get_high_disagreement_alerts(self) -> list[dict]:
        """Get alerts for symbols with high signal disagreement."""
        return self._signal_consensus.get_high_disagreement_alerts()

    def get_trade_journal_entries(self, limit: int = 50) -> list[dict]:
        """Get recent trade journal entries."""
        return self._trade_journal.get_entries(limit=limit)

    def get_trade_quality_stats(self) -> dict:
        """Get trade quality statistics."""
        return self._trade_journal.get_quality_stats()

    def record_trade_for_analytics(
        self,
        symbol: str,
        strategy: str,
        pnl: float,
        entry_time: datetime,
        exit_time: datetime,
    ) -> None:
        """Record a completed trade in analytics systems."""
        # Record in rolling metrics (uses add_trade with dict)
        self._rolling_metrics.add_trade({
            "pnl": pnl,
            "close_time": exit_time,
            "open_time": entry_time,
            "symbol": symbol,
        })

        # Record in session performance (uses record_trade with open_time, close_time, pnl, symbol)
        self._session_performance.record_trade(
            open_time=entry_time,
            close_time=exit_time,
            pnl=pnl,
            symbol=symbol,
        )

        # Calculate holding hours for strategy comparison
        holding_hours = (exit_time - entry_time).total_seconds() / 3600.0 if entry_time and exit_time else 0

        # Record in strategy comparison (uses record_trade with strategy, symbol, pnl, holding_hours, timestamp)
        self._strategy_comparison.record_trade(
            strategy=strategy,
            symbol=symbol,
            pnl=pnl,
            holding_hours=holding_hours,
            timestamp=exit_time,
        )

    def record_signal_for_analytics(
        self,
        symbol: str,
        agent: str,
        direction: str,
        confidence: float,
    ) -> None:
        """Record a signal in analytics systems."""
        # Record in signal consensus
        self._signal_consensus.record_signal(symbol, agent, direction, confidence)

        # Record in strategy comparison
        self._strategy_comparison.record_signal(agent, symbol, direction, confidence)

    def update_position_risk(
        self,
        symbol: str,
        position_value: float,
        var_contribution: float,
        portfolio_weight: float,
    ) -> None:
        """Update position risk data for heatmap."""
        self._risk_heatmap.update_position(
            symbol=symbol,
            position_value=position_value,
            var_contribution=var_contribution,
            portfolio_weight=portfolio_weight,
        )

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active

    def set_broadcast_callback(
        self, callback: Callable[[dict], Coroutine[Any, Any, None]]
    ) -> None:
        """Set callback for broadcasting events to WebSocket clients."""
        self._broadcast_callback = callback

    def get_latest_event_record(self) -> dict | None:
        """Get and clear the latest event record for broadcasting."""
        record = self._latest_event_record
        self._latest_event_record = None
        return record

    async def get_agent_health_summary(self) -> dict[str, Any]:
        """
        Get comprehensive agent health summary using AgentStatusTracker.

        Returns system-wide health metrics including:
        - Total, active, idle, error, stopped agent counts
        - Average and minimum health scores
        - System health status (healthy/degraded/critical)
        """
        # Update tracker with current orchestrator data
        if self._orchestrator:
            try:
                orchestrator_status = self._orchestrator.get_status()
                agents_data = orchestrator_status.get("agents", {})

                # Register and update all agents in the tracker
                for agent_status in agents_data.get("signal", []):
                    if agent_status:
                        name = agent_status.get("name")
                        if name:
                            self._agent_tracker.register_agent_sync(name, AgentType.SIGNAL)
                            # Update status based on running state
                            running = agent_status.get("running", False)
                            if running:
                                await self._agent_tracker.update_status(
                                    name, TrackerAgentStatus.ACTIVE
                                )
                            else:
                                await self._agent_tracker.update_status(
                                    name, TrackerAgentStatus.STOPPED
                                )
                            # Record events
                            events_processed = agent_status.get("events_processed", 0)
                            if events_processed > 0:
                                await self._agent_tracker.record_event_processed(name)

                # Update core agents
                agent_type_map = {
                    "cio": AgentType.DECISION,
                    "risk": AgentType.VALIDATION,
                    "compliance": AgentType.VALIDATION,
                    "execution": AgentType.EXECUTION,
                    "surveillance": AgentType.SURVEILLANCE,
                    "transaction_reporting": AgentType.REPORTING,
                }
                for agent_key, agent_type in agent_type_map.items():
                    agent_status = agents_data.get(agent_key)
                    if agent_status:
                        name = agent_status.get("name")
                        if name:
                            self._agent_tracker.register_agent_sync(name, agent_type)
                            running = agent_status.get("running", False)
                            if running:
                                await self._agent_tracker.update_status(
                                    name, TrackerAgentStatus.ACTIVE
                                )
                            else:
                                await self._agent_tracker.update_status(
                                    name, TrackerAgentStatus.STOPPED
                                )

            except Exception as e:
                logger.warning(f"Error updating agent tracker: {e}")

        # Get system health from tracker
        system_health = await self._agent_tracker.get_system_health()

        return {
            "total_agents": system_health.total_agents,
            "active_agents": system_health.active_agents,
            "idle_agents": system_health.idle_agents,
            "error_agents": system_health.error_agents,
            "stopped_agents": system_health.stopped_agents,
            "avg_health_score": system_health.avg_health_score,
            "min_health_score": system_health.min_health_score,
            "min_health_agent": system_health.min_health_agent,
            "total_events_processed": system_health.total_events_processed,
            "total_errors": system_health.total_errors,
            "system_health": system_health.system_health,
            "last_updated": system_health.last_updated.isoformat(),
        }

    def get_agent_tracker(self) -> AgentStatusTracker:
        """Get the agent status tracker for advanced monitoring."""
        return self._agent_tracker


# =============================================================================
# Dashboard Server
# =============================================================================

class DashboardServer:
    """
    FastAPI-based dashboard server with WebSocket support.

    Provides:
    - REST API endpoints for trading system data
    - WebSocket endpoint for real-time updates
    - Static file serving for dashboard UI
    - Integration with EventBus
    """

    def __init__(
        self,
        event_bus: "EventBus | None" = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        templates_dir: str | None = None,
    ):
        self._event_bus = event_bus
        self._host = host
        self._port = port

        # Resolve templates directory
        if templates_dir:
            self._templates_dir = Path(templates_dir)
        else:
            self._templates_dir = Path(__file__).parent / "templates"

        # Initialize components
        self._connection_manager = ConnectionManager()
        self._state = DashboardState(event_bus)

        # Background tasks
        self._update_task: asyncio.Task | None = None
        self._running = False

        # Create FastAPI app
        self._app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Handle startup and shutdown."""
            await self._startup()
            yield
            await self._shutdown()

        app = FastAPI(
            title="AI Trading Firm Dashboard",
            description="Real-time trading system monitoring dashboard",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Enable CORS for development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount static files if templates directory exists
        if self._templates_dir.exists():
            app.mount(
                "/static",
                StaticFiles(directory=str(self._templates_dir)),
                name="static"
            )

        # Register routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: FastAPI) -> None:
        """Register all API routes."""

        # =================================================================
        # Health Endpoint
        # =================================================================

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return JSONResponse({
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "websocket_connections": self._connection_manager.connection_count,
                "kill_switch_active": self._state.is_kill_switch_active(),
            })

        # =================================================================
        # Dashboard UI
        # =================================================================

        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve the dashboard HTML."""
            index_path = self._templates_dir / "index.html"
            if index_path.exists():
                return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
            else:
                return HTMLResponse(
                    content="<h1>Dashboard template not found</h1>",
                    status_code=404
                )

        # =================================================================
        # REST API Endpoints
        # =================================================================

        @app.get("/api/agents")
        async def get_agents():
            """Get all agent statuses from the trading system."""
            return JSONResponse({"agents": self._state.get_agents()})

        @app.get("/api/positions")
        async def get_positions():
            """Get all open positions from IB broker."""
            positions = await self._state.get_positions_async()
            return JSONResponse({"positions": positions})

        @app.get("/api/signals")
        async def get_signals():
            """Get current signals from all strategy agents."""
            return JSONResponse({"signals": self._state.get_signals()})

        @app.get("/api/decisions")
        async def get_decisions():
            """Get recent CIO decisions."""
            return JSONResponse({"decisions": self._state.get_decisions()})

        @app.get("/api/metrics")
        async def get_metrics():
            """Get portfolio metrics including real P&L from broker."""
            metrics = await self._state.get_metrics_async()
            return JSONResponse({"metrics": metrics})

        @app.get("/api/risk")
        async def get_risk():
            """Get risk limit statuses from risk agent."""
            return JSONResponse({
                "limits": self._state.get_risk_limits(),
                "kill_switch_active": self._state.is_kill_switch_active(),
            })

        @app.get("/api/events")
        async def get_events():
            """Get recent events."""
            return JSONResponse({"events": self._state.get_events()})

        @app.get("/api/alerts")
        async def get_alerts():
            """Get recent alerts."""
            return JSONResponse({"alerts": self._state.get_alerts()})

        @app.get("/api/closed_positions")
        async def get_closed_positions():
            """Get recently closed positions with realized P&L."""
            return JSONResponse({"closed_positions": self._state.get_closed_positions()})

        @app.get("/api/equity")
        async def get_equity():
            """Get equity curve data."""
            return JSONResponse({"equity": self._state.get_equity_curve()})

        @app.get("/api/status")
        async def get_status():
            """Get full dashboard status."""
            # Fetch positions from broker asynchronously
            positions = await self._state.get_positions_async()
            # Get metrics with position data for consistent P&L calculation
            metrics = await self._state.get_metrics_async(positions)
            return JSONResponse({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "agents": self._state.get_agents(),
                "positions": positions,
                "signals": self._state.get_signals(),
                "risk": {
                    "limits": self._state.get_risk_limits(),
                    "kill_switch_active": self._state.is_kill_switch_active(),
                },
            })

        # =================================================================
        # Advanced Analytics Endpoints (Phase 8)
        # =================================================================

        @app.get("/api/analytics/rolling-metrics")
        async def get_rolling_metrics():
            """Get rolling Sharpe/Sortino ratios across time periods."""
            return JSONResponse({"rolling_metrics": self._state.get_rolling_metrics()})

        @app.get("/api/analytics/session-performance")
        async def get_session_performance():
            """Get performance breakdown by trading session."""
            return JSONResponse({"session_performance": self._state.get_session_performance()})

        @app.get("/api/analytics/strategy-comparison")
        async def get_strategy_comparison():
            """Get strategy performance comparison and rankings."""
            return JSONResponse({
                "strategies": self._state.get_strategy_comparison(),
                "ranking": self._state.get_strategy_ranking(),
            })

        @app.get("/api/analytics/risk-heatmap")
        async def get_risk_heatmap():
            """Get risk heatmap data for positions."""
            return JSONResponse({"risk_heatmap": self._state.get_risk_heatmap()})

        @app.get("/api/analytics/signal-consensus")
        async def get_signal_consensus():
            """Get signal consensus and disagreement data."""
            return JSONResponse({
                "consensus": self._state.get_signal_consensus(),
                "high_disagreement": self._state.get_high_disagreement_alerts(),
            })

        @app.get("/api/analytics/trade-journal")
        async def get_trade_journal():
            """Get trade journal entries."""
            return JSONResponse({
                "entries": self._state.get_trade_journal_entries(),
                "quality_stats": self._state.get_trade_quality_stats(),
            })

        # =================================================================
        # WebSocket Endpoint
        # =================================================================

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self._connection_manager.connect(websocket)

            try:
                # CRITICAL FIX: Wrap broker data fetching in try-except to handle disconnects
                # Send initial state (use async methods for broker data)
                # Fetch positions first, then calculate metrics from them for consistency
                initial_positions = []
                initial_metrics = {}
                broker_error = None

                try:
                    initial_positions = await self._state.get_positions_async()
                    initial_metrics = await self._state.get_metrics_async(positions=initial_positions)
                except Exception as e:
                    logger.warning(f"Error fetching broker data for initial state: {e}")
                    broker_error = str(e)
                    # Use empty defaults - client will see loading skeleton
                    initial_positions = []
                    initial_metrics = {
                        "total_pnl": 0, "today_pnl": 0, "unrealized_pnl": 0,
                        "realized_pnl": 0, "sharpe_ratio": 0, "win_rate": 0,
                        "drawdown": 0, "position_count": 0
                    }

                initial_payload = {
                    "metrics": initial_metrics,
                    "agents": self._state.get_agents(),
                    "positions": initial_positions,
                    "closed_positions": self._state.get_closed_positions(),
                    "signals": self._state.get_signals(),
                    "decisions": self._state.get_decisions(),
                    "alerts": self._state.get_alerts(),
                    "risk": {
                        "limits": self._state.get_risk_limits(),
                    },
                    "equity": self._state.get_equity_curve(),
                    # Advanced analytics data (Phase 8)
                    "rolling_metrics": self._state.get_rolling_metrics(),
                    "session_performance": self._state.get_session_performance(),
                    "strategy_comparison": self._state.get_strategy_comparison(),
                    "strategy_ranking": self._state.get_strategy_ranking(),
                    "signal_consensus": self._state.get_signal_consensus(),
                }

                # Add broker error indicator if there was a problem
                if broker_error:
                    initial_payload["broker_error"] = broker_error

                await websocket.send_json({
                    "type": "initial",
                    "payload": initial_payload
                })

                # Listen for client messages (e.g., kill switch commands)
                while True:
                    try:
                        data = await websocket.receive_json()
                        await self._handle_client_message(data)
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.debug(f"WebSocket receive error: {e}")
                        break

            finally:
                await self._connection_manager.disconnect(websocket)

        @app.post("/api/kill-switch")
        async def toggle_kill_switch(active: bool = True):
            """Toggle the kill switch."""
            # This would typically trigger a KillSwitchEvent through the event bus
            return JSONResponse({
                "status": "acknowledged",
                "kill_switch_active": active,
            })

        @app.post("/api/agent/toggle")
        async def toggle_agent(agent_name: str, enabled: bool):
            """
            Toggle an agent's enabled state at runtime.

            Args:
                agent_name: Name of the agent to toggle
                enabled: Whether to enable (True) or disable (False)

            Returns:
                JSON response with the result
            """
            result = await self._state.toggle_agent(agent_name, enabled)

            # Broadcast agent state change to all clients
            await self._connection_manager.broadcast({
                "type": "agent_toggle",
                "payload": result
            })

            # Also broadcast an alert
            action_word = "enabled" if enabled else "disabled"
            llm_warning = " (LLM agent - uses API tokens)" if result.get("uses_llm") else ""
            await self._connection_manager.broadcast({
                "type": "alert",
                "payload": {
                    "title": "Agent Toggle",
                    "message": f"Agent {agent_name} {action_word}{llm_warning}",
                    "severity": "info" if enabled else "warning",
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                }
            })

            return JSONResponse(result)

        @app.get("/api/agent/states")
        async def get_agent_states():
            """Get all agent enabled/disabled states."""
            return JSONResponse({
                "states": self._state.get_agent_states(),
                "llm_agents": list(self._state._llm_agents),
                "essential_agents": list(self._state._essential_agents),
            })

    async def _handle_client_message(self, data: dict) -> None:
        """Handle incoming WebSocket messages from clients."""
        action = data.get("action")

        if action == "kill_switch":
            active = data.get("active", False)
            logger.warning(f"Kill switch toggled via dashboard: active={active}")
            # In a real system, this would publish a KillSwitchEvent

            # Broadcast kill switch state
            await self._connection_manager.broadcast({
                "type": "alert",
                "payload": {
                    "title": "Kill Switch",
                    "message": f"Kill switch {'activated' if active else 'deactivated'} via dashboard",
                    "severity": "critical" if active else "info",
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                }
            })

        elif action == "toggle_agent":
            agent_name = data.get("agent_name")
            enabled = data.get("enabled", True)
            if agent_name:
                result = await self._state.toggle_agent(agent_name, enabled)
                logger.info(f"Agent toggled via WebSocket: {agent_name} -> enabled={enabled}")

                # Broadcast agent state change
                await self._connection_manager.broadcast({
                    "type": "agent_toggle",
                    "payload": result
                })

                # Broadcast alert
                action_word = "enabled" if enabled else "disabled"
                llm_warning = " (LLM agent)" if result.get("uses_llm") else ""
                await self._connection_manager.broadcast({
                    "type": "alert",
                    "payload": {
                        "title": "Agent Toggle",
                        "message": f"{agent_name} {action_word}{llm_warning}",
                        "severity": "info" if enabled else "warning",
                        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    }
                })

    async def _startup(self) -> None:
        """Handle server startup."""
        await self._state.initialize()
        await self._connection_manager.start_broadcaster()

        # Start periodic update task
        self._running = True
        self._update_task = asyncio.create_task(self._periodic_update_loop())

        logger.info(f"Dashboard server started on {self._host}:{self._port}")

    async def _shutdown(self) -> None:
        """Handle server shutdown."""
        self._running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        await self._connection_manager.stop_broadcaster()
        logger.info("Dashboard server stopped")

    async def _periodic_update_loop(self) -> None:
        """
        Background task that pushes updates to WebSocket clients.

        Updates are pushed every 500ms to provide real-time feedback.
        """
        update_counter = 0
        while self._running:
            try:
                await asyncio.sleep(0.5)  # 500ms update interval

                if self._connection_manager.connection_count == 0:
                    continue

                update_counter += 1

                # Broadcast positions first (use async to get from broker)
                positions = await self._state.get_positions_async()
                await self._connection_manager.broadcast({
                    "type": "positions",
                    "payload": positions,
                })

                # Broadcast metrics (calculated from positions for consistency)
                metrics = await self._state.get_metrics_async(positions=positions)
                await self._connection_manager.broadcast({
                    "type": "metrics",
                    "payload": metrics,
                })

                # Broadcast latest event if available
                latest_event = self._state.get_latest_event_record()
                if latest_event:
                    await self._connection_manager.broadcast({
                        "type": "event",
                        "payload": latest_event,
                    })

                # Broadcast agents every update
                await self._connection_manager.broadcast({
                    "type": "agents",
                    "payload": self._state.get_agents(),
                })

                # Broadcast closed positions (every 4th update)
                if update_counter % 4 == 0:
                    await self._connection_manager.broadcast({
                        "type": "closed_positions",
                        "payload": self._state.get_closed_positions(),
                    })

                # Broadcast signals
                await self._connection_manager.broadcast({
                    "type": "signals",
                    "payload": self._state.get_signals(),
                })

                # Broadcast decisions (every 2nd update to reduce load)
                if update_counter % 2 == 0:
                    await self._connection_manager.broadcast({
                        "type": "decisions",
                        "payload": self._state.get_decisions(),
                    })

                # Broadcast risk
                await self._connection_manager.broadcast({
                    "type": "risk",
                    "payload": {
                        "limits": self._state.get_risk_limits(),
                    }
                })

                # Broadcast equity curve (every 4th update to reduce load)
                if update_counter % 4 == 0:
                    await self._connection_manager.broadcast({
                        "type": "equity",
                        "payload": self._state.get_equity_curve(),
                    })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Periodic update error: {e}")
                await asyncio.sleep(1.0)

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self._app

    @property
    def state(self) -> DashboardState:
        """Get the dashboard state manager."""
        return self._state

    @property
    def connection_manager(self) -> ConnectionManager:
        """Get the WebSocket connection manager."""
        return self._connection_manager

    def set_orchestrator(self, orchestrator: Any) -> None:
        """
        Connect the dashboard to the trading system orchestrator.

        This enables the dashboard to pull real-time data from:
        - Broker (positions, P&L)
        - Risk Agent (limits, VaR)
        - Signal Agents (status)
        - CIO Agent (decisions)

        Args:
            orchestrator: TradingFirmOrchestrator instance
        """
        self._state.set_orchestrator(orchestrator)
        logger.info("Dashboard server connected to orchestrator")

    async def broadcast_event(self, event_type: str, payload: dict) -> None:
        """Broadcast a custom event to all WebSocket clients."""
        await self._connection_manager.broadcast({
            "type": event_type,
            "payload": payload,
        })


# =============================================================================
# Factory Function
# =============================================================================

def create_dashboard_server(
    event_bus: "EventBus | None" = None,
    orchestrator: Any = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    templates_dir: str | None = None,
) -> DashboardServer:
    """
    Create a dashboard server instance.

    Args:
        event_bus: Optional EventBus instance for real-time updates
        orchestrator: Optional TradingFirmOrchestrator for real data
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 8080)
        templates_dir: Optional custom templates directory

    Returns:
        Configured DashboardServer instance

    Example:
        from core.event_bus import EventBus
        from dashboard.server import create_dashboard_server
        import uvicorn

        event_bus = EventBus()
        server = create_dashboard_server(event_bus=event_bus, port=8080)

        # Run with uvicorn
        uvicorn.run(server.app, host="0.0.0.0", port=8080)
    """
    server = DashboardServer(
        event_bus=event_bus,
        host=host,
        port=port,
        templates_dir=templates_dir,
    )

    # Connect to orchestrator if provided
    if orchestrator is not None:
        server.set_orchestrator(orchestrator)

    return server


# =============================================================================
# Standalone Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Create server without event bus (standalone mode)
    server = create_dashboard_server(port=8080)

    # Run with uvicorn
    uvicorn.run(server.app, host="0.0.0.0", port=8080)

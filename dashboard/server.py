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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.events import Event

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "status": self.status.value,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "event_count": self.event_count,
            "latency_ms": round(self.latency_ms, 2),
            "error_message": self.error_message,
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
    total_pnl: float = 0.0
    today_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    drawdown: float = 0.0
    position_count: int = 0
    total_trades: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_pnl": round(self.total_pnl, 2),
            "today_pnl": round(self.today_pnl, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "win_rate": round(self.win_rate, 3),
            "drawdown": round(self.drawdown, 4),
            "position_count": self.position_count,
            "total_trades": self.total_trades,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


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

        # Agent tracking
        self._agents: dict[str, AgentInfo] = {}

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

        # Equity curve data
        self._equity_curve: deque[tuple[str, float]] = deque(maxlen=500)

        # Kill switch state
        self._kill_switch_active = False

        # Alerts
        self._alerts: deque[dict] = deque(maxlen=100)

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize state and subscribe to events."""
        if self._event_bus:
            await self._subscribe_to_events()
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
                elif event.event_type == EventType.FILL:
                    self._handle_fill_event(event)
                elif event.event_type == EventType.RISK_ALERT:
                    self._handle_risk_alert_event(event)
                elif event.event_type == EventType.KILL_SWITCH:
                    self._handle_kill_switch_event(event)
                elif event.event_type == EventType.MARKET_DATA:
                    self._handle_market_data_event(event)

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
        decision = DecisionInfo(
            decision_id=event.event_id,
            symbol=audit_dict.get("symbol", ""),
            direction=audit_dict.get("action", "").upper() if audit_dict.get("action") else "HOLD",
            quantity=audit_dict.get("quantity", 0),
            conviction=audit_dict.get("conviction_score", 0.0),
            timestamp=event.timestamp,
            rationale=audit_dict.get("rationale", ""),
        )
        self._decisions.appendleft(decision)

    def _handle_fill_event(self, event: "Event") -> None:
        """Handle fill event - update positions."""
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
            if side == "buy":
                # Add to position
                total_cost = (pos.entry_price * pos.quantity) + (fill_price * fill_qty)
                pos.quantity += fill_qty
                if pos.quantity > 0:
                    pos.entry_price = total_cost / pos.quantity
            else:
                # Reduce position
                pos.quantity -= fill_qty
                if pos.quantity <= 0:
                    del self._positions[symbol]
                    return
            pos.current_price = fill_price
        else:
            self._positions[symbol] = PositionInfo(
                symbol=symbol,
                quantity=fill_qty if side == "buy" else -fill_qty,
                entry_price=fill_price,
                current_price=fill_price,
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
        """Add a point to the equity curve."""
        self._equity_curve.append((timestamp, value))

    def get_agents(self) -> list[dict]:
        """Get all agent statuses."""
        return [agent.to_dict() for agent in self._agents.values()]

    def get_positions(self) -> list[dict]:
        """Get all positions."""
        return [pos.to_dict() for pos in self._positions.values()]

    def get_signals(self) -> list[dict]:
        """Get current signals from all agents."""
        return [signal.to_dict() for signal in self._signals.values()]

    def get_decisions(self) -> list[dict]:
        """Get recent decisions."""
        return [dec.to_dict() for dec in self._decisions]

    def get_metrics(self) -> dict:
        """Get portfolio metrics."""
        return self._metrics.to_dict()

    def get_risk_limits(self) -> list[dict]:
        """Get risk limit statuses."""
        return [limit.to_dict() for limit in self._risk_limits.values()]

    def get_equity_curve(self) -> dict:
        """Get equity curve data."""
        labels = []
        values = []
        for ts, val in self._equity_curve:
            labels.append(ts)
            values.append(val)
        return {"labels": labels, "values": values}

    def get_events(self) -> list[dict]:
        """Get recent events."""
        return list(self._events)

    def get_alerts(self) -> list[dict]:
        """Get recent alerts."""
        return list(self._alerts)

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active


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
            """Get all agent statuses."""
            return JSONResponse({"agents": self._state.get_agents()})

        @app.get("/api/positions")
        async def get_positions():
            """Get all open positions."""
            return JSONResponse({"positions": self._state.get_positions()})

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
            """Get portfolio metrics."""
            return JSONResponse({"metrics": self._state.get_metrics()})

        @app.get("/api/risk")
        async def get_risk():
            """Get risk limit statuses."""
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

        @app.get("/api/equity")
        async def get_equity():
            """Get equity curve data."""
            return JSONResponse({"equity": self._state.get_equity_curve()})

        @app.get("/api/status")
        async def get_status():
            """Get full dashboard status."""
            return JSONResponse({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": self._state.get_metrics(),
                "agents": self._state.get_agents(),
                "positions": self._state.get_positions(),
                "signals": self._state.get_signals(),
                "risk": {
                    "limits": self._state.get_risk_limits(),
                    "kill_switch_active": self._state.is_kill_switch_active(),
                },
            })

        # =================================================================
        # WebSocket Endpoint
        # =================================================================

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self._connection_manager.connect(websocket)

            try:
                # Send initial state
                await websocket.send_json({
                    "type": "initial",
                    "payload": {
                        "metrics": self._state.get_metrics(),
                        "agents": self._state.get_agents(),
                        "positions": self._state.get_positions(),
                        "signals": self._state.get_signals(),
                        "decisions": self._state.get_decisions(),
                        "risk": {
                            "limits": self._state.get_risk_limits(),
                        },
                        "equity": self._state.get_equity_curve(),
                    }
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
        while self._running:
            try:
                await asyncio.sleep(0.5)  # 500ms update interval

                if self._connection_manager.connection_count == 0:
                    continue

                # Broadcast metrics
                await self._connection_manager.broadcast({
                    "type": "metrics",
                    "payload": self._state.get_metrics(),
                })

                # Broadcast agents (less frequently)
                await self._connection_manager.broadcast({
                    "type": "agents",
                    "payload": self._state.get_agents(),
                })

                # Broadcast positions
                await self._connection_manager.broadcast({
                    "type": "positions",
                    "payload": self._state.get_positions(),
                })

                # Broadcast signals
                await self._connection_manager.broadcast({
                    "type": "signals",
                    "payload": self._state.get_signals(),
                })

                # Broadcast risk
                await self._connection_manager.broadcast({
                    "type": "risk",
                    "payload": {
                        "limits": self._state.get_risk_limits(),
                    }
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
    host: str = "0.0.0.0",
    port: int = 8080,
    templates_dir: str | None = None,
) -> DashboardServer:
    """
    Create a dashboard server instance.

    Args:
        event_bus: Optional EventBus instance for real-time updates
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
    return DashboardServer(
        event_bus=event_bus,
        host=host,
        port=port,
        templates_dir=templates_dir,
    )


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

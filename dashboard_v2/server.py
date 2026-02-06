"""
Dashboard V2 Server - Bloomberg Terminal Style
===============================================

Professional-grade trading dashboard with advanced KPIs,
6-panel layout, and real-time WebSocket updates.

Runs on port 8082 alongside the existing dashboard (port 8081).

Endpoints:
    GET  /                    - Main dashboard UI
    GET  /api/v2/performance  - Performance KPIs
    GET  /api/v2/risk         - Risk metrics
    GET  /api/v2/signals      - Current signal state
    GET  /api/v2/strategies   - Strategy attribution
    GET  /api/v2/positions    - Open + closed positions
    GET  /api/v2/execution    - Execution quality metrics
    GET  /api/v2/kpis         - All KPIs in one call
    WS   /ws/v2               - Real-time updates
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from dashboard_v2.kpi_calculator import KPICalculator, KPIResult, TradeRecord

if TYPE_CHECKING:
    from core.event_bus import EventBus

logger = logging.getLogger(__name__)


# =============================================================================
# Connection Manager (WebSocket)
# =============================================================================

class ConnectionManagerV2:
    """Manages WebSocket connections for V2 dashboard."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


# =============================================================================
# V2 Dashboard Server
# =============================================================================

class DashboardV2Server:
    """
    Bloomberg-style trading dashboard server.

    Accesses the same orchestrator/broker/agents as the V1 dashboard
    but presents data in a professional terminal-style layout with
    advanced KPIs.
    """

    def __init__(
        self,
        event_bus: "EventBus | None" = None,
        host: str = "0.0.0.0",
        port: int = 8082,
    ):
        self._event_bus = event_bus
        self._host = host
        self._port = port
        self._templates_dir = Path(__file__).parent / "templates"

        # Connection manager
        self._ws_manager = ConnectionManagerV2()

        # KPI calculator
        self._kpi_calc = KPICalculator()

        # Orchestrator references (set via set_orchestrator)
        self._orchestrator: Any = None
        self._broker: Any = None
        self._risk_agent: Any = None
        self._cio_agent: Any = None
        self._execution_agent: Any = None
        self._signal_agents: list = []
        self._compliance_agent: Any = None

        # Cached data
        self._equity_curve: deque[float] = deque(maxlen=10000)
        self._closed_trades: deque[TradeRecord] = deque(maxlen=5000)
        self._last_kpis: KPIResult = KPIResult()
        self._signal_history: deque[dict] = deque(maxlen=200)

        # Background task
        self._update_task: asyncio.Task | None = None
        self._running = False

        # Create app
        self._app = self._create_app()

    @property
    def app(self) -> FastAPI:
        return self._app

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Connect to the trading system orchestrator."""
        self._orchestrator = orchestrator
        if hasattr(orchestrator, '_broker'):
            self._broker = orchestrator._broker
        if hasattr(orchestrator, '_risk_agent'):
            self._risk_agent = orchestrator._risk_agent
        if hasattr(orchestrator, '_cio_agent'):
            self._cio_agent = orchestrator._cio_agent
        if hasattr(orchestrator, '_execution_agent'):
            self._execution_agent = orchestrator._execution_agent
        if hasattr(orchestrator, '_signal_agents'):
            self._signal_agents = orchestrator._signal_agents
        if hasattr(orchestrator, '_compliance_agent'):
            self._compliance_agent = orchestrator._compliance_agent
        logger.info("Dashboard V2 connected to orchestrator")

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self._running = True
            self._update_task = asyncio.create_task(self._periodic_update())
            yield
            self._running = False
            if self._update_task:
                self._update_task.cancel()

        app = FastAPI(
            title="AI Trading Firm - Bloomberg V2",
            version="2.0.0",
            lifespan=lifespan,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_routes(app)
        return app

    def _register_routes(self, app: FastAPI) -> None:
        """Register all API routes."""

        @app.get("/", response_class=HTMLResponse)
        async def index():
            template_path = self._templates_dir / "index.html"
            if template_path.exists():
                return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
            return HTMLResponse(content="<h1>Dashboard V2 - Template not found</h1>")

        @app.get("/api/v2/performance")
        async def get_performance():
            return JSONResponse(content=self._get_performance_data())

        @app.get("/api/v2/risk")
        async def get_risk():
            return JSONResponse(content=self._get_risk_data())

        @app.get("/api/v2/signals")
        async def get_signals():
            return JSONResponse(content=self._get_signals_data())

        @app.get("/api/v2/strategies")
        async def get_strategies():
            return JSONResponse(content=self._get_strategy_data())

        @app.get("/api/v2/positions")
        async def get_positions():
            data = await self._get_positions_data()
            return JSONResponse(content=data)

        @app.get("/api/v2/execution")
        async def get_execution():
            return JSONResponse(content=self._get_execution_data())

        @app.get("/api/v2/kpis")
        async def get_all_kpis():
            """All KPIs in one call for initial page load."""
            positions_data = await self._get_positions_data()
            return JSONResponse(content={
                "performance": self._get_performance_data(),
                "risk": self._get_risk_data(),
                "signals": self._get_signals_data(),
                "strategies": self._get_strategy_data(),
                "positions": positions_data,
                "execution": self._get_execution_data(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        @app.websocket("/ws/v2")
        async def websocket_endpoint(ws: WebSocket):
            await self._ws_manager.connect(ws)
            try:
                while True:
                    # Keep connection alive, receive client messages
                    data = await ws.receive_text()
                    # Client can send commands like {"action": "refresh"}
                    try:
                        msg = json.loads(data)
                        if msg.get("action") == "refresh":
                            positions_data = await self._get_positions_data()
                            await ws.send_json({
                                "type": "full_update",
                                "performance": self._get_performance_data(),
                                "risk": self._get_risk_data(),
                                "signals": self._get_signals_data(),
                                "strategies": self._get_strategy_data(),
                                "positions": positions_data,
                                "execution": self._get_execution_data(),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                self._ws_manager.disconnect(ws)

    # =========================================================================
    # Periodic Update (broadcasts to all WS clients)
    # =========================================================================

    async def _periodic_update(self) -> None:
        """Send updates to all connected WebSocket clients every 2 seconds."""
        while self._running:
            try:
                await asyncio.sleep(2.0)
                if self._ws_manager.count == 0:
                    continue

                # Refresh data
                self._refresh_equity_and_trades()

                positions_data = await self._get_positions_data()
                await self._ws_manager.broadcast({
                    "type": "update",
                    "performance": self._get_performance_data(),
                    "risk": self._get_risk_data(),
                    "signals": self._get_signals_data(),
                    "strategies": self._get_strategy_data(),
                    "positions": positions_data,
                    "execution": self._get_execution_data(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"V2 periodic update error: {e}")

    # =========================================================================
    # Data Refresh
    # =========================================================================

    def _refresh_equity_and_trades(self) -> None:
        """Pull latest equity curve and closed trades from orchestrator/CIO."""
        if not self._orchestrator:
            return

        # Get equity from V1 dashboard state if available
        if hasattr(self._orchestrator, '_dashboard_server'):
            ds = self._orchestrator._dashboard_server
            if hasattr(ds, '_state'):
                state = ds._state
                # Equity curve
                if hasattr(state, '_equity_curve') and state._equity_curve:
                    self._equity_curve.clear()
                    for entry in state._equity_curve:
                        # entry is (iso_ts, value, unix_ms)
                        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                            self._equity_curve.append(float(entry[1]))

        # Closed trades from CIO
        if self._cio_agent and hasattr(self._cio_agent, '_closed_positions'):
            closed = list(self._cio_agent._closed_positions)
            self._closed_trades.clear()
            for pos in closed:
                pnl = getattr(pos, 'realized_pnl', 0.0) or getattr(pos, 'pnl', 0.0) or 0.0
                r_mult = getattr(pos, 'r_multiple', 0.0) or 0.0
                strategy = getattr(pos, 'source_agent', '') or getattr(pos, 'strategy', '') or ''
                symbol = getattr(pos, 'symbol', '') or ''
                # Hold time
                entry_time = getattr(pos, 'entry_time', None)
                exit_time = getattr(pos, 'exit_time', None) or getattr(pos, 'close_time', None)
                hold_hours = 0.0
                if entry_time and exit_time:
                    try:
                        delta = exit_time - entry_time
                        hold_hours = delta.total_seconds() / 3600.0
                    except Exception:
                        pass

                self._closed_trades.append(TradeRecord(
                    pnl=pnl,
                    r_multiple=r_mult,
                    strategy=strategy,
                    symbol=symbol,
                    hold_time_hours=hold_hours,
                ))

        # Calculate KPIs
        equity = list(self._equity_curve) if self._equity_curve else []
        trades = list(self._closed_trades)
        if equity or trades:
            self._last_kpis = self._kpi_calc.calculate_all(equity, trades)

    # =========================================================================
    # Data Getters (for API endpoints)
    # =========================================================================

    def _get_performance_data(self) -> dict:
        """Performance panel data."""
        self._refresh_equity_and_trades()
        kpis = self._last_kpis

        # Equity curve for chart (last 500 points)
        equity = list(self._equity_curve)
        equity_chart = equity[-500:] if len(equity) > 500 else equity

        # Daily P&L from equity curve
        daily_pnl = []
        if len(equity) >= 2:
            for i in range(max(0, len(equity) - 30), len(equity)):
                if i > 0:
                    daily_pnl.append(round(equity[i] - equity[i - 1], 2))

        # Get today P&L from V1 dashboard if available
        today_pnl = 0.0
        unrealized_pnl = 0.0
        realized_pnl = 0.0
        if self._orchestrator and hasattr(self._orchestrator, '_dashboard_server'):
            ds = self._orchestrator._dashboard_server
            if hasattr(ds, '_state') and hasattr(ds._state, '_metrics'):
                metrics = ds._state._metrics
                today_pnl = metrics.today_pnl
                unrealized_pnl = metrics.unrealized_pnl
                realized_pnl = metrics.realized_pnl

        return {
            "kpis": kpis.to_dict(),
            "equity_curve": equity_chart,
            "daily_pnl": daily_pnl,
            "today_pnl": round(today_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
        }

    def _get_risk_data(self) -> dict:
        """Risk panel data."""
        data = {
            "var_95": 0.0,
            "cvar_975": 0.0,
            "current_drawdown_pct": 0.0,
            "max_drawdown_pct": self._last_kpis.max_drawdown_pct,
            "current_leverage": 0.0,
            "max_leverage": 3.0,
            "weekly_loss_pct": 0.0,
            "max_weekly_loss_pct": 5.0,
            "rolling_drawdown_pct": 0.0,
            "max_rolling_drawdown_pct": 4.0,
            "kill_switch_active": False,
            "correlation_matrix": {},
            "drawdown_curve": [],
        }

        if self._risk_agent:
            try:
                risk_state = None
                if hasattr(self._risk_agent, '_risk_state'):
                    risk_state = self._risk_agent._risk_state
                if risk_state:
                    data["var_95"] = round(getattr(risk_state, 'current_var', 0.0) or 0.0, 2)
                    data["cvar_975"] = round(getattr(risk_state, 'cvar_975', 0.0) or 0.0, 2)
                    data["current_drawdown_pct"] = round(
                        getattr(risk_state, 'current_drawdown_pct', 0.0) or 0.0, 4
                    )
                    data["current_leverage"] = round(
                        getattr(risk_state, 'current_leverage', 0.0) or 0.0, 2
                    )

                if hasattr(self._risk_agent, '_weekly_loss_pct'):
                    data["weekly_loss_pct"] = round(self._risk_agent._weekly_loss_pct or 0.0, 4)
                if hasattr(self._risk_agent, '_rolling_drawdown_pct'):
                    data["rolling_drawdown_pct"] = round(
                        self._risk_agent._rolling_drawdown_pct or 0.0, 4
                    )
                if hasattr(self._risk_agent, '_kill_switch_active'):
                    data["kill_switch_active"] = bool(self._risk_agent._kill_switch_active)

                # Risk limits from config
                if hasattr(self._risk_agent, '_config') and self._risk_agent._config:
                    params = getattr(self._risk_agent._config, 'parameters', {}) or {}
                    data["max_leverage"] = params.get("max_leverage", 3.0)
                    data["max_weekly_loss_pct"] = params.get("max_weekly_loss_pct", 5.0)
                    data["max_rolling_drawdown_pct"] = params.get("max_rolling_drawdown_pct", 4.0)
            except Exception as e:
                logger.debug(f"Risk data error: {e}")

        # Drawdown curve from equity
        equity = list(self._equity_curve)
        if equity:
            dd_curve = []
            peak = equity[0]
            for val in equity[-200:]:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak if peak > 0 else 0.0
                dd_curve.append(round(-dd * 100, 3))
            data["drawdown_curve"] = dd_curve

        return data

    def _get_signals_data(self) -> dict:
        """Signal flow panel data."""
        agents_data = []

        for agent in self._signal_agents:
            try:
                agent_name = getattr(agent, 'name', type(agent).__name__)
                enabled = getattr(agent, '_enabled', True)
                status = "active" if enabled else "disabled"

                # Last signal from agent
                last_signal = {}
                if hasattr(agent, '_last_signal') and agent._last_signal:
                    sig = agent._last_signal
                    last_signal = {
                        "direction": getattr(sig, 'direction', 'FLAT'),
                        "confidence": round(getattr(sig, 'confidence', 0.0), 3),
                        "strength": round(getattr(sig, 'strength', 0.0), 3),
                        "symbol": getattr(sig, 'symbol', ''),
                    }

                agents_data.append({
                    "name": agent_name,
                    "enabled": enabled,
                    "status": status,
                    "last_signal": last_signal,
                })
            except Exception:
                continue

        # CIO consensus
        consensus = {}
        if self._cio_agent:
            try:
                if hasattr(self._cio_agent, '_last_decision_summary'):
                    consensus = self._cio_agent._last_decision_summary or {}
                elif hasattr(self._cio_agent, 'get_status'):
                    cio_status = self._cio_agent.get_status()
                    consensus = {
                        "decision_mode": cio_status.get("decision_mode", "NORMAL"),
                        "tracked_positions": cio_status.get("tracked_positions", 0),
                    }
            except Exception:
                pass

        return {
            "agents": agents_data,
            "consensus": consensus,
            "agent_count": len(agents_data),
            "active_count": sum(1 for a in agents_data if a.get("enabled")),
        }

    def _get_strategy_data(self) -> dict:
        """Strategy attribution panel data."""
        kpis = self._last_kpis
        strategies = []

        for strat_name in kpis.strategy_trades:
            pnl = kpis.strategy_pnl.get(strat_name, 0.0)
            wr = kpis.strategy_win_rate.get(strat_name, 0.0)
            trades = kpis.strategy_trades.get(strat_name, 0)
            strategies.append({
                "name": strat_name,
                "pnl": round(pnl, 2),
                "win_rate": round(wr, 4),
                "trades": trades,
                "contribution_pct": round(
                    (pnl / kpis.total_pnl * 100) if kpis.total_pnl != 0 else 0.0, 2
                ),
            })

        # Sort by P&L descending
        strategies.sort(key=lambda s: s["pnl"], reverse=True)

        # Capital allocation
        allocation = {}
        if self._orchestrator and hasattr(self._orchestrator, '_capital_governor'):
            gov = self._orchestrator._capital_governor
            if gov and hasattr(gov, 'get_allocations'):
                try:
                    allocation = gov.get_allocations()
                except Exception:
                    pass

        return {
            "strategies": strategies,
            "total_pnl": round(kpis.total_pnl, 2),
            "total_trades": kpis.total_trades,
            "allocation": allocation,
        }

    async def _get_positions_data(self) -> dict:
        """Positions panel data."""
        open_positions = []
        closed_positions = []

        # Open positions from broker
        if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected:
            try:
                portfolio_state = await self._broker.get_portfolio_state()
                if portfolio_state and hasattr(portfolio_state, 'positions'):
                    for symbol, pos in portfolio_state.positions.items():
                        quantity = getattr(pos, 'quantity', 0)
                        if quantity == 0:
                            continue
                        avg_cost = getattr(pos, 'avg_cost', 0.0) or 0.0
                        market_value = getattr(pos, 'market_value', 0.0) or 0.0
                        unrealized_pnl = getattr(pos, 'unrealized_pnl', 0.0) or 0.0
                        realized_pnl = getattr(pos, 'realized_pnl', 0.0) or 0.0
                        # Position class has no market_price field - derive from market_value/qty
                        market_price = abs(market_value / quantity) if quantity != 0 else 0.0

                        # Compute entry price (divide by multiplier for futures)
                        from core.contract_specs import CONTRACT_SPECS
                        multiplier = 1.0
                        sym_upper = symbol.upper()
                        # Try exact match, then 3-char, 2-char prefixes
                        for candidate in [sym_upper, sym_upper[:3], sym_upper[:2]]:
                            if candidate in CONTRACT_SPECS:
                                multiplier = CONTRACT_SPECS[candidate].multiplier
                                break
                        entry_price = avg_cost / multiplier if multiplier != 0 else avg_cost

                        # Get CIO tracked data for TP/SL/R-Multiple
                        tp_price = None
                        sl_price = None
                        r_multiple = 0.0
                        entry_time = None
                        strategy = ""
                        stop_at_breakeven = False
                        trailing_active = False

                        if self._cio_agent and hasattr(self._cio_agent, '_tracked_positions'):
                            tracked = self._cio_agent._tracked_positions.get(symbol)
                            if tracked:
                                tp_price = getattr(tracked, 'take_profit_price', None)
                                sl_price = getattr(tracked, 'stop_loss_price', None)
                                r_multiple = getattr(tracked, 'r_multiple', 0.0) or 0.0
                                entry_time = getattr(tracked, 'entry_time', None)
                                strategy = getattr(tracked, 'source_agent', '') or ''
                                stop_at_breakeven = getattr(
                                    tracked, 'stop_moved_to_breakeven', False
                                )
                                trailing_active = getattr(
                                    tracked, 'trailing_stop_active', False
                                )

                        direction = "LONG" if quantity > 0 else "SHORT"
                        pnl_pct = (unrealized_pnl / abs(avg_cost * quantity) * 100
                                   if avg_cost * quantity != 0 else 0.0)

                        open_positions.append({
                            "symbol": symbol,
                            "direction": direction,
                            "quantity": abs(quantity),
                            "entry_price": round(entry_price, 4),
                            "current_price": round(market_price, 4),
                            "pnl": round(unrealized_pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "r_multiple": round(r_multiple, 2),
                            "tp_price": round(tp_price, 4) if tp_price else None,
                            "sl_price": round(sl_price, 4) if sl_price else None,
                            "strategy": strategy,
                            "entry_time": entry_time.isoformat() if entry_time else None,
                            "stop_at_breakeven": stop_at_breakeven,
                            "trailing_active": trailing_active,
                        })
            except Exception as e:
                logger.debug(f"Positions data error: {e}")

        # Closed positions (last 50)
        trades = list(self._closed_trades)[-50:]
        for t in reversed(trades):
            closed_positions.append({
                "symbol": t.symbol,
                "strategy": t.strategy,
                "pnl": round(t.pnl, 2),
                "r_multiple": round(t.r_multiple, 2),
                "hold_time_hours": round(t.hold_time_hours, 1),
            })

        return {
            "open": open_positions,
            "closed": closed_positions,
            "open_count": len(open_positions),
            "closed_count": len(closed_positions),
        }

    def _get_execution_data(self) -> dict:
        """Execution quality panel data."""
        data = {
            "fill_rate_pct": 100.0,
            "avg_slippage_bps": 0.0,
            "vwap_deviation_bps": 0.0,
            "implementation_shortfall_bps": 0.0,
            "zombie_orders": 0,
            "avg_fill_latency_ms": 0.0,
            "total_orders": 0,
            "total_fills": 0,
            "algo_distribution": {},
        }

        if self._execution_agent:
            try:
                if hasattr(self._execution_agent, 'get_status'):
                    exec_status = self._execution_agent.get_status()
                    data["total_orders"] = exec_status.get("total_orders", 0)
                    data["total_fills"] = exec_status.get("total_fills", 0)
                    data["avg_slippage_bps"] = round(
                        exec_status.get("avg_slippage_bps", 0.0), 2
                    )

                    # Fill rate
                    if data["total_orders"] > 0:
                        data["fill_rate_pct"] = round(
                            data["total_fills"] / data["total_orders"] * 100, 1
                        )

                # Zombie orders
                if hasattr(self._execution_agent, '_zombie_orders'):
                    data["zombie_orders"] = len(self._execution_agent._zombie_orders)
                elif hasattr(self._execution_agent, '_pending_orders'):
                    data["zombie_orders"] = len(self._execution_agent._pending_orders)

                # Fill quality stats
                if hasattr(self._execution_agent, '_fill_quality_stats'):
                    fq = self._execution_agent._fill_quality_stats
                    data["vwap_deviation_bps"] = round(
                        getattr(fq, 'avg_vwap_deviation_bps', 0.0), 2
                    )
                    data["implementation_shortfall_bps"] = round(
                        getattr(fq, 'avg_impl_shortfall_bps', 0.0), 2
                    )
                    data["avg_fill_latency_ms"] = round(
                        getattr(fq, 'avg_latency_ms', 0.0), 1
                    )

                # Algo distribution
                if hasattr(self._execution_agent, '_algo_usage'):
                    data["algo_distribution"] = dict(self._execution_agent._algo_usage)
            except Exception as e:
                logger.debug(f"Execution data error: {e}")

        return data


# =============================================================================
# Factory
# =============================================================================

def create_dashboard_v2_server(
    event_bus: "EventBus | None" = None,
    orchestrator: Any = None,
    host: str = "0.0.0.0",
    port: int = 8082,
) -> DashboardV2Server:
    """Create a V2 dashboard server instance."""
    server = DashboardV2Server(
        event_bus=event_bus,
        host=host,
        port=port,
    )
    if orchestrator is not None:
        server.set_orchestrator(orchestrator)
    return server

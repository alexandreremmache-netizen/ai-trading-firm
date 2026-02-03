"""
Broker Sync Module
==================

Bridges IB broker data to the dashboard for real-time P&L and performance metrics.

This module periodically syncs portfolio data from Interactive Brokers to the dashboard,
ensuring actual account values, positions, and P&L are displayed correctly.

Features:
- Periodic portfolio state sync from IBBroker
- Real-time P&L calculation from IB positions
- Position sync to PositionView component
- Performance metrics updates
- WebSocket broadcasting of updates

Per CLAUDE.md:
- Observable and auditable system
- Event-driven where possible, scheduled sync where necessary
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.broker import IBBroker, PortfolioState, Position
    from dashboard.server import DashboardServer, DashboardState
    from dashboard.components.performance_metrics import PerformanceMetrics
    from dashboard.components.position_view import PositionView


logger = logging.getLogger(__name__)


@dataclass
class BrokerSyncConfig:
    """Configuration for broker sync."""
    # Sync intervals
    portfolio_sync_interval_seconds: float = 2.0  # Sync portfolio every 2 seconds
    performance_sync_interval_seconds: float = 5.0  # Update performance metrics every 5 seconds

    # Initial delay before first sync (wait for broker connection)
    initial_delay_seconds: float = 5.0

    # Enable/disable specific syncs
    sync_positions: bool = True
    sync_pnl: bool = True
    sync_performance: bool = True

    # Broadcast settings
    broadcast_enabled: bool = True


@dataclass
class RealTimePnL:
    """Real-time P&L data structure for dashboard display."""
    # Account values
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0

    # P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0

    # Position stats
    position_count: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0

    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    broker_timestamp: datetime | None = None

    # Account info
    account_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "net_liquidation": round(self.net_liquidation, 2),
            "total_cash": round(self.total_cash, 2),
            "buying_power": round(self.buying_power, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "position_count": self.position_count,
            "gross_exposure": round(self.gross_exposure, 2),
            "net_exposure": round(self.net_exposure, 2),
            "long_exposure": round(self.long_exposure, 2),
            "short_exposure": round(self.short_exposure, 2),
            "timestamp": self.timestamp.isoformat(),
            "broker_timestamp": self.broker_timestamp.isoformat() if self.broker_timestamp else None,
            "account_id": self.account_id,
        }


@dataclass
class PositionData:
    """Position data structure for dashboard display."""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    weight_pct: float
    exchange: str = "SMART"
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": round(self.avg_cost, 4),
            "market_value": round(self.market_value, 2),
            "current_price": round(self.current_price, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "weight_pct": round(self.weight_pct, 2),
            "exchange": self.exchange,
            "currency": self.currency,
        }


class BrokerMetricsSync:
    """
    Syncs broker data to dashboard metrics.

    Periodically fetches portfolio state from IBBroker and updates
    the dashboard with real P&L and performance metrics.

    Usage:
        broker_sync = BrokerMetricsSync(broker, dashboard_server)
        await broker_sync.start()

        # Later...
        await broker_sync.stop()
    """

    def __init__(
        self,
        broker: "IBBroker",
        dashboard_server: "DashboardServer | None" = None,
        config: BrokerSyncConfig | None = None,
        performance_metrics: "PerformanceMetrics | None" = None,
        position_view: "PositionView | None" = None,
    ):
        """
        Initialize broker sync.

        Args:
            broker: IBBroker instance for fetching portfolio data
            dashboard_server: Optional DashboardServer for WebSocket broadcasting
            config: Sync configuration
            performance_metrics: Optional PerformanceMetrics instance to update
            position_view: Optional PositionView instance to update
        """
        self._broker = broker
        self._dashboard_server = dashboard_server
        self._config = config or BrokerSyncConfig()
        self._performance_metrics = performance_metrics
        self._position_view = position_view

        # Current state
        self._current_pnl = RealTimePnL()
        self._positions: dict[str, PositionData] = {}
        self._initial_capital: float | None = None
        self._day_start_value: float | None = None
        self._day_start_date: datetime | None = None

        # Sync tasks
        self._sync_task: asyncio.Task | None = None
        self._running = False

        # Callbacks for custom handling
        self._pnl_callbacks: list[Callable[[RealTimePnL], None]] = []
        self._position_callbacks: list[Callable[[dict[str, PositionData]], None]] = []

        logger.info(
            f"BrokerMetricsSync initialized: "
            f"portfolio_interval={self._config.portfolio_sync_interval_seconds}s"
        )

    async def start(self) -> None:
        """Start the sync tasks."""
        if self._running:
            logger.warning("BrokerMetricsSync already running")
            return

        self._running = True

        # Wait for broker to connect
        logger.info(f"Waiting {self._config.initial_delay_seconds}s for broker connection...")
        await asyncio.sleep(self._config.initial_delay_seconds)

        # Start sync task
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("BrokerMetricsSync started")

    async def stop(self) -> None:
        """Stop the sync tasks."""
        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info("BrokerMetricsSync stopped")

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        last_performance_sync = datetime.now(timezone.utc)

        while self._running:
            try:
                # Sync portfolio state
                await self._sync_portfolio()

                # Sync performance metrics less frequently
                now = datetime.now(timezone.utc)
                if (now - last_performance_sync).total_seconds() >= self._config.performance_sync_interval_seconds:
                    await self._sync_performance()
                    last_performance_sync = now

                # Broadcast updates
                if self._config.broadcast_enabled and self._dashboard_server:
                    await self._broadcast_updates()

                await asyncio.sleep(self._config.portfolio_sync_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in broker sync loop: {e}")
                await asyncio.sleep(self._config.portfolio_sync_interval_seconds)

    async def _sync_portfolio(self) -> None:
        """Sync portfolio state from broker."""
        if not self._broker.is_connected:
            logger.debug("Broker not connected, skipping portfolio sync")
            return

        try:
            # Get portfolio state from IB
            portfolio_state = await self._broker.get_portfolio_state()

            # Calculate P&L metrics
            self._update_pnl_from_portfolio(portfolio_state)

            # Update positions
            if self._config.sync_positions:
                self._update_positions_from_portfolio(portfolio_state)

            # Notify callbacks
            for callback in self._pnl_callbacks:
                try:
                    callback(self._current_pnl)
                except Exception as e:
                    logger.exception(f"Error in P&L callback: {e}")

            for callback in self._position_callbacks:
                try:
                    callback(self._positions)
                except Exception as e:
                    logger.exception(f"Error in position callback: {e}")

            # Update dashboard state if available
            if self._dashboard_server:
                self._update_dashboard_state()

            # Update position view if available
            if self._position_view and self._config.sync_positions:
                self._update_position_view(portfolio_state)

        except Exception as e:
            logger.exception(f"Error syncing portfolio: {e}")

    def _update_pnl_from_portfolio(self, portfolio: "PortfolioState") -> None:
        """Calculate P&L metrics from portfolio state."""
        now = datetime.now(timezone.utc)

        # Track initial capital (first sync value)
        if self._initial_capital is None:
            self._initial_capital = portfolio.net_liquidation
            logger.info(f"Initial capital set to: ${self._initial_capital:,.2f}")

        # Track day start value
        today = now.date()
        if self._day_start_date is None or self._day_start_date != today:
            self._day_start_value = portfolio.net_liquidation
            self._day_start_date = today
            logger.info(f"Day start value set to: ${self._day_start_value:,.2f}")

        # Calculate exposure
        long_exposure = 0.0
        short_exposure = 0.0
        total_unrealized = 0.0
        total_realized = 0.0

        for pos in portfolio.positions.values():
            if pos.quantity > 0:
                long_exposure += pos.market_value
            else:
                short_exposure += abs(pos.market_value)
            total_unrealized += pos.unrealized_pnl
            total_realized += pos.realized_pnl

        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # Calculate daily P&L percentage
        daily_pnl_pct = 0.0
        if self._day_start_value and self._day_start_value > 0:
            daily_pnl_pct = (portfolio.daily_pnl / self._day_start_value) * 100

        # Total P&L (from initial capital)
        total_pnl = portfolio.net_liquidation - (self._initial_capital or portfolio.net_liquidation)

        self._current_pnl = RealTimePnL(
            net_liquidation=portfolio.net_liquidation,
            total_cash=portfolio.total_cash,
            buying_power=portfolio.buying_power,
            daily_pnl=portfolio.daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            unrealized_pnl=total_unrealized,
            realized_pnl=total_realized,
            total_pnl=total_pnl,
            position_count=len(portfolio.positions),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            timestamp=now,
            broker_timestamp=portfolio.timestamp,
            account_id=portfolio.account_id,
        )

    def _update_positions_from_portfolio(self, portfolio: "PortfolioState") -> None:
        """Update positions from portfolio state."""
        self._positions.clear()

        total_value = portfolio.net_liquidation

        for symbol, pos in portfolio.positions.items():
            # Calculate current price from market value
            current_price = 0.0
            if pos.quantity != 0:
                current_price = abs(pos.market_value / pos.quantity)

            # Calculate unrealized P&L percentage
            unrealized_pnl_pct = 0.0
            cost_basis = pos.quantity * pos.avg_cost
            if cost_basis != 0:
                unrealized_pnl_pct = (pos.unrealized_pnl / abs(cost_basis)) * 100

            # Calculate weight
            weight_pct = 0.0
            if total_value > 0:
                weight_pct = (abs(pos.market_value) / total_value) * 100

            self._positions[symbol] = PositionData(
                symbol=symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_value=pos.market_value,
                current_price=current_price,
                unrealized_pnl=pos.unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                realized_pnl=pos.realized_pnl,
                weight_pct=weight_pct,
                exchange=pos.exchange,
                currency=pos.currency,
            )

    def _update_dashboard_state(self) -> None:
        """Update dashboard state with current data."""
        if not self._dashboard_server:
            return

        state = self._dashboard_server.state

        # Update metrics
        state.update_metrics({
            "total_pnl": self._current_pnl.total_pnl,
            "today_pnl": self._current_pnl.daily_pnl,
            "drawdown": 0.0,  # Will be calculated by performance metrics
            "position_count": self._current_pnl.position_count,
        })

        # Update equity curve
        state.update_equity_curve(
            self._current_pnl.timestamp.strftime("%H:%M:%S"),
            self._current_pnl.net_liquidation,
        )

    def _update_position_view(self, portfolio: "PortfolioState") -> None:
        """Update PositionView component with broker positions."""
        if not self._position_view:
            return

        # Set cash
        self._position_view.set_cash(portfolio.total_cash)

        # Update each position
        for symbol, pos in portfolio.positions.items():
            current_price = 0.0
            if pos.quantity != 0:
                current_price = abs(pos.market_value / pos.quantity)

            self._position_view.update_position(
                symbol=symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                current_price=current_price,
            )

    async def _sync_performance(self) -> None:
        """Sync performance metrics."""
        if not self._performance_metrics:
            return

        try:
            # Update equity in performance metrics
            await self._performance_metrics.update_equity(
                equity_value=self._current_pnl.net_liquidation,
                unrealized_pnl=self._current_pnl.unrealized_pnl,
            )
        except Exception as e:
            logger.exception(f"Error syncing performance metrics: {e}")

    async def _broadcast_updates(self) -> None:
        """Broadcast updates to WebSocket clients."""
        if not self._dashboard_server:
            return

        try:
            # Broadcast P&L update
            await self._dashboard_server.broadcast_event(
                "pnl",
                self._current_pnl.to_dict(),
            )

            # Broadcast positions update
            positions_list = [p.to_dict() for p in self._positions.values()]
            await self._dashboard_server.broadcast_event(
                "broker_positions",
                {
                    "positions": positions_list,
                    "count": len(positions_list),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            logger.debug(f"Error broadcasting updates: {e}")

    # =========================================================================
    # Public Methods
    # =========================================================================

    def get_current_pnl(self) -> RealTimePnL:
        """Get current P&L data."""
        return self._current_pnl

    def get_positions(self) -> dict[str, PositionData]:
        """Get current positions."""
        return self._positions.copy()

    def on_pnl_update(self, callback: Callable[[RealTimePnL], None]) -> None:
        """Register callback for P&L updates."""
        self._pnl_callbacks.append(callback)

    def on_position_update(self, callback: Callable[[dict[str, PositionData]], None]) -> None:
        """Register callback for position updates."""
        self._position_callbacks.append(callback)

    async def force_sync(self) -> None:
        """Force an immediate sync."""
        await self._sync_portfolio()
        await self._sync_performance()
        if self._config.broadcast_enabled and self._dashboard_server:
            await self._broadcast_updates()

    def to_dict(self) -> dict[str, Any]:
        """Export current state as dictionary."""
        return {
            "pnl": self._current_pnl.to_dict(),
            "positions": {sym: pos.to_dict() for sym, pos in self._positions.items()},
            "config": {
                "portfolio_sync_interval": self._config.portfolio_sync_interval_seconds,
                "performance_sync_interval": self._config.performance_sync_interval_seconds,
                "sync_positions": self._config.sync_positions,
                "sync_pnl": self._config.sync_pnl,
            },
            "status": {
                "running": self._running,
                "broker_connected": self._broker.is_connected if self._broker else False,
                "initial_capital": self._initial_capital,
                "day_start_value": self._day_start_value,
            },
        }


def create_broker_sync(
    broker: "IBBroker",
    dashboard_server: "DashboardServer | None" = None,
    performance_metrics: "PerformanceMetrics | None" = None,
    position_view: "PositionView | None" = None,
    portfolio_interval: float = 2.0,
    performance_interval: float = 5.0,
) -> BrokerMetricsSync:
    """
    Factory function to create BrokerMetricsSync.

    Args:
        broker: IBBroker instance
        dashboard_server: Optional DashboardServer for WebSocket
        performance_metrics: Optional PerformanceMetrics to update
        position_view: Optional PositionView to update
        portfolio_interval: Seconds between portfolio syncs
        performance_interval: Seconds between performance syncs

    Returns:
        Configured BrokerMetricsSync instance
    """
    config = BrokerSyncConfig(
        portfolio_sync_interval_seconds=portfolio_interval,
        performance_sync_interval_seconds=performance_interval,
    )

    return BrokerMetricsSync(
        broker=broker,
        dashboard_server=dashboard_server,
        config=config,
        performance_metrics=performance_metrics,
        position_view=position_view,
    )

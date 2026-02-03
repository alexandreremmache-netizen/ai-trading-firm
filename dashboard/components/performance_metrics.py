"""
Performance Metrics Tracker
===========================

P&L and performance analytics component for the trading system dashboard.

Provides comprehensive performance tracking including:
- Total and daily returns
- Risk-adjusted metrics (Sharpe, Sortino)
- Drawdown analysis
- Win rate and profit factor
- Rolling calculations (1d, 1w, 1m, YTD, inception)
- Benchmark comparison (SPY)
- Equity curve for charting

Integrates with:
- core/attribution.py for TWR/MWR calculations
- core/var_calculator.py for risk metrics patterns

Features:
- Thread-safe with asyncio locks
- WebSocket-ready export to dict
- Export for dashboard charts (Chart.js compatible)
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from enum import Enum
from typing import Any, TYPE_CHECKING

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

if TYPE_CHECKING:
    from core.attribution import PerformanceAttribution


logger = logging.getLogger(__name__)


class RollingPeriod(Enum):
    """Standard rolling periods for performance calculations."""
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1m"
    MONTH_3 = "3m"
    MONTH_6 = "6m"
    YTD = "ytd"
    YEAR_1 = "1y"
    INCEPTION = "inception"


# Number of days for each period
PERIOD_DAYS: dict[RollingPeriod, int | None] = {
    RollingPeriod.DAY_1: 1,
    RollingPeriod.WEEK_1: 7,
    RollingPeriod.MONTH_1: 30,
    RollingPeriod.MONTH_3: 90,
    RollingPeriod.MONTH_6: 180,
    RollingPeriod.YTD: None,  # Calculated dynamically
    RollingPeriod.YEAR_1: 365,
    RollingPeriod.INCEPTION: None,  # All available data
}


@dataclass
class DailyPnL:
    """
    Daily P&L record for performance tracking.

    Captures all relevant metrics for a single trading day.
    """
    date: date
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    trades_count: int = 0
    winners: int = 0
    losers: int = 0
    commissions: float = 0.0
    slippage: float = 0.0
    opening_equity: float = 0.0
    closing_equity: float = 0.0
    high_equity: float = 0.0
    low_equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def return_pct(self) -> float:
        """Daily return as percentage."""
        if self.opening_equity <= 0:
            return 0.0
        return (self.net_pnl / self.opening_equity) * 100

    @property
    def win_rate(self) -> float:
        """Daily win rate."""
        total = self.winners + self.losers
        if total == 0:
            return 0.0
        return self.winners / total

    @property
    def avg_pnl_per_trade(self) -> float:
        """Average P&L per trade for the day."""
        if self.trades_count == 0:
            return 0.0
        return self.net_pnl / self.trades_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "date": self.date.isoformat(),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "trades_count": self.trades_count,
            "winners": self.winners,
            "losers": self.losers,
            "commissions": round(self.commissions, 2),
            "slippage": round(self.slippage, 2),
            "opening_equity": round(self.opening_equity, 2),
            "closing_equity": round(self.closing_equity, 2),
            "return_pct": round(self.return_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
        }


@dataclass
class EquityCurvePoint:
    """Single point on the equity curve."""
    timestamp: datetime
    equity_value: float
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    high_water_mark: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for charting."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": round(self.equity_value, 2),
            "drawdown": round(self.drawdown, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "high_water_mark": round(self.high_water_mark, 2),
        }


@dataclass
class RollingMetrics:
    """Metrics for a specific rolling period."""
    period: RollingPeriod
    start_date: datetime
    end_date: datetime
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_count: int = 0
    volatility: float = 0.0
    calmar_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "period": self.period.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return": round(self.annualized_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4) if self.sortino_ratio != float("inf") else None,
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4) if self.profit_factor != float("inf") else None,
            "trades_count": self.trades_count,
            "volatility": round(self.volatility, 4),
            "calmar_ratio": round(self.calmar_ratio, 4) if self.calmar_ratio != float("inf") else None,
        }


@dataclass
class BenchmarkComparison:
    """Comparison metrics against a benchmark (e.g., SPY)."""
    benchmark_symbol: str
    period: RollingPeriod
    portfolio_return: float = 0.0
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    active_return: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "benchmark_symbol": self.benchmark_symbol,
            "period": self.period.value,
            "portfolio_return": round(self.portfolio_return, 4),
            "benchmark_return": round(self.benchmark_return, 4),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "correlation": round(self.correlation, 4),
            "tracking_error": round(self.tracking_error, 4),
            "information_ratio": round(self.information_ratio, 4) if self.information_ratio != float("inf") else None,
            "active_return": round(self.active_return, 4),
        }


@dataclass
class PerformanceTradeRecord:
    """Record of a single trade for performance metrics calculation."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    entry_price: float
    exit_price: float | None = None
    exit_timestamp: datetime | None = None
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    is_closed: bool = False
    strategy: str = ""

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0

    @property
    def is_loser(self) -> bool:
        """Check if trade was a loss."""
        return self.net_pnl < 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "commission": round(self.commission, 2),
            "slippage": round(self.slippage, 2),
            "is_closed": self.is_closed,
            "strategy": self.strategy,
        }


class PerformanceMetrics:
    """
    Comprehensive performance metrics tracker.

    Tracks and calculates all key performance indicators for the
    trading system dashboard including returns, risk metrics,
    drawdown analysis, and benchmark comparison.

    Thread-safe for async usage with WebSocket streaming.
    """

    # Annualization factor (trading days per year)
    TRADING_DAYS_PER_YEAR = 252

    # Default risk-free rate (annual)
    DEFAULT_RISK_FREE_RATE = 0.05

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        benchmark_symbol: str = "SPY",
        attribution: PerformanceAttribution | None = None,
    ):
        """
        Initialize performance metrics tracker.

        Args:
            initial_capital: Starting capital for return calculations
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
            benchmark_symbol: Symbol for benchmark comparison (default: SPY)
            attribution: Optional PerformanceAttribution instance for TWR/MWR
        """
        self._initial_capital = initial_capital
        self._current_equity = initial_capital
        self._risk_free_rate = risk_free_rate
        self._benchmark_symbol = benchmark_symbol
        self._attribution = attribution

        # Thread safety
        self._lock = asyncio.Lock()

        # Trade tracking
        self._trades: dict[str, PerformanceTradeRecord] = {}
        self._closed_trades: list[PerformanceTradeRecord] = []
        self._trade_counter = 0

        # Equity curve (timestamp, equity_value)
        self._equity_curve: list[EquityCurvePoint] = []
        self._high_water_mark = initial_capital

        # Daily P&L tracking
        self._daily_pnl: dict[date, DailyPnL] = {}

        # Returns series for calculations
        self._daily_returns: deque = deque(maxlen=1000)

        # Benchmark returns (if available)
        self._benchmark_returns: deque = deque(maxlen=1000)
        self._benchmark_prices: list[tuple[datetime, float]] = []

        # Current metrics cache
        self._metrics_cache: dict[str, Any] = {}
        self._last_metrics_update: datetime | None = None
        self._cache_valid_seconds = 1.0  # Cache validity period

        # Inception date
        self._inception_date = datetime.now(timezone.utc)

        # Initialize first equity point
        self._equity_curve.append(EquityCurvePoint(
            timestamp=self._inception_date,
            equity_value=initial_capital,
            drawdown=0.0,
            drawdown_pct=0.0,
            high_water_mark=initial_capital,
        ))

        logger.info(
            f"PerformanceMetrics initialized: capital={initial_capital}, "
            f"benchmark={benchmark_symbol}, rf_rate={risk_free_rate}"
        )

    # =========================================================================
    # TRADE RECORDING
    # =========================================================================

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        strategy: str = "",
        entry_time: datetime | None = None,
        exit_time: datetime | None = None,
    ) -> str:
        """
        Record a completed trade and update metrics.

        Args:
            symbol: Instrument symbol
            side: "buy" or "sell"
            quantity: Number of units traded
            entry_price: Entry price
            exit_price: Exit price
            commission: Total commission (entry + exit)
            slippage: Execution slippage
            strategy: Strategy name
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp

        Returns:
            Trade ID
        """
        async with self._lock:
            self._trade_counter += 1
            trade_id = f"TRD-{self._trade_counter:08d}"

            now = datetime.now(timezone.utc)
            entry_time = entry_time or now
            exit_time = exit_time or now

            # Calculate P&L
            multiplier = 1 if side.lower() == "buy" else -1
            gross_pnl = (exit_price - entry_price) * quantity * multiplier
            net_pnl = gross_pnl - commission - abs(slippage)

            trade = PerformanceTradeRecord(
                trade_id=trade_id,
                timestamp=entry_time,
                symbol=symbol,
                side=side.lower(),
                quantity=quantity,
                entry_price=entry_price,
                exit_price=exit_price,
                exit_timestamp=exit_time,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                commission=commission,
                slippage=slippage,
                is_closed=True,
                strategy=strategy,
            )

            self._trades[trade_id] = trade
            self._closed_trades.append(trade)

            # Update current equity
            self._current_equity += net_pnl

            # Update high water mark
            if self._current_equity > self._high_water_mark:
                self._high_water_mark = self._current_equity

            # Calculate current drawdown
            drawdown = self._high_water_mark - self._current_equity
            drawdown_pct = (drawdown / self._high_water_mark) if self._high_water_mark > 0 else 0.0

            # Add equity curve point
            self._equity_curve.append(EquityCurvePoint(
                timestamp=exit_time,
                equity_value=self._current_equity,
                drawdown=drawdown,
                drawdown_pct=drawdown_pct,
                high_water_mark=self._high_water_mark,
            ))

            # Update daily P&L
            trade_date = exit_time.date()
            if trade_date not in self._daily_pnl:
                self._daily_pnl[trade_date] = DailyPnL(
                    date=trade_date,
                    opening_equity=self._current_equity - net_pnl,
                )

            daily = self._daily_pnl[trade_date]
            daily.gross_pnl += gross_pnl
            daily.net_pnl += net_pnl
            daily.trades_count += 1
            daily.commissions += commission
            daily.slippage += abs(slippage)
            daily.closing_equity = self._current_equity
            daily.realized_pnl += net_pnl

            if net_pnl > 0:
                daily.winners += 1
            elif net_pnl < 0:
                daily.losers += 1

            # Update daily high/low
            if self._current_equity > daily.high_equity or daily.high_equity == 0:
                daily.high_equity = self._current_equity
            if self._current_equity < daily.low_equity or daily.low_equity == 0:
                daily.low_equity = self._current_equity

            # Invalidate metrics cache
            self._last_metrics_update = None

            logger.debug(
                f"Recorded trade {trade_id}: {symbol} {side} {quantity} @ "
                f"{entry_price}->{exit_price}, P&L={net_pnl:.2f}"
            )

            return trade_id

    async def update_equity(
        self,
        equity_value: float,
        unrealized_pnl: float = 0.0,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update current equity value (for periodic snapshots).

        Args:
            equity_value: Current portfolio equity
            unrealized_pnl: Current unrealized P&L
            timestamp: Snapshot timestamp
        """
        async with self._lock:
            timestamp = timestamp or datetime.now(timezone.utc)

            self._current_equity = equity_value

            # Update high water mark
            if equity_value > self._high_water_mark:
                self._high_water_mark = equity_value

            # Calculate drawdown
            drawdown = self._high_water_mark - equity_value
            drawdown_pct = (drawdown / self._high_water_mark) if self._high_water_mark > 0 else 0.0

            # Add equity curve point
            self._equity_curve.append(EquityCurvePoint(
                timestamp=timestamp,
                equity_value=equity_value,
                drawdown=drawdown,
                drawdown_pct=drawdown_pct,
                high_water_mark=self._high_water_mark,
            ))

            # Update daily unrealized P&L
            today = timestamp.date()
            if today in self._daily_pnl:
                self._daily_pnl[today].unrealized_pnl = unrealized_pnl
                self._daily_pnl[today].closing_equity = equity_value

            # Calculate and store daily return
            if len(self._equity_curve) >= 2:
                prev_equity = self._equity_curve[-2].equity_value
                if prev_equity > 0:
                    daily_return = (equity_value - prev_equity) / prev_equity
                    self._daily_returns.append(daily_return)

            # Invalidate cache
            self._last_metrics_update = None

    async def update_benchmark_price(
        self,
        price: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update benchmark price for comparison calculations.

        Args:
            price: Current benchmark price
            timestamp: Price timestamp
        """
        async with self._lock:
            timestamp = timestamp or datetime.now(timezone.utc)
            self._benchmark_prices.append((timestamp, price))

            # Calculate benchmark return if we have previous price
            if len(self._benchmark_prices) >= 2:
                prev_price = self._benchmark_prices[-2][1]
                if prev_price > 0:
                    bench_return = (price - prev_price) / prev_price
                    self._benchmark_returns.append(bench_return)

    # =========================================================================
    # METRICS CALCULATIONS
    # =========================================================================

    async def calculate_metrics(self) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary with all calculated metrics
        """
        async with self._lock:
            # Check cache validity
            now = datetime.now(timezone.utc)
            if (
                self._last_metrics_update is not None
                and (now - self._last_metrics_update).total_seconds() < self._cache_valid_seconds
            ):
                return self._metrics_cache

            metrics = {
                "timestamp": now.isoformat(),
                "initial_capital": self._initial_capital,
                "current_equity": round(self._current_equity, 2),
                "high_water_mark": round(self._high_water_mark, 2),
            }

            # Total return
            total_return = self._current_equity - self._initial_capital
            total_return_pct = (total_return / self._initial_capital) if self._initial_capital > 0 else 0.0
            metrics["total_return"] = round(total_return, 2)
            metrics["total_return_pct"] = round(total_return_pct * 100, 4)

            # Daily return (latest)
            if self._daily_returns:
                metrics["daily_return"] = round(list(self._daily_returns)[-1] * 100, 4)
            else:
                metrics["daily_return"] = 0.0

            # Sharpe ratio
            metrics["sharpe_ratio"] = round(self._calculate_sharpe_ratio(), 4)

            # Sortino ratio
            sortino = self._calculate_sortino_ratio()
            metrics["sortino_ratio"] = round(sortino, 4) if sortino != float("inf") else None

            # Drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics()
            metrics["max_drawdown"] = round(drawdown_metrics["max_drawdown"], 2)
            metrics["max_drawdown_pct"] = round(drawdown_metrics["max_drawdown_pct"] * 100, 4)
            metrics["current_drawdown"] = round(drawdown_metrics["current_drawdown"], 2)
            metrics["current_drawdown_pct"] = round(drawdown_metrics["current_drawdown_pct"] * 100, 4)

            # Win rate and profit factor
            trade_metrics = self._calculate_trade_metrics()
            metrics["win_rate"] = round(trade_metrics["win_rate"] * 100, 2)
            metrics["profit_factor"] = (
                round(trade_metrics["profit_factor"], 4)
                if trade_metrics["profit_factor"] != float("inf")
                else None
            )
            metrics["total_trades"] = trade_metrics["total_trades"]
            metrics["winning_trades"] = trade_metrics["winning_trades"]
            metrics["losing_trades"] = trade_metrics["losing_trades"]
            metrics["avg_win"] = round(trade_metrics["avg_win"], 2)
            metrics["avg_loss"] = round(trade_metrics["avg_loss"], 2)
            metrics["largest_win"] = round(trade_metrics["largest_win"], 2)
            metrics["largest_loss"] = round(trade_metrics["largest_loss"], 2)

            # Additional metrics
            metrics["volatility"] = round(self._calculate_volatility() * 100, 4)
            metrics["calmar_ratio"] = self._calculate_calmar_ratio()

            # Days since inception
            metrics["days_active"] = (now - self._inception_date).days

            # Update cache
            self._metrics_cache = metrics
            self._last_metrics_update = now

            return metrics

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if not HAS_NUMPY or len(self._daily_returns) < 2:
            return 0.0

        returns = np.array(list(self._daily_returns))

        # Filter invalid values
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0

        # Daily risk-free rate
        daily_rf = self._risk_free_rate / self.TRADING_DAYS_PER_YEAR

        # Excess returns
        mean_excess = np.mean(returns) - daily_rf
        std_returns = np.std(returns, ddof=1)

        if std_returns < 1e-12:
            return 0.0

        # Annualize
        sharpe = (mean_excess / std_returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(sharpe) if np.isfinite(sharpe) else 0.0

    def _calculate_sortino_ratio(self) -> float:
        """Calculate annualized Sortino ratio."""
        if not HAS_NUMPY or len(self._daily_returns) < 2:
            return 0.0

        returns = np.array(list(self._daily_returns))

        # Filter invalid values
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0

        # Daily risk-free rate (MAR)
        daily_rf = self._risk_free_rate / self.TRADING_DAYS_PER_YEAR

        mean_excess = np.mean(returns) - daily_rf

        # Downside returns only
        downside_returns = returns[returns < daily_rf]
        if len(downside_returns) == 0:
            return float("inf") if mean_excess > 0 else 0.0

        downside_deviations = downside_returns - daily_rf
        downside_std = np.std(downside_deviations, ddof=1)

        if downside_std < 1e-12:
            return 0.0

        # Annualize
        sortino = (mean_excess / downside_std) * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(sortino) if np.isfinite(sortino) else 0.0

    def _calculate_drawdown_metrics(self) -> dict[str, float]:
        """Calculate drawdown metrics from equity curve."""
        if not self._equity_curve:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "current_drawdown": 0.0,
                "current_drawdown_pct": 0.0,
            }

        # Get max drawdown from equity curve
        max_dd = 0.0
        max_dd_pct = 0.0

        for point in self._equity_curve:
            if point.drawdown > max_dd:
                max_dd = point.drawdown
            if point.drawdown_pct > max_dd_pct:
                max_dd_pct = point.drawdown_pct

        # Current drawdown
        current_point = self._equity_curve[-1]

        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "current_drawdown": current_point.drawdown,
            "current_drawdown_pct": current_point.drawdown_pct,
        }

    def _calculate_trade_metrics(self) -> dict[str, Any]:
        """Calculate trade-based metrics."""
        if not self._closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        winners = [t for t in self._closed_trades if t.is_winner]
        losers = [t for t in self._closed_trades if t.is_loser]

        total_trades = len(self._closed_trades)
        winning_trades = len(winners)
        losing_trades = len(losers)

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_wins = sum(t.net_pnl for t in winners)
        gross_losses = abs(sum(t.net_pnl for t in losers))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)

        # Average win/loss
        avg_win = gross_wins / winning_trades if winning_trades > 0 else 0.0
        avg_loss = gross_losses / losing_trades if losing_trades > 0 else 0.0

        # Largest win/loss
        largest_win = max((t.net_pnl for t in winners), default=0.0)
        largest_loss = min((t.net_pnl for t in losers), default=0.0)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
        }

    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if not HAS_NUMPY or len(self._daily_returns) < 2:
            return 0.0

        returns = np.array(list(self._daily_returns))
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            return 0.0

        daily_vol = np.std(returns, ddof=1)
        annualized_vol = daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(annualized_vol) if np.isfinite(annualized_vol) else 0.0

    def _calculate_calmar_ratio(self) -> float | None:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        drawdown_metrics = self._calculate_drawdown_metrics()
        max_dd_pct = drawdown_metrics["max_drawdown_pct"]

        if max_dd_pct < 0.0001:  # Essentially no drawdown
            return None

        # Annualized return
        days_active = (datetime.now(timezone.utc) - self._inception_date).days
        if days_active <= 0:
            return None

        total_return = (self._current_equity - self._initial_capital) / self._initial_capital
        years = days_active / 365.25

        if years > 0 and (1 + total_return) > 0:
            annualized_return = ((1 + total_return) ** (1 / years)) - 1
        else:
            return None

        calmar = annualized_return / max_dd_pct

        return round(calmar, 4) if not np.isinf(calmar) else None

    # =========================================================================
    # ROLLING CALCULATIONS
    # =========================================================================

    async def get_rolling_metrics(
        self,
        period: RollingPeriod,
    ) -> RollingMetrics:
        """
        Calculate metrics for a rolling period.

        Args:
            period: Rolling period (1d, 1w, 1m, YTD, inception)

        Returns:
            RollingMetrics for the specified period
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            # Determine start date based on period
            if period == RollingPeriod.YTD:
                start_date = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            elif period == RollingPeriod.INCEPTION:
                start_date = self._inception_date
            else:
                days = PERIOD_DAYS.get(period, 1)
                start_date = now - timedelta(days=days) if days else self._inception_date

            # Filter equity curve points in period
            points_in_period = [
                p for p in self._equity_curve
                if p.timestamp >= start_date
            ]

            if not points_in_period:
                return RollingMetrics(
                    period=period,
                    start_date=start_date,
                    end_date=now,
                )

            # Get starting and ending equity
            start_equity = points_in_period[0].equity_value
            end_equity = points_in_period[-1].equity_value

            # Total return
            total_return = end_equity - start_equity
            total_return_pct = (total_return / start_equity) if start_equity > 0 else 0.0

            # Annualized return
            days_in_period = (now - start_date).days
            years = days_in_period / 365.25
            if years > 0 and (1 + total_return_pct) > 0:
                annualized_return = ((1 + total_return_pct) ** (1 / years)) - 1
            else:
                annualized_return = total_return_pct

            # Get returns in period
            returns_in_period = []
            for i in range(1, len(points_in_period)):
                prev_eq = points_in_period[i-1].equity_value
                curr_eq = points_in_period[i].equity_value
                if prev_eq > 0:
                    returns_in_period.append((curr_eq - prev_eq) / prev_eq)

            # Sharpe and Sortino for period
            sharpe = self._calculate_sharpe_from_returns(returns_in_period)
            sortino = self._calculate_sortino_from_returns(returns_in_period)

            # Max drawdown in period
            max_dd = max((p.drawdown for p in points_in_period), default=0.0)
            max_dd_pct = max((p.drawdown_pct for p in points_in_period), default=0.0)

            # Trades in period
            trades_in_period = [
                t for t in self._closed_trades
                if t.exit_timestamp and t.exit_timestamp >= start_date
            ]

            # Win rate and profit factor for period
            winners = [t for t in trades_in_period if t.is_winner]
            losers = [t for t in trades_in_period if t.is_loser]

            win_rate = len(winners) / len(trades_in_period) if trades_in_period else 0.0

            gross_wins = sum(t.net_pnl for t in winners)
            gross_losses = abs(sum(t.net_pnl for t in losers))
            profit_factor = (
                gross_wins / gross_losses if gross_losses > 0
                else (float("inf") if gross_wins > 0 else 0.0)
            )

            # Volatility for period
            volatility = 0.0
            if HAS_NUMPY and len(returns_in_period) >= 2:
                returns_arr = np.array(returns_in_period)
                returns_arr = returns_arr[np.isfinite(returns_arr)]
                if len(returns_arr) >= 2:
                    volatility = float(np.std(returns_arr, ddof=1) * np.sqrt(self.TRADING_DAYS_PER_YEAR))

            # Calmar ratio
            calmar = annualized_return / max_dd_pct if max_dd_pct > 0.0001 else float("inf")

            return RollingMetrics(
                period=period,
                start_date=start_date,
                end_date=now,
                total_return=total_return,
                total_return_pct=total_return_pct,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                max_drawdown_pct=max_dd_pct,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trades_count=len(trades_in_period),
                volatility=volatility,
                calmar_ratio=calmar,
            )

    def _calculate_sharpe_from_returns(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio from a list of returns."""
        if not HAS_NUMPY or len(returns) < 2:
            return 0.0

        returns_arr = np.array(returns)
        returns_arr = returns_arr[np.isfinite(returns_arr)]

        if len(returns_arr) < 2:
            return 0.0

        daily_rf = self._risk_free_rate / self.TRADING_DAYS_PER_YEAR
        mean_excess = np.mean(returns_arr) - daily_rf
        std_returns = np.std(returns_arr, ddof=1)

        if std_returns < 1e-12:
            return 0.0

        sharpe = (mean_excess / std_returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return float(sharpe) if np.isfinite(sharpe) else 0.0

    def _calculate_sortino_from_returns(self, returns: list[float]) -> float:
        """Calculate Sortino ratio from a list of returns."""
        if not HAS_NUMPY or len(returns) < 2:
            return 0.0

        returns_arr = np.array(returns)
        returns_arr = returns_arr[np.isfinite(returns_arr)]

        if len(returns_arr) < 2:
            return 0.0

        daily_rf = self._risk_free_rate / self.TRADING_DAYS_PER_YEAR
        mean_excess = np.mean(returns_arr) - daily_rf

        downside = returns_arr[returns_arr < daily_rf]
        if len(downside) == 0:
            return float("inf") if mean_excess > 0 else 0.0

        downside_std = np.std(downside - daily_rf, ddof=1)
        if downside_std < 1e-12:
            return 0.0

        sortino = (mean_excess / downside_std) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return float(sortino) if np.isfinite(sortino) else 0.0

    async def get_all_rolling_metrics(self) -> dict[str, RollingMetrics]:
        """
        Get metrics for all standard rolling periods.

        Returns:
            Dictionary mapping period names to RollingMetrics
        """
        results = {}
        for period in RollingPeriod:
            metrics = await self.get_rolling_metrics(period)
            results[period.value] = metrics
        return results

    # =========================================================================
    # BENCHMARK COMPARISON
    # =========================================================================

    async def get_benchmark_comparison(
        self,
        period: RollingPeriod = RollingPeriod.INCEPTION,
    ) -> BenchmarkComparison:
        """
        Calculate benchmark comparison metrics.

        Args:
            period: Period for comparison

        Returns:
            BenchmarkComparison with alpha, beta, correlation, etc.
        """
        async with self._lock:
            if not HAS_NUMPY:
                return BenchmarkComparison(
                    benchmark_symbol=self._benchmark_symbol,
                    period=period,
                )

            # Get period dates
            now = datetime.now(timezone.utc)
            if period == RollingPeriod.YTD:
                start_date = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            elif period == RollingPeriod.INCEPTION:
                start_date = self._inception_date
            else:
                days = PERIOD_DAYS.get(period, 1)
                start_date = now - timedelta(days=days) if days else self._inception_date

            # Get portfolio returns in period
            portfolio_returns = list(self._daily_returns)
            benchmark_returns = list(self._benchmark_returns)

            # Ensure equal length
            min_len = min(len(portfolio_returns), len(benchmark_returns))
            if min_len < 2:
                return BenchmarkComparison(
                    benchmark_symbol=self._benchmark_symbol,
                    period=period,
                )

            port_arr = np.array(portfolio_returns[-min_len:])
            bench_arr = np.array(benchmark_returns[-min_len:])

            # Filter invalid values
            valid_mask = np.isfinite(port_arr) & np.isfinite(bench_arr)
            port_arr = port_arr[valid_mask]
            bench_arr = bench_arr[valid_mask]

            if len(port_arr) < 2:
                return BenchmarkComparison(
                    benchmark_symbol=self._benchmark_symbol,
                    period=period,
                )

            # Calculate returns
            portfolio_return = float(np.sum(port_arr))
            benchmark_return = float(np.sum(bench_arr))
            active_return = portfolio_return - benchmark_return

            # Beta (covariance / variance)
            cov_matrix = np.cov(port_arr, bench_arr)
            variance_bench = np.var(bench_arr, ddof=1)
            beta = cov_matrix[0, 1] / variance_bench if variance_bench > 1e-12 else 0.0

            # Alpha (Jensen's alpha)
            # alpha = portfolio_return - (rf + beta * (benchmark_return - rf))
            rf_period = self._risk_free_rate * (len(port_arr) / self.TRADING_DAYS_PER_YEAR)
            alpha = portfolio_return - (rf_period + beta * (benchmark_return - rf_period))

            # Correlation
            correlation = float(np.corrcoef(port_arr, bench_arr)[0, 1])

            # Tracking error (std of active returns)
            active_returns = port_arr - bench_arr
            tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(self.TRADING_DAYS_PER_YEAR))

            # Information ratio
            avg_active_return = np.mean(active_returns) * self.TRADING_DAYS_PER_YEAR
            information_ratio = avg_active_return / tracking_error if tracking_error > 1e-12 else float("inf")

            return BenchmarkComparison(
                benchmark_symbol=self._benchmark_symbol,
                period=period,
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                alpha=float(alpha),
                beta=float(beta),
                correlation=correlation if np.isfinite(correlation) else 0.0,
                tracking_error=tracking_error,
                information_ratio=float(information_ratio) if np.isfinite(information_ratio) else float("inf"),
                active_return=active_return,
            )

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    async def get_equity_curve(
        self,
        max_points: int | None = None,
        period: RollingPeriod | None = None,
    ) -> list[tuple[datetime, float]]:
        """
        Get equity curve for charting.

        Args:
            max_points: Maximum number of points to return (None for all)
            period: Optional period filter

        Returns:
            List of (timestamp, equity_value) tuples
        """
        async with self._lock:
            points = self._equity_curve.copy()

            # Filter by period if specified
            if period:
                now = datetime.now(timezone.utc)
                if period == RollingPeriod.YTD:
                    start_date = datetime(now.year, 1, 1, tzinfo=timezone.utc)
                elif period == RollingPeriod.INCEPTION:
                    start_date = self._inception_date
                else:
                    days = PERIOD_DAYS.get(period, 1)
                    start_date = now - timedelta(days=days) if days else self._inception_date

                points = [p for p in points if p.timestamp >= start_date]

            # Downsample if needed
            if max_points and len(points) > max_points:
                step = len(points) // max_points
                points = points[::step]

            return [(p.timestamp, p.equity_value) for p in points]

    async def get_equity_curve_for_chart(
        self,
        max_points: int = 100,
        period: RollingPeriod | None = None,
    ) -> dict[str, Any]:
        """
        Get equity curve data formatted for Chart.js.

        Args:
            max_points: Maximum data points
            period: Optional period filter

        Returns:
            Chart.js compatible data object
        """
        async with self._lock:
            points = self._equity_curve.copy()

            # Filter by period
            if period:
                now = datetime.now(timezone.utc)
                if period == RollingPeriod.YTD:
                    start_date = datetime(now.year, 1, 1, tzinfo=timezone.utc)
                elif period == RollingPeriod.INCEPTION:
                    start_date = self._inception_date
                else:
                    days = PERIOD_DAYS.get(period, 1)
                    start_date = now - timedelta(days=days) if days else self._inception_date

                points = [p for p in points if p.timestamp >= start_date]

            # Downsample
            if len(points) > max_points:
                step = len(points) // max_points
                points = points[::step]

            return {
                "labels": [p.timestamp.isoformat() for p in points],
                "datasets": [
                    {
                        "label": "Portfolio Equity",
                        "data": [round(p.equity_value, 2) for p in points],
                        "borderColor": "rgb(75, 192, 192)",
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "fill": True,
                    },
                    {
                        "label": "High Water Mark",
                        "data": [round(p.high_water_mark, 2) for p in points],
                        "borderColor": "rgb(54, 162, 235)",
                        "borderDash": [5, 5],
                        "fill": False,
                    },
                ],
            }

    async def get_drawdown_chart(
        self,
        max_points: int = 100,
        period: RollingPeriod | None = None,
    ) -> dict[str, Any]:
        """
        Get drawdown data formatted for Chart.js.

        Args:
            max_points: Maximum data points
            period: Optional period filter

        Returns:
            Chart.js compatible data object
        """
        async with self._lock:
            points = self._equity_curve.copy()

            # Filter by period
            if period:
                now = datetime.now(timezone.utc)
                if period == RollingPeriod.YTD:
                    start_date = datetime(now.year, 1, 1, tzinfo=timezone.utc)
                elif period == RollingPeriod.INCEPTION:
                    start_date = self._inception_date
                else:
                    days = PERIOD_DAYS.get(period, 1)
                    start_date = now - timedelta(days=days) if days else self._inception_date

                points = [p for p in points if p.timestamp >= start_date]

            # Downsample
            if len(points) > max_points:
                step = len(points) // max_points
                points = points[::step]

            return {
                "labels": [p.timestamp.isoformat() for p in points],
                "datasets": [
                    {
                        "label": "Drawdown %",
                        "data": [round(-p.drawdown_pct * 100, 2) for p in points],
                        "borderColor": "rgb(255, 99, 132)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "fill": True,
                    },
                ],
            }

    async def get_daily_pnl_series(
        self,
        days: int | None = None,
    ) -> list[DailyPnL]:
        """
        Get daily P&L series.

        Args:
            days: Number of recent days (None for all)

        Returns:
            List of DailyPnL records sorted by date
        """
        async with self._lock:
            sorted_days = sorted(self._daily_pnl.values(), key=lambda d: d.date)

            if days:
                sorted_days = sorted_days[-days:]

            return sorted_days

    async def get_daily_pnl_chart(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get daily P&L data formatted for Chart.js bar chart.

        Args:
            days: Number of days to include

        Returns:
            Chart.js compatible data object
        """
        daily_data = await self.get_daily_pnl_series(days)

        return {
            "labels": [d.date.isoformat() for d in daily_data],
            "datasets": [
                {
                    "label": "Daily P&L",
                    "data": [round(d.net_pnl, 2) for d in daily_data],
                    "backgroundColor": [
                        "rgba(75, 192, 192, 0.6)" if d.net_pnl >= 0
                        else "rgba(255, 99, 132, 0.6)"
                        for d in daily_data
                    ],
                    "borderColor": [
                        "rgb(75, 192, 192)" if d.net_pnl >= 0
                        else "rgb(255, 99, 132)"
                        for d in daily_data
                    ],
                    "borderWidth": 1,
                },
            ],
        }

    async def export_metrics_for_dashboard(self) -> dict[str, Any]:
        """
        Export all metrics in a format suitable for dashboard display.

        Returns:
            Comprehensive metrics dictionary
        """
        # Get base metrics
        metrics = await self.calculate_metrics()

        # Get rolling metrics
        rolling = await self.get_all_rolling_metrics()
        metrics["rolling"] = {k: v.to_dict() for k, v in rolling.items()}

        # Get benchmark comparison
        benchmark = await self.get_benchmark_comparison()
        metrics["benchmark"] = benchmark.to_dict()

        # Get charts data
        metrics["charts"] = {
            "equity_curve": await self.get_equity_curve_for_chart(),
            "drawdown": await self.get_drawdown_chart(),
            "daily_pnl": await self.get_daily_pnl_chart(),
        }

        # Get TWR/MWR if attribution is available
        if self._attribution:
            try:
                comparison = self._attribution.get_return_comparison()
                metrics["twr_mwr"] = comparison
            except Exception as e:
                logger.warning(f"Failed to get TWR/MWR comparison: {e}")
                metrics["twr_mwr"] = None

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """
        Synchronous export for WebSocket streaming.

        Returns:
            Basic metrics dictionary
        """
        # Build metrics synchronously (for non-async contexts)
        total_return = self._current_equity - self._initial_capital
        total_return_pct = (total_return / self._initial_capital) if self._initial_capital > 0 else 0.0

        drawdown_metrics = self._calculate_drawdown_metrics()
        trade_metrics = self._calculate_trade_metrics()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initial_capital": self._initial_capital,
            "current_equity": round(self._current_equity, 2),
            "high_water_mark": round(self._high_water_mark, 2),
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct * 100, 4),
            "sharpe_ratio": round(self._calculate_sharpe_ratio(), 4),
            "sortino_ratio": round(self._calculate_sortino_ratio(), 4),
            "max_drawdown_pct": round(drawdown_metrics["max_drawdown_pct"] * 100, 4),
            "current_drawdown_pct": round(drawdown_metrics["current_drawdown_pct"] * 100, 4),
            "win_rate": round(trade_metrics["win_rate"] * 100, 2),
            "profit_factor": (
                round(trade_metrics["profit_factor"], 4)
                if trade_metrics["profit_factor"] != float("inf")
                else None
            ),
            "total_trades": trade_metrics["total_trades"],
            "days_active": (datetime.now(timezone.utc) - self._inception_date).days,
        }

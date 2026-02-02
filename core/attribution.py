"""
Performance Attribution
=======================

Tracks and attributes P&L to strategies, enabling performance analysis
and dynamic weight adjustment. Essential for institutional-grade
portfolio management and compliance.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class TradeOutcome(Enum):
    """Trade outcome classification."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class TradeRecord:
    """Record of a single trade for attribution."""
    trade_id: str
    timestamp: datetime
    strategy: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    entry_price: float
    exit_price: float | None = None
    exit_timestamp: datetime | None = None
    realized_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    is_closed: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def gross_pnl(self) -> float:
        """Gross P&L before costs."""
        if not self.is_closed or self.exit_price is None:
            return 0.0
        multiplier = 1 if self.side == "buy" else -1
        return (self.exit_price - self.entry_price) * self.quantity * multiplier

    @property
    def net_pnl(self) -> float:
        """Net P&L after costs."""
        return self.gross_pnl - self.commission - abs(self.slippage)

    @property
    def outcome(self) -> TradeOutcome:
        """Classify trade outcome."""
        if not self.is_closed:
            return TradeOutcome.BREAKEVEN
        if self.net_pnl > 0:
            return TradeOutcome.WIN
        elif self.net_pnl < 0:
            return TradeOutcome.LOSS
        else:
            return TradeOutcome.BREAKEVEN

    @property
    def holding_period_hours(self) -> float | None:
        """Calculate holding period in hours."""
        if self.exit_timestamp is None:
            return None
        delta = self.exit_timestamp - self.timestamp
        return delta.total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "realized_pnl": self.realized_pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "is_closed": self.is_closed,
            "outcome": self.outcome.value,
            "net_pnl": self.net_pnl,
        }


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a strategy."""
    strategy: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    total_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_hours: float = 0.0
    returns: list[float] = field(default_factory=list)
    pnl_history: list[tuple[datetime, float]] = field(default_factory=list)
    risk_free_rate: float = 0.05  # Annual risk-free rate for Sharpe calculation
    data_frequency: str = "daily"  # "daily", "hourly", "minute", "trade" for annualization

    # Annualization factors for different frequencies
    ANNUALIZATION_FACTORS = {
        "yearly": 1,
        "monthly": 12,
        "weekly": 52,
        "daily": 252,  # Trading days per year
        "hourly": 252 * 6.5,  # 6.5 hours per trading day
        "minute": 252 * 6.5 * 60,
        "trade": 252,  # Default to daily-equivalent for trade-by-trade
    }

    @property
    def annualization_factor(self) -> float:
        """Get annualization factor based on data frequency."""
        return self.ANNUALIZATION_FACTORS.get(self.data_frequency, 252)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross wins / gross losses)."""
        wins = sum(r for r in self.returns if r > 0)
        losses = abs(sum(r for r in self.returns if r < 0))
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    @property
    def sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio using excess returns.

        Uses proper annualization based on data frequency:
        - daily: sqrt(252)
        - hourly: sqrt(252 * 6.5)
        - minute: sqrt(252 * 6.5 * 60)
        etc.
        """
        if len(self.returns) < 2:
            return 0.0

        returns_array = np.array(self.returns)
        ann_factor = self.annualization_factor

        # Convert annual risk-free rate to per-period rate
        period_rf = self.risk_free_rate / ann_factor

        # Calculate excess return
        mean_excess_return = np.mean(returns_array) - period_rf
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualize: multiply mean by factor, std by sqrt(factor)
        # Sharpe = (mean * factor) / (std * sqrt(factor)) = mean / std * sqrt(factor)
        return (mean_excess_return / std_return) * np.sqrt(ann_factor)

    @property
    def sortino_ratio(self) -> float:
        """
        Calculate Sortino ratio using excess returns and downside deviation.

        Uses proper annualization based on data frequency.
        """
        if len(self.returns) < 2:
            return 0.0

        returns_array = np.array(self.returns)
        ann_factor = self.annualization_factor

        # Convert annual risk-free rate to per-period (used as MAR)
        period_rf = self.risk_free_rate / ann_factor
        mean_excess_return = np.mean(returns_array) - period_rf

        # Downside deviation: returns below MAR (risk-free rate)
        downside_returns = returns_array[returns_array < period_rf]

        if len(downside_returns) == 0:
            return float("inf") if mean_excess_return > 0 else 0.0

        # Calculate downside std from deviations below MAR
        downside_deviations = downside_returns - period_rf
        downside_std = np.std(downside_deviations, ddof=1)

        if downside_std == 0:
            return 0.0

        return (mean_excess_return / downside_std) * np.sqrt(ann_factor)

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.pnl_history) == 0:
            return 0.0
        cumulative = np.cumsum([p[1] for p in self.pnl_history])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return float(np.min(drawdowns))

    @property
    def expectancy(self) -> float:
        """Calculate trade expectancy (expected P&L per trade)."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "gross_pnl": self.gross_pnl,
            "profit_factor": self.profit_factor if self.profit_factor != float("inf") else None,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio if self.sortino_ratio != float("inf") else None,
            "max_drawdown": self.max_drawdown,
            "expectancy": self.expectancy,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_holding_hours": self.avg_holding_hours,
        }


class PerformanceAttribution:
    """
    Comprehensive performance attribution system.

    Features:
    - Trade-to-strategy mapping
    - P&L attribution by strategy
    - Risk-adjusted metrics (Sharpe, Sortino)
    - Win rate and profit factor tracking
    - Rolling performance windows
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize attribution system.

        Args:
            config: Configuration with:
                - rolling_window_days: Window for rolling metrics (default: 30)
                - risk_free_rate: Annual risk-free rate (default: 0.05)
        """
        self._config = config or {}
        self._rolling_window_days = self._config.get("rolling_window_days", 30)
        self._risk_free_rate = self._config.get("risk_free_rate", 0.05)

        # Trade storage
        self._trades: dict[str, TradeRecord] = {}
        self._open_trades: dict[str, TradeRecord] = {}  # By trade_id

        # Strategy metrics cache
        self._strategy_metrics: dict[str, StrategyMetrics] = {}

        # Portfolio-level tracking
        self._daily_pnl: list[tuple[datetime, float]] = []
        self._total_pnl = 0.0

        # Trade counter
        self._trade_counter = 0

        logger.info("PerformanceAttribution initialized")

    def record_trade_entry(
        self,
        strategy: str,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        commission: float = 0.0,
        tags: dict[str, str] | None = None
    ) -> str:
        """
        Record a new trade entry.

        Args:
            strategy: Strategy name
            symbol: Instrument symbol
            side: "buy" or "sell"
            quantity: Number of units
            entry_price: Entry price
            commission: Entry commission
            tags: Optional tags for categorization

        Returns:
            Trade ID for future reference
        """
        self._trade_counter += 1
        trade_id = f"TRD-{self._trade_counter:08d}"

        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=datetime.now(timezone.utc),
            strategy=strategy,
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            entry_price=entry_price,
            commission=commission,
            tags=tags or {},
        )

        self._trades[trade_id] = trade
        self._open_trades[trade_id] = trade

        # Ensure strategy exists in metrics
        if strategy not in self._strategy_metrics:
            self._strategy_metrics[strategy] = StrategyMetrics(strategy=strategy)

        logger.debug(f"Recorded trade entry: {trade_id} - {strategy} {side} {quantity} {symbol}")

        return trade_id

    def record_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> TradeRecord | None:
        """
        Record trade exit and calculate P&L.

        Args:
            trade_id: Trade ID from entry
            exit_price: Exit price
            commission: Exit commission
            slippage: Execution slippage

        Returns:
            Closed TradeRecord, or None if not found
        """
        if trade_id not in self._trades:
            logger.warning(f"Trade not found: {trade_id}")
            return None

        trade = self._trades[trade_id]
        if trade.is_closed:
            logger.warning(f"Trade already closed: {trade_id}")
            return trade

        # Update trade record
        now = datetime.now(timezone.utc)
        trade.exit_price = exit_price
        trade.exit_timestamp = now
        trade.commission += commission
        trade.slippage = slippage
        trade.is_closed = True
        trade.realized_pnl = trade.net_pnl

        # Remove from open trades
        self._open_trades.pop(trade_id, None)

        # Update strategy metrics
        self._update_strategy_metrics(trade)

        # Update portfolio totals
        self._total_pnl += trade.net_pnl
        self._daily_pnl.append((now, trade.net_pnl))

        logger.debug(f"Closed trade: {trade_id} - P&L: {trade.net_pnl:.2f}")

        return trade

    def _update_strategy_metrics(self, trade: TradeRecord) -> None:
        """Update strategy metrics with closed trade."""
        strategy = trade.strategy

        if strategy not in self._strategy_metrics:
            self._strategy_metrics[strategy] = StrategyMetrics(strategy=strategy)

        metrics = self._strategy_metrics[strategy]

        # Update counters
        metrics.total_trades += 1

        if trade.outcome == TradeOutcome.WIN:
            metrics.winning_trades += 1
        elif trade.outcome == TradeOutcome.LOSS:
            metrics.losing_trades += 1
        else:
            metrics.breakeven_trades += 1

        # Update P&L totals
        metrics.total_pnl += trade.net_pnl
        metrics.gross_pnl += trade.gross_pnl
        metrics.total_commission += trade.commission
        metrics.total_slippage += abs(trade.slippage)

        # Update extremes
        if trade.net_pnl > metrics.largest_win:
            metrics.largest_win = trade.net_pnl
        if trade.net_pnl < metrics.largest_loss:
            metrics.largest_loss = trade.net_pnl

        # Update averages
        wins = [t.net_pnl for t in self._trades.values()
                if t.strategy == strategy and t.is_closed and t.outcome == TradeOutcome.WIN]
        losses = [t.net_pnl for t in self._trades.values()
                  if t.strategy == strategy and t.is_closed and t.outcome == TradeOutcome.LOSS]

        metrics.avg_win = np.mean(wins) if wins else 0.0
        metrics.avg_loss = np.mean(losses) if losses else 0.0

        # Update holding time
        holding_times = [t.holding_period_hours for t in self._trades.values()
                        if t.strategy == strategy and t.is_closed and t.holding_period_hours is not None]
        metrics.avg_holding_hours = np.mean(holding_times) if holding_times else 0.0

        # Store returns for ratio calculations
        metrics.returns.append(trade.net_pnl)
        metrics.pnl_history.append((trade.exit_timestamp or datetime.now(timezone.utc), trade.net_pnl))

    def get_strategy_metrics(self, strategy: str) -> StrategyMetrics | None:
        """Get metrics for a specific strategy."""
        return self._strategy_metrics.get(strategy)

    def get_all_strategy_metrics(self) -> dict[str, StrategyMetrics]:
        """Get metrics for all strategies."""
        return dict(self._strategy_metrics)

    def get_strategy_pnl(self, strategy: str) -> float:
        """Get total P&L for a strategy."""
        metrics = self._strategy_metrics.get(strategy)
        return metrics.total_pnl if metrics else 0.0

    def get_strategy_sharpe(self, strategy: str) -> float:
        """Get Sharpe ratio for a strategy."""
        metrics = self._strategy_metrics.get(strategy)
        return metrics.sharpe_ratio if metrics else 0.0

    def get_strategy_win_rate(self, strategy: str) -> float:
        """Get win rate for a strategy."""
        metrics = self._strategy_metrics.get(strategy)
        return metrics.win_rate if metrics else 0.0

    def get_rolling_metrics(
        self,
        strategy: str,
        days: int | None = None
    ) -> StrategyMetrics | None:
        """
        Calculate metrics for a rolling window.

        Args:
            strategy: Strategy name
            days: Window size (default: configured rolling_window_days)

        Returns:
            StrategyMetrics for the window, or None if insufficient data
        """
        if days is None:
            days = self._rolling_window_days

        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)

        # Get trades in window
        trades_in_window = [
            t for t in self._trades.values()
            if t.strategy == strategy
            and t.is_closed
            and t.exit_timestamp is not None
            and t.exit_timestamp.timestamp() > cutoff
        ]

        if not trades_in_window:
            return None

        # Calculate metrics for window
        metrics = StrategyMetrics(strategy=f"{strategy}_rolling_{days}d")

        for trade in trades_in_window:
            metrics.total_trades += 1
            if trade.outcome == TradeOutcome.WIN:
                metrics.winning_trades += 1
            elif trade.outcome == TradeOutcome.LOSS:
                metrics.losing_trades += 1

            metrics.total_pnl += trade.net_pnl
            metrics.returns.append(trade.net_pnl)
            metrics.pnl_history.append((trade.exit_timestamp, trade.net_pnl))

        return metrics

    def get_pnl_attribution(self) -> dict[str, float]:
        """
        Get P&L attribution by strategy.

        Returns:
            Dictionary mapping strategy to total P&L
        """
        return {
            strategy: metrics.total_pnl
            for strategy, metrics in self._strategy_metrics.items()
        }

    def get_pnl_contribution(self) -> dict[str, float]:
        """
        Get P&L contribution percentages by strategy.

        Returns:
            Dictionary mapping strategy to contribution percentage
        """
        if self._total_pnl == 0:
            return {}

        return {
            strategy: (metrics.total_pnl / self._total_pnl) * 100
            for strategy, metrics in self._strategy_metrics.items()
        }

    def get_recommended_weights(
        self,
        method: str = "sharpe"
    ) -> dict[str, float]:
        """
        Calculate recommended strategy weights based on performance.

        Args:
            method: Weighting method - "sharpe", "win_rate", "profit_factor", "equal"

        Returns:
            Normalized weights by strategy
        """
        if not self._strategy_metrics:
            return {}

        weights = {}

        for strategy, metrics in self._strategy_metrics.items():
            if method == "sharpe":
                # Use Sharpe ratio (minimum 0)
                weights[strategy] = max(0, metrics.sharpe_ratio)
            elif method == "win_rate":
                # Use win rate
                weights[strategy] = metrics.win_rate
            elif method == "profit_factor":
                # Use profit factor (cap at 5 for stability)
                pf = metrics.profit_factor
                weights[strategy] = min(pf, 5.0) if pf != float("inf") else 5.0
            else:  # equal
                weights[strategy] = 1.0

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Equal weights fallback
            n = len(weights)
            weights = {k: 1.0 / n for k in weights}

        return weights

    def get_symbol_attribution(self) -> dict[str, float]:
        """Get P&L attribution by symbol."""
        attribution: dict[str, float] = defaultdict(float)

        for trade in self._trades.values():
            if trade.is_closed:
                attribution[trade.symbol] += trade.net_pnl

        return dict(attribution)

    def get_open_trades(self, strategy: str | None = None) -> list[TradeRecord]:
        """
        Get open trades.

        Args:
            strategy: Filter by strategy (optional)

        Returns:
            List of open trades
        """
        trades = list(self._open_trades.values())

        if strategy:
            trades = [t for t in trades if t.strategy == strategy]

        return trades

    def get_trade_history(
        self,
        strategy: str | None = None,
        symbol: str | None = None,
        limit: int = 100
    ) -> list[TradeRecord]:
        """
        Get trade history.

        Args:
            strategy: Filter by strategy (optional)
            symbol: Filter by symbol (optional)
            limit: Maximum trades to return

        Returns:
            List of trade records
        """
        trades = list(self._trades.values())

        if strategy:
            trades = [t for t in trades if t.strategy == strategy]
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        # Sort by timestamp descending
        trades.sort(key=lambda t: t.timestamp, reverse=True)

        return trades[:limit]

    def get_daily_pnl_series(self, days: int = 30) -> list[tuple[datetime, float]]:
        """Get daily P&L time series."""
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        return [
            (ts, pnl) for ts, pnl in self._daily_pnl
            if ts.timestamp() > cutoff
        ]

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio-level summary."""
        total_trades = sum(m.total_trades for m in self._strategy_metrics.values())
        total_wins = sum(m.winning_trades for m in self._strategy_metrics.values())
        total_losses = sum(m.losing_trades for m in self._strategy_metrics.values())

        return {
            "total_pnl": self._total_pnl,
            "total_trades": total_trades,
            "winning_trades": total_wins,
            "losing_trades": total_losses,
            "win_rate": total_wins / max(total_wins + total_losses, 1),
            "open_trades": len(self._open_trades),
            "strategies_tracked": len(self._strategy_metrics),
            "top_strategy": max(
                self._strategy_metrics.items(),
                key=lambda x: x[1].total_pnl,
                default=(None, None)
            )[0],
        }

    def get_status(self) -> dict[str, Any]:
        """Get attribution system status for monitoring."""
        return {
            "total_trades_recorded": len(self._trades),
            "open_trades": len(self._open_trades),
            "strategies_tracked": len(self._strategy_metrics),
            "total_portfolio_pnl": self._total_pnl,
            "strategy_summary": {
                strategy: {
                    "pnl": metrics.total_pnl,
                    "trades": metrics.total_trades,
                    "win_rate": metrics.win_rate,
                    "sharpe": metrics.sharpe_ratio,
                }
                for strategy, metrics in self._strategy_metrics.items()
            },
        }

    def export_to_dataframe(self):
        """Export trades to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for export_to_dataframe()")
            return None

        data = [trade.to_dict() for trade in self._trades.values()]
        return pd.DataFrame(data)

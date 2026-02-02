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
from datetime import datetime, timezone, timedelta
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

        # Filter out NaN and inf values before calculation
        valid_mask = np.isfinite(returns_array)
        returns_array = returns_array[valid_mask]

        if len(returns_array) < 2:
            return 0.0

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

        # P1-15: TWR/MWR tracking
        self._portfolio_values: list[tuple[datetime, float]] = []  # (timestamp, NAV)
        self._cash_flows: list[tuple[datetime, float]] = []  # (timestamp, amount) + for deposits

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

    # =========================================================================
    # P1-15: TWR/MWR CALCULATIONS
    # =========================================================================

    def record_portfolio_value(self, timestamp: datetime, value: float) -> None:
        """
        P1-15: Record portfolio NAV for TWR calculation.

        Should be called daily (or at each valuation point).
        """
        self._portfolio_values.append((timestamp, value))

    def record_cash_flow(self, timestamp: datetime, amount: float) -> None:
        """
        P1-15: Record external cash flow for MWR calculation.

        Args:
            timestamp: When the cash flow occurred
            amount: Positive for deposits, negative for withdrawals
        """
        self._cash_flows.append((timestamp, amount))

    def calculate_twr(self, start_date: datetime | None = None) -> float:
        """
        P1-15: Calculate Time-Weighted Return (TWR).

        TWR eliminates the effect of cash flows, showing pure investment
        performance. Used for comparing manager skill regardless of
        deposit/withdrawal timing.

        Formula: TWR = ((1 + r1) * (1 + r2) * ... * (1 + rn)) - 1
        where rn is the sub-period return between cash flows.

        Returns:
            Annualized TWR as decimal (e.g., 0.15 = 15%)
        """
        if len(self._portfolio_values) < 2:
            return 0.0

        # Sort values by timestamp
        sorted_values = sorted(self._portfolio_values, key=lambda x: x[0])

        if start_date:
            sorted_values = [(t, v) for t, v in sorted_values if t >= start_date]
            if len(sorted_values) < 2:
                return 0.0

        # Calculate sub-period returns
        # For each period between cash flows, calculate the return
        cash_flow_dict = {t.date(): amt for t, amt in self._cash_flows}

        cumulative_return = 1.0
        prev_value = sorted_values[0][1]
        periods = 0

        for i in range(1, len(sorted_values)):
            curr_date, curr_value = sorted_values[i]
            prev_date = sorted_values[i-1][0]

            # Check for cash flow at start of this period
            cash_flow = cash_flow_dict.get(curr_date.date(), 0)

            # Adjust previous value for cash flow (add at start of period)
            adjusted_prev = prev_value + cash_flow

            if adjusted_prev > 0:
                period_return = curr_value / adjusted_prev
                cumulative_return *= period_return
                periods += 1

            prev_value = curr_value

        if periods == 0:
            return 0.0

        # Calculate total return
        total_return = cumulative_return - 1

        # Annualize if we have date range
        first_date = sorted_values[0][0]
        last_date = sorted_values[-1][0]
        days = (last_date - first_date).days

        if days > 0:
            years = days / 365.25
            if years > 0 and cumulative_return > 0:
                annualized = (cumulative_return ** (1 / years)) - 1
                return annualized

        return total_return

    def calculate_mwr(self, start_date: datetime | None = None) -> float:
        """
        P1-15: Calculate Money-Weighted Return (MWR / IRR).

        MWR reflects actual investor experience including the timing
        of deposits and withdrawals. Higher weight given to returns
        when more capital was invested.

        Uses Newton-Raphson iteration to solve for IRR.

        Returns:
            Annualized MWR as decimal (e.g., 0.12 = 12%)
        """
        if len(self._portfolio_values) < 2:
            return 0.0

        sorted_values = sorted(self._portfolio_values, key=lambda x: x[0])
        sorted_flows = sorted(self._cash_flows, key=lambda x: x[0])

        if start_date:
            sorted_values = [(t, v) for t, v in sorted_values if t >= start_date]
            sorted_flows = [(t, a) for t, a in sorted_flows if t >= start_date]
            if len(sorted_values) < 2:
                return 0.0

        first_date = sorted_values[0][0]
        last_date = sorted_values[-1][0]
        total_days = (last_date - first_date).days

        if total_days <= 0:
            return 0.0

        # Build cash flow series: initial investment + flows + final value
        # CF0 = -initial_value (investment out)
        # CFi = -cash_flows (deposits negative, withdrawals positive)
        # CFn = +final_value (proceeds)

        initial_value = sorted_values[0][1]
        final_value = sorted_values[-1][1]

        cash_flows_with_timing = []
        # Initial outflow
        cash_flows_with_timing.append((0, -initial_value))

        # External cash flows
        for cf_date, cf_amount in sorted_flows:
            days_from_start = (cf_date - first_date).days
            # Deposits are cash OUT (negative), withdrawals are cash IN (positive)
            cash_flows_with_timing.append((days_from_start, -cf_amount))

        # Final value as inflow
        cash_flows_with_timing.append((total_days, final_value))

        # Solve for IRR using Newton-Raphson
        def npv(rate: float) -> float:
            """Calculate NPV at given daily rate."""
            total = 0.0
            for days, cf in cash_flows_with_timing:
                if rate > -1:
                    total += cf / ((1 + rate) ** days)
            return total

        def npv_derivative(rate: float) -> float:
            """Derivative of NPV for Newton-Raphson."""
            total = 0.0
            for days, cf in cash_flows_with_timing:
                if rate > -1 and days > 0:
                    total -= days * cf / ((1 + rate) ** (days + 1))
            return total

        # Initial guess - simple return
        simple_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0
        daily_rate = simple_return / total_days if total_days > 0 else 0

        # Newton-Raphson iteration
        max_iterations = 100
        tolerance = 1e-10

        for _ in range(max_iterations):
            f_val = npv(daily_rate)
            f_deriv = npv_derivative(daily_rate)

            if abs(f_deriv) < 1e-15:
                break

            new_rate = daily_rate - f_val / f_deriv

            if abs(new_rate - daily_rate) < tolerance:
                daily_rate = new_rate
                break

            daily_rate = new_rate

        # Annualize the daily rate
        annualized_mwr = ((1 + daily_rate) ** 365.25) - 1

        return annualized_mwr

    def get_return_comparison(self) -> dict:
        """
        P1-15: Get TWR vs MWR comparison.

        A large difference between TWR and MWR indicates poor timing
        of deposits/withdrawals relative to market performance.

        Returns:
            Dictionary with TWR, MWR, and difference analysis
        """
        twr = self.calculate_twr()
        mwr = self.calculate_mwr()
        diff = mwr - twr

        return {
            "twr": twr,
            "twr_pct": twr * 100,
            "mwr": mwr,
            "mwr_pct": mwr * 100,
            "difference": diff,
            "difference_pct": diff * 100,
            "timing_impact": "positive" if diff > 0.01 else ("negative" if diff < -0.01 else "neutral"),
            "interpretation": (
                "Good cash flow timing - deposits before rallies, withdrawals before declines"
                if diff > 0.01 else (
                    "Poor cash flow timing - deposits before declines, withdrawals before rallies"
                    if diff < -0.01 else "Cash flow timing had minimal impact"
                )
            ),
        }

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


# =============================================================================
# SECTOR/FACTOR EXPOSURE CONSTRAINTS (#P6)
# =============================================================================

@dataclass
class ExposureLimit:
    """Limit definition for a sector or factor (#P6)."""
    name: str
    limit_type: str  # "absolute", "percentage"
    max_long: float
    max_short: float
    max_gross: float
    max_net: float | None = None


class SectorFactorExposureManager:
    """
    Manages sector and factor exposure constraints (#P6).

    Tracks and enforces limits on:
    - Sector exposures (Technology, Healthcare, etc.)
    - Factor exposures (Value, Momentum, Size, etc.)
    - Geographic exposures
    """

    # Default sector classification (simplified)
    DEFAULT_SECTOR_MAP: dict[str, str] = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
        "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
        "ES": "Index", "NQ": "Index", "YM": "Index",
        "CL": "Commodities", "GC": "Commodities", "NG": "Commodities",
    }

    def __init__(self, portfolio_value: float = 1_000_000):
        self._portfolio_value = portfolio_value
        self._sector_map = dict(self.DEFAULT_SECTOR_MAP)
        self._limits: dict[str, ExposureLimit] = {}
        self._current_exposures: dict[str, dict[str, float]] = {}  # type -> {name: exposure}

        # Default sector limits (as % of portfolio)
        self._initialize_default_limits()

    def _initialize_default_limits(self):
        """Set up default exposure limits."""
        default_sectors = ["Technology", "Financials", "Healthcare", "Energy",
                         "Consumer Discretionary", "Industrials", "Materials",
                         "Utilities", "Real Estate", "Communications", "Staples"]

        for sector in default_sectors:
            self._limits[f"sector:{sector}"] = ExposureLimit(
                name=sector,
                limit_type="percentage",
                max_long=25.0,  # 25% max long
                max_short=15.0,  # 15% max short
                max_gross=30.0,  # 30% max gross
                max_net=20.0,   # 20% max net
            )

        # Factor limits
        for factor in ["Value", "Momentum", "Size", "Quality", "Volatility"]:
            self._limits[f"factor:{factor}"] = ExposureLimit(
                name=factor,
                limit_type="absolute",
                max_long=1.5,   # 1.5 beta max long
                max_short=0.5,  # 0.5 beta max short
                max_gross=2.0,  # 2.0 beta gross
            )

    def set_sector(self, symbol: str, sector: str) -> None:
        """Set sector classification for a symbol."""
        self._sector_map[symbol] = sector

    def set_limit(self, limit: ExposureLimit, limit_type: str = "sector") -> None:
        """Set an exposure limit."""
        key = f"{limit_type}:{limit.name}"
        self._limits[key] = limit

    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value for percentage calculations."""
        self._portfolio_value = value

    def calculate_sector_exposures(
        self,
        positions: dict[str, float]  # symbol -> notional value
    ) -> dict[str, dict]:
        """
        Calculate current sector exposures (#P6).

        Args:
            positions: Map of symbols to notional positions

        Returns:
            Exposure by sector
        """
        exposures: dict[str, dict] = {}

        for symbol, notional in positions.items():
            sector = self._sector_map.get(symbol, "Other")

            if sector not in exposures:
                exposures[sector] = {
                    "long": 0.0,
                    "short": 0.0,
                    "gross": 0.0,
                    "net": 0.0,
                    "symbols": [],
                }

            if notional > 0:
                exposures[sector]["long"] += notional
            else:
                exposures[sector]["short"] += abs(notional)

            exposures[sector]["gross"] += abs(notional)
            exposures[sector]["net"] += notional
            exposures[sector]["symbols"].append(symbol)

        # Convert to percentages
        for sector, exp in exposures.items():
            if self._portfolio_value > 0:
                exp["long_pct"] = exp["long"] / self._portfolio_value * 100
                exp["short_pct"] = exp["short"] / self._portfolio_value * 100
                exp["gross_pct"] = exp["gross"] / self._portfolio_value * 100
                exp["net_pct"] = exp["net"] / self._portfolio_value * 100

        self._current_exposures["sector"] = exposures
        return exposures

    def check_exposure_limits(
        self,
        positions: dict[str, float]
    ) -> list[dict]:
        """
        Check all exposure limits and return violations (#P6).

        Args:
            positions: Map of symbols to notional positions

        Returns:
            List of limit violations
        """
        violations = []
        sector_exposures = self.calculate_sector_exposures(positions)

        for sector, exp in sector_exposures.items():
            limit_key = f"sector:{sector}"
            limit = self._limits.get(limit_key)

            if limit is None:
                continue

            # Check each limit type
            if exp.get("long_pct", 0) > limit.max_long:
                violations.append({
                    "type": "sector",
                    "name": sector,
                    "limit": "max_long",
                    "current": exp["long_pct"],
                    "limit_value": limit.max_long,
                    "breach": exp["long_pct"] - limit.max_long,
                })

            if exp.get("short_pct", 0) > limit.max_short:
                violations.append({
                    "type": "sector",
                    "name": sector,
                    "limit": "max_short",
                    "current": exp["short_pct"],
                    "limit_value": limit.max_short,
                    "breach": exp["short_pct"] - limit.max_short,
                })

            if exp.get("gross_pct", 0) > limit.max_gross:
                violations.append({
                    "type": "sector",
                    "name": sector,
                    "limit": "max_gross",
                    "current": exp["gross_pct"],
                    "limit_value": limit.max_gross,
                    "breach": exp["gross_pct"] - limit.max_gross,
                })

        return violations

    def get_exposure_summary(self) -> dict:
        """Get summary of all exposures for monitoring."""
        return {
            "portfolio_value": self._portfolio_value,
            "sector_exposures": self._current_exposures.get("sector", {}),
            "limits_defined": len(self._limits),
        }


# =============================================================================
# CASH MANAGEMENT (#P7)
# =============================================================================

class CashManager:
    """
    Manages portfolio cash and liquidity (#P7).

    Handles:
    - Cash balance tracking
    - Minimum cash reserves
    - Cash sweep logic
    - T+2 settlement tracking
    """

    def __init__(
        self,
        initial_cash: float = 0.0,
        min_cash_reserve_pct: float = 5.0,
        target_cash_pct: float = 10.0
    ):
        self._cash_balance = initial_cash
        self._min_reserve_pct = min_cash_reserve_pct
        self._target_cash_pct = target_cash_pct
        self._pending_settlements: list[dict] = []
        self._cash_history: list[tuple[datetime, float, str]] = []

    def update_cash(self, amount: float, reason: str) -> float:
        """
        Update cash balance (#P7).

        Args:
            amount: Cash change (positive = inflow, negative = outflow)
            reason: Reason for cash change

        Returns:
            New cash balance
        """
        self._cash_balance += amount
        self._cash_history.append((datetime.now(timezone.utc), amount, reason))
        return self._cash_balance

    def get_available_cash(self, portfolio_value: float) -> float:
        """
        Get cash available for trading (#P7).

        Subtracts minimum reserve and pending settlements.

        Args:
            portfolio_value: Total portfolio value

        Returns:
            Available cash for new positions
        """
        min_reserve = portfolio_value * (self._min_reserve_pct / 100)
        pending_outflows = sum(
            s["amount"] for s in self._pending_settlements
            if s["amount"] < 0
        )
        return max(0, self._cash_balance - min_reserve + pending_outflows)

    def add_pending_settlement(
        self,
        amount: float,
        settlement_date: datetime,
        trade_id: str
    ) -> None:
        """
        Add pending settlement (#P7).

        Args:
            amount: Settlement amount (positive = receive, negative = pay)
            settlement_date: Expected settlement date
            trade_id: Associated trade ID
        """
        self._pending_settlements.append({
            "amount": amount,
            "settlement_date": settlement_date,
            "trade_id": trade_id,
            "added": datetime.now(timezone.utc),
        })

    def process_settlements(self) -> list[dict]:
        """
        Process due settlements (#P7).

        Returns:
            List of processed settlements
        """
        now = datetime.now(timezone.utc)
        processed = []
        remaining = []

        for settlement in self._pending_settlements:
            if settlement["settlement_date"] <= now:
                self.update_cash(settlement["amount"], f"settlement:{settlement['trade_id']}")
                processed.append(settlement)
            else:
                remaining.append(settlement)

        self._pending_settlements = remaining
        return processed

    def calculate_cash_sweep(self, portfolio_value: float) -> dict:
        """
        Calculate cash sweep to maintain target allocation (#P7).

        Args:
            portfolio_value: Total portfolio value

        Returns:
            Sweep recommendation
        """
        target_cash = portfolio_value * (self._target_cash_pct / 100)
        current_cash = self._cash_balance
        excess = current_cash - target_cash

        if excess > 0:
            return {
                "action": "invest",
                "amount": excess,
                "reason": "excess_cash",
                "current_cash_pct": (current_cash / portfolio_value * 100) if portfolio_value > 0 else 0,
                "target_cash_pct": self._target_cash_pct,
            }
        elif excess < -portfolio_value * 0.02:  # More than 2% below target
            return {
                "action": "raise_cash",
                "amount": abs(excess),
                "reason": "below_target",
                "current_cash_pct": (current_cash / portfolio_value * 100) if portfolio_value > 0 else 0,
                "target_cash_pct": self._target_cash_pct,
            }
        else:
            return {
                "action": "none",
                "current_cash_pct": (current_cash / portfolio_value * 100) if portfolio_value > 0 else 0,
                "target_cash_pct": self._target_cash_pct,
            }

    def get_cash_status(self) -> dict:
        """Get cash management status."""
        return {
            "cash_balance": self._cash_balance,
            "pending_settlements": len(self._pending_settlements),
            "pending_inflows": sum(s["amount"] for s in self._pending_settlements if s["amount"] > 0),
            "pending_outflows": sum(s["amount"] for s in self._pending_settlements if s["amount"] < 0),
            "min_reserve_pct": self._min_reserve_pct,
            "target_cash_pct": self._target_cash_pct,
        }


# =============================================================================
# DIVIDEND HANDLING (#P8)
# =============================================================================

@dataclass
class DividendRecord:
    """Record of a dividend event (#P8)."""
    symbol: str
    ex_date: datetime
    record_date: datetime
    pay_date: datetime
    amount_per_share: float
    dividend_type: str  # "regular", "special", "return_of_capital"
    shares_held: int
    total_amount: float = field(init=False)
    currency: str = "USD"
    withheld_tax: float = 0.0

    def __post_init__(self):
        self.total_amount = self.amount_per_share * self.shares_held


class DividendManager:
    """
    Manages dividend tracking and processing (#P8).

    Handles:
    - Ex-date tracking
    - Dividend accrual
    - Tax withholding
    - DRIP (dividend reinvestment)
    """

    def __init__(self, enable_drip: bool = False, default_tax_rate: float = 0.0):
        self._enable_drip = enable_drip
        self._default_tax_rate = default_tax_rate
        self._upcoming_dividends: list[DividendRecord] = []
        self._received_dividends: list[DividendRecord] = []
        self._accrued_dividends: float = 0.0

    def add_upcoming_dividend(
        self,
        symbol: str,
        ex_date: datetime,
        record_date: datetime,
        pay_date: datetime,
        amount_per_share: float,
        shares_held: int,
        dividend_type: str = "regular"
    ) -> DividendRecord:
        """
        Register an upcoming dividend (#P8).

        Args:
            symbol: Stock symbol
            ex_date: Ex-dividend date
            record_date: Record date
            pay_date: Payment date
            amount_per_share: Dividend per share
            shares_held: Number of shares held
            dividend_type: Type of dividend

        Returns:
            Dividend record
        """
        record = DividendRecord(
            symbol=symbol,
            ex_date=ex_date,
            record_date=record_date,
            pay_date=pay_date,
            amount_per_share=amount_per_share,
            dividend_type=dividend_type,
            shares_held=shares_held,
            withheld_tax=amount_per_share * shares_held * self._default_tax_rate,
        )

        self._upcoming_dividends.append(record)
        return record

    def process_ex_dates(
        self,
        current_positions: dict[str, int],
        as_of: datetime | None = None
    ) -> list[DividendRecord]:
        """
        Process dividends going ex (#P8).

        Args:
            current_positions: Current share positions
            as_of: Processing date (default: now)

        Returns:
            Dividends going ex
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        going_ex = []
        remaining = []

        for div in self._upcoming_dividends:
            if div.ex_date <= as_of:
                # Update shares based on current position
                current_shares = current_positions.get(div.symbol, 0)
                if current_shares > 0:
                    div.shares_held = current_shares
                    div.total_amount = div.amount_per_share * current_shares
                    div.withheld_tax = div.total_amount * self._default_tax_rate
                    self._accrued_dividends += div.total_amount - div.withheld_tax
                    going_ex.append(div)
            else:
                remaining.append(div)

        self._upcoming_dividends = remaining
        return going_ex

    def process_payments(self, as_of: datetime | None = None) -> list[dict]:
        """
        Process dividend payments (#P8).

        Args:
            as_of: Processing date (default: now)

        Returns:
            Processed payments
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        payments = []
        remaining = []

        for div in self._received_dividends:
            if div.pay_date <= as_of:
                net_amount = div.total_amount - div.withheld_tax
                payments.append({
                    "symbol": div.symbol,
                    "gross_amount": div.total_amount,
                    "tax_withheld": div.withheld_tax,
                    "net_amount": net_amount,
                    "shares": div.shares_held,
                    "type": div.dividend_type,
                    "drip": self._enable_drip,
                })
                self._accrued_dividends -= net_amount
            else:
                remaining.append(div)

        self._received_dividends = remaining
        return payments

    def get_dividend_forecast(self, days: int = 90) -> dict:
        """Get forecast of upcoming dividends."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=days)
        upcoming = [d for d in self._upcoming_dividends if d.pay_date <= cutoff]

        return {
            "count": len(upcoming),
            "total_gross": sum(d.total_amount for d in upcoming),
            "total_tax": sum(d.withheld_tax for d in upcoming),
            "total_net": sum(d.total_amount - d.withheld_tax for d in upcoming),
            "by_symbol": {
                d.symbol: d.total_amount - d.withheld_tax for d in upcoming
            },
        }

    def get_ytd_dividends(self) -> dict:
        """Get year-to-date dividend summary."""
        current_year = datetime.now(timezone.utc).year
        ytd = [d for d in self._received_dividends if d.pay_date.year == current_year]

        return {
            "total_gross": sum(d.total_amount for d in ytd),
            "total_tax": sum(d.withheld_tax for d in ytd),
            "total_net": sum(d.total_amount - d.withheld_tax for d in ytd),
            "dividend_count": len(ytd),
        }


# =============================================================================
# CORPORATE ACTION PROCESSING (#P9)
# =============================================================================

class CorporateActionType(Enum):
    """Types of corporate actions (#P9)."""
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    SPIN_OFF = "spin_off"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    NAME_CHANGE = "name_change"
    SYMBOL_CHANGE = "symbol_change"
    RIGHTS_ISSUE = "rights_issue"
    TENDER_OFFER = "tender_offer"
    DELISTING = "delisting"


@dataclass
class CorporateAction:
    """Corporate action record (#P9)."""
    action_type: CorporateActionType
    symbol: str
    effective_date: datetime
    details: dict  # Action-specific details
    processed: bool = False
    processed_date: datetime | None = None


class CorporateActionProcessor:
    """
    Processes corporate actions (#P9).

    Handles:
    - Stock splits and reverse splits
    - Mergers and acquisitions
    - Spin-offs
    - Symbol changes
    """

    def __init__(self):
        self._pending_actions: list[CorporateAction] = []
        self._processed_actions: list[CorporateAction] = []

    def add_corporate_action(self, action: CorporateAction) -> None:
        """Add a corporate action to process."""
        self._pending_actions.append(action)

    def process_split(
        self,
        action: CorporateAction,
        current_shares: int,
        cost_basis: float
    ) -> dict:
        """
        Process stock split (#P9).

        Args:
            action: Split action
            current_shares: Current share count
            cost_basis: Current cost basis

        Returns:
            Adjusted position details
        """
        ratio = action.details.get("ratio", 1.0)  # e.g., 4.0 for 4:1 split

        if action.action_type == CorporateActionType.REVERSE_SPLIT:
            # Reverse split: fewer shares, higher price
            new_shares = int(current_shares / ratio)
            new_cost_per_share = cost_basis / current_shares * ratio if current_shares > 0 else 0
        else:
            # Forward split: more shares, lower price
            new_shares = int(current_shares * ratio)
            new_cost_per_share = cost_basis / current_shares / ratio if current_shares > 0 else 0

        return {
            "action_type": action.action_type.value,
            "symbol": action.symbol,
            "ratio": ratio,
            "old_shares": current_shares,
            "new_shares": new_shares,
            "old_cost_basis": cost_basis,
            "new_cost_basis": cost_basis,  # Total basis unchanged
            "new_cost_per_share": new_cost_per_share,
        }

    def process_spinoff(
        self,
        action: CorporateAction,
        parent_shares: int,
        parent_cost_basis: float
    ) -> dict:
        """
        Process spin-off (#P9).

        Args:
            action: Spin-off action
            parent_shares: Parent company shares held
            parent_cost_basis: Parent cost basis

        Returns:
            New position and adjusted basis
        """
        ratio = action.details.get("ratio", 0.1)  # e.g., 0.1 = 1 spinoff share per 10 parent
        new_symbol = action.details.get("new_symbol", "SPINOFF")
        basis_allocation = action.details.get("basis_allocation", 0.1)  # % of basis to spinoff

        spinoff_shares = int(parent_shares * ratio)
        spinoff_basis = parent_cost_basis * basis_allocation
        parent_new_basis = parent_cost_basis * (1 - basis_allocation)

        return {
            "action_type": "spin_off",
            "parent_symbol": action.symbol,
            "spinoff_symbol": new_symbol,
            "parent_shares": parent_shares,
            "spinoff_shares": spinoff_shares,
            "parent_old_basis": parent_cost_basis,
            "parent_new_basis": parent_new_basis,
            "spinoff_basis": spinoff_basis,
        }

    def process_merger(
        self,
        action: CorporateAction,
        target_shares: int,
        target_cost_basis: float
    ) -> dict:
        """
        Process merger/acquisition (#P9).

        Args:
            action: Merger action
            target_shares: Target company shares held
            target_cost_basis: Target cost basis

        Returns:
            Conversion details
        """
        acquirer = action.details.get("acquirer", "ACQUIRER")
        exchange_ratio = action.details.get("exchange_ratio", 1.0)
        cash_per_share = action.details.get("cash_per_share", 0.0)

        new_shares = int(target_shares * exchange_ratio)
        cash_received = target_shares * cash_per_share

        # Simplified basis calculation (actual may involve gain recognition)
        if new_shares > 0:
            new_cost_per_share = (target_cost_basis - cash_received) / new_shares
        else:
            new_cost_per_share = 0

        return {
            "action_type": "merger",
            "target_symbol": action.symbol,
            "acquirer_symbol": acquirer,
            "target_shares": target_shares,
            "new_shares": new_shares,
            "exchange_ratio": exchange_ratio,
            "cash_per_share": cash_per_share,
            "total_cash": cash_received,
            "new_cost_basis": target_cost_basis - cash_received,
            "new_cost_per_share": new_cost_per_share,
        }

    def process_pending_actions(
        self,
        positions: dict[str, tuple[int, float]],  # symbol -> (shares, cost_basis)
        as_of: datetime | None = None
    ) -> list[dict]:
        """
        Process all pending corporate actions (#P9).

        Args:
            positions: Current positions
            as_of: Processing date

        Returns:
            List of processed action results
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        results = []
        remaining = []

        for action in self._pending_actions:
            if action.effective_date <= as_of:
                position = positions.get(action.symbol)
                if position is None:
                    remaining.append(action)
                    continue

                shares, cost_basis = position

                if action.action_type in (CorporateActionType.STOCK_SPLIT, CorporateActionType.REVERSE_SPLIT):
                    result = self.process_split(action, shares, cost_basis)
                elif action.action_type == CorporateActionType.SPIN_OFF:
                    result = self.process_spinoff(action, shares, cost_basis)
                elif action.action_type in (CorporateActionType.MERGER, CorporateActionType.ACQUISITION):
                    result = self.process_merger(action, shares, cost_basis)
                else:
                    result = {"action_type": action.action_type.value, "symbol": action.symbol, "status": "manual_review"}

                action.processed = True
                action.processed_date = as_of
                self._processed_actions.append(action)
                results.append(result)
            else:
                remaining.append(action)

        self._pending_actions = remaining
        return results

    def get_pending_actions(self) -> list[dict]:
        """Get list of pending corporate actions."""
        return [
            {
                "type": a.action_type.value,
                "symbol": a.symbol,
                "effective_date": a.effective_date.isoformat(),
                "details": a.details,
            }
            for a in self._pending_actions
        ]


# =============================================================================
# TAX LOT MANAGEMENT (#P10)
# =============================================================================

@dataclass
class TaxLot:
    """Individual tax lot for a position (#P10)."""
    lot_id: str
    symbol: str
    purchase_date: datetime
    quantity: int
    cost_per_share: float
    total_cost: float = field(init=False)
    remaining_quantity: int = field(init=False)
    wash_sale_disallowed: float = 0.0
    holding_period_days: int = field(init=False)

    def __post_init__(self):
        self.total_cost = self.quantity * self.cost_per_share
        self.remaining_quantity = self.quantity
        self.holding_period_days = (datetime.now(timezone.utc) - self.purchase_date).days

    @property
    def is_long_term(self) -> bool:
        """Check if lot qualifies for long-term capital gains (>1 year)."""
        return self.holding_period_days > 365

    @property
    def adjusted_cost_basis(self) -> float:
        """Get cost basis adjusted for wash sale disallowance."""
        return self.total_cost + self.wash_sale_disallowed


class TaxLotManager:
    """
    Manages tax lots for cost basis tracking (#P10).

    Supports:
    - FIFO (First In First Out)
    - LIFO (Last In First Out)
    - Specific identification
    - Average cost
    - Highest cost
    - Lowest cost
    """

    def __init__(self, default_method: str = "fifo"):
        self._default_method = default_method
        self._lots: dict[str, list[TaxLot]] = {}  # symbol -> lots
        self._lot_counter = 0

    def add_lot(
        self,
        symbol: str,
        purchase_date: datetime,
        quantity: int,
        cost_per_share: float
    ) -> TaxLot:
        """
        Add a new tax lot (#P10).

        Args:
            symbol: Stock symbol
            purchase_date: Purchase date
            quantity: Number of shares
            cost_per_share: Cost per share

        Returns:
            Created tax lot
        """
        self._lot_counter += 1
        lot = TaxLot(
            lot_id=f"LOT-{self._lot_counter:06d}",
            symbol=symbol,
            purchase_date=purchase_date,
            quantity=quantity,
            cost_per_share=cost_per_share,
        )

        if symbol not in self._lots:
            self._lots[symbol] = []
        self._lots[symbol].append(lot)

        return lot

    def select_lots_for_sale(
        self,
        symbol: str,
        quantity: int,
        method: str | None = None
    ) -> list[tuple[TaxLot, int]]:
        """
        Select lots for a sale using specified method (#P10).

        Args:
            symbol: Stock symbol
            quantity: Shares to sell
            method: Selection method (fifo, lifo, hifo, lofo, specific)

        Returns:
            List of (lot, shares_to_sell) tuples
        """
        if method is None:
            method = self._default_method

        lots = self._lots.get(symbol, [])
        available_lots = [lot for lot in lots if lot.remaining_quantity > 0]

        if not available_lots:
            return []

        # Sort based on method
        if method == "fifo":
            available_lots.sort(key=lambda l: l.purchase_date)
        elif method == "lifo":
            available_lots.sort(key=lambda l: l.purchase_date, reverse=True)
        elif method == "hifo":  # Highest cost first
            available_lots.sort(key=lambda l: l.cost_per_share, reverse=True)
        elif method == "lofo":  # Lowest cost first
            available_lots.sort(key=lambda l: l.cost_per_share)
        elif method == "ltfo":  # Long-term first
            available_lots.sort(key=lambda l: (not l.is_long_term, l.purchase_date))

        selected = []
        remaining_to_sell = quantity

        for lot in available_lots:
            if remaining_to_sell <= 0:
                break

            shares_from_lot = min(lot.remaining_quantity, remaining_to_sell)
            selected.append((lot, shares_from_lot))
            remaining_to_sell -= shares_from_lot

        # Warn if insufficient lots to cover requested quantity
        if remaining_to_sell > 0:
            logger.warning(
                f"Insufficient lots for {symbol}: requested {quantity}, "
                f"can only sell {quantity - remaining_to_sell}, "
                f"remaining {remaining_to_sell} shares not covered"
            )

        return selected

    def execute_sale(
        self,
        symbol: str,
        quantity: int,
        sale_price: float,
        sale_date: datetime,
        method: str | None = None
    ) -> dict:
        """
        Execute a sale and calculate gain/loss (#P10).

        Args:
            symbol: Stock symbol
            quantity: Shares to sell
            sale_price: Sale price per share
            sale_date: Sale date
            method: Lot selection method

        Returns:
            Sale details with gain/loss
        """
        lots_to_sell = self.select_lots_for_sale(symbol, quantity, method)

        if not lots_to_sell:
            return {"error": "no_lots_available"}

        total_proceeds = quantity * sale_price
        total_cost = 0
        short_term_gain = 0
        long_term_gain = 0
        lots_used = []

        for lot, shares in lots_to_sell:
            cost = shares * lot.cost_per_share
            total_cost += cost
            gain = (shares * sale_price) - cost

            if lot.is_long_term:
                long_term_gain += gain
            else:
                short_term_gain += gain

            lot.remaining_quantity -= shares
            lots_used.append({
                "lot_id": lot.lot_id,
                "shares_sold": shares,
                "cost_per_share": lot.cost_per_share,
                "holding_days": lot.holding_period_days,
                "is_long_term": lot.is_long_term,
                "gain_loss": gain,
            })

        return {
            "symbol": symbol,
            "shares_sold": quantity,
            "sale_price": sale_price,
            "proceeds": total_proceeds,
            "cost_basis": total_cost,
            "total_gain_loss": total_proceeds - total_cost,
            "short_term_gain": short_term_gain,
            "long_term_gain": long_term_gain,
            "lots_used": lots_used,
            "method": method or self._default_method,
        }

    def get_lots_summary(self, symbol: str) -> dict:
        """Get summary of lots for a symbol."""
        lots = self._lots.get(symbol, [])
        active_lots = [l for l in lots if l.remaining_quantity > 0]

        if not active_lots:
            return {"symbol": symbol, "total_shares": 0, "lots": []}

        return {
            "symbol": symbol,
            "total_shares": sum(l.remaining_quantity for l in active_lots),
            "total_cost_basis": sum(l.remaining_quantity * l.cost_per_share for l in active_lots),
            "avg_cost": sum(l.remaining_quantity * l.cost_per_share for l in active_lots) / sum(l.remaining_quantity for l in active_lots),
            "oldest_lot_date": min(l.purchase_date for l in active_lots).isoformat(),
            "long_term_shares": sum(l.remaining_quantity for l in active_lots if l.is_long_term),
            "short_term_shares": sum(l.remaining_quantity for l in active_lots if not l.is_long_term),
            "lot_count": len(active_lots),
        }


# =============================================================================
# BRINSON PERFORMANCE ATTRIBUTION (#P11)
# =============================================================================

class BrinsonAttributor:
    """
    Brinson performance attribution model (#P11).

    Decomposes portfolio return into:
    - Allocation effect (sector weight decisions)
    - Selection effect (security selection within sectors)
    - Interaction effect (combined effect)
    """

    def __init__(self):
        self._attribution_history: list[dict] = []

    def calculate_attribution(
        self,
        portfolio_weights: dict[str, float],  # sector -> weight
        portfolio_returns: dict[str, float],  # sector -> return
        benchmark_weights: dict[str, float],  # sector -> weight
        benchmark_returns: dict[str, float],  # sector -> return
    ) -> dict:
        """
        Calculate Brinson attribution (#P11).

        Args:
            portfolio_weights: Portfolio sector weights
            portfolio_returns: Portfolio sector returns
            benchmark_weights: Benchmark sector weights
            benchmark_returns: Benchmark sector returns

        Returns:
            Attribution breakdown
        """
        sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

        allocation_effects = {}
        selection_effects = {}
        interaction_effects = {}

        total_portfolio_return = sum(
            portfolio_weights.get(s, 0) * portfolio_returns.get(s, 0)
            for s in sectors
        )
        total_benchmark_return = sum(
            benchmark_weights.get(s, 0) * benchmark_returns.get(s, 0)
            for s in sectors
        )

        for sector in sectors:
            wp = portfolio_weights.get(sector, 0)
            wb = benchmark_weights.get(sector, 0)
            rp = portfolio_returns.get(sector, 0)
            rb = benchmark_returns.get(sector, 0)

            # Allocation effect: (wp - wb) * rb
            allocation_effects[sector] = (wp - wb) * rb

            # Selection effect: wb * (rp - rb)
            selection_effects[sector] = wb * (rp - rb)

            # Interaction effect: (wp - wb) * (rp - rb)
            interaction_effects[sector] = (wp - wb) * (rp - rb)

        # Sum up effects
        total_allocation = sum(allocation_effects.values())
        total_selection = sum(selection_effects.values())
        total_interaction = sum(interaction_effects.values())

        # Active return = allocation + selection + interaction
        active_return = total_portfolio_return - total_benchmark_return

        result = {
            "portfolio_return": total_portfolio_return,
            "benchmark_return": total_benchmark_return,
            "active_return": active_return,
            "allocation_effect": total_allocation,
            "selection_effect": total_selection,
            "interaction_effect": total_interaction,
            "sum_of_effects": total_allocation + total_selection + total_interaction,
            "by_sector": {
                sector: {
                    "portfolio_weight": portfolio_weights.get(sector, 0),
                    "benchmark_weight": benchmark_weights.get(sector, 0),
                    "portfolio_return": portfolio_returns.get(sector, 0),
                    "benchmark_return": benchmark_returns.get(sector, 0),
                    "allocation": allocation_effects.get(sector, 0),
                    "selection": selection_effects.get(sector, 0),
                    "interaction": interaction_effects.get(sector, 0),
                }
                for sector in sectors
            },
        }

        self._attribution_history.append(result)
        return result

    def get_cumulative_attribution(self, periods: int = 30) -> dict:
        """Get cumulative attribution over multiple periods."""
        recent = self._attribution_history[-periods:]
        if not recent:
            return {"error": "no_attribution_data"}

        return {
            "periods": len(recent),
            "cumulative_active_return": sum(a["active_return"] for a in recent),
            "cumulative_allocation": sum(a["allocation_effect"] for a in recent),
            "cumulative_selection": sum(a["selection_effect"] for a in recent),
            "cumulative_interaction": sum(a["interaction_effect"] for a in recent),
        }


# =============================================================================
# BENCHMARK TRACKING (#P12)
# =============================================================================

@dataclass
class BenchmarkData:
    """Benchmark data point (#P12)."""
    timestamp: datetime
    value: float
    return_pct: float | None = None


class BenchmarkTracker:
    """
    Tracks portfolio performance against benchmarks (#P12).

    Supports multiple benchmarks and calculates:
    - Tracking error
    - Information ratio
    - Active return
    - Beta and alpha
    """

    def __init__(self):
        self._benchmarks: dict[str, list[BenchmarkData]] = {}
        self._portfolio_values: list[tuple[datetime, float]] = []
        self._active_benchmark: str | None = None

    def add_benchmark(self, name: str) -> None:
        """Add a benchmark to track."""
        if name not in self._benchmarks:
            self._benchmarks[name] = []

    def set_active_benchmark(self, name: str) -> None:
        """Set the primary benchmark for comparison."""
        self._active_benchmark = name

    def record_benchmark_value(self, benchmark: str, timestamp: datetime, value: float) -> None:
        """Record benchmark value."""
        if benchmark not in self._benchmarks:
            self.add_benchmark(benchmark)

        data = self._benchmarks[benchmark]
        if data:
            prev_value = data[-1].value
            return_pct = (value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            return_pct = None

        data.append(BenchmarkData(timestamp=timestamp, value=value, return_pct=return_pct))

    def record_portfolio_value(self, timestamp: datetime, value: float) -> None:
        """Record portfolio value."""
        self._portfolio_values.append((timestamp, value))

    def calculate_tracking_error(
        self,
        benchmark: str | None = None,
        lookback_days: int = 30
    ) -> float | None:
        """
        Calculate tracking error vs benchmark (#P12).

        Tracking error = std dev of (portfolio return - benchmark return)

        Args:
            benchmark: Benchmark name (uses active if None)
            lookback_days: Days of history to use

        Returns:
            Annualized tracking error
        """
        benchmark = benchmark or self._active_benchmark
        if benchmark is None or benchmark not in self._benchmarks:
            return None

        benchmark_data = self._benchmarks[benchmark][-lookback_days:]
        portfolio_data = self._portfolio_values[-lookback_days:]

        if len(benchmark_data) < 2 or len(portfolio_data) < 2:
            return None

        # Calculate returns
        benchmark_returns = [d.return_pct for d in benchmark_data if d.return_pct is not None]
        portfolio_returns = []
        for i in range(1, len(portfolio_data)):
            prev_val = portfolio_data[i-1][1]
            curr_val = portfolio_data[i][1]
            if prev_val > 0:
                portfolio_returns.append((curr_val - prev_val) / prev_val)

        # Match lengths
        n = min(len(benchmark_returns), len(portfolio_returns))
        if n < 2:
            return None

        benchmark_returns = benchmark_returns[-n:]
        portfolio_returns = portfolio_returns[-n:]

        # Calculate active returns
        active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]

        # Tracking error (annualized)
        if len(active_returns) > 1:
            variance = sum((r - sum(active_returns)/len(active_returns))**2 for r in active_returns) / (len(active_returns) - 1)
            daily_te = variance ** 0.5
            annualized_te = daily_te * (252 ** 0.5)  # Annualize
            return annualized_te

        return None

    def calculate_information_ratio(
        self,
        benchmark: str | None = None,
        lookback_days: int = 252
    ) -> float | None:
        """
        Calculate information ratio (#P12).

        IR = Active Return / Tracking Error

        Args:
            benchmark: Benchmark name
            lookback_days: Days of history

        Returns:
            Information ratio
        """
        tracking_error = self.calculate_tracking_error(benchmark, lookback_days)
        if tracking_error is None or tracking_error == 0:
            return None

        benchmark = benchmark or self._active_benchmark
        benchmark_data = self._benchmarks.get(benchmark, [])[-lookback_days:]
        portfolio_data = self._portfolio_values[-lookback_days:]

        if len(benchmark_data) < 2 or len(portfolio_data) < 2:
            return None

        # Total returns
        benchmark_total = (benchmark_data[-1].value / benchmark_data[0].value - 1) if benchmark_data[0].value > 0 else 0
        portfolio_total = (portfolio_data[-1][1] / portfolio_data[0][1] - 1) if portfolio_data[0][1] > 0 else 0

        active_return = portfolio_total - benchmark_total

        return active_return / tracking_error

    def get_benchmark_comparison(self, benchmark: str | None = None) -> dict:
        """Get comparison of portfolio vs benchmark."""
        benchmark = benchmark or self._active_benchmark
        if benchmark is None:
            return {"error": "no_benchmark_set"}

        benchmark_data = self._benchmarks.get(benchmark, [])
        portfolio_data = self._portfolio_values

        if not benchmark_data or not portfolio_data:
            return {"error": "insufficient_data"}

        # Calculate metrics
        tracking_error = self.calculate_tracking_error(benchmark)
        info_ratio = self.calculate_information_ratio(benchmark)

        return {
            "benchmark": benchmark,
            "portfolio_current": portfolio_data[-1][1] if portfolio_data else None,
            "benchmark_current": benchmark_data[-1].value if benchmark_data else None,
            "tracking_error_annualized": tracking_error,
            "information_ratio": info_ratio,
            "data_points_portfolio": len(portfolio_data),
            "data_points_benchmark": len(benchmark_data),
        }


# =============================================================================
# PORTFOLIO HEAT MAP VISUALIZATION (#P13)
# =============================================================================

class PortfolioHeatMapGenerator:
    """
    Generates heat map data for portfolio visualization (#P13).

    Creates visualizations for:
    - Sector/asset performance
    - Risk contribution
    - Correlation matrix
    - P&L by position
    """

    @staticmethod
    def generate_performance_heatmap(
        positions: dict[str, dict],  # symbol -> {return_pct, weight, ...}
        group_by: str = "sector"
    ) -> dict:
        """
        Generate performance heat map data (#P13).

        Args:
            positions: Position data with returns
            group_by: Grouping field (sector, asset_class, etc.)

        Returns:
            Heat map data structure
        """
        # Group positions
        groups: dict[str, list] = {}
        for symbol, data in positions.items():
            group = data.get(group_by, "Other")
            if group not in groups:
                groups[group] = []
            groups[group].append({
                "symbol": symbol,
                "return_pct": data.get("return_pct", 0),
                "weight": data.get("weight", 0),
                "pnl": data.get("pnl", 0),
            })

        # Calculate group-level metrics
        heatmap = []
        for group, items in groups.items():
            total_weight = sum(i["weight"] for i in items)
            weighted_return = sum(i["return_pct"] * i["weight"] for i in items) / total_weight if total_weight > 0 else 0

            heatmap.append({
                "name": group,
                "value": weighted_return,  # Return for color scaling
                "weight": total_weight,
                "count": len(items),
                "items": sorted(items, key=lambda x: x["return_pct"], reverse=True),
            })

        return {
            "type": "performance",
            "group_by": group_by,
            "data": sorted(heatmap, key=lambda x: x["value"], reverse=True),
            "min_value": min(h["value"] for h in heatmap) if heatmap else 0,
            "max_value": max(h["value"] for h in heatmap) if heatmap else 0,
        }

    @staticmethod
    def generate_risk_contribution_heatmap(
        risk_contributions: dict[str, float],  # symbol -> risk contribution %
        positions: dict[str, dict]
    ) -> dict:
        """
        Generate risk contribution heat map (#P13).

        Args:
            risk_contributions: Risk contribution by symbol
            positions: Position data for grouping

        Returns:
            Heat map data structure
        """
        heatmap = []
        total_risk = sum(abs(v) for v in risk_contributions.values())

        for symbol, contribution in risk_contributions.items():
            pos_data = positions.get(symbol, {})
            heatmap.append({
                "symbol": symbol,
                "risk_contribution": contribution,
                "risk_contribution_pct": contribution / total_risk * 100 if total_risk > 0 else 0,
                "weight": pos_data.get("weight", 0),
                "sector": pos_data.get("sector", "Other"),
            })

        return {
            "type": "risk_contribution",
            "total_risk": total_risk,
            "data": sorted(heatmap, key=lambda x: abs(x["risk_contribution"]), reverse=True),
        }

    @staticmethod
    def generate_correlation_heatmap(
        correlation_matrix: dict[str, dict[str, float]]  # symbol -> {symbol -> corr}
    ) -> dict:
        """
        Generate correlation matrix heat map (#P13).

        Args:
            correlation_matrix: Pairwise correlations

        Returns:
            Heat map data structure
        """
        symbols = list(correlation_matrix.keys())
        n = len(symbols)

        # Build matrix for visualization
        matrix = []
        for i, sym1 in enumerate(symbols):
            row = []
            for j, sym2 in enumerate(symbols):
                corr = correlation_matrix.get(sym1, {}).get(sym2, 0)
                row.append({
                    "row": i,
                    "col": j,
                    "row_symbol": sym1,
                    "col_symbol": sym2,
                    "correlation": corr,
                })
            matrix.append(row)

        return {
            "type": "correlation",
            "symbols": symbols,
            "size": n,
            "matrix": matrix,
            "min_correlation": -1,
            "max_correlation": 1,
        }

    @staticmethod
    def generate_pnl_heatmap(
        daily_pnl: dict[str, list[float]],  # symbol -> [daily pnl values]
        dates: list[str]
    ) -> dict:
        """
        Generate P&L calendar heat map (#P13).

        Args:
            daily_pnl: Daily P&L by symbol
            dates: List of date strings

        Returns:
            Heat map data structure for calendar view
        """
        # Aggregate daily P&L across symbols
        total_daily = [0.0] * len(dates)
        for symbol, pnl_list in daily_pnl.items():
            for i, pnl in enumerate(pnl_list[:len(dates)]):
                total_daily[i] += pnl

        calendar_data = [
            {"date": d, "pnl": p, "is_positive": p >= 0}
            for d, p in zip(dates, total_daily)
        ]

        return {
            "type": "pnl_calendar",
            "dates": dates,
            "data": calendar_data,
            "total_pnl": sum(total_daily),
            "positive_days": sum(1 for p in total_daily if p >= 0),
            "negative_days": sum(1 for p in total_daily if p < 0),
            "max_gain": max(total_daily) if total_daily else 0,
            "max_loss": min(total_daily) if total_daily else 0,
        }

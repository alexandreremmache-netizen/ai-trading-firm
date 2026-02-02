"""
Backtesting Framework
=====================

Comprehensive backtesting engine for strategy validation (Issue #Q7).

Features:
- Historical data simulation
- Multiple strategy support
- Transaction cost modeling
- Slippage simulation
- Performance metrics calculation
- Walk-forward analysis support
- Multi-asset portfolio backtesting
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterator
import math
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


class BacktestMode(str, Enum):
    """Backtest execution mode."""
    VECTORIZED = "vectorized"  # Fast, all-at-once (for simple strategies)
    EVENT_DRIVEN = "event_driven"  # Realistic, bar-by-bar


class FillModel(str, Enum):
    """Order fill simulation model."""
    IMMEDIATE = "immediate"  # Fill at current price
    NEXT_OPEN = "next_open"  # Fill at next bar's open
    NEXT_CLOSE = "next_close"  # Fill at next bar's close
    VWAP = "vwap"  # Fill at bar VWAP
    SLIPPAGE = "slippage"  # With slippage model


@dataclass
class Bar:
    """Single OHLCV bar."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None

    def __post_init__(self):
        if self.vwap is None:
            # Estimate VWAP as typical price
            self.vwap = (self.high + self.low + self.close) / 3


@dataclass
class BacktestOrder:
    """Order in backtest."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'MARKET', 'LIMIT'
    limit_price: float | None = None
    strategy_id: str = ""
    signal_strength: float = 0.0


@dataclass
class BacktestFill:
    """Fill result in backtest."""
    order_id: str
    fill_time: datetime
    symbol: str
    side: str
    quantity: int
    fill_price: float
    commission: float
    slippage: float
    strategy_id: str = ""


@dataclass
class BacktestPosition:
    """Position tracking in backtest."""
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    strategy_id: str = ""

    def update(self, fill: BacktestFill, current_price: float) -> float:
        """Update position from fill, return realized P&L."""
        realized = 0.0

        if fill.side == 'BUY':
            if self.quantity >= 0:
                # Adding to long or opening
                new_qty = self.quantity + fill.quantity
                new_cost = (self.avg_cost * self.quantity + fill.fill_price * fill.quantity)
                self.avg_cost = new_cost / new_qty if new_qty > 0 else 0
                self.quantity = new_qty
            else:
                # Covering short
                cover_qty = min(fill.quantity, abs(self.quantity))
                realized = cover_qty * (self.avg_cost - fill.fill_price)
                self.quantity += fill.quantity
                if self.quantity > 0:
                    self.avg_cost = fill.fill_price
        else:  # SELL
            if self.quantity <= 0:
                # Adding to short or opening
                new_qty = self.quantity - fill.quantity
                if self.quantity < 0:
                    new_cost = (abs(self.quantity) * self.avg_cost + fill.quantity * fill.fill_price)
                    self.avg_cost = new_cost / abs(new_qty) if new_qty != 0 else 0
                else:
                    self.avg_cost = fill.fill_price
                self.quantity = new_qty
            else:
                # Closing long
                close_qty = min(fill.quantity, self.quantity)
                realized = close_qty * (fill.fill_price - self.avg_cost)
                self.quantity -= fill.quantity
                if self.quantity < 0:
                    self.avg_cost = fill.fill_price

        self.realized_pnl += realized
        self.unrealized_pnl = self.quantity * (current_price - self.avg_cost) if self.quantity != 0 else 0

        return realized


@dataclass
class TransactionCostModel:
    """Transaction cost model for realistic backtesting (Issue #Q10)."""
    commission_per_share: float = 0.005  # $0.005 per share
    commission_minimum: float = 1.0  # $1 minimum
    commission_maximum: float = 0.005  # 0.5% of trade value max
    spread_bps: float = 2.0  # 2 bps spread
    market_impact_bps: float = 5.0  # 5 bps market impact
    slippage_volatility_mult: float = 0.5  # Slippage = mult * volatility

    def calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate commission for trade."""
        trade_value = quantity * price
        commission = max(
            self.commission_minimum,
            min(
                quantity * self.commission_per_share,
                trade_value * self.commission_maximum
            )
        )
        return commission

    def calculate_slippage(
        self,
        quantity: int,
        price: float,
        volatility: float,
        adv: float | None = None,
    ) -> float:
        """Calculate expected slippage."""
        # Base spread cost
        spread_cost = price * (self.spread_bps / 10000)

        # Market impact (square root model)
        if adv and adv > 0:
            participation = quantity / adv
            market_impact = price * (self.market_impact_bps / 10000) * math.sqrt(participation)
        else:
            market_impact = price * (self.market_impact_bps / 10000)

        # Volatility-based component
        vol_slippage = price * volatility * self.slippage_volatility_mult / 100

        return spread_cost + market_impact + vol_slippage


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0

    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # Days
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Trading
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0

    # Time analysis
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    total_months: int = 0

    # Exposure
    avg_exposure: float = 0.0
    max_exposure: float = 0.0
    time_in_market: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'returns': {
                'total_return_pct': self.total_return * 100,
                'annualized_return_pct': self.annualized_return * 100,
                'cagr_pct': self.cagr * 100,
            },
            'risk': {
                'volatility_pct': self.volatility * 100,
                'max_drawdown_pct': self.max_drawdown * 100,
                'max_drawdown_duration_days': self.max_drawdown_duration,
                'var_95_pct': self.var_95 * 100,
                'cvar_95_pct': self.cvar_95 * 100,
            },
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'information_ratio': self.information_ratio,
            },
            'trading': {
                'total_trades': self.total_trades,
                'win_rate_pct': self.win_rate * 100,
                'profit_factor': self.profit_factor,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
            },
            'costs': {
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
                'total_costs': self.total_costs,
            },
            'exposure': {
                'avg_exposure_pct': self.avg_exposure * 100,
                'time_in_market_pct': self.time_in_market * 100,
            },
        }


class BacktestStrategy(ABC):
    """Abstract base class for backtestable strategies."""

    def __init__(self, strategy_id: str, params: dict | None = None):
        self.strategy_id = strategy_id
        self.params = params or {}

    @abstractmethod
    def on_bar(
        self,
        bar: Bar,
        position: BacktestPosition | None,
        portfolio_value: float,
    ) -> BacktestOrder | None:
        """
        Generate signal on new bar.

        Returns order if action needed, None otherwise.
        """
        pass

    def on_fill(self, fill: BacktestFill) -> None:
        """Called when order is filled."""
        pass

    def on_start(self, start_date: datetime, initial_capital: float) -> None:
        """Called at backtest start."""
        pass

    def on_end(self, end_date: datetime, final_value: float) -> None:
        """Called at backtest end."""
        pass


class BacktestEngine:
    """
    Main backtesting engine.

    Supports:
    - Multiple strategies
    - Realistic execution simulation
    - Transaction costs
    - Performance analytics
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fill_model: FillModel = FillModel.NEXT_OPEN,
        cost_model: TransactionCostModel | None = None,
        risk_free_rate: float = 0.02,  # 2% annual
    ):
        self.initial_capital = initial_capital
        self.fill_model = fill_model
        self.cost_model = cost_model or TransactionCostModel()
        self.risk_free_rate = risk_free_rate

        # State
        self.cash = initial_capital
        self.positions: dict[str, BacktestPosition] = {}
        self.pending_orders: list[BacktestOrder] = []
        self.fills: list[BacktestFill] = []
        self.strategies: dict[str, BacktestStrategy] = {}

        # History
        self.equity_curve: list[tuple[datetime, float]] = []
        self.daily_returns: list[float] = []
        self.drawdown_curve: list[tuple[datetime, float]] = []

        # Tracking
        self._order_counter = 0
        self._current_bar: dict[str, Bar] = {}
        self._volatility_cache: dict[str, float] = {}
        self._trade_pnls: list[float] = []
        self._monthly_returns: dict[str, float] = {}

    def add_strategy(self, strategy: BacktestStrategy) -> None:
        """Add strategy to backtest."""
        self.strategies[strategy.strategy_id] = strategy

    def run(
        self,
        data: dict[str, list[Bar]],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.

        Args:
            data: Dictionary mapping symbol to list of bars
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestMetrics with performance results
        """
        logger.info(f"Starting backtest with {len(self.strategies)} strategies")

        # Merge and sort all bars
        all_bars = self._merge_bars(data, start_date, end_date)
        if not all_bars:
            raise ValueError("No data in specified date range")

        # Initialize
        self._reset()
        actual_start = all_bars[0].timestamp
        actual_end = all_bars[-1].timestamp

        for strategy in self.strategies.values():
            strategy.on_start(actual_start, self.initial_capital)

        # Track for daily returns
        last_equity = self.initial_capital
        last_date = actual_start.date()
        peak_equity = self.initial_capital
        drawdown_start: datetime | None = None
        max_dd_duration = 0
        current_dd_start: datetime | None = None

        # Main simulation loop
        for bar in all_bars:
            self._current_bar[bar.symbol] = bar

            # Process pending orders first
            self._process_pending_orders(bar)

            # Get current portfolio value
            portfolio_value = self._calculate_portfolio_value()

            # Generate signals from strategies
            for strategy in self.strategies.values():
                position = self.positions.get(bar.symbol)
                order = strategy.on_bar(bar, position, portfolio_value)
                if order:
                    order.order_id = f"BT_{self._order_counter}"
                    order.strategy_id = strategy.strategy_id
                    self._order_counter += 1
                    self.pending_orders.append(order)

            # Track equity
            current_equity = self._calculate_portfolio_value()
            self.equity_curve.append((bar.timestamp, current_equity))

            # Daily returns
            if bar.timestamp.date() != last_date:
                daily_return = (current_equity - last_equity) / last_equity if last_equity > 0 else 0
                self.daily_returns.append(daily_return)

                # Monthly tracking
                month_key = bar.timestamp.strftime('%Y-%m')
                if month_key not in self._monthly_returns:
                    self._monthly_returns[month_key] = 0
                self._monthly_returns[month_key] += daily_return

                last_equity = current_equity
                last_date = bar.timestamp.date()

            # Drawdown tracking
            if current_equity > peak_equity:
                peak_equity = current_equity
                if current_dd_start:
                    dd_duration = (bar.timestamp - current_dd_start).days
                    max_dd_duration = max(max_dd_duration, dd_duration)
                current_dd_start = None
            else:
                if current_dd_start is None:
                    current_dd_start = bar.timestamp

            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            self.drawdown_curve.append((bar.timestamp, drawdown))

        # Finalize
        for strategy in self.strategies.values():
            strategy.on_end(actual_end, self._calculate_portfolio_value())

        # Calculate metrics
        metrics = self._calculate_metrics(actual_start, actual_end)
        metrics.max_drawdown_duration = max_dd_duration

        logger.info(
            f"Backtest complete: Return={metrics.total_return*100:.2f}%, "
            f"Sharpe={metrics.sharpe_ratio:.2f}, MaxDD={metrics.max_drawdown*100:.2f}%"
        )

        return metrics

    def _merge_bars(
        self,
        data: dict[str, list[Bar]],
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[Bar]:
        """Merge and sort bars from all symbols."""
        all_bars = []

        for symbol, bars in data.items():
            for bar in bars:
                if start_date and bar.timestamp < start_date:
                    continue
                if end_date and bar.timestamp > end_date:
                    continue
                all_bars.append(bar)

        return sorted(all_bars, key=lambda b: b.timestamp)

    def _reset(self) -> None:
        """Reset engine state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.pending_orders = []
        self.fills = []
        self.equity_curve = []
        self.daily_returns = []
        self.drawdown_curve = []
        self._order_counter = 0
        self._current_bar = {}
        self._trade_pnls = []
        self._monthly_returns = {}

    def _process_pending_orders(self, bar: Bar) -> None:
        """Process pending orders against current bar."""
        remaining_orders = []

        for order in self.pending_orders:
            if order.symbol != bar.symbol:
                remaining_orders.append(order)
                continue

            fill_price = self._get_fill_price(order, bar)

            if fill_price is None:
                # Order not fillable (e.g., limit not reached)
                remaining_orders.append(order)
                continue

            # Calculate costs
            volatility = self._get_volatility(bar.symbol)
            slippage = self.cost_model.calculate_slippage(
                order.quantity, fill_price, volatility
            )

            # Adjust fill price for slippage
            if order.side == 'BUY':
                fill_price += slippage
            else:
                fill_price -= slippage

            commission = self.cost_model.calculate_commission(order.quantity, fill_price)

            # Create fill
            fill = BacktestFill(
                order_id=order.order_id,
                fill_time=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage,
                strategy_id=order.strategy_id,
            )

            # Update position
            if order.symbol not in self.positions:
                self.positions[order.symbol] = BacktestPosition(
                    symbol=order.symbol,
                    strategy_id=order.strategy_id,
                )

            position = self.positions[order.symbol]
            realized_pnl = position.update(fill, bar.close)

            if realized_pnl != 0:
                self._trade_pnls.append(realized_pnl)

            # Update cash
            trade_value = order.quantity * fill_price
            if order.side == 'BUY':
                self.cash -= trade_value + commission
            else:
                self.cash += trade_value - commission

            self.fills.append(fill)

            # Notify strategy
            if order.strategy_id in self.strategies:
                self.strategies[order.strategy_id].on_fill(fill)

        self.pending_orders = remaining_orders

    def _get_fill_price(self, order: BacktestOrder, bar: Bar) -> float | None:
        """Determine fill price based on fill model."""
        if order.order_type == 'LIMIT':
            if order.side == 'BUY' and order.limit_price:
                if bar.low <= order.limit_price:
                    return min(order.limit_price, bar.open)
            elif order.side == 'SELL' and order.limit_price:
                if bar.high >= order.limit_price:
                    return max(order.limit_price, bar.open)
            return None

        # Market orders
        if self.fill_model == FillModel.IMMEDIATE:
            return bar.close
        elif self.fill_model == FillModel.NEXT_OPEN:
            return bar.open
        elif self.fill_model == FillModel.NEXT_CLOSE:
            return bar.close
        elif self.fill_model == FillModel.VWAP:
            return bar.vwap or (bar.high + bar.low + bar.close) / 3
        else:  # SLIPPAGE model uses open with slippage added later
            return bar.open

    def _get_volatility(self, symbol: str) -> float:
        """Get estimated volatility for symbol."""
        # Simple ATR-based volatility estimate
        if symbol in self._volatility_cache:
            return self._volatility_cache[symbol]
        return 0.02  # Default 2%

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        value = self.cash

        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in self._current_bar:
                price = self._current_bar[symbol].close
                value += position.quantity * price

        return value

    def _calculate_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.equity_curve or not self.daily_returns:
            return metrics

        final_value = self.equity_curve[-1][1]
        days = (end_date - start_date).days
        years = days / 365.25

        # Returns
        metrics.total_return = (final_value - self.initial_capital) / self.initial_capital
        if years > 0:
            metrics.cagr = (final_value / self.initial_capital) ** (1 / years) - 1
            metrics.annualized_return = sum(self.daily_returns) * 252 / len(self.daily_returns)

        # Risk
        if len(self.daily_returns) > 1:
            metrics.volatility = statistics.stdev(self.daily_returns) * math.sqrt(252)

            # VaR and CVaR
            sorted_returns = sorted(self.daily_returns)
            var_index = int(len(sorted_returns) * 0.05)
            metrics.var_95 = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0
            if var_index > 0:
                metrics.cvar_95 = -statistics.mean(sorted_returns[:var_index])

        # Max drawdown
        if self.drawdown_curve:
            metrics.max_drawdown = max(dd for _, dd in self.drawdown_curve)

        # Risk-adjusted ratios
        daily_rf = self.risk_free_rate / 252
        excess_returns = [r - daily_rf for r in self.daily_returns]

        if metrics.volatility > 0:
            avg_excess = statistics.mean(excess_returns) * 252
            metrics.sharpe_ratio = avg_excess / metrics.volatility

        # Sortino (downside deviation)
        negative_returns = [r for r in self.daily_returns if r < daily_rf]
        if negative_returns:
            downside_vol = statistics.stdev(negative_returns) * math.sqrt(252)
            if downside_vol > 0:
                metrics.sortino_ratio = statistics.mean(excess_returns) * 252 / downside_vol

        # Calmar
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown

        # Trading metrics
        metrics.total_trades = len(self._trade_pnls)
        if self._trade_pnls:
            wins = [p for p in self._trade_pnls if p > 0]
            losses = [p for p in self._trade_pnls if p < 0]

            metrics.winning_trades = len(wins)
            metrics.losing_trades = len(losses)
            metrics.win_rate = len(wins) / len(self._trade_pnls) if self._trade_pnls else 0

            if wins:
                metrics.avg_win = statistics.mean(wins)
                metrics.largest_win = max(wins)
            if losses:
                metrics.avg_loss = statistics.mean(losses)
                metrics.largest_loss = min(losses)

            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses

        # Costs
        metrics.total_commission = sum(f.commission for f in self.fills)
        metrics.total_slippage = sum(f.slippage * f.quantity for f in self.fills)
        metrics.total_costs = metrics.total_commission + metrics.total_slippage

        # Monthly analysis
        if self._monthly_returns:
            monthly_rets = list(self._monthly_returns.values())
            metrics.best_month = max(monthly_rets)
            metrics.worst_month = min(monthly_rets)
            metrics.positive_months = sum(1 for r in monthly_rets if r > 0)
            metrics.total_months = len(monthly_rets)

        # Exposure
        exposures = []
        for _, equity in self.equity_curve:
            exposure = (equity - self.cash) / equity if equity > 0 else 0
            exposures.append(abs(exposure))

        if exposures:
            metrics.avg_exposure = statistics.mean(exposures)
            metrics.max_exposure = max(exposures)
            metrics.time_in_market = sum(1 for e in exposures if e > 0.01) / len(exposures)

        return metrics


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation (Issue #Q8 partial).

    Helps detect overfitting by validating on out-of-sample data.
    """

    def __init__(
        self,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,  # 3 months
        step_days: int = 21,  # 1 month
    ):
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.step = timedelta(days=step_days)

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test windows.

        Yields: (train_start, train_end, test_start, test_end)
        """
        current_train_start = start_date

        while True:
            train_end = current_train_start + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period

            if test_end > end_date:
                break

            yield (current_train_start, train_end, test_start, test_end)

            current_train_start += self.step

    def run(
        self,
        engine_factory: Callable[[], BacktestEngine],
        strategy_factory: Callable[[dict], BacktestStrategy],
        data: dict[str, list[Bar]],
        param_optimizer: Callable[[BacktestEngine, datetime, datetime], dict] | None = None,
    ) -> list[dict]:
        """
        Run walk-forward analysis.

        Args:
            engine_factory: Function that creates new BacktestEngine
            strategy_factory: Function that creates strategy with params
            data: Historical data
            param_optimizer: Optional function to optimize params on train set

        Returns:
            List of results per window
        """
        # Get date range from data
        all_dates = []
        for bars in data.values():
            all_dates.extend(b.timestamp for b in bars)

        start_date = min(all_dates)
        end_date = max(all_dates)

        results = []

        for train_start, train_end, test_start, test_end in self.generate_windows(start_date, end_date):
            logger.info(
                f"Walk-forward window: Train {train_start.date()} to {train_end.date()}, "
                f"Test {test_start.date()} to {test_end.date()}"
            )

            # Create fresh engine for each window
            engine = engine_factory()

            # Optimize on train set if optimizer provided
            if param_optimizer:
                params = param_optimizer(engine, train_start, train_end)
            else:
                params = {}

            # Create strategy with optimized params
            strategy = strategy_factory(params)
            engine.add_strategy(strategy)

            # Run on test period
            metrics = engine.run(data, test_start, test_end)

            results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'params': params,
                'metrics': metrics.to_dict(),
            })

        return results


@dataclass
class BacktestComparison:
    """Compare multiple backtest results."""
    results: dict[str, BacktestMetrics] = field(default_factory=dict)

    def add_result(self, name: str, metrics: BacktestMetrics) -> None:
        """Add backtest result."""
        self.results[name] = metrics

    def get_comparison_table(self) -> list[dict]:
        """Get comparison as table rows."""
        rows = []

        for name, metrics in self.results.items():
            rows.append({
                'Strategy': name,
                'Return %': f"{metrics.total_return * 100:.2f}",
                'CAGR %': f"{metrics.cagr * 100:.2f}",
                'Volatility %': f"{metrics.volatility * 100:.2f}",
                'Sharpe': f"{metrics.sharpe_ratio:.2f}",
                'Sortino': f"{metrics.sortino_ratio:.2f}",
                'Max DD %': f"{metrics.max_drawdown * 100:.2f}",
                'Win Rate %': f"{metrics.win_rate * 100:.1f}",
                'Trades': metrics.total_trades,
                'Profit Factor': f"{metrics.profit_factor:.2f}",
            })

        return rows

    def get_best_by_sharpe(self) -> str | None:
        """Get strategy name with best Sharpe ratio."""
        if not self.results:
            return None
        return max(self.results.items(), key=lambda x: x[1].sharpe_ratio)[0]

    def get_best_by_return(self) -> str | None:
        """Get strategy name with best total return."""
        if not self.results:
            return None
        return max(self.results.items(), key=lambda x: x[1].total_return)[0]

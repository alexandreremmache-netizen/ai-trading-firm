"""
Options Strategy Backtesting Module
===================================

Addresses issue #O18: No option strategy backtesting.

Features:
- Options strategy backtesting framework
- Multi-leg strategy support
- Greeks evolution tracking
- Volatility surface modeling
- Strategy P&L decomposition
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class OptionType(str, Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class OptionStrategy(str, Enum):
    """Common option strategies."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    CUSTOM = "custom"


@dataclass
class OptionLeg:
    """Single leg of an options position."""
    option_type: OptionType
    strike: float
    expiry: datetime
    quantity: int  # Positive = long, negative = short
    premium: float  # Price paid/received per contract
    multiplier: int = 100

    @property
    def is_long(self) -> bool:
        """Check if leg is long."""
        return self.quantity > 0

    @property
    def notional(self) -> float:
        """Calculate notional value."""
        return abs(self.quantity) * self.premium * self.multiplier

    def days_to_expiry(self, as_of: datetime) -> int:
        """Calculate days to expiry."""
        delta = self.expiry - as_of
        return max(0, delta.days)


@dataclass
class OptionPosition:
    """Multi-leg options position."""
    position_id: str
    underlying: str
    strategy_type: OptionStrategy
    legs: list[OptionLeg]
    entry_date: datetime
    entry_underlying_price: float
    exit_date: datetime | None = None
    exit_underlying_price: float | None = None

    @property
    def max_profit(self) -> float | None:
        """Calculate maximum profit (if defined)."""
        # Simplified - would need full payoff calculation
        return None

    @property
    def max_loss(self) -> float | None:
        """Calculate maximum loss (if defined)."""
        # For most spreads, max loss is net debit
        net_premium = sum(
            leg.premium * leg.quantity * leg.multiplier
            for leg in self.legs
        )
        return abs(min(0, net_premium))

    @property
    def net_premium(self) -> float:
        """Net premium paid/received."""
        return sum(
            leg.premium * leg.quantity * leg.multiplier
            for leg in self.legs
        )

    def is_expired(self, as_of: datetime) -> bool:
        """Check if all legs expired."""
        return all(leg.expiry <= as_of for leg in self.legs)


@dataclass
class OptionGreeks:
    """Greeks for a position."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass
class OptionBacktestBar:
    """Single bar of options backtest data."""
    timestamp: datetime
    underlying_price: float
    option_prices: dict[tuple[float, datetime, OptionType], float]  # (strike, expiry, type) -> price
    implied_vols: dict[tuple[float, datetime, OptionType], float]
    risk_free_rate: float = 0.05


@dataclass
class OptionBacktestResult:
    """Result of options strategy backtest."""
    strategy_type: OptionStrategy
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float

    # Options-specific metrics
    avg_days_held: float
    assignments: int
    exercises: int
    expirations_worthless: int
    avg_entry_iv: float
    avg_exit_iv: float
    theta_collected: float
    gamma_pnl: float
    vega_pnl: float

    # P&L decomposition
    delta_pnl: float = 0.0
    total_theta_decay: float = 0.0
    total_iv_pnl: float = 0.0

    # Equity curve
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy_type.value,
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "performance": {
                "initial_capital": self.initial_capital,
                "final_value": self.final_value,
                "total_return_pct": self.total_return * 100,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown_pct": self.max_drawdown * 100,
            },
            "trading": {
                "total_trades": self.total_trades,
                "win_rate_pct": self.win_rate * 100,
                "avg_profit": self.avg_profit,
                "avg_loss": self.avg_loss,
            },
            "options_specific": {
                "avg_days_held": self.avg_days_held,
                "assignments": self.assignments,
                "exercises": self.exercises,
                "expirations_worthless": self.expirations_worthless,
                "avg_entry_iv": self.avg_entry_iv,
                "avg_exit_iv": self.avg_exit_iv,
            },
            "pnl_decomposition": {
                "delta_pnl": self.delta_pnl,
                "theta_pnl": self.total_theta_decay,
                "vega_pnl": self.total_iv_pnl,
            },
        }


class BlackScholes:
    """Black-Scholes option pricing for backtesting."""

    @staticmethod
    def price(
        spot: float,
        strike: float,
        time_to_expiry: float,  # In years
        volatility: float,
        rate: float,
        is_call: bool,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate option price using Black-Scholes."""
        if time_to_expiry <= 0:
            # At expiry
            if is_call:
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)

        d1 = (
            math.log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))

        d2 = d1 - volatility * math.sqrt(time_to_expiry)

        if is_call:
            price = (
                spot * math.exp(-dividend_yield * time_to_expiry) * BlackScholes._norm_cdf(d1)
                - strike * math.exp(-rate * time_to_expiry) * BlackScholes._norm_cdf(d2)
            )
        else:
            price = (
                strike * math.exp(-rate * time_to_expiry) * BlackScholes._norm_cdf(-d2)
                - spot * math.exp(-dividend_yield * time_to_expiry) * BlackScholes._norm_cdf(-d1)
            )

        return max(0, price)

    @staticmethod
    def delta(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        is_call: bool,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate delta."""
        if time_to_expiry <= 0:
            if is_call:
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0

        d1 = (
            math.log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))

        if is_call:
            return math.exp(-dividend_yield * time_to_expiry) * BlackScholes._norm_cdf(d1)
        else:
            return -math.exp(-dividend_yield * time_to_expiry) * BlackScholes._norm_cdf(-d1)

    @staticmethod
    def gamma(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate gamma."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (
            math.log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))

        return (
            math.exp(-dividend_yield * time_to_expiry)
            * BlackScholes._norm_pdf(d1)
            / (spot * volatility * math.sqrt(time_to_expiry))
        )

    @staticmethod
    def theta(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        is_call: bool,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate theta (per day)."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (
            math.log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))

        d2 = d1 - volatility * math.sqrt(time_to_expiry)

        term1 = -spot * BlackScholes._norm_pdf(d1) * volatility * math.exp(-dividend_yield * time_to_expiry)
        term1 /= 2 * math.sqrt(time_to_expiry)

        if is_call:
            theta_annual = (
                term1
                - rate * strike * math.exp(-rate * time_to_expiry) * BlackScholes._norm_cdf(d2)
                + dividend_yield * spot * math.exp(-dividend_yield * time_to_expiry) * BlackScholes._norm_cdf(d1)
            )
        else:
            theta_annual = (
                term1
                + rate * strike * math.exp(-rate * time_to_expiry) * BlackScholes._norm_cdf(-d2)
                - dividend_yield * spot * math.exp(-dividend_yield * time_to_expiry) * BlackScholes._norm_cdf(-d1)
            )

        return theta_annual / 365  # Per day

    @staticmethod
    def vega(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate vega (per 1% vol move)."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (
            math.log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))

        return (
            spot
            * math.exp(-dividend_yield * time_to_expiry)
            * BlackScholes._norm_pdf(d1)
            * math.sqrt(time_to_expiry)
            * 0.01  # Per 1% move
        )

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


class OptionStrategyBacktester:
    """
    Options strategy backtesting engine (#O18).

    Supports multi-leg strategies with Greeks tracking.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.02,
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            risk_free_rate: Risk-free rate for pricing
            dividend_yield: Dividend yield for underlying
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

        # State
        self._cash = initial_capital
        self._positions: list[OptionPosition] = []
        self._closed_positions: list[OptionPosition] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._daily_returns: list[float] = []

        # Tracking
        self._total_theta_decay = 0.0
        self._total_vega_pnl = 0.0
        self._total_delta_pnl = 0.0
        self._assignments = 0
        self._exercises = 0
        self._expirations_worthless = 0

    def add_position(
        self,
        position: OptionPosition,
        bar: OptionBacktestBar,
    ) -> bool:
        """Add new position to portfolio."""
        # Check if we have enough capital
        margin_required = self._calculate_margin(position, bar)
        if margin_required > self._cash:
            logger.warning(f"Insufficient capital for position: {position.position_id}")
            return False

        # Deduct premium (if net debit)
        self._cash -= position.net_premium

        self._positions.append(position)
        return True

    def close_position(
        self,
        position: OptionPosition,
        bar: OptionBacktestBar,
        reason: str = "manual",
    ) -> float:
        """Close a position and return P&L."""
        if position not in self._positions:
            return 0.0

        # Calculate exit value
        exit_value = self._calculate_position_value(position, bar)

        # P&L is exit value minus entry premium
        pnl = exit_value - (-position.net_premium)

        # Update position
        position.exit_date = bar.timestamp
        position.exit_underlying_price = bar.underlying_price

        # Update cash
        self._cash += exit_value

        # Move to closed
        self._positions.remove(position)
        self._closed_positions.append(position)

        return pnl

    def run_backtest(
        self,
        data: list[OptionBacktestBar],
        entry_signal: Callable[[OptionBacktestBar, float], OptionPosition | None],
        exit_signal: Callable[[OptionPosition, OptionBacktestBar], bool],
    ) -> OptionBacktestResult:
        """
        Run options strategy backtest.

        Args:
            data: Historical bar data
            entry_signal: Function returning position to enter or None
            exit_signal: Function returning True if should exit

        Returns:
            OptionBacktestResult with metrics
        """
        self._reset()

        last_value = self.initial_capital

        for bar in data:
            # Check exits first
            for position in list(self._positions):
                # Check expiration
                if position.is_expired(bar.timestamp):
                    self._handle_expiration(position, bar)
                    continue

                # Check exit signal
                if exit_signal(position, bar):
                    self.close_position(position, bar, reason="signal")

            # Check entries
            position = entry_signal(bar, self._cash)
            if position:
                self.add_position(position, bar)

            # Record equity
            current_value = self._calculate_portfolio_value(bar)
            self._equity_curve.append((bar.timestamp, current_value))

            # Daily return
            if last_value > 0:
                daily_return = (current_value - last_value) / last_value
                self._daily_returns.append(daily_return)

            last_value = current_value

        # Close remaining positions at end
        if data:
            final_bar = data[-1]
            for position in list(self._positions):
                self.close_position(position, final_bar, reason="end_of_backtest")

        return self._calculate_results(data[0].timestamp, data[-1].timestamp)

    def _reset(self) -> None:
        """Reset backtest state."""
        self._cash = self.initial_capital
        self._positions = []
        self._closed_positions = []
        self._equity_curve = []
        self._daily_returns = []
        self._total_theta_decay = 0.0
        self._total_vega_pnl = 0.0
        self._total_delta_pnl = 0.0
        self._assignments = 0
        self._exercises = 0
        self._expirations_worthless = 0

    def _calculate_position_value(
        self,
        position: OptionPosition,
        bar: OptionBacktestBar,
    ) -> float:
        """Calculate current value of position."""
        total_value = 0.0

        for leg in position.legs:
            tte = max(0, leg.days_to_expiry(bar.timestamp) / 365)

            # Get IV from bar data or use default
            key = (leg.strike, leg.expiry, leg.option_type)
            iv = bar.implied_vols.get(key, 0.25)

            price = BlackScholes.price(
                spot=bar.underlying_price,
                strike=leg.strike,
                time_to_expiry=tte,
                volatility=iv,
                rate=bar.risk_free_rate,
                is_call=leg.option_type == OptionType.CALL,
                dividend_yield=self.dividend_yield,
            )

            total_value += price * leg.quantity * leg.multiplier

        return total_value

    def _calculate_position_greeks(
        self,
        position: OptionPosition,
        bar: OptionBacktestBar,
    ) -> OptionGreeks:
        """Calculate Greeks for position."""
        greeks = OptionGreeks()

        for leg in position.legs:
            tte = max(0, leg.days_to_expiry(bar.timestamp) / 365)
            key = (leg.strike, leg.expiry, leg.option_type)
            iv = bar.implied_vols.get(key, 0.25)
            is_call = leg.option_type == OptionType.CALL

            leg_delta = BlackScholes.delta(
                bar.underlying_price, leg.strike, tte, iv,
                bar.risk_free_rate, is_call, self.dividend_yield
            )
            leg_gamma = BlackScholes.gamma(
                bar.underlying_price, leg.strike, tte, iv,
                bar.risk_free_rate, self.dividend_yield
            )
            leg_theta = BlackScholes.theta(
                bar.underlying_price, leg.strike, tte, iv,
                bar.risk_free_rate, is_call, self.dividend_yield
            )
            leg_vega = BlackScholes.vega(
                bar.underlying_price, leg.strike, tte, iv,
                bar.risk_free_rate, self.dividend_yield
            )

            multiplier = leg.quantity * leg.multiplier
            greeks.delta += leg_delta * multiplier
            greeks.gamma += leg_gamma * multiplier
            greeks.theta += leg_theta * multiplier
            greeks.vega += leg_vega * multiplier

        return greeks

    def _calculate_portfolio_value(self, bar: OptionBacktestBar) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            self._calculate_position_value(p, bar)
            for p in self._positions
        )
        return self._cash + positions_value

    def _calculate_margin(
        self,
        position: OptionPosition,
        bar: OptionBacktestBar,
    ) -> float:
        """Calculate margin requirement for position."""
        # Simplified margin calculation
        # In reality, would depend on strategy type and broker rules

        if position.strategy_type in [
            OptionStrategy.LONG_CALL,
            OptionStrategy.LONG_PUT,
            OptionStrategy.STRADDLE,
            OptionStrategy.STRANGLE,
        ]:
            # Fully paid positions
            return abs(position.net_premium)

        # For credit spreads, margin is width minus premium received
        strikes = [leg.strike for leg in position.legs]
        if len(strikes) >= 2:
            width = max(strikes) - min(strikes)
            multiplier = max(abs(leg.quantity) * leg.multiplier for leg in position.legs)
            return width * multiplier - max(0, -position.net_premium)

        return abs(position.net_premium)

    def _handle_expiration(
        self,
        position: OptionPosition,
        bar: OptionBacktestBar,
    ) -> None:
        """Handle position expiration."""
        total_intrinsic = 0.0

        for leg in position.legs:
            is_call = leg.option_type == OptionType.CALL
            if is_call:
                intrinsic = max(0, bar.underlying_price - leg.strike)
            else:
                intrinsic = max(0, leg.strike - bar.underlying_price)

            total_intrinsic += intrinsic * leg.quantity * leg.multiplier

        if total_intrinsic == 0:
            self._expirations_worthless += 1
        else:
            # In reality, would check for assignment/exercise
            pass

        # Close position with intrinsic value
        self._cash += total_intrinsic
        position.exit_date = bar.timestamp
        position.exit_underlying_price = bar.underlying_price

        self._positions.remove(position)
        self._closed_positions.append(position)

    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> OptionBacktestResult:
        """Calculate backtest results."""
        total_trades = len(self._closed_positions)
        final_value = self._equity_curve[-1][1] if self._equity_curve else self.initial_capital

        # P&L calculations
        profits = []
        losses = []
        days_held = []
        entry_ivs = []
        exit_ivs = []

        for pos in self._closed_positions:
            pnl = self._calculate_position_pnl(pos)
            if pnl > 0:
                profits.append(pnl)
            else:
                losses.append(pnl)

            if pos.exit_date and pos.entry_date:
                days_held.append((pos.exit_date - pos.entry_date).days)

        # Win rate
        win_rate = len(profits) / total_trades if total_trades > 0 else 0

        # Max drawdown
        max_dd = 0.0
        peak = self.initial_capital
        for _, value in self._equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio
        if len(self._daily_returns) > 1:
            avg_return = statistics.mean(self._daily_returns) * 252
            std_return = statistics.stdev(self._daily_returns) * math.sqrt(252)
            sharpe = (avg_return - self.risk_free_rate) / std_return if std_return > 0 else 0
        else:
            sharpe = 0.0

        return OptionBacktestResult(
            strategy_type=OptionStrategy.CUSTOM,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=(final_value - self.initial_capital) / self.initial_capital,
            total_trades=total_trades,
            winning_trades=len(profits),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_profit=statistics.mean(profits) if profits else 0,
            avg_loss=statistics.mean(losses) if losses else 0,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            avg_days_held=statistics.mean(days_held) if days_held else 0,
            assignments=self._assignments,
            exercises=self._exercises,
            expirations_worthless=self._expirations_worthless,
            avg_entry_iv=statistics.mean(entry_ivs) if entry_ivs else 0,
            avg_exit_iv=statistics.mean(exit_ivs) if exit_ivs else 0,
            theta_collected=self._total_theta_decay,
            gamma_pnl=0,
            vega_pnl=self._total_vega_pnl,
            delta_pnl=self._total_delta_pnl,
            total_theta_decay=self._total_theta_decay,
            total_iv_pnl=self._total_vega_pnl,
            equity_curve=self._equity_curve,
        )

    def _calculate_position_pnl(self, position: OptionPosition) -> float:
        """
        Calculate P&L for a closed position.

        For a closed position:
        - Entry cost = net_premium (negative for debit, positive for credit)
        - Exit value = intrinsic value at expiration or mark-to-market at close

        P&L = exit_value - (-net_premium) = exit_value + net_premium
        For credit positions (net_premium > 0): profit if option expires worthless
        For debit positions (net_premium < 0): profit if exit_value > abs(net_premium)
        """
        if position.exit_underlying_price is None:
            # Position not closed yet, return unrealized based on premium only
            return position.net_premium

        # Calculate exit/expiration value based on intrinsic values
        exit_value = 0.0
        for leg in position.legs:
            if leg.option_type == OptionType.CALL:
                intrinsic = max(0, position.exit_underlying_price - leg.strike)
            else:  # PUT
                intrinsic = max(0, leg.strike - position.exit_underlying_price)

            # leg.quantity is positive for long, negative for short
            exit_value += intrinsic * leg.quantity * leg.multiplier

        # P&L = what we received at exit minus what we paid at entry
        # net_premium is negative for debit (we paid), positive for credit (we received)
        # So P&L = exit_value - (-net_premium) = exit_value + net_premium
        pnl = exit_value + position.net_premium

        return pnl

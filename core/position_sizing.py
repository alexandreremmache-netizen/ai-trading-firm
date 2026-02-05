"""
Position Sizing Module
======================

Advanced position sizing using Kelly Criterion and related methods.
Provides optimal position sizing with risk management constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

import numpy as np


# =============================================================================
# QUICK WIN #7: Turn-of-Month Effect
# =============================================================================
def get_turn_of_month_multiplier(date: datetime | None = None) -> float:
    """
    QUICK WIN #7: Return position size multiplier based on turn-of-month effect.

    Last 4 trading days + first 3 trading days of month show
    statistically higher returns due to institutional cash flows
    (pension fund rebalancing, monthly deposits, etc.).

    Returns:
        1.0 (normal) or 1.25 (turn of month)
    """
    if date is None:
        date = datetime.now()

    day = date.day

    # Get last day of month
    if date.month == 12:
        next_month = date.replace(year=date.year + 1, month=1, day=1)
    else:
        next_month = date.replace(month=date.month + 1, day=1)
    last_day = (next_month - timedelta(days=1)).day

    # First 3 days of month
    if day <= 3:
        return 1.25

    # Last 4 days of month
    if day >= last_day - 3:
        return 1.25

    return 1.0


logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methodology."""
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    VOL_TARGET = "vol_target"
    FIXED_FRACTIONAL = "fixed_fractional"
    EQUAL_WEIGHT = "equal_weight"
    MEAN_VARIANCE = "mean_variance"  # #P5
    RISK_PARITY = "risk_parity"  # #P5
    MIN_VARIANCE = "min_variance"  # #P5
    HRP = "hrp"  # Phase 3: Hierarchical Risk Parity
    RESAMPLED = "resampled"  # Phase 4: Resampled Efficiency (Michaud)
    TURNOVER_PENALIZED = "turnover_penalized"  # Phase 5: Transaction cost aware


# =============================================================================
# Phase 5.2: Transaction Cost Configuration
# =============================================================================

@dataclass
class TransactionCostConfig:
    """
    Configuration for transaction costs (Phase 5.2).

    Transaction costs are critical for portfolio optimization as they can
    significantly impact realized returns. This config supports:
    - Fixed costs (per trade)
    - Variable costs (proportional to trade value)
    - Market impact (proportional to trade size relative to volume)
    - Turnover penalty for portfolio rebalancing

    Default values are conservative estimates for institutional trading.
    """
    # Fixed cost per trade (e.g., commission)
    fixed_cost_per_trade: float = 0.0

    # Variable cost as basis points of trade value (e.g., 10 bps = 0.10%)
    variable_cost_bps: float = 10.0

    # Market impact coefficient (for larger trades)
    # Impact = coefficient * sqrt(trade_size / avg_daily_volume)
    market_impact_coefficient: float = 0.1

    # Turnover penalty lambda for optimization
    # Higher = stronger preference for current weights (less trading)
    turnover_penalty_lambda: float = 0.5

    # Minimum improvement required to trigger rebalance (as %)
    min_trade_improvement_pct: float = 0.3

    # Bid-ask spread by asset class (in bps)
    spread_by_asset_class: dict[str, float] = field(default_factory=lambda: {
        "equity": 5.0,
        "future": 2.0,
        "forex": 1.0,
        "commodity": 10.0,
        "etf": 3.0,
    })

    def get_total_cost_bps(self, asset_class: str = "equity") -> float:
        """Get total round-trip cost in basis points."""
        spread = self.spread_by_asset_class.get(asset_class, 5.0)
        return self.variable_cost_bps + spread

    def calculate_trade_cost(
        self,
        trade_value: float,
        asset_class: str = "equity",
        avg_daily_volume: float | None = None,
    ) -> float:
        """
        Calculate total cost for a trade.

        Args:
            trade_value: Absolute value of trade
            asset_class: Asset class for spread lookup
            avg_daily_volume: Average daily volume (for impact)

        Returns:
            Total cost in currency
        """
        # Fixed cost
        cost = self.fixed_cost_per_trade

        # Variable cost (commission + spread)
        total_bps = self.get_total_cost_bps(asset_class)
        cost += trade_value * (total_bps / 10000)

        # Market impact (if volume provided)
        if avg_daily_volume is not None and avg_daily_volume > 0:
            participation_rate = trade_value / avg_daily_volume
            impact = self.market_impact_coefficient * np.sqrt(participation_rate)
            cost += trade_value * impact

        return cost


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    symbol: str
    method: SizingMethod
    raw_fraction: float  # Raw Kelly or calculated fraction
    adjusted_fraction: float  # After constraints and adjustments
    position_size_pct: float  # As % of portfolio
    position_value: float  # In currency
    contracts: int | None = None  # For futures
    rationale: str = ""
    adjustments: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "method": self.method.value,
            "raw_fraction": self.raw_fraction,
            "adjusted_fraction": self.adjusted_fraction,
            "position_size_pct": self.position_size_pct,
            "position_value": self.position_value,
            "contracts": self.contracts,
            "rationale": self.rationale,
            "adjustments": self.adjustments,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StrategyStats:
    """
    Statistics for a trading strategy (used in Kelly calculation).

    IMPORTANT: avg_win and avg_loss must be RETURNS (percentages as decimals),
    NOT dollar P&L values. For example:
    - avg_win = 0.02 means 2% average winning trade
    - avg_loss = 0.01 means 1% average losing trade

    If you have dollar P&L, divide by position size to get returns.
    """
    win_rate: float  # Probability of winning trade (0.0 to 1.0)
    avg_win: float  # Average winning trade return as decimal (e.g., 0.02 = 2%)
    avg_loss: float  # Average losing trade return as decimal (positive number)
    volatility: float  # Strategy return volatility (annualized)
    sharpe_ratio: float = 0.0
    n_trades: int = 0

    def __post_init__(self):
        """Validate inputs are in expected ranges."""
        # Validate win_rate is a probability
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"win_rate must be between 0 and 1, got {self.win_rate}")

        # Warn if avg_win/avg_loss look like dollar amounts (> 1.0 = 100%)
        if self.avg_win > 1.0:
            import logging
            logging.getLogger(__name__).warning(
                f"avg_win={self.avg_win} > 1.0 (100%) - this looks like a dollar amount. "
                "Kelly formula expects returns as decimals (e.g., 0.02 for 2%)."
            )
        if self.avg_loss > 1.0:
            import logging
            logging.getLogger(__name__).warning(
                f"avg_loss={self.avg_loss} > 1.0 (100%) - this looks like a dollar amount. "
                "Kelly formula expects returns as decimals (e.g., 0.01 for 1%)."
            )

    @property
    def edge(self) -> float:
        """Calculate edge (expected return per trade)."""
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)

    @property
    def kelly_fraction(self) -> float:
        """
        Calculate full Kelly fraction for optimal position sizing.

        The Kelly criterion, developed by John L. Kelly Jr. at Bell Labs (1956),
        determines the optimal fraction of capital to risk on a series of
        favorable bets to maximize long-term geometric growth rate.

        Kelly formula: f* = (bp - q) / b

        Where:
            f* = optimal fraction of bankroll to wager
            b  = odds received on the bet (avg_win / avg_loss)
                 Also called "odds" or "win/loss ratio"
            p  = probability of winning (win_rate)
            q  = probability of losing (1 - p)

        Derivation (simplified):
            We want to maximize E[log(1 + f*X)] where X is the bet outcome.
            Taking the derivative and setting to zero:
            d/df[p*log(1+f*b) + q*log(1-f)] = 0
            p*b/(1+f*b) - q/(1-f) = 0
            Solving for f: f* = (bp - q) / b

        Example:
            Win rate = 60% (p=0.6), avg_win = 2%, avg_loss = 1%
            b = 0.02 / 0.01 = 2.0
            f* = (2.0 * 0.6 - 0.4) / 2.0 = 0.8 / 2.0 = 0.40 (40%)

        Properties of Kelly:
            - Maximizes long-term growth rate (geometric mean)
            - Never risks bankruptcy (f* is always < 100% for finite odds)
            - Drawdowns can be severe (half-Kelly often preferred in practice)

        Returns:
            Optimal fraction of capital to risk (0.0 to ~1.0)
            Returns 0.0 if inputs are invalid or expectancy is negative.

        Note:
            In practice, "half-Kelly" or "quarter-Kelly" is often used to
            reduce volatility at the cost of some growth rate.
        """
        # KELLY-P0-1: Guard against division by zero or near-zero avg_loss
        if self.avg_loss <= 0 or self.avg_loss < 1e-10:
            import logging
            logging.getLogger(__name__).warning(
                f"avg_loss is zero or near-zero ({self.avg_loss}), returning 0 for Kelly fraction"
            )
            return 0.0

        # Calculate b (odds): how much you win per unit risked
        b = self.avg_win / self.avg_loss
        p = self.win_rate  # Probability of winning
        q = 1 - p  # Probability of losing

        # Apply Kelly formula: f* = (b*p - q) / b
        # Equivalent to: f* = p - q/b
        kelly = (b * p - q) / b

        # KELLY-P0-2: Handle negative Kelly (negative expectancy strategy)
        # If f* < 0, the strategy has negative expected value - don't bet
        if kelly < 0:
            import logging
            logging.getLogger(__name__).warning(
                f"Negative Kelly fraction ({kelly:.4f}) indicates negative expectancy. "
                f"Strategy has negative edge: win_rate={p:.2%}, avg_win={self.avg_win:.4f}, "
                f"avg_loss={self.avg_loss:.4f}. Returning 0."
            )
            return 0.0

        return kelly

    @classmethod
    def from_dollar_pnl(
        cls,
        wins: list[float],
        losses: list[float],
        position_sizes: list[float],
        volatility: float = 0.15
    ) -> "StrategyStats":
        """
        Create StrategyStats from dollar P&L values.

        Converts dollar P&L to returns for proper Kelly calculation.

        Args:
            wins: List of winning trade P&L in dollars
            losses: List of losing trade P&L in dollars (positive numbers)
            position_sizes: List of position sizes for each trade
            volatility: Strategy volatility (default 15%)

        Returns:
            StrategyStats with properly calculated returns
        """
        if not wins and not losses:
            return cls(win_rate=0.5, avg_win=0.0, avg_loss=0.0, volatility=volatility)

        n_wins = len(wins)
        n_losses = len(losses)
        n_total = n_wins + n_losses

        win_rate = n_wins / n_total if n_total > 0 else 0.5

        # Convert to returns using position sizes
        # Assuming position_sizes align with trades (all wins then all losses)
        avg_pos_size = np.mean(position_sizes) if position_sizes else 1.0

        avg_win_return = np.mean(wins) / avg_pos_size if wins else 0.0
        avg_loss_return = np.mean(losses) / avg_pos_size if losses else 0.0

        # Calculate Sharpe from trade returns
        all_returns = [w / avg_pos_size for w in wins] + [-l / avg_pos_size for l in losses]
        sharpe = np.mean(all_returns) / np.std(all_returns) if all_returns and np.std(all_returns) > 0 else 0.0

        return cls(
            win_rate=win_rate,
            avg_win=avg_win_return,
            avg_loss=avg_loss_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            n_trades=n_total
        )


class PositionSizer:
    """
    Advanced position sizing calculator.

    Features:
    - Kelly Criterion with variants
    - Volatility targeting
    - Correlation-adjusted sizing
    - Risk limit constraints
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize position sizer.

        Args:
            config: Configuration with:
                - method: Default sizing method (default: "kelly")
                - kelly_fraction: Fractional Kelly multiplier (default: 0.25 = quarter-Kelly)
                - max_position_pct: Maximum position size (default: 3%)
                - min_position_pct: Minimum position size (default: 0.5%)
                - max_total_exposure_pct: Maximum total portfolio exposure (default: 50%)
                - vol_target: Target volatility (default: 15%)
                - correlation_discount: Discount for correlated positions (default: True)
                - max_kelly_raw: Maximum raw Kelly fraction before multiplier (default: 0.15 = 15%)
        """
        self._config = config or {}
        self._default_method = SizingMethod[self._config.get("method", "kelly").upper()]
        # IMPROVED: Use quarter-Kelly (0.25x) instead of half-Kelly for safer sizing
        self._kelly_fraction = self._config.get("kelly_fraction", 0.25)
        self._use_half_kelly = self._config.get("use_half_kelly", False)  # Deprecated, use kelly_fraction
        # IMPROVED: Reduce max position from 10% to 3% to limit concentration risk
        self._max_position_pct = self._config.get("max_position_pct", 3.0)
        self._min_position_pct = self._config.get("min_position_pct", 0.5)
        # NEW: Maximum total portfolio exposure (gross)
        self._max_total_exposure_pct = self._config.get("max_total_exposure_pct", 50.0)
        self._vol_target = self._config.get("vol_target", 0.15)  # 15% annual
        self._correlation_discount = self._config.get("correlation_discount", True)
        # NEW: Cap raw Kelly to prevent outsized positions even with good edge
        self._max_kelly_raw = self._config.get("max_kelly_raw", 0.15)  # 15% max raw Kelly

        # Strategy statistics cache
        self._strategy_stats: dict[str, StrategyStats] = {}

        # Correlation data
        self._correlations: dict[tuple[str, str], float] = {}

        logger.info(
            f"PositionSizer initialized: method={self._default_method.value}, "
            f"kelly_fraction={self._kelly_fraction}, max_position={self._max_position_pct}%, "
            f"max_total_exposure={self._max_total_exposure_pct}%, max_kelly_raw={self._max_kelly_raw}"
        )

    def update_strategy_stats(self, strategy: str, stats: StrategyStats) -> None:
        """
        Update statistics for a strategy.

        Args:
            strategy: Strategy name
            stats: Strategy statistics
        """
        self._strategy_stats[strategy] = stats
        logger.debug(f"Updated stats for {strategy}: win_rate={stats.win_rate:.2%}, kelly={stats.kelly_fraction:.2%}")

    def update_correlations(self, correlations: dict[tuple[str, str], float]) -> None:
        """
        Update correlation data.

        Args:
            correlations: Dictionary of (symbol1, symbol2) -> correlation
        """
        self._correlations = correlations

    def calculate_kelly_size(
        self,
        strategy: str,
        portfolio_value: float,
        kelly_variant: SizingMethod = SizingMethod.QUARTER_KELLY,
        current_exposure_pct: float = 0.0
    ) -> PositionSizeResult:
        """
        Calculate position size using Kelly Criterion with conservative constraints.

        IMPROVED money management:
        - Uses quarter-Kelly (0.25x) by default for safety
        - Caps raw Kelly at 15% before applying fraction
        - Enforces 3% max position size
        - Considers total portfolio exposure limit (50%)
        - Requires minimum 50 trades for statistical significance

        Args:
            strategy: Strategy name
            portfolio_value: Total portfolio value
            kelly_variant: Kelly variant to use
            current_exposure_pct: Current total portfolio exposure as % (0-100)

        Returns:
            PositionSizeResult with calculated size
        """
        stats = self._strategy_stats.get(strategy)
        if stats is None:
            logger.warning(f"No stats for strategy {strategy}, using minimum size")
            return PositionSizeResult(
                symbol=strategy,
                method=kelly_variant,
                raw_fraction=0,
                adjusted_fraction=self._min_position_pct / 100,
                position_size_pct=self._min_position_pct,
                position_value=portfolio_value * self._min_position_pct / 100,
                rationale="No strategy statistics available",
            )

        adjustments = []

        # IMPROVED: Require minimum trades for statistical significance
        MIN_TRADES_FOR_KELLY = 50
        if stats.n_trades < MIN_TRADES_FOR_KELLY:
            logger.warning(
                f"Insufficient trades for Kelly ({stats.n_trades}/{MIN_TRADES_FOR_KELLY}), "
                f"using minimum size for {strategy}"
            )
            return PositionSizeResult(
                symbol=strategy,
                method=kelly_variant,
                raw_fraction=0,
                adjusted_fraction=self._min_position_pct / 100,
                position_size_pct=self._min_position_pct,
                position_value=portfolio_value * self._min_position_pct / 100,
                rationale=f"Insufficient trades ({stats.n_trades}/{MIN_TRADES_FOR_KELLY})",
                adjustments=["Minimum size due to low sample count"],
            )

        # Calculate raw Kelly fraction
        kelly = stats.kelly_fraction
        adjustments.append(f"Raw Kelly: {kelly*100:.2f}%")

        # IMPROVED: Cap raw Kelly BEFORE applying fraction multiplier
        # This prevents outsized positions even with seemingly good edge
        if kelly > self._max_kelly_raw:
            kelly = self._max_kelly_raw
            adjustments.append(f"Raw Kelly capped at {self._max_kelly_raw*100:.1f}%")

        # Apply Kelly fraction (default: 0.25x = quarter-Kelly)
        adjusted = kelly * self._kelly_fraction
        adjustments.append(f"Applied {self._kelly_fraction}x Kelly ({adjusted*100:.2f}%)")

        # IMPROVED: Enforce strict max position size (3% default)
        max_position_frac = self._max_position_pct / 100
        if adjusted > max_position_frac:
            adjusted = max_position_frac
            adjustments.append(f"Capped at max position {self._max_position_pct}%")

        # NEW: Consider total exposure limit
        # If portfolio is already 40% exposed and limit is 50%, only allow 10% more
        if current_exposure_pct > 0:
            remaining_exposure_pct = max(0, self._max_total_exposure_pct - current_exposure_pct)
            remaining_exposure_frac = remaining_exposure_pct / 100
            if adjusted > remaining_exposure_frac:
                adjusted = remaining_exposure_frac
                adjustments.append(
                    f"Limited by total exposure ({current_exposure_pct:.1f}% used, "
                    f"{remaining_exposure_pct:.1f}% remaining)"
                )

        # Apply minimum floor
        if adjusted < self._min_position_pct / 100 and kelly > 0:
            adjusted = self._min_position_pct / 100
            adjustments.append(f"Floor at min {self._min_position_pct}%")

        if adjusted < 0:
            adjusted = 0
            adjustments.append("Negative Kelly, set to zero")

        position_value = portfolio_value * adjusted

        return PositionSizeResult(
            symbol=strategy,
            method=kelly_variant,
            raw_fraction=kelly,
            adjusted_fraction=adjusted,
            position_size_pct=adjusted * 100,
            position_value=position_value,
            rationale=f"Kelly: win_rate={stats.win_rate:.2%}, edge={stats.edge:.4f}, n_trades={stats.n_trades}",
            adjustments=adjustments,
        )

    def calculate_vol_target_size(
        self,
        symbol: str,
        portfolio_value: float,
        asset_volatility: float,
        target_vol: float | None = None,
        max_position_multiplier: float = 3.0
    ) -> PositionSizeResult:
        """
        Calculate position size using volatility targeting.

        Position size = (target_vol / asset_vol) * portfolio_value

        Args:
            symbol: Asset symbol
            portfolio_value: Total portfolio value
            asset_volatility: Asset's annualized volatility
            target_vol: Target portfolio volatility (default: config value)
            max_position_multiplier: Maximum leverage multiplier to prevent
                excessive positions in low volatility environments (default: 3.0)

        Returns:
            PositionSizeResult with calculated size
        """
        if target_vol is None:
            target_vol = self._vol_target

        adjustments = []

        if asset_volatility <= 0:
            logger.warning(f"Invalid volatility for {symbol}: {asset_volatility}")
            return PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.VOL_TARGET,
                raw_fraction=0,
                adjusted_fraction=0,
                position_size_pct=0,
                position_value=0,
                rationale="Invalid asset volatility",
            )

        # Calculate raw fraction
        raw_fraction = target_vol / asset_volatility
        adjusted = raw_fraction

        # SIZE-P1-1: Limit leverage in low volatility environments
        # This prevents excessive position sizes when asset volatility is unusually low
        if adjusted > max_position_multiplier:
            logger.warning(
                f"Vol target sizing for {symbol}: raw_fraction={raw_fraction:.2f} exceeds "
                f"max_position_multiplier={max_position_multiplier:.1f}. "
                f"Asset vol={asset_volatility:.2%} may be unusually low."
            )
            adjusted = max_position_multiplier
            adjustments.append(f"Capped at max leverage multiplier {max_position_multiplier:.1f}x")

        # Apply constraints
        if adjusted > self._max_position_pct / 100:
            adjusted = self._max_position_pct / 100
            adjustments.append(f"Capped at max {self._max_position_pct}%")

        if adjusted < self._min_position_pct / 100:
            adjusted = self._min_position_pct / 100
            adjustments.append(f"Floor at min {self._min_position_pct}%")

        position_value = portfolio_value * adjusted

        return PositionSizeResult(
            symbol=symbol,
            method=SizingMethod.VOL_TARGET,
            raw_fraction=raw_fraction,
            adjusted_fraction=adjusted,
            position_size_pct=adjusted * 100,
            position_value=position_value,
            rationale=f"Vol target: target={target_vol:.1%}, asset_vol={asset_volatility:.1%}",
            adjustments=adjustments,
        )

    def calculate_correlation_adjusted_size(
        self,
        symbol: str,
        base_size_pct: float,
        existing_positions: dict[str, float],
        portfolio_value: float
    ) -> PositionSizeResult:
        """
        Adjust position size based on correlation with existing positions.

        IMPROVED: More aggressive correlation discount to reduce concentration risk:
        - Starts discounting at 0.5 correlation (was 0.7)
        - Maximum discount of 70% (was 45%)
        - Considers cumulative correlation effect across multiple positions

        Args:
            symbol: Symbol to size
            base_size_pct: Base position size before adjustment
            existing_positions: Current positions (symbol -> value)
            portfolio_value: Total portfolio value

        Returns:
            PositionSizeResult with correlation-adjusted size
        """
        if not self._correlation_discount:
            return PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.KELLY,
                raw_fraction=base_size_pct / 100,
                adjusted_fraction=base_size_pct / 100,
                position_size_pct=base_size_pct,
                position_value=portfolio_value * base_size_pct / 100,
                rationale="Correlation adjustment disabled",
            )

        adjustments = []
        max_correlation = 0.0
        correlated_positions = []  # Track all correlated positions

        # Find all correlations with existing positions
        for existing_symbol, existing_value in existing_positions.items():
            if existing_value == 0:
                continue

            # Get correlation (check both orderings)
            corr = self._correlations.get((symbol, existing_symbol))
            if corr is None:
                corr = self._correlations.get((existing_symbol, symbol))
            if corr is None:
                continue

            # IMPROVED: Track all significant correlations
            if abs(corr) > 0.3:  # Track correlations > 0.3
                correlated_positions.append((existing_symbol, corr, existing_value))

            if abs(corr) > abs(max_correlation):
                max_correlation = corr

        # IMPROVED: Apply more aggressive correlation discount
        # Start discounting at 0.5 correlation (was 0.7)
        discount = 1.0
        if abs(max_correlation) > 0.5:
            # Linear discount: 0% at 0.5, up to 70% at 1.0
            discount = 1.0 - (abs(max_correlation) - 0.5) * 1.4
            discount = max(0.30, discount)  # Minimum 30% of base size (was 55%)
            adjustments.append(
                f"Correlation discount: {(1-discount)*100:.1f}% (max_corr={max_correlation:.2f})"
            )

        # IMPROVED: Additional cumulative discount for multiple correlated positions
        # Each additional correlated position reduces size further
        if len(correlated_positions) > 1:
            cumulative_discount = 1.0 - (len(correlated_positions) - 1) * 0.10
            cumulative_discount = max(0.5, cumulative_discount)  # Max 50% additional reduction
            discount *= cumulative_discount
            adjustments.append(
                f"Cumulative correlation: {len(correlated_positions)} correlated positions, "
                f"additional {(1-cumulative_discount)*100:.0f}% reduction"
            )

        adjusted_pct = base_size_pct * discount

        # Apply constraints
        if adjusted_pct > self._max_position_pct:
            adjusted_pct = self._max_position_pct
            adjustments.append(f"Capped at max {self._max_position_pct}%")

        if adjusted_pct < self._min_position_pct:
            adjusted_pct = self._min_position_pct
            adjustments.append(f"Floor at min {self._min_position_pct}%")

        return PositionSizeResult(
            symbol=symbol,
            method=SizingMethod.KELLY,
            raw_fraction=base_size_pct / 100,
            adjusted_fraction=adjusted_pct / 100,
            position_size_pct=adjusted_pct,
            position_value=portfolio_value * adjusted_pct / 100,
            rationale=f"Correlation-adjusted: base={base_size_pct:.1f}%, max_corr={max_correlation:.2f}, "
                      f"correlated_positions={len(correlated_positions)}",
            adjustments=adjustments,
        )

    def calculate_optimal_size(
        self,
        symbol: str,
        strategy: str,
        portfolio_value: float,
        asset_volatility: float | None = None,
        existing_positions: dict[str, float] | None = None,
        method: SizingMethod | None = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using configured method.

        This is the main entry point that combines all sizing logic.

        Args:
            symbol: Asset symbol
            strategy: Strategy name
            portfolio_value: Total portfolio value
            asset_volatility: Asset volatility (for vol targeting)
            existing_positions: Current positions (for correlation adjustment)
            method: Override default method

        Returns:
            PositionSizeResult with optimal size
        """
        if method is None:
            method = self._default_method

        # Start with base calculation
        if method in [SizingMethod.KELLY, SizingMethod.HALF_KELLY, SizingMethod.QUARTER_KELLY]:
            # Use Kelly
            variant = SizingMethod.HALF_KELLY if self._use_half_kelly else method
            result = self.calculate_kelly_size(strategy, portfolio_value, variant)

        elif method == SizingMethod.VOL_TARGET and asset_volatility is not None:
            # Use volatility targeting
            result = self.calculate_vol_target_size(
                symbol, portfolio_value, asset_volatility
            )

        elif method == SizingMethod.FIXED_FRACTIONAL:
            # Fixed fraction (simple)
            fraction = self._config.get("fixed_fraction", 0.05)
            result = PositionSizeResult(
                symbol=symbol,
                method=method,
                raw_fraction=fraction,
                adjusted_fraction=fraction,
                position_size_pct=fraction * 100,
                position_value=portfolio_value * fraction,
                rationale=f"Fixed {fraction*100:.1f}% allocation",
            )

        else:
            # Equal weight fallback
            result = PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.EQUAL_WEIGHT,
                raw_fraction=self._min_position_pct / 100,
                adjusted_fraction=self._min_position_pct / 100,
                position_size_pct=self._min_position_pct,
                position_value=portfolio_value * self._min_position_pct / 100,
                rationale="Equal weight fallback",
            )

        # Apply correlation adjustment if we have existing positions
        if existing_positions and self._correlation_discount:
            corr_result = self.calculate_correlation_adjusted_size(
                symbol=symbol,
                base_size_pct=result.position_size_pct,
                existing_positions=existing_positions,
                portfolio_value=portfolio_value,
            )

            # Merge adjustments
            corr_result.adjustments = result.adjustments + corr_result.adjustments
            result = corr_result

        # QUICK WIN #7: Apply turn-of-month multiplier
        if self._config.get("use_turn_of_month_boost", False):
            tom_multiplier = get_turn_of_month_multiplier()
            if tom_multiplier > 1.0:
                # Check if asset class is eligible (default: equities and futures)
                tom_asset_classes = self._config.get("tom_asset_classes", ["equity", "future"])
                # Simple heuristic: futures symbols are typically 2-3 uppercase letters
                is_future = len(symbol) <= 4 and symbol.isupper() and not symbol.startswith("$")

                if is_future or "equity" in tom_asset_classes:
                    original_pct = result.position_size_pct
                    boosted_pct = min(
                        original_pct * tom_multiplier,
                        self._max_position_pct  # Don't exceed max
                    )
                    boost_applied = boosted_pct / original_pct if original_pct > 0 else 1.0

                    # Create new result with boosted size
                    result = PositionSizeResult(
                        symbol=result.symbol,
                        method=result.method,
                        raw_fraction=result.raw_fraction,
                        adjusted_fraction=boosted_pct / 100,
                        position_size_pct=boosted_pct,
                        position_value=portfolio_value * boosted_pct / 100,
                        rationale=result.rationale,
                        adjustments=result.adjustments + [
                            f"TOM boost: {boost_applied:.0%} ({original_pct:.1f}% -> {boosted_pct:.1f}%)"
                        ],
                    )
                    logger.debug(
                        f"Turn-of-month boost applied to {symbol}: "
                        f"{original_pct:.1f}% -> {boosted_pct:.1f}%"
                    )

        return result

    def calculate_contracts(
        self,
        position_value: float,
        price: float,
        multiplier: float = 1.0
    ) -> int:
        """
        Calculate number of contracts for a given position value.

        Args:
            position_value: Target position value
            price: Current price
            multiplier: Contract multiplier

        Returns:
            Number of contracts (rounded down)
        """
        if price <= 0 or multiplier <= 0:
            return 0

        contract_value = price * multiplier
        contracts = int(position_value / contract_value)

        return max(0, contracts)

    def get_strategy_stats(self, strategy: str) -> StrategyStats | None:
        """Get stored statistics for a strategy."""
        return self._strategy_stats.get(strategy)

    def get_all_stats(self) -> dict[str, StrategyStats]:
        """Get all strategy statistics."""
        return dict(self._strategy_stats)

    def get_optimal_portfolio_weights(
        self,
        strategies: list[str],
        portfolio_value: float
    ) -> dict[str, float]:
        """
        Calculate optimal weights for multiple strategies.

        Uses normalized Kelly fractions.

        Args:
            strategies: List of strategy names
            portfolio_value: Total portfolio value

        Returns:
            Dictionary of strategy to weight (0-1)
        """
        kelly_fractions = {}

        for strategy in strategies:
            stats = self._strategy_stats.get(strategy)
            if stats and stats.kelly_fraction > 0:
                if self._use_half_kelly:
                    kelly_fractions[strategy] = stats.kelly_fraction * 0.5
                else:
                    kelly_fractions[strategy] = stats.kelly_fraction
            else:
                kelly_fractions[strategy] = self._min_position_pct / 100

        # Normalize if total exceeds 100%
        total = sum(kelly_fractions.values())
        if total > 1.0:
            kelly_fractions = {k: v / total for k, v in kelly_fractions.items()}

        return kelly_fractions

    def get_status(self) -> dict[str, Any]:
        """Get sizer status for monitoring."""
        return {
            "default_method": self._default_method.value,
            "kelly_fraction": self._kelly_fraction,
            "max_kelly_raw": self._max_kelly_raw,
            "max_position_pct": self._max_position_pct,
            "min_position_pct": self._min_position_pct,
            "max_total_exposure_pct": self._max_total_exposure_pct,
            "vol_target": self._vol_target,
            "correlation_discount": self._correlation_discount,
            "strategies_tracked": len(self._strategy_stats),
            "correlation_pairs": len(self._correlations),
            "supports_portfolio_optimization": True,  # #P5
            # Money management summary
            "money_management": {
                "kelly_type": "quarter-kelly" if self._kelly_fraction <= 0.25 else "fractional-kelly",
                "max_single_position": f"{self._max_position_pct}%",
                "max_total_exposure": f"{self._max_total_exposure_pct}%",
                "raw_kelly_cap": f"{self._max_kelly_raw*100}%",
            }
        }

    # =========================================================================
    # PORTFOLIO OPTIMIZATION (#P5)
    # =========================================================================

    def optimize_portfolio_mean_variance(
        self,
        symbols: list[str],
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        portfolio_value: float,
        target_return: float | None = None,
        risk_free_rate: float = 0.02
    ) -> dict[str, PositionSizeResult]:
        """
        Mean-variance portfolio optimization (Markowitz) (#P5).

        Finds the optimal portfolio weights that maximize the Sharpe ratio
        (or minimize variance for a target return).

        Uses quadratic programming to solve:
        - Max: (w'μ - rf) / sqrt(w'Σw)  [max Sharpe]
        - Or: Min: w'Σw  s.t. w'μ = target  [min var for target return]

        IMPORTANT (PM-11): For better estimation stability, callers should use
        a shrunk covariance matrix. The VaRCalculator.calculate_shrunk_covariance()
        method provides Ledoit-Wolf shrinkage which is recommended for:
        - Small sample sizes (< 10x the number of assets)
        - High-dimensional portfolios
        - Out-of-sample performance

        Example:
            from core.var_calculator import VaRCalculator
            var_calc = VaRCalculator()
            shrunk_cov = var_calc.calculate_shrunk_covariance(returns_df)
            results = sizer.optimize_portfolio_mean_variance(
                symbols, expected_returns, shrunk_cov, portfolio_value
            )

        Args:
            symbols: List of asset symbols
            expected_returns: Expected returns by symbol (annual)
            covariance_matrix: Covariance matrix (NxN numpy array).
                Recommended: Use shrunk covariance from VaRCalculator.
            portfolio_value: Total portfolio value
            target_return: Target portfolio return (None = max Sharpe)
            risk_free_rate: Risk-free rate for Sharpe calculation

        Returns:
            Dictionary of symbol to PositionSizeResult with optimal weights
        """
        n = len(symbols)
        if n == 0:
            return {}

        # Convert expected returns to array
        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])

        # Ensure covariance matrix is positive semi-definite
        try:
            # Add small regularization for numerical stability
            cov = covariance_matrix + np.eye(n) * 1e-8
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix not positive definite, applying fix")
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        if target_return is not None:
            # Minimize variance for target return
            weights = self._solve_min_variance(mu, cov, target_return)
            method_name = "mean_variance_target"
        else:
            # Maximize Sharpe ratio
            weights = self._solve_max_sharpe(mu, cov, risk_free_rate)
            method_name = "mean_variance_sharpe"

        # Apply constraints
        weights = self._apply_weight_constraints(weights)

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        logger.info(
            f"Mean-variance optimization: return={portfolio_return:.2%}, "
            f"vol={portfolio_vol:.2%}, sharpe={sharpe:.2f}"
        )

        # Create results
        results = {}
        for i, symbol in enumerate(symbols):
            w = weights[i]
            results[symbol] = PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.MEAN_VARIANCE,
                raw_fraction=w,
                adjusted_fraction=w,
                position_size_pct=w * 100,
                position_value=portfolio_value * w,
                rationale=f"MV optimization: E[r]={expected_returns.get(symbol, 0):.2%}, Sharpe={sharpe:.2f}",
                adjustments=[method_name],
            )

        return results

    def _solve_max_sharpe(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        rf: float
    ) -> np.ndarray:
        """
        Solve for maximum Sharpe ratio portfolio.

        Uses the analytical solution for the tangency portfolio:
        w = (Σ^-1)(μ - rf) / 1'(Σ^-1)(μ - rf)
        """
        n = len(mu)

        try:
            # Compute inverse of covariance
            cov_inv = np.linalg.inv(cov)

            # Excess returns
            excess_returns = mu - rf

            # Tangency portfolio weights (before normalization)
            raw_weights = cov_inv @ excess_returns

            # Normalize to sum to 1
            if np.sum(raw_weights) != 0:
                weights = raw_weights / np.sum(raw_weights)
            else:
                weights = np.ones(n) / n

            return weights

        except np.linalg.LinAlgError:
            logger.warning("Could not invert covariance matrix, using equal weights")
            return np.ones(n) / n

    def _solve_min_variance(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_return: float
    ) -> np.ndarray:
        """
        Solve for minimum variance portfolio with target return.

        Uses Lagrange multipliers for constrained optimization.
        """
        n = len(mu)

        try:
            cov_inv = np.linalg.inv(cov)

            # Calculate A, B, C, D for the efficient frontier
            ones = np.ones(n)
            A = ones @ cov_inv @ ones
            B = ones @ cov_inv @ mu
            C = mu @ cov_inv @ mu
            D = A * C - B * B

            if D == 0:
                return np.ones(n) / n

            # Optimal weights
            lambda1 = (C - B * target_return) / D
            lambda2 = (A * target_return - B) / D

            weights = lambda1 * (cov_inv @ ones) + lambda2 * (cov_inv @ mu)

            return weights

        except np.linalg.LinAlgError:
            logger.warning("Could not solve min variance, using equal weights")
            return np.ones(n) / n

    def optimize_portfolio_risk_parity(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        portfolio_value: float,
        risk_budgets: dict[str, float] | None = None
    ) -> dict[str, PositionSizeResult]:
        """
        Risk parity portfolio optimization (#P5).

        Allocates positions so each contributes equally to portfolio risk.
        Risk contribution_i = w_i * (Σw)_i / σ_p

        If risk_budgets provided, allocates according to those proportions
        instead of equal risk.

        Args:
            symbols: List of asset symbols
            covariance_matrix: Covariance matrix (NxN numpy array)
            portfolio_value: Total portfolio value
            risk_budgets: Optional risk budget allocation (sums to 1)

        Returns:
            Dictionary of symbol to PositionSizeResult
        """
        n = len(symbols)
        if n == 0:
            return {}

        # Default to equal risk budgets
        if risk_budgets is None:
            budgets = np.ones(n) / n
        else:
            budgets = np.array([risk_budgets.get(s, 1.0 / n) for s in symbols])
            budgets = budgets / np.sum(budgets)  # Normalize

        # Iterative algorithm to find risk parity weights
        weights = self._solve_risk_parity(covariance_matrix, budgets)

        # Apply constraints
        weights = self._apply_weight_constraints(weights)

        # Calculate risk contributions
        cov = covariance_matrix
        port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        marginal_risk = np.dot(cov, weights)
        risk_contributions = weights * marginal_risk / port_vol if port_vol > 0 else np.zeros(n)

        logger.info(
            f"Risk parity optimization: portfolio_vol={port_vol:.2%}, "
            f"n_assets={n}"
        )

        # Create results
        results = {}
        for i, symbol in enumerate(symbols):
            w = weights[i]
            rc = risk_contributions[i]
            results[symbol] = PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.RISK_PARITY,
                raw_fraction=w,
                adjusted_fraction=w,
                position_size_pct=w * 100,
                position_value=portfolio_value * w,
                rationale=f"Risk parity: risk_contrib={rc:.2%}, budget={budgets[i]:.2%}",
                adjustments=[f"risk_contribution={rc:.4f}"],
            )

        return results

    def _solve_risk_parity(
        self,
        cov: np.ndarray,
        budgets: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8
    ) -> np.ndarray:
        """
        Solve risk parity optimization using cyclical coordinate descent.

        This is the Spinu (2013) algorithm for risk budgeting.
        """
        n = len(budgets)

        # Initialize with equal weights
        weights = np.ones(n) / n

        for iteration in range(max_iter):
            weights_old = weights.copy()

            # Update each weight
            for i in range(n):
                # Marginal risk contribution
                sigma = np.sqrt(cov[i, i])
                cov_w = np.dot(cov[i, :], weights)

                if sigma > 0:
                    # Update weight to match risk budget
                    w_new = budgets[i] / sigma * np.sqrt(np.dot(weights, np.dot(cov, weights)))
                    w_new = w_new / (cov_w / (weights[i] * sigma) if weights[i] > 0 else 1)
                    weights[i] = max(0.001, w_new)  # Ensure positive

            # Normalize
            weights = weights / np.sum(weights)

            # Check convergence
            if np.max(np.abs(weights - weights_old)) < tol:
                logger.debug(f"Risk parity converged in {iteration + 1} iterations")
                break

        return weights

    def optimize_portfolio_min_variance(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        portfolio_value: float
    ) -> dict[str, PositionSizeResult]:
        """
        Minimum variance portfolio optimization (#P5).

        Finds the portfolio with lowest possible variance (risk).
        Useful for defensive positioning.

        Args:
            symbols: List of asset symbols
            covariance_matrix: Covariance matrix (NxN numpy array)
            portfolio_value: Total portfolio value

        Returns:
            Dictionary of symbol to PositionSizeResult
        """
        n = len(symbols)
        if n == 0:
            return {}

        try:
            cov_inv = np.linalg.inv(covariance_matrix)
            ones = np.ones(n)

            # Min variance weights: w = Σ^-1 * 1 / (1' * Σ^-1 * 1)
            raw_weights = cov_inv @ ones
            weights = raw_weights / np.sum(raw_weights)

        except np.linalg.LinAlgError:
            logger.warning("Could not invert covariance matrix, using equal weights")
            weights = np.ones(n) / n

        # Apply constraints
        weights = self._apply_weight_constraints(weights)

        # Calculate portfolio variance
        port_var = np.dot(weights, np.dot(covariance_matrix, weights))
        port_vol = np.sqrt(port_var)

        logger.info(f"Min variance optimization: portfolio_vol={port_vol:.2%}")

        # Create results
        results = {}
        for i, symbol in enumerate(symbols):
            w = weights[i]
            results[symbol] = PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.MIN_VARIANCE,
                raw_fraction=w,
                adjusted_fraction=w,
                position_size_pct=w * 100,
                position_value=portfolio_value * w,
                rationale=f"Min variance: port_vol={port_vol:.2%}",
            )

        return results

    # =========================================================================
    # Phase 3: HIERARCHICAL RISK PARITY (HRP)
    # =========================================================================

    def optimize_portfolio_hrp(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        portfolio_value: float,
        linkage_method: str = "single"
    ) -> dict[str, PositionSizeResult]:
        """
        Hierarchical Risk Parity (HRP) portfolio optimization (Phase 3).

        HRP is a portfolio allocation method developed by Marcos López de Prado
        that addresses limitations of traditional mean-variance optimization.

        Key benefits over MVO:
        - No matrix inversion required (numerically stable)
        - Works with singular covariance matrices
        - Produces well-diversified portfolios
        - Better out-of-sample performance
        - Handles high-dimensional asset spaces

        Algorithm:
        1. Build distance matrix from correlation
        2. Hierarchical clustering (single linkage by default)
        3. Quasi-diagonalization of covariance
        4. Recursive bisection for weight allocation

        Args:
            symbols: List of asset symbols
            covariance_matrix: Covariance matrix (NxN numpy array)
            portfolio_value: Total portfolio value
            linkage_method: Clustering method ('single', 'ward', 'complete', 'average')

        Returns:
            Dictionary of symbol to PositionSizeResult

        Reference: López de Prado, M. (2016). "Building Diversified Portfolios that
                   Outperform Out-of-Sample", Journal of Portfolio Management
        """
        n = len(symbols)
        if n == 0:
            return {}

        if n == 1:
            # Single asset, allocate 100%
            return {
                symbols[0]: PositionSizeResult(
                    symbol=symbols[0],
                    method=SizingMethod.HRP,
                    raw_fraction=1.0,
                    adjusted_fraction=1.0,
                    position_size_pct=100.0,
                    position_value=portfolio_value,
                    rationale="Single asset: 100% allocation",
                )
            }

        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
        except ImportError:
            logger.warning("scipy not available for HRP, falling back to equal weights")
            return self._equal_weight_fallback(symbols, portfolio_value)

        # Step 1: Convert covariance to correlation
        corr_matrix = self._cov_to_corr(covariance_matrix)

        # Step 2: Compute distance matrix from correlation
        # Distance = sqrt(0.5 * (1 - correlation))
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Step 3: Perform hierarchical clustering
        # Convert distance matrix to condensed form
        condensed_dist = squareform(dist_matrix, checks=False)

        # Handle potential NaN/Inf values
        condensed_dist = np.nan_to_num(condensed_dist, nan=1.0, posinf=1.0, neginf=0.0)

        # Perform hierarchical clustering
        link = linkage(condensed_dist, method=linkage_method)

        # Step 4: Get quasi-diagonalization ordering
        sorted_indices = leaves_list(link)

        # Step 5: Recursive bisection to get weights
        weights = self._hrp_recursive_bisection(covariance_matrix, sorted_indices)

        # Apply constraints
        weights = self._apply_weight_constraints(weights)

        # Calculate portfolio statistics
        port_var = np.dot(weights, np.dot(covariance_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))

        # Calculate diversification ratio
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0

        logger.info(
            f"HRP optimization: portfolio_vol={port_vol:.2%}, "
            f"diversification_ratio={div_ratio:.2f}, n_assets={n}"
        )

        # Create results
        results = {}
        for i, symbol in enumerate(symbols):
            w = weights[i]
            asset_vol = individual_vols[i]

            results[symbol] = PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.HRP,
                raw_fraction=w,
                adjusted_fraction=w,
                position_size_pct=w * 100,
                position_value=portfolio_value * w,
                rationale=f"HRP: asset_vol={asset_vol:.2%}, cluster_order={sorted_indices.tolist().index(i)}",
                adjustments=[f"diversification_ratio={div_ratio:.2f}"],
            )

        return results

    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-8  # Avoid division by zero

        corr = cov / np.outer(std, std)

        # Clean up numerical issues
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        return corr

    def _hrp_recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_indices: np.ndarray
    ) -> np.ndarray:
        """
        Recursive bisection step of HRP algorithm.

        Allocates weights by recursively splitting the sorted assets
        into clusters and allocating based on inverse variance.

        Args:
            cov: Covariance matrix
            sorted_indices: Quasi-diagonalization ordering from clustering

        Returns:
            Array of portfolio weights
        """
        n = len(sorted_indices)
        weights = np.ones(n)

        # List of clusters to process: (start_idx, end_idx, current_weight)
        clusters = [(0, n)]

        while clusters:
            start, end = clusters.pop(0)
            size = end - start

            if size == 1:
                continue

            # Split into two clusters
            mid = (start + end) // 2
            left_indices = sorted_indices[start:mid]
            right_indices = sorted_indices[mid:end]

            # Calculate cluster variances
            left_var = self._get_cluster_variance(cov, left_indices)
            right_var = self._get_cluster_variance(cov, right_indices)

            # Allocate based on inverse variance
            total_inv_var = 1/left_var + 1/right_var if left_var > 0 and right_var > 0 else 2.0
            left_weight = (1/left_var) / total_inv_var if left_var > 0 else 0.5
            right_weight = 1.0 - left_weight

            # Apply weights to cluster members
            for idx in range(start, mid):
                weights[sorted_indices[idx]] *= left_weight
            for idx in range(mid, end):
                weights[sorted_indices[idx]] *= right_weight

            # Add subclusters for further processing
            if mid - start > 1:
                clusters.append((start, mid))
            if end - mid > 1:
                clusters.append((mid, end))

        return weights

    def _get_cluster_variance(
        self,
        cov: np.ndarray,
        indices: np.ndarray
    ) -> float:
        """
        Calculate variance of a cluster (sub-portfolio).

        Uses inverse-variance weighting within the cluster.

        Args:
            cov: Full covariance matrix
            indices: Indices of assets in the cluster

        Returns:
            Cluster variance
        """
        if len(indices) == 0:
            return 1e-8

        if len(indices) == 1:
            return cov[indices[0], indices[0]]

        # Extract sub-covariance matrix
        sub_cov = cov[np.ix_(indices, indices)]

        # Inverse-variance weights within cluster
        inv_diag = 1.0 / np.diag(sub_cov)
        inv_diag = np.nan_to_num(inv_diag, nan=1.0, posinf=1.0)
        cluster_weights = inv_diag / np.sum(inv_diag)

        # Cluster variance
        cluster_var = np.dot(cluster_weights, np.dot(sub_cov, cluster_weights))

        return max(cluster_var, 1e-8)

    def _equal_weight_fallback(
        self,
        symbols: list[str],
        portfolio_value: float
    ) -> dict[str, PositionSizeResult]:
        """Fallback to equal weights when HRP cannot be computed."""
        n = len(symbols)
        w = 1.0 / n

        return {
            symbol: PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.EQUAL_WEIGHT,
                raw_fraction=w,
                adjusted_fraction=w,
                position_size_pct=w * 100,
                position_value=portfolio_value * w,
                rationale="Fallback to equal weights",
            )
            for symbol in symbols
        }

    # =========================================================================
    # Phase 4: Resampled Efficiency (Michaud)
    # =========================================================================

    def optimize_portfolio_resampled(
        self,
        symbols: list[str],
        returns_matrix: np.ndarray,
        portfolio_value: float,
        n_simulations: int = 500,
        risk_aversion: float = 2.5,
        min_weight: float = 0.0,
        max_weight: float = 0.40,
    ) -> dict[str, PositionSizeResult]:
        """
        Resampled Efficiency Frontier optimization (Michaud, 1998).

        Traditional mean-variance optimization suffers from estimation error:
        small changes in expected returns lead to large weight changes.

        Resampled efficiency addresses this by:
        1. Bootstrap sampling return/covariance estimates
        2. Running MVO on each sample
        3. Averaging the weights across samples

        This produces more stable, diversified portfolios with better
        out-of-sample performance.

        Research finding: Resampled efficiency reduces weight variance
        by 40-60% and improves out-of-sample Sharpe by 10-20%.

        Args:
            symbols: List of asset symbols
            returns_matrix: T x N matrix of historical returns
            portfolio_value: Total portfolio value
            n_simulations: Number of bootstrap samples (default: 500)
            risk_aversion: Risk aversion parameter for MVO
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            Dictionary of symbol to PositionSizeResult
        """
        n_assets = len(symbols)
        T, N = returns_matrix.shape

        if N != n_assets:
            logger.error(f"Returns matrix columns ({N}) != number of symbols ({n_assets})")
            return self._equal_weight_fallback(symbols, portfolio_value)

        if T < 30:
            logger.warning(f"Insufficient data for resampling ({T} < 30 observations)")
            return self._equal_weight_fallback(symbols, portfolio_value)

        # Collect weights from all simulations
        all_weights = np.zeros((n_simulations, n_assets))

        for sim in range(n_simulations):
            # Bootstrap sample: sample T observations with replacement
            sample_indices = np.random.choice(T, size=T, replace=True)
            sample_returns = returns_matrix[sample_indices, :]

            # Estimate parameters from bootstrap sample
            sample_mean = np.mean(sample_returns, axis=0)
            sample_cov = np.cov(sample_returns, rowvar=False)

            # Ensure covariance is valid
            if sample_cov.ndim == 0:
                sample_cov = np.array([[sample_cov]])
            elif sample_cov.ndim == 1:
                sample_cov = np.diag(sample_cov)

            # Run MVO on this sample
            weights = self._mean_variance_optimize(
                sample_mean, sample_cov, risk_aversion, min_weight, max_weight
            )

            all_weights[sim, :] = weights

        # Average weights across all simulations
        avg_weights = np.mean(all_weights, axis=0)

        # Normalize to sum to 1
        avg_weights = avg_weights / np.sum(avg_weights)

        # Calculate weight statistics for diagnostics
        weight_std = np.std(all_weights, axis=0)
        weight_cv = weight_std / (avg_weights + 1e-8)  # Coefficient of variation

        # Calculate full-sample statistics for comparison
        full_mean = np.mean(returns_matrix, axis=0)
        full_cov = np.cov(returns_matrix, rowvar=False)
        if full_cov.ndim == 0:
            full_cov = np.array([[full_cov]])
        elif full_cov.ndim == 1:
            full_cov = np.diag(full_cov)

        # Portfolio metrics
        port_return = np.dot(avg_weights, full_mean) * 252  # Annualized
        port_vol = np.sqrt(np.dot(avg_weights, np.dot(full_cov, avg_weights)) * 252)

        logger.info(
            f"Resampled optimization: n_sims={n_simulations}, "
            f"exp_return={port_return:.2%}, exp_vol={port_vol:.2%}, "
            f"avg_weight_cv={np.mean(weight_cv):.2f}"
        )

        # Create results
        results = {}
        for i, symbol in enumerate(symbols):
            w = avg_weights[i]

            results[symbol] = PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.RESAMPLED,
                raw_fraction=w,
                adjusted_fraction=w,
                position_size_pct=w * 100,
                position_value=portfolio_value * w,
                rationale=(
                    f"Resampled MVO: weight_std={weight_std[i]:.3f}, "
                    f"cv={weight_cv[i]:.2f}"
                ),
                adjustments=[
                    f"n_simulations={n_simulations}",
                    f"port_return={port_return:.2%}",
                    f"port_vol={port_vol:.2%}",
                ],
            )

        return results

    def _mean_variance_optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> np.ndarray:
        """
        Single mean-variance optimization for use in resampling.

        Solves: max w'*mu - (lambda/2)*w'*Sigma*w
        Subject to: sum(w) = 1, min <= w <= max

        Args:
            expected_returns: Expected return vector
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            min_weight: Minimum weight
            max_weight: Maximum weight

        Returns:
            Optimal weight vector
        """
        n = len(expected_returns)

        try:
            # Unconstrained solution
            sigma_inv = np.linalg.inv(covariance_matrix + np.eye(n) * 1e-8)
            raw_weights = (1 / risk_aversion) * sigma_inv @ expected_returns

            # Project to simplex and apply bounds
            weights = np.clip(raw_weights, min_weight, max_weight)

            # Normalize
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n) / n

            return weights

        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply position size constraints to weights.

        - Ensures non-negative (long-only)
        - Ensures each position is within min/max limits
        - Normalizes to sum to 1 (or less if constraints bind)
        """
        n = len(weights)

        # Ensure non-negative (long only)
        weights = np.maximum(weights, 0)

        # Apply max constraint
        max_weight = self._max_position_pct / 100
        weights = np.minimum(weights, max_weight)

        # Apply min constraint (set small weights to 0)
        min_weight = self._min_position_pct / 100
        weights[weights < min_weight / 2] = 0  # Remove very small positions

        # Re-normalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(n) / n

        return weights

    # =========================================================================
    # DRAWDOWN-BASED SIZING (P2)
    # =========================================================================

    def calculate_drawdown_adjusted_size(
        self,
        symbol: str,
        base_size_pct: float,
        portfolio_value: float,
        current_drawdown: float,
        max_acceptable_drawdown: float = 0.10,  # IMPROVED: Reduced from 20% to 10%
        drawdown_sensitivity: float = 1.5  # IMPROVED: Start reducing earlier (was 2.0)
    ) -> PositionSizeResult:
        """
        Adjust position size based on current portfolio drawdown.

        IMPROVED: More aggressive drawdown protection:
        - Starts reducing at ANY drawdown (not just above threshold)
        - Halts new positions at 10% drawdown (was 20%)
        - More gradual reduction curve (sensitivity 1.5 vs 2.0)
        - Tiered response: warning at 3%, reduce at 5%, halt at 10%

        Formula: adjusted_size = base_size * (1 - (drawdown / max_dd) ^ sensitivity)

        Args:
            symbol: Asset symbol
            base_size_pct: Base position size as percentage
            portfolio_value: Total portfolio value
            current_drawdown: Current drawdown as decimal (e.g., 0.10 = 10%)
            max_acceptable_drawdown: Maximum acceptable drawdown (default: 10%)
            drawdown_sensitivity: How aggressively to reduce size (default: 1.5)

        Returns:
            PositionSizeResult with drawdown-adjusted size
        """
        adjustments = []

        # Ensure drawdown is positive
        dd = abs(current_drawdown)

        # IMPROVED: Tiered drawdown response thresholds
        DRAWDOWN_WARNING = 0.03   # 3%: Start logging warnings
        DRAWDOWN_REDUCE = 0.05   # 5%: Actively reduce position sizes
        DRAWDOWN_SEVERE = 0.08   # 8%: Minimal new positions
        DRAWDOWN_HALT = 0.10     # 10%: Halt all new positions

        if dd <= 0:
            # No drawdown, use base size
            return PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.VOL_TARGET,
                raw_fraction=base_size_pct / 100,
                adjusted_fraction=base_size_pct / 100,
                position_size_pct=base_size_pct,
                position_value=portfolio_value * base_size_pct / 100,
                rationale="No drawdown, using base size",
                adjustments=["drawdown_adjustment=1.0"],
            )

        # IMPROVED: Tiered response based on drawdown severity
        if dd >= DRAWDOWN_HALT:
            # HALT: No new positions
            adjustments.append(f"HALTED: Drawdown {dd:.1%} >= halt threshold {DRAWDOWN_HALT:.0%}")
            return PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.VOL_TARGET,
                raw_fraction=base_size_pct / 100,
                adjusted_fraction=0.0,
                position_size_pct=0.0,
                position_value=0.0,
                rationale=f"Drawdown halt: {dd:.1%} exceeds {DRAWDOWN_HALT:.0%} limit",
                adjustments=adjustments,
            )

        if dd >= DRAWDOWN_SEVERE:
            # SEVERE: Maximum 20% of base size
            reduction_factor = 0.20
            adjustments.append(f"SEVERE drawdown ({dd:.1%}): max 20% of base size")
        elif dd >= DRAWDOWN_REDUCE:
            # REDUCE: 20-60% of base size based on drawdown
            # Linear interpolation from 60% at 5% dd to 20% at 8% dd
            reduction_factor = 0.60 - (dd - DRAWDOWN_REDUCE) / (DRAWDOWN_SEVERE - DRAWDOWN_REDUCE) * 0.40
            adjustments.append(f"REDUCE mode ({dd:.1%}): {reduction_factor*100:.0f}% of base size")
        elif dd >= DRAWDOWN_WARNING:
            # WARNING: 60-90% of base size
            # Linear interpolation from 90% at 3% dd to 60% at 5% dd
            reduction_factor = 0.90 - (dd - DRAWDOWN_WARNING) / (DRAWDOWN_REDUCE - DRAWDOWN_WARNING) * 0.30
            adjustments.append(f"WARNING mode ({dd:.1%}): {reduction_factor*100:.0f}% of base size")
        else:
            # Small drawdown: 90-100% of base size
            reduction_factor = 1.0 - (dd / DRAWDOWN_WARNING) * 0.10
            adjustments.append(f"Minor drawdown ({dd:.1%}): {reduction_factor*100:.0f}% of base size")

        # Apply reduction
        adjusted_pct = base_size_pct * reduction_factor

        # Apply constraints
        if adjusted_pct > self._max_position_pct:
            adjusted_pct = self._max_position_pct
            adjustments.append(f"Capped at max {self._max_position_pct}%")

        if adjusted_pct < self._min_position_pct and reduction_factor > 0:
            adjusted_pct = self._min_position_pct
            adjustments.append(f"Floor at min {self._min_position_pct}%")

        return PositionSizeResult(
            symbol=symbol,
            method=SizingMethod.VOL_TARGET,
            raw_fraction=base_size_pct / 100,
            adjusted_fraction=adjusted_pct / 100,
            position_size_pct=adjusted_pct,
            position_value=portfolio_value * adjusted_pct / 100,
            rationale=f"Drawdown-adjusted: base={base_size_pct:.1f}%, dd={dd:.1%}, factor={reduction_factor:.2f}",
            adjustments=adjustments,
        )

    def calculate_volatility_targeted_size(
        self,
        symbol: str,
        portfolio_value: float,
        asset_volatility: float,
        target_contribution_vol: float,
        current_portfolio_vol: float | None = None,
        max_vol_contribution: float = 0.05
    ) -> PositionSizeResult:
        """
        Calculate position size targeting specific volatility contribution (P2).

        Sizes position so its marginal contribution to portfolio volatility
        equals the target. More sophisticated than simple vol targeting.

        Args:
            symbol: Asset symbol
            portfolio_value: Total portfolio value
            asset_volatility: Asset's annualized volatility
            target_contribution_vol: Target volatility contribution (e.g., 0.02 = 2%)
            current_portfolio_vol: Current portfolio volatility (optional)
            max_vol_contribution: Maximum vol contribution per position (default: 5%)

        Returns:
            PositionSizeResult with volatility-targeted size
        """
        adjustments = []

        if asset_volatility <= 0:
            return PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.VOL_TARGET,
                raw_fraction=0,
                adjusted_fraction=0,
                position_size_pct=0,
                position_value=0,
                rationale="Invalid asset volatility",
            )

        # Basic calculation: weight = target_vol / asset_vol
        raw_weight = target_contribution_vol / asset_volatility
        adjusted_weight = raw_weight

        # Cap vol contribution
        if target_contribution_vol > max_vol_contribution:
            adjusted_weight = max_vol_contribution / asset_volatility
            adjustments.append(f"Vol contribution capped at {max_vol_contribution:.1%}")

        # If we have portfolio context, adjust for correlation
        if current_portfolio_vol is not None and current_portfolio_vol > 0:
            # Simple adjustment: reduce if adding to high vol portfolio
            vol_headroom = max(0, self._vol_target - current_portfolio_vol)
            if vol_headroom < target_contribution_vol:
                adjusted_weight = vol_headroom / asset_volatility
                adjustments.append(f"Reduced for portfolio vol headroom: {vol_headroom:.2%}")

        # Apply standard constraints
        if adjusted_weight > self._max_position_pct / 100:
            adjusted_weight = self._max_position_pct / 100
            adjustments.append(f"Capped at max {self._max_position_pct}%")

        if adjusted_weight < self._min_position_pct / 100:
            adjusted_weight = self._min_position_pct / 100
            adjustments.append(f"Floor at min {self._min_position_pct}%")

        return PositionSizeResult(
            symbol=symbol,
            method=SizingMethod.VOL_TARGET,
            raw_fraction=raw_weight,
            adjusted_fraction=adjusted_weight,
            position_size_pct=adjusted_weight * 100,
            position_value=portfolio_value * adjusted_weight,
            rationale=f"Vol-targeted: target_contrib={target_contribution_vol:.2%}, asset_vol={asset_volatility:.2%}",
            adjustments=adjustments,
        )

    def calculate_risk_parity_weights(
        self,
        symbols: list[str],
        volatilities: dict[str, float],
        correlations: dict[tuple[str, str], float] | None = None
    ) -> dict[str, float]:
        """
        Calculate risk parity weights from volatilities (P2 simplified version).

        Simpler than full optimization - uses inverse volatility weighting
        with optional correlation adjustment.

        Args:
            symbols: List of symbols
            volatilities: Symbol volatilities
            correlations: Optional pairwise correlations

        Returns:
            Dictionary of symbol to weight (sums to 1.0)
        """
        if not symbols or not volatilities:
            return {}

        # Get volatilities for all symbols
        vols = []
        valid_symbols = []
        for s in symbols:
            vol = volatilities.get(s)
            if vol is not None and vol > 0:
                vols.append(vol)
                valid_symbols.append(s)

        if not vols:
            return {s: 1.0 / len(symbols) for s in symbols}

        vols = np.array(vols)

        # Inverse volatility weighting (basic risk parity)
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)

        # Optional correlation adjustment
        if correlations is not None and len(valid_symbols) > 1:
            # Build correlation matrix
            n = len(valid_symbols)
            corr_matrix = np.eye(n)
            for i, s1 in enumerate(valid_symbols):
                for j, s2 in enumerate(valid_symbols):
                    if i != j:
                        corr = correlations.get((s1, s2)) or correlations.get((s2, s1))
                        if corr is not None:
                            corr_matrix[i, j] = corr

            # Build covariance matrix
            vol_diag = np.diag(vols)
            cov_matrix = vol_diag @ corr_matrix @ vol_diag

            # Iteratively adjust weights for equal risk contribution
            for _ in range(10):  # Simple iteration
                port_vol = np.sqrt(weights @ cov_matrix @ weights)
                if port_vol <= 0:
                    break

                marginal_risk = cov_matrix @ weights
                risk_contrib = weights * marginal_risk / port_vol

                # Target equal contribution
                target_contrib = port_vol / n
                adjustment = target_contrib / (risk_contrib + 1e-10)
                adjustment = np.clip(adjustment, 0.5, 2.0)  # Limit adjustment

                weights = weights * adjustment
                weights = weights / np.sum(weights)

        return {valid_symbols[i]: float(weights[i]) for i in range(len(valid_symbols))}

    def get_efficient_frontier(
        self,
        symbols: list[str],
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        n_points: int = 20,
        risk_free_rate: float = 0.02
    ) -> list[dict]:
        """
        Calculate the efficient frontier (#P5).

        Returns a series of optimal portfolios from minimum variance
        to maximum return.

        Args:
            symbols: List of asset symbols
            expected_returns: Expected returns by symbol
            covariance_matrix: Covariance matrix
            n_points: Number of points on the frontier
            risk_free_rate: Risk-free rate

        Returns:
            List of dicts with return, risk, sharpe, weights for each point
        """
        n = len(symbols)
        if n == 0:
            return []

        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])

        # Find min and max possible returns
        min_return = np.min(mu)
        max_return = np.max(mu)

        frontier = []
        target_returns = np.linspace(min_return, max_return, n_points)

        for target in target_returns:
            weights = self._solve_min_variance(mu, covariance_matrix, target)
            weights = self._apply_weight_constraints(weights)

            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

            frontier.append({
                "target_return": target,
                "actual_return": port_return,
                "volatility": port_vol,
                "sharpe_ratio": sharpe,
                "weights": {symbols[i]: weights[i] for i in range(n)},
            })

        return frontier

    # =========================================================================
    # Phase 5.2: Transaction Cost Aware Optimization
    # =========================================================================

    def optimize_portfolio_turnover_penalized(
        self,
        symbols: list[str],
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        portfolio_value: float,
        current_weights: dict[str, float] | None = None,
        cost_config: TransactionCostConfig | None = None,
        risk_aversion: float = 2.5,
    ) -> dict[str, PositionSizeResult]:
        """
        Transaction cost aware portfolio optimization (Phase 5.2).

        Extends mean-variance optimization to account for transaction costs.
        This is critical for practical portfolio management where frequent
        rebalancing can erode returns.

        The objective function is:
        max: w'μ - (λ/2)w'Σw - γ||w - w_current||₁ * costs

        Where:
        - w'μ = expected return
        - (λ/2)w'Σw = risk penalty
        - γ||w - w_current||₁ = turnover penalty weighted by trading costs

        Key features:
        - Penalizes deviations from current portfolio (reduces turnover)
        - Accounts for asset-class specific trading costs
        - Implements minimum trade threshold (don't rebalance for tiny improvements)
        - Returns no-trade recommendation if costs exceed benefits

        Research finding: Transaction costs of 50bps round-trip can reduce
        optimal rebalancing frequency by 40-60%.

        Args:
            symbols: List of asset symbols
            expected_returns: Expected returns by symbol (annual)
            covariance_matrix: Covariance matrix (NxN numpy array)
            portfolio_value: Total portfolio value
            current_weights: Current portfolio weights (default: equal weight)
            cost_config: Transaction cost configuration
            risk_aversion: Risk aversion parameter

        Returns:
            Dictionary of symbol to PositionSizeResult with optimal weights
        """
        n = len(symbols)
        if n == 0:
            return {}

        # Default to equal weights if no current portfolio
        if current_weights is None:
            current_weights = {s: 1.0 / n for s in symbols}

        # Default cost config
        if cost_config is None:
            cost_config = TransactionCostConfig()

        # Convert to arrays
        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])
        w_current = np.array([current_weights.get(s, 0.0) for s in symbols])

        # Ensure covariance matrix is valid
        cov = covariance_matrix + np.eye(n) * 1e-8

        # Calculate optimal weights without turnover penalty (baseline)
        try:
            cov_inv = np.linalg.inv(cov)
            w_unconstrained = (1 / risk_aversion) * cov_inv @ mu
            w_unconstrained = np.clip(w_unconstrained, 0, None)
            if np.sum(w_unconstrained) > 0:
                w_unconstrained = w_unconstrained / np.sum(w_unconstrained)
            else:
                w_unconstrained = np.ones(n) / n
        except np.linalg.LinAlgError:
            w_unconstrained = np.ones(n) / n

        # Calculate turnover and trading costs
        turnover = np.abs(w_unconstrained - w_current)
        total_turnover_pct = np.sum(turnover)

        # Estimate total trading cost
        avg_cost_bps = cost_config.get_total_cost_bps()
        total_trade_value = total_turnover_pct * portfolio_value
        total_cost = cost_config.calculate_trade_cost(total_trade_value)
        cost_as_return = total_cost / portfolio_value if portfolio_value > 0 else 0

        # Calculate benefit of rebalancing (expected improvement)
        new_return = np.dot(w_unconstrained, mu)
        old_return = np.dot(w_current, mu)
        improvement = new_return - old_return

        # Check minimum improvement threshold
        min_improvement = cost_config.min_trade_improvement_pct / 100

        # Decision: rebalance or not?
        if improvement < cost_as_return + min_improvement:
            # Don't rebalance - cost exceeds benefit
            logger.info(
                f"Turnover-penalized: No rebalance. "
                f"Improvement={improvement:.4f}, Cost={cost_as_return:.4f}, "
                f"Min threshold={min_improvement:.4f}"
            )

            # Return current weights
            results = {}
            for i, symbol in enumerate(symbols):
                w = w_current[i]
                results[symbol] = PositionSizeResult(
                    symbol=symbol,
                    method=SizingMethod.TURNOVER_PENALIZED,
                    raw_fraction=w_unconstrained[i],
                    adjusted_fraction=w,
                    position_size_pct=w * 100,
                    position_value=portfolio_value * w,
                    rationale=f"No rebalance: cost ({cost_as_return:.2%}) > benefit ({improvement:.2%})",
                    adjustments=[
                        "no_trade_decision",
                        f"turnover={total_turnover_pct:.2%}",
                        f"cost_bps={avg_cost_bps:.1f}",
                    ],
                )
            return results

        # Rebalance with turnover penalty
        # Use iterative approach to find optimal weights with penalty
        lambda_turnover = cost_config.turnover_penalty_lambda

        weights = self._solve_turnover_penalized_mvo(
            mu, cov, w_current, risk_aversion, lambda_turnover, cost_config
        )

        # Apply constraints
        weights = self._apply_weight_constraints(weights)

        # Calculate final metrics
        new_turnover = np.abs(weights - w_current)
        final_turnover_pct = np.sum(new_turnover)
        final_trade_value = final_turnover_pct * portfolio_value
        final_cost = cost_config.calculate_trade_cost(final_trade_value)

        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        net_improvement = (port_return - np.dot(w_current, mu)) - final_cost / portfolio_value

        logger.info(
            f"Turnover-penalized optimization: turnover={final_turnover_pct:.2%}, "
            f"cost=${final_cost:.2f}, net_improvement={net_improvement:.4f}"
        )

        # Create results
        results = {}
        for i, symbol in enumerate(symbols):
            w = weights[i]
            trade_size = abs(w - w_current[i])

            results[symbol] = PositionSizeResult(
                symbol=symbol,
                method=SizingMethod.TURNOVER_PENALIZED,
                raw_fraction=w_unconstrained[i],
                adjusted_fraction=w,
                position_size_pct=w * 100,
                position_value=portfolio_value * w,
                rationale=f"TC-aware: trade={trade_size:.2%}, port_vol={port_vol:.2%}",
                adjustments=[
                    f"turnover_penalty={lambda_turnover:.2f}",
                    f"trade_cost=${final_cost:.2f}",
                    f"net_improvement={net_improvement:.4f}",
                ],
            )

        return results

    def _solve_turnover_penalized_mvo(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w_current: np.ndarray,
        risk_aversion: float,
        lambda_turnover: float,
        cost_config: TransactionCostConfig,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Solve MVO with turnover penalty using proximal gradient descent.

        The L1 penalty on turnover is non-differentiable, so we use
        soft-thresholding (proximal operator).

        Args:
            mu: Expected returns
            cov: Covariance matrix
            w_current: Current weights
            risk_aversion: Risk aversion
            lambda_turnover: Turnover penalty weight
            cost_config: Cost configuration
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Optimal weights
        """
        n = len(mu)
        weights = w_current.copy()

        # Cost per unit turnover (in return space)
        cost_bps = cost_config.get_total_cost_bps()
        cost_per_unit = cost_bps / 10000  # Convert to decimal

        # Step size (should be < 1 / max_eigenvalue)
        max_eig = np.max(np.abs(np.linalg.eigvals(cov)))
        step = 0.5 / (risk_aversion * max_eig + 1e-8)

        for iteration in range(max_iter):
            weights_old = weights.copy()

            # Gradient of quadratic part: -μ + λ*Σ*w
            grad = -mu + risk_aversion * cov @ weights

            # Gradient step
            weights_temp = weights - step * grad

            # Proximal step (soft-thresholding for L1 penalty)
            threshold = step * lambda_turnover * cost_per_unit
            diff = weights_temp - w_current

            # Soft-thresholding: shrink towards current weights
            shrunk = np.sign(diff) * np.maximum(np.abs(diff) - threshold, 0)
            weights = w_current + shrunk

            # Project to simplex (sum to 1, non-negative)
            weights = np.clip(weights, 0, None)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n) / n

            # Check convergence
            if np.max(np.abs(weights - weights_old)) < tol:
                logger.debug(f"Turnover-penalized MVO converged in {iteration + 1} iterations")
                break

        return weights

    def calculate_rebalancing_threshold(
        self,
        symbols: list[str],
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        cost_config: TransactionCostConfig | None = None,
        expected_returns: dict[str, float] | None = None,
        holding_period_days: int = 30,
    ) -> dict[str, Any]:
        """
        Calculate whether rebalancing is worthwhile given costs (Phase 5.2).

        This provides a decision framework for when to rebalance, considering:
        - Trading costs (commissions, spread, market impact)
        - Expected drift if not rebalancing
        - Time horizon until next rebalance

        Args:
            symbols: Asset symbols
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            cost_config: Transaction cost configuration
            expected_returns: Expected returns (for benefit calculation)
            holding_period_days: Expected days until next rebalance

        Returns:
            Dictionary with recommendation and analysis
        """
        if cost_config is None:
            cost_config = TransactionCostConfig()

        n = len(symbols)

        # Calculate turnover
        turnover_by_asset = {}
        total_turnover = 0.0

        for symbol in symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            diff = abs(target - current)
            turnover_by_asset[symbol] = diff
            total_turnover += diff

        # Calculate trading costs
        trade_value = total_turnover * portfolio_value
        total_cost = cost_config.calculate_trade_cost(trade_value)
        cost_as_pct = (total_cost / portfolio_value) * 100 if portfolio_value > 0 else 0

        # Calculate expected benefit (if returns provided)
        expected_benefit = 0.0
        if expected_returns:
            current_return = sum(
                current_weights.get(s, 0) * expected_returns.get(s, 0)
                for s in symbols
            )
            target_return = sum(
                target_weights.get(s, 0) * expected_returns.get(s, 0)
                for s in symbols
            )
            # Annualized benefit, adjusted for holding period
            expected_benefit = (target_return - current_return) * (holding_period_days / 365)

        # Make recommendation
        benefit_exceeds_cost = expected_benefit > (total_cost / portfolio_value)
        min_threshold_met = total_turnover > (cost_config.min_trade_improvement_pct / 100)

        should_rebalance = benefit_exceeds_cost or min_threshold_met

        return {
            "recommendation": "REBALANCE" if should_rebalance else "HOLD",
            "total_turnover_pct": total_turnover * 100,
            "turnover_by_asset": turnover_by_asset,
            "estimated_cost_dollars": total_cost,
            "estimated_cost_pct": cost_as_pct,
            "expected_benefit_pct": expected_benefit * 100,
            "cost_benefit_ratio": total_cost / (expected_benefit * portfolio_value) if expected_benefit > 0 else float('inf'),
            "min_threshold_pct": cost_config.min_trade_improvement_pct,
            "holding_period_days": holding_period_days,
        }

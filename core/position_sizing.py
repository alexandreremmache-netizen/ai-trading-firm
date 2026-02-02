"""
Position Sizing Module
======================

Advanced position sizing using Kelly Criterion and related methods.
Provides optimal position sizing with risk management constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methodology."""
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    VOL_TARGET = "vol_target"
    FIXED_FRACTIONAL = "fixed_fractional"
    EQUAL_WEIGHT = "equal_weight"


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
        Calculate full Kelly fraction.

        Kelly formula: f* = (bp - q) / b
        where:
            b = avg_win / avg_loss (win/loss ratio)
            p = win_rate
            q = 1 - p (loss rate)

        Returns the optimal fraction of capital to risk.
        """
        if self.avg_loss == 0:
            return 0.0

        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = 1 - p

        kelly = (b * p - q) / b

        return max(0, kelly)  # Never negative

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
                - use_half_kelly: Use half Kelly (default: True)
                - max_position_pct: Maximum position size (default: 10%)
                - min_position_pct: Minimum position size (default: 1%)
                - vol_target: Target volatility (default: 15%)
                - correlation_discount: Discount for correlated positions (default: True)
        """
        self._config = config or {}
        self._default_method = SizingMethod[self._config.get("method", "kelly").upper()]
        self._use_half_kelly = self._config.get("use_half_kelly", True)
        self._max_position_pct = self._config.get("max_position_pct", 10.0)
        self._min_position_pct = self._config.get("min_position_pct", 1.0)
        self._vol_target = self._config.get("vol_target", 0.15)  # 15% annual
        self._correlation_discount = self._config.get("correlation_discount", True)

        # Strategy statistics cache
        self._strategy_stats: dict[str, StrategyStats] = {}

        # Correlation data
        self._correlations: dict[tuple[str, str], float] = {}

        logger.info(
            f"PositionSizer initialized: method={self._default_method.value}, "
            f"half_kelly={self._use_half_kelly}, max={self._max_position_pct}%"
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
        kelly_variant: SizingMethod = SizingMethod.HALF_KELLY
    ) -> PositionSizeResult:
        """
        Calculate position size using Kelly Criterion.

        Args:
            strategy: Strategy name
            portfolio_value: Total portfolio value
            kelly_variant: Kelly variant to use

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

        # Calculate Kelly fraction
        kelly = stats.kelly_fraction
        adjustments = []

        # Apply Kelly variant
        if kelly_variant == SizingMethod.HALF_KELLY:
            adjusted = kelly * 0.5
            adjustments.append("Applied half-Kelly (0.5x)")
        elif kelly_variant == SizingMethod.QUARTER_KELLY:
            adjusted = kelly * 0.25
            adjustments.append("Applied quarter-Kelly (0.25x)")
        else:
            adjusted = kelly

        # Apply constraints
        if adjusted > self._max_position_pct / 100:
            adjusted = self._max_position_pct / 100
            adjustments.append(f"Capped at max {self._max_position_pct}%")

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
            rationale=f"Kelly: win_rate={stats.win_rate:.2%}, edge={stats.edge:.4f}",
            adjustments=adjustments,
        )

    def calculate_vol_target_size(
        self,
        symbol: str,
        portfolio_value: float,
        asset_volatility: float,
        target_vol: float | None = None
    ) -> PositionSizeResult:
        """
        Calculate position size using volatility targeting.

        Position size = (target_vol / asset_vol) * portfolio_value

        Args:
            symbol: Asset symbol
            portfolio_value: Total portfolio value
            asset_volatility: Asset's annualized volatility
            target_vol: Target portfolio volatility (default: config value)

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

        Reduces size if highly correlated with existing holdings.

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

        # Find highest correlation with existing positions
        for existing_symbol, existing_value in existing_positions.items():
            if existing_value == 0:
                continue

            # Get correlation (check both orderings)
            corr = self._correlations.get((symbol, existing_symbol))
            if corr is None:
                corr = self._correlations.get((existing_symbol, symbol))
            if corr is None:
                continue

            if abs(corr) > abs(max_correlation):
                max_correlation = corr

        # Apply correlation discount
        # High correlation (>0.7) reduces position size
        discount = 1.0
        if abs(max_correlation) > 0.7:
            discount = 1 - (abs(max_correlation) - 0.7) * 1.5  # Up to 45% discount
            discount = max(0.55, discount)  # At least 55% of base size
            adjustments.append(f"Correlation discount: {(1-discount)*100:.1f}% (max_corr={max_correlation:.2f})")

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
            rationale=f"Correlation-adjusted: base={base_size_pct:.1f}%, max_corr={max_correlation:.2f}",
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
            "use_half_kelly": self._use_half_kelly,
            "max_position_pct": self._max_position_pct,
            "min_position_pct": self._min_position_pct,
            "vol_target": self._vol_target,
            "correlation_discount": self._correlation_discount,
            "strategies_tracked": len(self._strategy_stats),
            "correlation_pairs": len(self._correlations),
        }

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
    MEAN_VARIANCE = "mean_variance"  # #P5
    RISK_PARITY = "risk_parity"  # #P5
    MIN_VARIANCE = "min_variance"  # #P5


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
            "supports_portfolio_optimization": True,  # #P5
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
        max_acceptable_drawdown: float = 0.20,
        drawdown_sensitivity: float = 2.0
    ) -> PositionSizeResult:
        """
        Adjust position size based on current portfolio drawdown (P2).

        Reduces position sizes as drawdown increases to limit further losses
        and preserve capital during adverse periods.

        Formula: adjusted_size = base_size * (1 - (drawdown / max_dd) ^ sensitivity)

        Args:
            symbol: Asset symbol
            base_size_pct: Base position size as percentage
            portfolio_value: Total portfolio value
            current_drawdown: Current drawdown as decimal (e.g., 0.10 = 10%)
            max_acceptable_drawdown: Maximum acceptable drawdown (default: 20%)
            drawdown_sensitivity: How aggressively to reduce size (default: 2.0)
                Higher values = more aggressive reduction

        Returns:
            PositionSizeResult with drawdown-adjusted size
        """
        adjustments = []

        # Ensure drawdown is positive
        dd = abs(current_drawdown)

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

        # Calculate reduction factor
        dd_ratio = min(dd / max_acceptable_drawdown, 1.0)
        reduction_factor = 1.0 - (dd_ratio ** drawdown_sensitivity)
        reduction_factor = max(0.1, reduction_factor)  # Minimum 10% of base size

        adjustments.append(f"Drawdown reduction: {(1-reduction_factor)*100:.1f}%")

        # Apply reduction
        adjusted_pct = base_size_pct * reduction_factor

        # Apply constraints
        if adjusted_pct > self._max_position_pct:
            adjusted_pct = self._max_position_pct
            adjustments.append(f"Capped at max {self._max_position_pct}%")

        if adjusted_pct < self._min_position_pct:
            adjusted_pct = self._min_position_pct
            adjustments.append(f"Floor at min {self._min_position_pct}%")

        # If drawdown exceeds max, halt new positions
        if dd >= max_acceptable_drawdown:
            adjusted_pct = 0
            adjustments.append(f"HALTED: Drawdown {dd:.1%} >= max {max_acceptable_drawdown:.1%}")

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

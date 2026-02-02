"""
Value at Risk Calculator
========================

Comprehensive VaR calculation with multiple methodologies:
- Parametric (Variance-Covariance)
- Historical Simulation
- Monte Carlo Simulation
- Incremental and Marginal VaR

Designed for institutional-grade risk management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methodology."""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    method: VaRMethod
    confidence_level: float  # e.g., 0.95 for 95%
    horizon_days: int
    var_absolute: float  # VaR in absolute terms (currency)
    var_pct: float  # VaR as percentage of portfolio
    expected_shortfall: float | None = None  # CVaR / Expected Shortfall
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon_days,
            "var_absolute": self.var_absolute,
            "var_pct": self.var_pct,
            "expected_shortfall": self.expected_shortfall,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class LiquidityProfile:
    """
    Liquidity profile for a position (#R6).

    Used to adjust VaR for liquidity risk factors:
    - Bid-ask spread (immediate cost)
    - Market impact (price movement from liquidating)
    - Time to liquidate (days to exit position)
    """
    symbol: str
    average_daily_volume: float  # ADV in units
    bid_ask_spread_bps: float  # Typical bid-ask spread in basis points
    volatility_daily: float  # Daily volatility for impact estimation
    position_size: float  # Current position in units

    @property
    def days_to_liquidate(self) -> float:
        """
        Estimate days to liquidate position.

        Assumes max 10% of ADV per day participation rate.
        """
        if self.average_daily_volume <= 0:
            return 10.0  # Default max
        participation_rate = 0.10  # 10% of ADV
        max_daily = self.average_daily_volume * participation_rate
        return min(abs(self.position_size) / max(max_daily, 1), 10.0)

    @property
    def market_impact_bps(self) -> float:
        """
        Estimate market impact in basis points.

        Uses square-root market impact model:
        Impact = sigma * sqrt(Q / ADV) * constant

        Where constant is typically 0.5-1.0 for developed markets.
        """
        if self.average_daily_volume <= 0:
            return 100.0  # Conservative default

        # Square-root model coefficient
        IMPACT_COEFFICIENT = 0.6

        # Participation rate
        participation = abs(self.position_size) / self.average_daily_volume

        # Impact in volatility terms, then convert to bps
        impact_vol_units = IMPACT_COEFFICIENT * np.sqrt(participation)
        impact_bps = impact_vol_units * self.volatility_daily * 10000

        return min(impact_bps, 500.0)  # Cap at 5%

    @property
    def total_liquidity_cost_bps(self) -> float:
        """Total expected liquidity cost in basis points."""
        # Half spread (one-way cost) + market impact
        half_spread = self.bid_ask_spread_bps / 2
        return half_spread + self.market_impact_bps


@dataclass
class LiquidityAdjustedVaRResult:
    """Result of liquidity-adjusted VaR calculation (#R6)."""
    base_var: VaRResult
    liquidity_var: float  # Additional VaR from liquidity risk
    total_var: float  # Base + Liquidity
    liquidation_cost: float  # Direct liquidation cost
    liquidation_time_var: float  # VaR from extended liquidation time
    by_position: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_var": self.base_var.to_dict(),
            "liquidity_var": self.liquidity_var,
            "total_var": self.total_var,
            "liquidation_cost": self.liquidation_cost,
            "liquidation_time_var": self.liquidation_time_var,
            "liquidity_var_pct_of_total": (self.liquidity_var / self.total_var * 100) if self.total_var > 0 else 0,
            "by_position": self.by_position,
        }


@dataclass
class IncrementalVaRResult:
    """Result of incremental/marginal VaR calculation."""
    symbol: str
    base_var: float  # Portfolio VaR without position
    new_var: float  # Portfolio VaR with position
    incremental_var: float  # Difference
    marginal_var: float  # VaR contribution per unit
    component_var: float  # VaR contribution of position
    pct_contribution: float  # Percentage of total VaR

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "base_var": self.base_var,
            "new_var": self.new_var,
            "incremental_var": self.incremental_var,
            "marginal_var": self.marginal_var,
            "component_var": self.component_var,
            "pct_contribution": self.pct_contribution,
        }


class VaRCalculator:
    """
    Comprehensive Value at Risk calculator.

    Supports multiple methodologies and provides detailed
    risk decomposition for portfolio management.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize VaR calculator.

        Args:
            config: Configuration with:
                - method: Default VaR method (default: "all")
                - confidence_level: Default confidence (default: 0.95)
                - horizon_days: Default horizon (default: 1)
                - monte_carlo_simulations: MC iterations (default: 10000)
                - decay_factor: EWMA decay for volatility (default: 0.94)
        """
        self._config = config or {}
        self._default_method = self._config.get("method", "all")
        self._confidence_level = self._config.get("confidence_level", 0.95)
        self._horizon_days = self._config.get("horizon_days", 1)
        self._mc_simulations = self._config.get("monte_carlo_simulations", 10000)
        self._decay_factor = self._config.get("decay_factor", 0.94)

        # Cache for covariance matrix
        self._cov_matrix: np.ndarray | None = None
        self._symbols: list[str] = []
        self._returns_data: dict[str, np.ndarray] = {}

        logger.info(
            f"VaRCalculator initialized: conf={self._confidence_level}, "
            f"horizon={self._horizon_days}d, MC={self._mc_simulations}"
        )

    def update_returns(
        self,
        returns_dict: dict[str, np.ndarray]
    ) -> None:
        """
        Update returns data for VaR calculation.

        Args:
            returns_dict: Dictionary mapping symbols to returns arrays
        """
        self._returns_data = returns_dict
        self._symbols = list(returns_dict.keys())
        self._cov_matrix = None  # Invalidate cache

    def _build_covariance_matrix(
        self,
        use_ewma: bool = True
    ) -> np.ndarray:
        """Build covariance matrix from returns data."""
        if self._cov_matrix is not None:
            return self._cov_matrix

        n = len(self._symbols)
        if n == 0:
            return np.array([[]])

        # Align returns
        min_len = min(len(r) for r in self._returns_data.values())
        returns_matrix = np.array([
            self._returns_data[s][-min_len:] for s in self._symbols
        ])

        if use_ewma:
            # EWMA covariance
            self._cov_matrix = self._ewma_covariance(returns_matrix.T)
        else:
            # Standard covariance
            self._cov_matrix = np.cov(returns_matrix)

        return self._cov_matrix

    def _ewma_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate EWMA covariance matrix.

        Args:
            returns: T x N matrix of returns

        Returns:
            N x N covariance matrix
        """
        T, n = returns.shape
        lambda_ = self._decay_factor

        # Calculate weights
        weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(T)])
        weights = weights[::-1]  # Reverse for chronological order
        weights /= weights.sum()  # Normalize

        # Demean returns
        means = np.average(returns, axis=0, weights=weights)
        centered = returns - means

        # Weighted covariance
        cov = np.zeros((n, n))
        for t in range(T):
            cov += weights[t] * np.outer(centered[t], centered[t])

        return cov

    def calculate_parametric_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
        horizon_days: int | None = None
    ) -> VaRResult:
        """
        Calculate Parametric (Variance-Covariance) VaR.

        Assumes returns are normally distributed.

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level (e.g., 0.95)
            horizon_days: Time horizon in days

        Returns:
            VaRResult with parametric VaR
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days

        # Build covariance matrix
        cov_matrix = self._build_covariance_matrix()

        # Build position weights vector
        symbols = self._symbols
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Scale for horizon
        portfolio_std_scaled = portfolio_std * np.sqrt(horizon_days)

        # Z-score for confidence level
        z_score = stats.norm.ppf(confidence_level)

        # VaR
        var_pct = z_score * portfolio_std_scaled
        var_absolute = var_pct * portfolio_value

        # Expected Shortfall (CVaR)
        es_multiplier = stats.norm.pdf(z_score) / (1 - confidence_level)
        es_pct = es_multiplier * portfolio_std_scaled
        es_absolute = es_pct * portfolio_value

        return VaRResult(
            method=VaRMethod.PARAMETRIC,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            var_absolute=var_absolute,
            var_pct=var_pct,
            expected_shortfall=es_absolute,
            details={
                "portfolio_volatility": portfolio_std,
                "z_score": z_score,
            }
        )

    def calculate_historical_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
        horizon_days: int | None = None
    ) -> VaRResult:
        """
        Calculate Historical Simulation VaR.

        Uses actual historical returns distribution.

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level
            horizon_days: Time horizon

        Returns:
            VaRResult with historical VaR
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days

        # Build portfolio returns series
        symbols = self._symbols
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Get aligned returns
        min_len = min(len(r) for r in self._returns_data.values())
        returns_matrix = np.array([
            self._returns_data[s][-min_len:] for s in symbols
        ])

        # Portfolio returns
        portfolio_returns = np.dot(weights, returns_matrix)

        # Scale for horizon (sum of returns for multi-day)
        if horizon_days > 1:
            # Use rolling windows
            scaled_returns = []
            for i in range(len(portfolio_returns) - horizon_days + 1):
                window_return = np.sum(portfolio_returns[i:i + horizon_days])
                scaled_returns.append(window_return)
            portfolio_returns = np.array(scaled_returns)

        # Calculate percentile
        var_pct = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var_absolute = var_pct * portfolio_value

        # Expected Shortfall - average of losses beyond VaR
        tail_returns = portfolio_returns[portfolio_returns <= -var_pct]
        if len(tail_returns) > 0:
            es_pct = -np.mean(tail_returns)
            es_absolute = es_pct * portfolio_value
        else:
            es_absolute = var_absolute

        return VaRResult(
            method=VaRMethod.HISTORICAL,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            var_absolute=var_absolute,
            var_pct=var_pct,
            expected_shortfall=es_absolute,
            details={
                "observations": len(portfolio_returns),
                "worst_loss": -np.min(portfolio_returns) * portfolio_value,
                "avg_loss": -np.mean(portfolio_returns[portfolio_returns < 0]) * portfolio_value,
            }
        )

    def calculate_monte_carlo_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
        horizon_days: int | None = None,
        n_simulations: int | None = None
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR.

        Simulates potential portfolio returns using the covariance structure.

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level
            horizon_days: Time horizon
            n_simulations: Number of simulations

        Returns:
            VaRResult with Monte Carlo VaR
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days
        if n_simulations is None:
            n_simulations = self._mc_simulations

        # Build covariance matrix and mean returns
        cov_matrix = self._build_covariance_matrix()
        symbols = self._symbols

        # Calculate mean returns
        mean_returns = np.array([
            np.mean(self._returns_data[s]) for s in symbols
        ])

        # Position weights
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Generate correlated random returns
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Handle non-positive definite matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            L = np.linalg.cholesky(cov_matrix)

        # Simulate returns
        # Generate random shocks with correlation structure
        random_returns = np.random.randn(n_simulations, len(symbols))
        correlated_shocks = random_returns @ L.T

        # Scale volatility for horizon (sqrt of time)
        # and add drift (linear in time)
        # Returns = drift * horizon + vol_shock * sqrt(horizon)
        drift_component = mean_returns * horizon_days
        vol_component = correlated_shocks * np.sqrt(horizon_days)

        # Total simulated returns (drift + volatility shock)
        simulated_returns = drift_component + vol_component

        # Portfolio returns
        portfolio_returns = np.dot(simulated_returns, weights)

        # Calculate VaR
        var_pct = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var_absolute = var_pct * portfolio_value

        # Expected Shortfall
        tail_returns = portfolio_returns[portfolio_returns <= -var_pct]
        if len(tail_returns) > 0:
            es_pct = -np.mean(tail_returns)
            es_absolute = es_pct * portfolio_value
        else:
            es_absolute = var_absolute

        return VaRResult(
            method=VaRMethod.MONTE_CARLO,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            var_absolute=var_absolute,
            var_pct=var_pct,
            expected_shortfall=es_absolute,
            details={
                "simulations": n_simulations,
                "mean_return": np.mean(portfolio_returns),
                "return_std": np.std(portfolio_returns),
                "worst_scenario": -np.min(portfolio_returns) * portfolio_value,
                "best_scenario": np.max(portfolio_returns) * portfolio_value,
            }
        )

    def calculate_all_methods(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
        horizon_days: int | None = None
    ) -> dict[VaRMethod, VaRResult]:
        """
        Calculate VaR using all methods.

        Args:
            positions: Position values
            portfolio_value: Total portfolio value
            confidence_level: Confidence level
            horizon_days: Time horizon

        Returns:
            Dictionary of method to VaRResult
        """
        results = {}

        results[VaRMethod.PARAMETRIC] = self.calculate_parametric_var(
            positions, portfolio_value, confidence_level, horizon_days
        )

        results[VaRMethod.HISTORICAL] = self.calculate_historical_var(
            positions, portfolio_value, confidence_level, horizon_days
        )

        results[VaRMethod.MONTE_CARLO] = self.calculate_monte_carlo_var(
            positions, portfolio_value, confidence_level, horizon_days
        )

        return results

    def calculate_incremental_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        new_position_symbol: str,
        new_position_value: float,
        method: VaRMethod = VaRMethod.PARAMETRIC
    ) -> IncrementalVaRResult:
        """
        Calculate incremental VaR for adding a new position.

        Args:
            positions: Current positions
            portfolio_value: Current portfolio value
            new_position_symbol: Symbol to add
            new_position_value: Value of new position
            method: VaR calculation method

        Returns:
            IncrementalVaRResult with decomposition
        """
        # Calculate base VaR
        if method == VaRMethod.PARAMETRIC:
            base_result = self.calculate_parametric_var(positions, portfolio_value)
        elif method == VaRMethod.HISTORICAL:
            base_result = self.calculate_historical_var(positions, portfolio_value)
        else:
            base_result = self.calculate_monte_carlo_var(positions, portfolio_value)

        base_var = base_result.var_absolute

        # Add new position and recalculate
        new_positions = dict(positions)
        new_positions[new_position_symbol] = new_positions.get(new_position_symbol, 0) + new_position_value
        new_portfolio_value = portfolio_value + new_position_value

        if method == VaRMethod.PARAMETRIC:
            new_result = self.calculate_parametric_var(new_positions, new_portfolio_value)
        elif method == VaRMethod.HISTORICAL:
            new_result = self.calculate_historical_var(new_positions, new_portfolio_value)
        else:
            new_result = self.calculate_monte_carlo_var(new_positions, new_portfolio_value)

        new_var = new_result.var_absolute

        # Incremental VaR
        incremental_var = new_var - base_var

        # Marginal VaR (per unit of position)
        marginal_var = incremental_var / abs(new_position_value) if new_position_value != 0 else 0

        # Component VaR
        weight = new_position_value / new_portfolio_value
        component_var = weight * new_var

        # Percentage contribution
        pct_contribution = (component_var / new_var * 100) if new_var > 0 else 0

        return IncrementalVaRResult(
            symbol=new_position_symbol,
            base_var=base_var,
            new_var=new_var,
            incremental_var=incremental_var,
            marginal_var=marginal_var,
            component_var=component_var,
            pct_contribution=pct_contribution,
        )

    def calculate_component_var(
        self,
        positions: dict[str, float],
        portfolio_value: float
    ) -> dict[str, IncrementalVaRResult]:
        """
        Calculate component VaR for all positions.

        Shows each position's contribution to total VaR.

        Args:
            positions: Current positions
            portfolio_value: Portfolio value

        Returns:
            Dictionary of symbol to IncrementalVaRResult
        """
        results = {}

        # Calculate total VaR
        total_result = self.calculate_parametric_var(positions, portfolio_value)
        total_var = total_result.var_absolute

        # For each position, calculate marginal contribution
        cov_matrix = self._build_covariance_matrix()
        symbols = self._symbols
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Portfolio standard deviation
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        port_std = np.sqrt(port_var)

        for i, symbol in enumerate(symbols):
            if positions.get(symbol, 0) == 0:
                continue

            # Marginal contribution to variance
            marginal_contrib = np.dot(cov_matrix[i, :], weights) / port_std

            # Component VaR
            weight = positions.get(symbol, 0) / portfolio_value
            component_var = weight * marginal_contrib * total_var / (port_std * stats.norm.ppf(self._confidence_level))

            # Percentage
            pct_contribution = (component_var / total_var * 100) if total_var > 0 else 0

            results[symbol] = IncrementalVaRResult(
                symbol=symbol,
                base_var=total_var,
                new_var=total_var,
                incremental_var=0,
                marginal_var=marginal_contrib * total_var / portfolio_value,
                component_var=component_var,
                pct_contribution=pct_contribution,
            )

        return results

    def stress_test_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        volatility_multiplier: float = 2.0,
        correlation_override: float | None = None
    ) -> VaRResult:
        """
        Calculate stressed VaR with increased volatility.

        Args:
            positions: Current positions
            portfolio_value: Portfolio value
            volatility_multiplier: Factor to increase volatility
            correlation_override: Override all correlations to this value

        Returns:
            VaRResult under stressed conditions
        """
        # Build stressed covariance matrix
        cov_matrix = self._build_covariance_matrix()

        # Extract volatilities
        vols = np.sqrt(np.diag(cov_matrix))

        # Stressed volatilities
        stressed_vols = vols * volatility_multiplier

        # Build correlation matrix
        if correlation_override is not None:
            # Override correlations (e.g., crisis scenario with high correlation)
            n = len(vols)
            corr_matrix = np.full((n, n), correlation_override)
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            # Keep original correlations
            std_outer = np.outer(vols, vols)
            corr_matrix = cov_matrix / std_outer
            np.fill_diagonal(corr_matrix, 1.0)

        # Rebuild covariance with stressed values
        stressed_std_outer = np.outer(stressed_vols, stressed_vols)
        stressed_cov = corr_matrix * stressed_std_outer

        # Temporarily replace covariance matrix
        old_cov = self._cov_matrix
        self._cov_matrix = stressed_cov

        # Calculate VaR
        result = self.calculate_parametric_var(positions, portfolio_value)
        result.details["stress_type"] = "volatility_scaled"
        result.details["volatility_multiplier"] = volatility_multiplier
        result.details["correlation_override"] = correlation_override

        # Restore original
        self._cov_matrix = old_cov

        return result

    # =========================================================================
    # LIQUIDITY-ADJUSTED VAR (#R6)
    # =========================================================================

    def calculate_liquidity_adjusted_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        liquidity_profiles: dict[str, LiquidityProfile],
        confidence_level: float | None = None,
        horizon_days: int | None = None,
        method: VaRMethod = VaRMethod.PARAMETRIC
    ) -> LiquidityAdjustedVaRResult:
        """
        Calculate liquidity-adjusted VaR (#R6).

        Incorporates three liquidity risk components:
        1. Direct liquidation cost (spread + market impact)
        2. Extended horizon VaR (for positions that take multiple days to liquidate)
        3. Liquidity-volatility correlation (liquidity often dries up during stress)

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            liquidity_profiles: Liquidity profile for each position
            confidence_level: Confidence level
            horizon_days: Base time horizon
            method: VaR calculation method for base VaR

        Returns:
            LiquidityAdjustedVaRResult with decomposition
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days

        # Step 1: Calculate base VaR
        if method == VaRMethod.PARAMETRIC:
            base_result = self.calculate_parametric_var(
                positions, portfolio_value, confidence_level, horizon_days
            )
        elif method == VaRMethod.HISTORICAL:
            base_result = self.calculate_historical_var(
                positions, portfolio_value, confidence_level, horizon_days
            )
        else:
            base_result = self.calculate_monte_carlo_var(
                positions, portfolio_value, confidence_level, horizon_days
            )

        # Step 2: Calculate liquidation costs and time-adjusted VaR per position
        total_liquidation_cost = 0.0
        total_time_adjustment = 0.0
        by_position = {}

        for symbol, position_value in positions.items():
            if abs(position_value) < 1:  # Skip negligible positions
                continue

            profile = liquidity_profiles.get(symbol)

            if profile:
                # Direct liquidation cost
                liq_cost = abs(position_value) * profile.total_liquidity_cost_bps / 10000

                # Time-adjusted VaR
                # If liquidation takes T days, we're exposed to market risk for T days
                # VaR scales with sqrt(T) for normal horizon
                # But we also need to account for the fact that we're gradually reducing
                # the position, so the average exposure is position/2 over the liquidation period
                days_to_liq = profile.days_to_liquidate

                if days_to_liq > horizon_days:
                    # Additional horizon adjustment
                    # The "effective" horizon is longer due to extended liquidation
                    effective_horizon = horizon_days + (days_to_liq - horizon_days) / 2
                    horizon_multiplier = np.sqrt(effective_horizon / horizon_days) - 1

                    # Position's contribution to VaR (approximation)
                    position_weight = abs(position_value) / portfolio_value
                    position_var_contrib = base_result.var_absolute * position_weight

                    # Time adjustment for this position
                    time_adj = position_var_contrib * horizon_multiplier
                else:
                    time_adj = 0.0

                total_liquidation_cost += liq_cost
                total_time_adjustment += time_adj

                by_position[symbol] = {
                    "position_value": position_value,
                    "days_to_liquidate": days_to_liq,
                    "liquidation_cost": liq_cost,
                    "time_adjustment": time_adj,
                    "spread_bps": profile.bid_ask_spread_bps,
                    "impact_bps": profile.market_impact_bps,
                    "total_cost_bps": profile.total_liquidity_cost_bps,
                }
            else:
                # No liquidity profile - use conservative defaults
                DEFAULT_SPREAD_BPS = 20  # 20 bps default spread
                DEFAULT_IMPACT_BPS = 50  # 50 bps default impact
                liq_cost = abs(position_value) * (DEFAULT_SPREAD_BPS / 2 + DEFAULT_IMPACT_BPS) / 10000
                total_liquidation_cost += liq_cost

                by_position[symbol] = {
                    "position_value": position_value,
                    "days_to_liquidate": 1.0,
                    "liquidation_cost": liq_cost,
                    "time_adjustment": 0.0,
                    "note": "No liquidity profile - using defaults",
                }

        # Step 3: Liquidity stress adjustment
        # During stress, liquidity often deteriorates - apply a multiplier
        STRESS_LIQUIDITY_MULTIPLIER = 1.5  # Costs increase 50% during stress
        stressed_liquidation_cost = total_liquidation_cost * STRESS_LIQUIDITY_MULTIPLIER

        # Total liquidity VaR component
        liquidity_var = stressed_liquidation_cost + total_time_adjustment

        # Total VaR = Base + Liquidity
        total_var = base_result.var_absolute + liquidity_var

        logger.info(
            f"Liquidity-adjusted VaR: base=${base_result.var_absolute:,.0f}, "
            f"liq_cost=${stressed_liquidation_cost:,.0f}, "
            f"time_adj=${total_time_adjustment:,.0f}, "
            f"total=${total_var:,.0f}"
        )

        return LiquidityAdjustedVaRResult(
            base_var=base_result,
            liquidity_var=liquidity_var,
            total_var=total_var,
            liquidation_cost=stressed_liquidation_cost,
            liquidation_time_var=total_time_adjustment,
            by_position=by_position,
        )

    def update_liquidity_profile(
        self,
        symbol: str,
        adv: float,
        bid_ask_spread_bps: float,
        position_size: float,
        volatility: float | None = None
    ) -> LiquidityProfile:
        """
        Create or update a liquidity profile for a symbol.

        Args:
            symbol: Instrument symbol
            adv: Average daily volume
            bid_ask_spread_bps: Typical bid-ask spread in basis points
            position_size: Current position size in units
            volatility: Daily volatility (uses returns data if None)

        Returns:
            LiquidityProfile instance
        """
        if volatility is None:
            # Try to get from returns data
            if symbol in self._returns_data and len(self._returns_data[symbol]) > 0:
                volatility = np.std(self._returns_data[symbol])
            else:
                volatility = 0.02  # Default 2% daily vol

        return LiquidityProfile(
            symbol=symbol,
            average_daily_volume=adv,
            bid_ask_spread_bps=bid_ask_spread_bps,
            volatility_daily=volatility,
            position_size=position_size,
        )

    def get_status(self) -> dict[str, Any]:
        """Get calculator status for monitoring."""
        return {
            "symbols_tracked": len(self._symbols),
            "default_confidence": self._confidence_level,
            "default_horizon": self._horizon_days,
            "monte_carlo_simulations": self._mc_simulations,
            "ewma_decay_factor": self._decay_factor,
            "covariance_cached": self._cov_matrix is not None,
            "supports_liquidity_adjustment": True,  # #R6
            "supports_jump_risk": True,  # #R9
        }

    # =========================================================================
    # JUMP RISK MODELING (#R9)
    # =========================================================================

    def calculate_jump_adjusted_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
        horizon_days: int | None = None,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.05,
        jump_std: float = 0.10,
        n_simulations: int | None = None
    ) -> VaRResult:
        """
        Calculate VaR with jump-diffusion model for fat tails (#R9).

        Uses Merton's jump-diffusion model to capture extreme events
        that normal distributions miss. Returns are modeled as:

        dS/S = μdt + σdW + J*dN

        Where:
        - dW is standard Brownian motion
        - dN is Poisson process with intensity λ (jump_intensity)
        - J is jump size ~ Normal(jump_mean, jump_std)

        This addresses the "fat tails" problem where market crashes
        are more frequent than normal distributions suggest.

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level (e.g., 0.99 for jump risk)
            horizon_days: Time horizon
            jump_intensity: Expected jumps per year (λ, default 0.1 = 1 every 10 years)
            jump_mean: Average jump size (default -5% for crash bias)
            jump_std: Jump size volatility (default 10%)
            n_simulations: Number of Monte Carlo paths

        Returns:
            VaRResult with jump-adjusted VaR and tail statistics
        """
        if confidence_level is None:
            confidence_level = max(self._confidence_level, 0.99)  # Higher for tail risk
        if horizon_days is None:
            horizon_days = self._horizon_days
        if n_simulations is None:
            n_simulations = self._mc_simulations

        # Build covariance matrix and mean returns
        cov_matrix = self._build_covariance_matrix()
        symbols = self._symbols

        if len(symbols) == 0:
            return VaRResult(
                method=VaRMethod.MONTE_CARLO,
                confidence_level=confidence_level,
                horizon_days=horizon_days,
                var_absolute=0.0,
                var_pct=0.0,
                expected_shortfall=0.0,
                details={"error": "No symbols in returns data"}
            )

        # Calculate mean returns and volatilities
        mean_returns = np.array([
            np.mean(self._returns_data[s]) for s in symbols
        ])

        # Position weights
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Cholesky decomposition for correlated returns
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            L = np.linalg.cholesky(cov_matrix)

        # Generate simulated portfolio returns with jumps
        portfolio_returns = np.zeros(n_simulations)
        jump_counts = np.zeros(n_simulations)

        # Daily jump probability from annual intensity
        daily_jump_prob = 1 - np.exp(-jump_intensity / 252 * horizon_days)

        for i in range(n_simulations):
            # Diffusion component (normal returns)
            random_shocks = np.random.randn(len(symbols))
            correlated_shocks = L @ random_shocks

            # Scale for horizon
            drift_component = mean_returns * horizon_days
            vol_component = correlated_shocks * np.sqrt(horizon_days)

            asset_returns = drift_component + vol_component

            # Jump component
            # Poisson process for number of jumps
            n_jumps = np.random.poisson(jump_intensity * horizon_days / 252)
            jump_counts[i] = n_jumps

            if n_jumps > 0:
                # Sum of jump sizes
                total_jump = np.sum(np.random.normal(jump_mean, jump_std, n_jumps))
                # Apply jump to all assets (systemic shock) with correlation
                asset_returns += total_jump * np.abs(weights)  # Correlated jumps

            # Portfolio return
            portfolio_returns[i] = np.dot(weights, asset_returns)

        # Calculate VaR
        var_pct = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var_absolute = var_pct * portfolio_value

        # Expected Shortfall (CVaR)
        tail_returns = portfolio_returns[portfolio_returns <= -var_pct]
        if len(tail_returns) > 0:
            es_pct = -np.mean(tail_returns)
            es_absolute = es_pct * portfolio_value
        else:
            es_absolute = var_absolute

        # Calculate tail statistics
        skewness = stats.skew(portfolio_returns)
        kurtosis_excess = stats.kurtosis(portfolio_returns)
        worst_return = np.min(portfolio_returns)
        jump_occurrence_rate = np.mean(jump_counts > 0)

        # Compare to normal VaR
        normal_var = self.calculate_parametric_var(
            positions, portfolio_value, confidence_level, horizon_days
        )

        tail_multiplier = var_absolute / normal_var.var_absolute if normal_var.var_absolute > 0 else 1.0

        logger.info(
            f"Jump-adjusted VaR: ${var_absolute:,.0f} ({var_pct*100:.2f}%), "
            f"tail_multiplier={tail_multiplier:.2f}x vs normal, "
            f"jump_rate={jump_occurrence_rate*100:.1f}% of scenarios"
        )

        return VaRResult(
            method=VaRMethod.MONTE_CARLO,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            var_absolute=var_absolute,
            var_pct=var_pct,
            expected_shortfall=es_absolute,
            details={
                "model": "jump_diffusion",
                "jump_intensity": jump_intensity,
                "jump_mean": jump_mean,
                "jump_std": jump_std,
                "simulations": n_simulations,
                "skewness": skewness,
                "excess_kurtosis": kurtosis_excess,
                "worst_scenario": -worst_return * portfolio_value,
                "jump_occurrence_rate": jump_occurrence_rate,
                "tail_multiplier_vs_normal": tail_multiplier,
                "normal_var": normal_var.var_absolute,
            }
        )

    def calculate_fat_tail_metrics(
        self,
        positions: dict[str, float],
        portfolio_value: float
    ) -> dict[str, Any]:
        """
        Calculate comprehensive fat tail risk metrics (#R9).

        Returns metrics that capture tail risk beyond standard VaR:
        - Skewness and kurtosis
        - Tail ratio (losses vs gains)
        - Maximum drawdown in historical returns
        - Jump risk premium

        Args:
            positions: Position values
            portfolio_value: Total portfolio value

        Returns:
            Dictionary of fat tail metrics
        """
        # Build portfolio returns
        symbols = self._symbols
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        min_len = min(len(r) for r in self._returns_data.values())
        returns_matrix = np.array([
            self._returns_data[s][-min_len:] for s in symbols
        ])

        portfolio_returns = np.dot(weights, returns_matrix)

        if len(portfolio_returns) < 20:
            return {"error": "Insufficient return history for tail analysis"}

        # Basic statistics
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        skewness = stats.skew(portfolio_returns)
        kurtosis_excess = stats.kurtosis(portfolio_returns)

        # Tail statistics
        lower_5 = np.percentile(portfolio_returns, 5)
        upper_95 = np.percentile(portfolio_returns, 95)
        lower_1 = np.percentile(portfolio_returns, 1)
        upper_99 = np.percentile(portfolio_returns, 99)

        # Tail ratio (asymmetry of tails)
        left_tail = np.mean(portfolio_returns[portfolio_returns < lower_5])
        right_tail = np.mean(portfolio_returns[portfolio_returns > upper_95])
        tail_ratio = abs(left_tail / right_tail) if right_tail != 0 else float('inf')

        # Expected shortfall at different levels
        es_95 = -np.mean(portfolio_returns[portfolio_returns <= lower_5]) * portfolio_value
        es_99 = -np.mean(portfolio_returns[portfolio_returns <= lower_1]) * portfolio_value

        # Maximum historical drawdown in returns series
        cumulative = np.cumsum(portfolio_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative)
        max_drawdown = np.max(drawdown)

        # Count extreme events (>3 std)
        extreme_negative = np.sum(portfolio_returns < mean_return - 3 * std_return)
        extreme_positive = np.sum(portfolio_returns > mean_return + 3 * std_return)
        expected_extreme = len(portfolio_returns) * 2 * stats.norm.cdf(-3)  # ~0.27% for normal

        extreme_ratio = (extreme_negative + extreme_positive) / max(expected_extreme, 0.1)

        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(portfolio_returns)

        return {
            "observations": len(portfolio_returns),
            "mean_return": mean_return,
            "volatility": std_return,
            "skewness": skewness,
            "excess_kurtosis": kurtosis_excess,
            "is_fat_tailed": kurtosis_excess > 1.0,
            "is_negatively_skewed": skewness < -0.5,
            "tail_ratio": tail_ratio,
            "percentiles": {
                "p1": lower_1,
                "p5": lower_5,
                "p95": upper_95,
                "p99": upper_99,
            },
            "expected_shortfall": {
                "es_95": es_95,
                "es_99": es_99,
            },
            "extreme_events": {
                "negative_3std": int(extreme_negative),
                "positive_3std": int(extreme_positive),
                "expected_if_normal": expected_extreme,
                "extreme_ratio": extreme_ratio,
            },
            "max_historical_drawdown": max_drawdown,
            "jarque_bera": {
                "statistic": jb_stat,
                "p_value": jb_pvalue,
                "is_normal": jb_pvalue > 0.05,
            },
            "risk_assessment": (
                "HIGH" if kurtosis_excess > 3 or extreme_ratio > 3
                else "MEDIUM" if kurtosis_excess > 1 or extreme_ratio > 1.5
                else "LOW"
            ),
        }

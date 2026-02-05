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

# NumPy and SciPy are required for VaR calculations
# Provide graceful error message if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None  # type: ignore

if not HAS_NUMPY or not HAS_SCIPY:
    logging.getLogger(__name__).error(
        "NumPy and/or SciPy not available - VaR calculations will be disabled. "
        "Install with: pip install numpy scipy"
    )


logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methodology."""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


class VolatilityRegime(Enum):
    """
    Volatility regime classification for regime-conditional VaR (Phase 2).

    Based on VIX levels and research findings:
    - Low vol periods often precede corrections (complacency)
    - High vol reduces VaR predictive accuracy
    - Crisis regimes require correlation adjustments
    """
    LOW = "low"          # VIX < 15 - complacency, tighter estimates
    NORMAL = "normal"    # VIX 15-20 - standard parameters
    HIGH = "high"        # VIX 20-30 - increased uncertainty
    CRISIS = "crisis"    # VIX > 30 - correlation breakdown


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
class RegimeConditionalVaRResult:
    """
    Result of regime-conditional VaR calculation (Phase 2).

    Provides VaR adjusted for current volatility regime with:
    - Regime-specific volatility scaling
    - Correlation floor adjustments
    - Confidence level adjustments
    """
    base_var: VaRResult
    regime: VolatilityRegime
    regime_adjusted_var: float
    regime_adjusted_es: float | None
    volatility_multiplier: float
    correlation_floor: float | None
    confidence_adjustment: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_var": self.base_var.to_dict(),
            "regime": self.regime.value,
            "regime_adjusted_var": self.regime_adjusted_var,
            "regime_adjusted_es": self.regime_adjusted_es,
            "volatility_multiplier": self.volatility_multiplier,
            "correlation_floor": self.correlation_floor,
            "confidence_adjustment": self.confidence_adjustment,
            "timestamp": self.timestamp.isoformat(),
            "var_increase_pct": (
                (self.regime_adjusted_var - self.base_var.var_absolute)
                / self.base_var.var_absolute * 100
                if self.base_var.var_absolute > 0 else 0
            ),
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

    def _shrink_covariance(
        self,
        sample_cov: np.ndarray,
        n_obs: int
    ) -> np.ndarray:
        """
        Apply Ledoit-Wolf shrinkage to stabilize covariance estimation (PM-11).

        Problem:
            Sample covariance matrices are noisy when n_obs < n_assets^2.
            This noise leads to poor portfolio optimization results:
            - Extreme weights
            - Poor out-of-sample performance
            - Numerical instability (near-singular matrices)

        Solution (Ledoit-Wolf 2004):
            Shrink the sample covariance towards a structured target:
            Cov_shrunk = alpha * Target + (1 - alpha) * Sample

        The target is a "constant correlation" matrix where:
            - All diagonal elements equal the average variance
            - All off-diagonal elements equal avg_var * avg_correlation

        This target has guaranteed good conditioning but loses information
        about individual correlations. The shrinkage intensity alpha balances
        the bias-variance tradeoff:
            - alpha = 0: Use sample covariance (high variance, low bias)
            - alpha = 1: Use target (low variance, high bias)

        The optimal alpha increases with:
            - Fewer observations (n_obs small)
            - More assets (n large)
            - Higher estimation noise in sample

        Args:
            sample_cov: Raw sample covariance matrix (n x n)
            n_obs: Number of observations used to estimate sample_cov

        Returns:
            Shrunk covariance matrix (n x n)

        Reference:
            Ledoit & Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix"
            Journal of Empirical Finance 11(1), 107-129
        """
        n = len(sample_cov)

        if n <= 1:
            return sample_cov

        # Step 1: Extract variances (diagonal elements)
        variances = np.diag(sample_cov)
        std_devs = np.sqrt(np.maximum(variances, 1e-10))

        # Step 2: Build correlation matrix from sample covariance
        # Corr_ij = Cov_ij / (sigma_i * sigma_j)
        std_outer = np.outer(std_devs, std_devs)
        std_outer = np.where(std_outer > 0, std_outer, 1.0)  # Avoid division by zero
        corr_matrix = sample_cov / std_outer
        np.fill_diagonal(corr_matrix, 1.0)  # Ensure diagonal is exactly 1

        # Step 3: Calculate target parameters
        # Average variance (for target diagonal)
        avg_var = np.mean(variances)

        # Average correlation (for target off-diagonal)
        # Only consider off-diagonal elements
        mask = ~np.eye(n, dtype=bool)
        if np.sum(mask) > 0:
            avg_corr = np.mean(corr_matrix[mask])
        else:
            avg_corr = 0.0

        # Step 4: Build shrinkage target (constant correlation matrix)
        # Target_ij = avg_var * (1 if i==j else avg_corr)
        # In matrix form: Target = avg_var * (I + avg_corr * (J - I))
        target = avg_var * (
            np.eye(n) + avg_corr * (np.ones((n, n)) - np.eye(n))
        )

        # Step 5: Calculate shrinkage intensity
        # Simple heuristic: intensity proportional to 1/n_obs
        # With k observations, estimation error ~ 1/sqrt(k)
        # Use k=10 as the "critical" number where we need 100% shrinkage
        SHRINKAGE_CONSTANT = 10.0
        shrinkage_intensity = min(1.0, SHRINKAGE_CONSTANT / max(n_obs, 1))

        # Step 6: Apply shrinkage formula
        # Cov_shrunk = alpha * Target + (1 - alpha) * Sample
        shrunk_cov = (
            shrinkage_intensity * target +
            (1 - shrinkage_intensity) * sample_cov
        )

        logger.debug(
            f"Applied Ledoit-Wolf shrinkage: intensity={shrinkage_intensity:.3f}, "
            f"n_obs={n_obs}, n_assets={n}"
        )

        return shrunk_cov

    def _build_covariance_matrix(
        self,
        use_ewma: bool = True,
        apply_shrinkage: bool = True
    ) -> np.ndarray:
        """
        Build covariance matrix from returns data.

        Args:
            use_ewma: Use EWMA weighting for recent observations (default True)
            apply_shrinkage: Apply Ledoit-Wolf shrinkage for stability (default True)

        Returns:
            Covariance matrix (n x n)
        """
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

        # Apply Ledoit-Wolf shrinkage for stability (PM-11)
        # Especially important with < 250 observations
        n_obs = min_len
        if apply_shrinkage and n_obs < 250:
            self._cov_matrix = self._shrink_covariance(self._cov_matrix, n_obs)
            logger.debug(
                f"Applied shrinkage due to limited observations: {n_obs} < 250"
            )

        # Ensure covariance matrix is positive semi-definite
        # Fix numerical issues that could lead to negative eigenvalues
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self._cov_matrix)
            if np.any(eigenvalues < 0):
                # Fix negative eigenvalues (numerical precision issue)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                self._cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except np.linalg.LinAlgError:
            # If eigendecomposition fails, add small diagonal for stability
            self._cov_matrix = self._cov_matrix + np.eye(n) * 1e-8

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

    def _incremental_ewma_update(self, new_returns: np.ndarray) -> None:
        """
        Perform incremental EWMA update of covariance matrix (PERF-P0-005).

        This avoids full O(n^2 * T) recalculation by applying a rank-1 update.
        The EWMA update formula is:
            Cov_t = lambda * Cov_{t-1} + (1 - lambda) * r_t * r_t'

        Where r_t is the demeaned return vector at time t.

        Args:
            new_returns: New return vector of shape (n,) for each asset

        Note:
            Call this method when new returns arrive instead of rebuilding
            the full covariance matrix from scratch.
        """
        if self._cov_matrix is None:
            # No existing matrix - need full computation first
            logger.warning(
                "Cannot perform incremental update without existing covariance. "
                "Call _build_covariance_matrix() first."
            )
            return

        lambda_ = self._decay_factor
        n = len(new_returns)

        # Validate dimensions
        if self._cov_matrix.shape[0] != n:
            logger.warning(
                f"Dimension mismatch: cov_matrix has {self._cov_matrix.shape[0]} assets, "
                f"new_returns has {n}. Skipping incremental update."
            )
            return

        # Demean the new returns using running mean (approximate)
        # For a more accurate approach, maintain a running mean estimate
        if hasattr(self, '_running_mean') and self._running_mean is not None:
            # Update running mean with EWMA
            self._running_mean = lambda_ * self._running_mean + (1 - lambda_) * new_returns
            centered_returns = new_returns - self._running_mean
        else:
            # Initialize running mean
            self._running_mean = new_returns.copy()
            centered_returns = new_returns  # First observation, no demeaning

        # Incremental EWMA update: Cov_new = lambda * Cov_old + (1-lambda) * outer(r, r)
        self._cov_matrix = (
            lambda_ * self._cov_matrix +
            (1 - lambda_) * np.outer(centered_returns, centered_returns)
        )

        logger.debug(
            f"Incremental EWMA update applied with decay factor {lambda_}"
        )

    def update_returns_incremental(
        self,
        new_returns: dict[str, float]
    ) -> None:
        """
        Update returns data incrementally and apply EWMA covariance update.

        This is more efficient than update_returns() when adding single
        observations, as it avoids full matrix recalculation.

        Args:
            new_returns: Dictionary mapping symbols to new return values
        """
        # Ensure symbols match
        if set(new_returns.keys()) != set(self._symbols):
            # Symbol set changed - need full rebuild
            logger.info(
                "Symbol set changed, falling back to full covariance rebuild"
            )
            # Add new returns to history
            for symbol, ret in new_returns.items():
                if symbol in self._returns_data:
                    self._returns_data[symbol] = np.append(
                        self._returns_data[symbol], ret
                    )
                else:
                    self._returns_data[symbol] = np.array([ret])
            self._symbols = list(self._returns_data.keys())
            self._cov_matrix = None  # Force full rebuild
            return

        # Build new returns vector in symbol order
        new_ret_vector = np.array([new_returns[s] for s in self._symbols])

        # Append to history
        for symbol, ret in new_returns.items():
            self._returns_data[symbol] = np.append(
                self._returns_data[symbol], ret
            )

        # Apply incremental update if we have existing covariance
        if self._cov_matrix is not None:
            self._incremental_ewma_update(new_ret_vector)
        # If no covariance matrix exists, it will be built on next calculation

    def get_covariance_for_optimization(
        self,
        force_shrinkage: bool = True,
        shrinkage_target_type: str = "constant_correlation"
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """
        Get covariance matrix optimized for mean-variance portfolio optimization (PM-11).

        This method provides a stabilized covariance matrix suitable for
        optimization, with explicit shrinkage control and metadata.

        Args:
            force_shrinkage: Always apply shrinkage regardless of sample size
            shrinkage_target_type: Type of shrinkage target
                - "constant_correlation": Ledoit-Wolf constant correlation target
                - "identity": Shrink towards scaled identity (spherical)

        Returns:
            Tuple of:
                - Shrunk covariance matrix (n x n)
                - List of symbols in matrix order
                - Metadata dict with shrinkage details

        Example:
            >>> cov, symbols, meta = var_calc.get_covariance_for_optimization()
            >>> # Use cov for mean-variance optimization
            >>> weights = optimize_portfolio(cov, expected_returns)
        """
        n = len(self._symbols)
        if n == 0:
            return np.array([[]]), [], {"error": "No symbols available"}

        # Get number of observations
        min_len = min(len(r) for r in self._returns_data.values())

        # Build returns matrix
        returns_matrix = np.array([
            self._returns_data[s][-min_len:] for s in self._symbols
        ])

        # Calculate raw covariance (EWMA)
        raw_cov = self._ewma_covariance(returns_matrix.T)

        # Determine if shrinkage is needed
        needs_shrinkage = force_shrinkage or min_len < 250

        if needs_shrinkage:
            if shrinkage_target_type == "identity":
                # Shrink towards scaled identity matrix
                avg_var = np.trace(raw_cov) / n
                target = avg_var * np.eye(n)
                SHRINKAGE_CONSTANT = 10.0
                shrinkage_intensity = min(1.0, SHRINKAGE_CONSTANT / max(min_len, 1))
                shrunk_cov = (
                    shrinkage_intensity * target +
                    (1 - shrinkage_intensity) * raw_cov
                )
            else:
                # Default: constant correlation (Ledoit-Wolf)
                shrunk_cov = self._shrink_covariance(raw_cov, min_len)
                SHRINKAGE_CONSTANT = 10.0
                shrinkage_intensity = min(1.0, SHRINKAGE_CONSTANT / max(min_len, 1))
        else:
            shrunk_cov = raw_cov
            shrinkage_intensity = 0.0

        # Ensure positive semi-definiteness
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(shrunk_cov)
            if np.any(eigenvalues < 0):
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                shrunk_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                psd_corrected = True
            else:
                psd_corrected = False
        except np.linalg.LinAlgError:
            shrunk_cov = shrunk_cov + np.eye(n) * 1e-8
            psd_corrected = True

        # Build metadata
        metadata = {
            "n_assets": n,
            "n_observations": min_len,
            "shrinkage_applied": needs_shrinkage,
            "shrinkage_intensity": shrinkage_intensity,
            "shrinkage_target": shrinkage_target_type if needs_shrinkage else None,
            "psd_corrected": psd_corrected,
            "condition_number": np.linalg.cond(shrunk_cov) if n > 0 else None,
            "min_eigenvalue": float(np.min(eigenvalues)) if n > 0 else None,
            "recommendation": (
                "STABLE" if shrinkage_intensity > 0 and min_len >= 50
                else "CAUTION - limited data" if min_len < 50
                else "OK"
            ),
        }

        logger.info(
            f"Covariance for optimization: {n} assets, {min_len} obs, "
            f"shrinkage={shrinkage_intensity:.3f}, cond={metadata['condition_number']:.1f}"
        )

        return shrunk_cov, list(self._symbols), metadata

    def calculate_parametric_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
        horizon_days: int | None = None
    ) -> VaRResult:
        """
        Calculate Parametric (Variance-Covariance) VaR.

        The parametric method assumes returns follow a normal distribution
        and uses the portfolio's standard deviation to estimate potential losses.

        VaR Formula:
            VaR = Z_alpha * sigma_p * sqrt(T) * Portfolio_Value

        Where:
            Z_alpha = inverse CDF of normal distribution at confidence level
                      (e.g., 1.645 for 95%, 2.326 for 99%)
            sigma_p = portfolio standard deviation (daily)
            T = time horizon in days
            sqrt(T) = square-root-of-time scaling (assumes IID returns)

        Portfolio variance is calculated as:
            sigma_p^2 = w' * Cov * w

        Where w is the vector of position weights and Cov is the covariance matrix.

        Expected Shortfall (CVaR) is also computed:
            ES = sigma_p * phi(Z_alpha) / (1 - alpha) * Portfolio_Value

        Where phi() is the standard normal PDF. ES represents the expected
        loss given that the loss exceeds VaR (i.e., the average of the tail).

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            horizon_days: Time horizon in days (default: 1)

        Returns:
            VaRResult with parametric VaR and Expected Shortfall

        Limitations:
            - Assumes normally distributed returns (underestimates tail risk)
            - Does not capture fat tails or skewness
            - Square-root-of-time scaling assumes no autocorrelation
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days

        # Step 1: Build covariance matrix from historical returns
        cov_matrix = self._build_covariance_matrix()

        # Step 2: Build position weights vector (w_i = position_value_i / total_value)
        # CRITICAL: Validate portfolio_value to prevent division by zero
        if portfolio_value <= 0:
            logger.warning(f"VaR calculation skipped: portfolio_value={portfolio_value} is <= 0")
            return VaRResult(
                method=VaRMethod.PARAMETRIC,
                confidence_level=confidence_level,
                horizon_days=horizon_days,
                var_absolute=0.0,
                var_pct=0.0,
                expected_shortfall=0.0,
                details={"error": "invalid_portfolio_value", "portfolio_value": portfolio_value},
            )

        symbols = self._symbols
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Step 3: Calculate portfolio variance using matrix form: Var(Rp) = w' * Cov * w
        # This captures all pairwise correlations between positions

        # Validate array shapes to prevent dimension mismatch crashes
        if len(weights) == 0:
            logger.warning("VaR calculation skipped: no position weights available")
            return VaRResult(
                method=VaRMethod.PARAMETRIC,
                confidence_level=confidence_level,
                horizon_days=horizon_days,
                var_absolute=0.0,
                var_pct=0.0,
                expected_shortfall=0.0,
                details={"error": "no_positions"},
            )

        if cov_matrix.shape[0] != len(weights) or cov_matrix.shape[1] != len(weights):
            logger.warning(
                f"VaR calculation skipped: shape mismatch - "
                f"cov_matrix={cov_matrix.shape}, weights={len(weights)}"
            )
            return VaRResult(
                method=VaRMethod.PARAMETRIC,
                confidence_level=confidence_level,
                horizon_days=horizon_days,
                var_absolute=0.0,
                var_pct=0.0,
                expected_shortfall=0.0,
                details={"error": "shape_mismatch"},
            )

        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        # Guard against numerical precision issues (should never be negative)
        # If variance IS negative, it indicates covariance matrix instability - LOG WARNING
        if portfolio_variance < 0:
            logger.warning(
                f"NUMERICAL ISSUE: Portfolio variance is negative ({portfolio_variance:.6f}). "
                "Clamping to 0. This may indicate covariance matrix instability - "
                "consider triggering matrix rebuild."
            )
        portfolio_variance = max(0.0, float(portfolio_variance))
        portfolio_std = np.sqrt(portfolio_variance)

        # Step 4: Scale volatility for time horizon using square-root-of-time rule
        # sigma_T = sigma_1 * sqrt(T), valid if returns are IID
        portfolio_std_scaled = portfolio_std * np.sqrt(horizon_days)

        # Step 5: Get Z-score for the confidence level
        # Z_0.95 = 1.645, Z_0.99 = 2.326 (one-tailed)
        z_score = stats.norm.ppf(confidence_level)

        # Step 6: Calculate VaR = Z * sigma * Portfolio_Value
        var_pct = z_score * portfolio_std_scaled
        var_absolute = var_pct * portfolio_value

        # Step 7: Calculate Expected Shortfall (CVaR)
        # ES = E[Loss | Loss > VaR] = sigma * phi(z) / (1-alpha)
        # This is the average loss in the tail beyond VaR
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

        Historical VaR uses the actual empirical distribution of past returns
        rather than assuming a parametric distribution (like normal).

        Method:
            1. Calculate historical portfolio returns using current weights
            2. Sort returns from worst to best
            3. Find the return at the (1-alpha) percentile
            4. VaR = -Percentile_return * Portfolio_Value

        For example, with 250 observations and 95% confidence:
            - Sort all 250 returns
            - VaR is the 13th worst return (250 * 0.05 = 12.5, round up)

        For multi-day horizons, we use overlapping windows to aggregate
        returns (sum of daily returns over the window period).

        Expected Shortfall (CVaR) is calculated as the average of all
        returns worse than the VaR threshold.

        Advantages over parametric VaR:
            - Captures fat tails and non-normal distributions
            - No distributional assumptions required
            - Naturally includes extreme historical events

        Limitations:
            - Limited by available historical data
            - Cannot extrapolate beyond historical worst case
            - Assumes past distribution predicts future

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level (e.g., 0.95)
            horizon_days: Time horizon in days

        Returns:
            VaRResult with historical VaR and Expected Shortfall
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days

        # CRITICAL: Validate portfolio_value to prevent division by zero
        if portfolio_value <= 0:
            logger.warning(f"Historical VaR calculation skipped: portfolio_value={portfolio_value} is <= 0")
            return VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=confidence_level,
                horizon_days=horizon_days,
                var_absolute=0.0,
                var_pct=0.0,
                expected_shortfall=0.0,
                details={"error": "invalid_portfolio_value", "portfolio_value": portfolio_value},
            )

        # Step 1: Build portfolio returns series using current position weights
        symbols = self._symbols
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Step 2: Get aligned returns (use shortest common history)
        min_len = min(len(r) for r in self._returns_data.values())
        returns_matrix = np.array([
            self._returns_data[s][-min_len:] for s in symbols
        ])

        # Step 3: Calculate portfolio returns: R_p,t = sum(w_i * R_i,t)
        portfolio_returns = np.dot(weights, returns_matrix)

        # Step 4: Scale for horizon using overlapping windows
        # For T-day VaR, we need T-day returns: R_T = R_1 + R_2 + ... + R_T
        if horizon_days > 1:
            scaled_returns = []
            for i in range(len(portfolio_returns) - horizon_days + 1):
                # Sum returns over the window (assumes log returns or small returns)
                window_return = np.sum(portfolio_returns[i:i + horizon_days])
                scaled_returns.append(window_return)
            portfolio_returns = np.array(scaled_returns)

        # Step 5: Calculate VaR as the (1-alpha) percentile of losses
        # Negative sign because we want the loss magnitude (positive number)
        var_pct = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var_absolute = var_pct * portfolio_value

        # Step 6: Calculate Expected Shortfall (CVaR)
        # ES = E[Loss | Loss > VaR] = average of all returns worse than -VaR
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
        # Guard against negative variance due to numerical precision issues
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        port_var = max(0.0, float(port_var))
        port_std = np.sqrt(port_var)

        if port_std < 1e-12:
            return results  # Cannot compute component VaR with zero portfolio std

        for i, symbol in enumerate(symbols):
            if positions.get(symbol, 0) == 0:
                continue

            # Marginal contribution to variance
            # Guard against division by zero
            if port_std < 1e-12:
                marginal_contrib = 0.0
            else:
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
            "supports_backtesting": True,  # P2: Kupiec test
            "supports_component_var": True,  # P2: Component VaR breakdown
            "supports_marginal_var": True,  # P2: Marginal VaR
            "supports_regime_conditional": True,  # Phase 2: Regime-conditional VaR
        }

    # =========================================================================
    # COMPONENT VAR BREAKDOWN (P2)
    # =========================================================================

    def calculate_component_var_breakdown(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate detailed component VaR breakdown for all positions (P2).

        Component VaR decomposes total portfolio VaR into contributions
        from each position. The sum of component VaRs equals total VaR.

        Formula: ComponentVaR_i = weight_i * MarginalVaR_i
                                = weight_i * (Cov(R_i, R_p) / sigma_p) * z * sigma_p
                                = weight_i * beta_i * VaR_p

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level

        Returns:
            Dictionary with component VaR breakdown and statistics
        """
        if confidence_level is None:
            confidence_level = self._confidence_level

        # Calculate total VaR first
        total_result = self.calculate_parametric_var(
            positions, portfolio_value, confidence_level
        )
        total_var = total_result.var_absolute

        # Build covariance matrix
        cov_matrix = self._build_covariance_matrix()
        symbols = self._symbols

        # Position weights
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Portfolio variance and std
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        port_var = max(0.0, float(port_var))
        port_std = np.sqrt(port_var)

        if port_std < 1e-12:
            return {
                "error": "Portfolio standard deviation too small",
                "total_var": total_var,
                "components": {},
            }

        z_score = stats.norm.ppf(confidence_level)

        # Calculate component VaR for each position
        components = {}
        total_component_var = 0.0

        for i, symbol in enumerate(symbols):
            position_value = positions.get(symbol, 0)
            if abs(position_value) < 1e-6:
                continue

            weight = position_value / portfolio_value

            # Marginal contribution to variance: d(sigma_p^2)/d(w_i) = 2 * Cov(i, p)
            # Cov(i, p) = sum_j(w_j * Cov(i,j)) = cov_matrix[i,:] @ weights
            cov_i_portfolio = np.dot(cov_matrix[i, :], weights)

            # Marginal VaR (per unit weight)
            # MarginalVaR = d(VaR)/d(w) = z * Cov(i,p) / sigma_p
            marginal_var_per_weight = z_score * cov_i_portfolio / port_std

            # Component VaR = weight * MarginalVaR
            component_var = weight * marginal_var_per_weight * portfolio_value

            # Beta to portfolio
            beta_to_portfolio = cov_i_portfolio / port_var if port_var > 0 else 0

            # Percentage of total VaR
            pct_of_total = (component_var / total_var * 100) if total_var > 0 else 0

            components[symbol] = {
                "position_value": position_value,
                "weight": weight,
                "component_var": component_var,
                "marginal_var_per_unit": marginal_var_per_weight,
                "beta_to_portfolio": beta_to_portfolio,
                "pct_of_total_var": pct_of_total,
                "standalone_volatility": np.sqrt(cov_matrix[i, i]),
                "diversification_benefit": (
                    weight * np.sqrt(cov_matrix[i, i]) * z_score * portfolio_value
                    - component_var
                ),
            }
            total_component_var += component_var

        # Verification: sum of component VaRs should equal total VaR
        decomposition_error = abs(total_component_var - total_var)

        return {
            "total_var": total_var,
            "confidence_level": confidence_level,
            "components": components,
            "summary": {
                "sum_component_var": total_component_var,
                "decomposition_error": decomposition_error,
                "decomposition_valid": decomposition_error < total_var * 0.01,
                "largest_contributor": max(
                    components.keys(),
                    key=lambda s: abs(components[s]["component_var"]),
                    default=None,
                ),
                "diversification_benefit_total": sum(
                    c["diversification_benefit"] for c in components.values()
                ),
            },
        }

    # =========================================================================
    # MARGINAL VAR CALCULATION (P2)
    # =========================================================================

    def calculate_marginal_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        target_symbol: str,
        position_change: float = 1.0,
        confidence_level: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate marginal VaR for a position change (P2).

        Marginal VaR measures the change in portfolio VaR per unit change
        in position. It helps in:
        - Position sizing decisions
        - Risk budgeting
        - Understanding VaR sensitivity

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            target_symbol: Symbol to calculate marginal VaR for
            position_change: Amount of position change to test
            confidence_level: Confidence level

        Returns:
            Dictionary with marginal VaR metrics
        """
        if confidence_level is None:
            confidence_level = self._confidence_level

        if target_symbol not in self._symbols:
            return {"error": f"Symbol {target_symbol} not in returns data"}

        # Current VaR
        current_var_result = self.calculate_parametric_var(
            positions, portfolio_value, confidence_level
        )
        current_var = current_var_result.var_absolute

        # Build covariance matrix
        cov_matrix = self._build_covariance_matrix()
        symbols = self._symbols
        target_idx = symbols.index(target_symbol)

        # Position weights
        weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

        # Portfolio variance and std
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        port_var = max(0.0, float(port_var))
        port_std = np.sqrt(port_var)

        z_score = stats.norm.ppf(confidence_level)

        # Analytical marginal VaR (derivative approach)
        # d(VaR)/d(position_i) = z * Cov(i, portfolio) / (sigma_p * portfolio_value)
        cov_i_portfolio = np.dot(cov_matrix[target_idx, :], weights)

        if port_std > 1e-12:
            analytical_marginal_var = z_score * cov_i_portfolio / port_std
        else:
            analytical_marginal_var = 0.0

        # Numerical marginal VaR (finite difference)
        # VaR with position increased
        positions_up = dict(positions)
        positions_up[target_symbol] = positions_up.get(target_symbol, 0) + position_change
        portfolio_value_up = portfolio_value + position_change

        var_up_result = self.calculate_parametric_var(
            positions_up, portfolio_value_up, confidence_level
        )
        var_up = var_up_result.var_absolute

        # VaR with position decreased
        positions_down = dict(positions)
        positions_down[target_symbol] = positions_down.get(target_symbol, 0) - position_change
        portfolio_value_down = portfolio_value - position_change

        var_down_result = self.calculate_parametric_var(
            positions_down, portfolio_value_down, confidence_level
        )
        var_down = var_down_result.var_absolute

        # Central difference numerical derivative
        numerical_marginal_var = (var_up - var_down) / (2 * position_change)

        # Incremental VaR (actual change from adding position_change)
        incremental_var = var_up - current_var

        return {
            "target_symbol": target_symbol,
            "current_position": positions.get(target_symbol, 0),
            "position_change": position_change,
            "current_var": current_var,
            "var_after_increase": var_up,
            "var_after_decrease": var_down,
            "marginal_var": {
                "analytical": analytical_marginal_var,
                "numerical": numerical_marginal_var,
                "per_unit": numerical_marginal_var,
                "per_percent": numerical_marginal_var * portfolio_value / 100,
            },
            "incremental_var": incremental_var,
            "var_elasticity": (
                (incremental_var / current_var) / (position_change / portfolio_value)
                if current_var > 0 and portfolio_value > 0
                else 0
            ),
            "recommendation": (
                "REDUCE" if numerical_marginal_var > current_var / portfolio_value * 1.5
                else "INCREASE" if numerical_marginal_var < current_var / portfolio_value * 0.5
                else "NEUTRAL"
            ),
        }

    # =========================================================================
    # VAR BACKTESTING WITH KUPIEC TEST (P2)
    # =========================================================================

    def backtest_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        historical_returns: np.ndarray | None = None,
        confidence_level: float | None = None,
        method: VaRMethod = VaRMethod.PARAMETRIC,
    ) -> dict[str, Any]:
        """
        Backtest VaR model using Kupiec's POF test (P2).

        The Kupiec test (Proportion of Failures) tests whether the number
        of VaR exceedances is consistent with the model's confidence level.

        H0: The VaR model is correctly specified
        H1: The VaR model is misspecified

        Test statistic: LR = -2 * ln[(1-p)^(n-x) * p^x] + 2 * ln[(1-x/n)^(n-x) * (x/n)^x]
        where:
        - p = 1 - confidence_level (expected exceedance rate)
        - n = number of observations
        - x = number of actual exceedances

        Under H0, LR ~ chi-squared(1)

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            historical_returns: Array of historical portfolio returns (optional)
            confidence_level: Confidence level for VaR
            method: VaR calculation method

        Returns:
            Dictionary with backtesting results including Kupiec test
        """
        if confidence_level is None:
            confidence_level = self._confidence_level

        # Get historical portfolio returns
        if historical_returns is None:
            symbols = self._symbols
            weights = np.array([
                positions.get(s, 0) / portfolio_value for s in symbols
            ])

            min_len = min(len(r) for r in self._returns_data.values())
            returns_matrix = np.array([
                self._returns_data[s][-min_len:] for s in symbols
            ])

            historical_returns = np.dot(weights, returns_matrix)

        n_obs = len(historical_returns)
        if n_obs < 30:
            return {
                "error": "Insufficient observations for backtesting",
                "n_observations": n_obs,
                "minimum_required": 30,
            }

        # Calculate rolling VaR and count exceedances
        # Use expanding window for VaR estimation
        exceedances = []
        var_estimates = []
        actual_returns = []

        # Start after having enough data for estimation
        estimation_window = min(250, n_obs // 2)

        for t in range(estimation_window, n_obs):
            # Estimate VaR using data up to t
            window_returns = historical_returns[:t]
            window_std = np.std(window_returns)
            z_score = stats.norm.ppf(confidence_level)

            if method == VaRMethod.PARAMETRIC:
                var_estimate = z_score * window_std
            elif method == VaRMethod.HISTORICAL:
                var_estimate = -np.percentile(
                    window_returns, (1 - confidence_level) * 100
                )
            else:
                var_estimate = z_score * window_std  # Default to parametric

            var_estimates.append(var_estimate)
            actual_return = historical_returns[t]
            actual_returns.append(actual_return)

            # Check for exceedance (loss exceeds VaR)
            exceedance = actual_return < -var_estimate
            exceedances.append(exceedance)

        n_backtest = len(exceedances)
        n_exceedances = sum(exceedances)

        # Expected exceedances
        expected_rate = 1 - confidence_level
        expected_exceedances = expected_rate * n_backtest

        # Kupiec's POF test
        p = expected_rate  # Expected exceedance probability
        x = n_exceedances  # Actual exceedances
        n = n_backtest

        # Observed exceedance rate
        observed_rate = x / n if n > 0 else 0

        # Log-likelihood ratio test statistic
        # LR = -2 * [ln(L0) - ln(L1)]
        # L0 = (1-p)^(n-x) * p^x  (null hypothesis: model is correct)
        # L1 = (1-x/n)^(n-x) * (x/n)^x  (alternative: use observed rate)

        # Handle edge cases
        if x == 0:
            # No exceedances
            log_L0 = (n - x) * np.log(1 - p)
            log_L1 = (n - x) * np.log(1) if n == x else (n - x) * np.log(1)
            lr_statistic = -2 * (log_L0 - log_L1) if n > x else 0
        elif x == n:
            # All exceedances
            log_L0 = x * np.log(p)
            log_L1 = x * np.log(1)
            lr_statistic = -2 * (log_L0 - log_L1)
        else:
            # Normal case
            log_L0 = (n - x) * np.log(1 - p) + x * np.log(p)
            log_L1 = (n - x) * np.log(1 - x / n) + x * np.log(x / n)
            lr_statistic = -2 * (log_L0 - log_L1)

        # P-value from chi-squared distribution with 1 degree of freedom
        kupiec_p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)

        # Model assessment
        if kupiec_p_value >= 0.05:
            model_assessment = "PASS"
            assessment_detail = "VaR model is not rejected at 5% significance level"
        else:
            if observed_rate > expected_rate:
                model_assessment = "FAIL_CONSERVATIVE"
                assessment_detail = "Model underestimates risk (too many exceedances)"
            else:
                model_assessment = "FAIL_AGGRESSIVE"
                assessment_detail = "Model overestimates risk (too few exceedances)"

        # Additional statistics
        # Christoffersen independence test (simplified)
        # Tests whether exceedances are independent (clustering test)
        if n_exceedances >= 2:
            # Count transitions
            n_00 = sum(1 for i in range(len(exceedances) - 1)
                      if not exceedances[i] and not exceedances[i + 1])
            n_01 = sum(1 for i in range(len(exceedances) - 1)
                      if not exceedances[i] and exceedances[i + 1])
            n_10 = sum(1 for i in range(len(exceedances) - 1)
                      if exceedances[i] and not exceedances[i + 1])
            n_11 = sum(1 for i in range(len(exceedances) - 1)
                      if exceedances[i] and exceedances[i + 1])

            # Transition probabilities
            pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
            pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0

            clustering_detected = abs(pi_01 - pi_11) > 0.1
        else:
            clustering_detected = False
            pi_01 = pi_11 = 0

        return {
            "method": method.value,
            "confidence_level": confidence_level,
            "backtest_period": {
                "n_observations": n_obs,
                "n_backtest_days": n_backtest,
                "estimation_window": estimation_window,
            },
            "exceedances": {
                "actual": n_exceedances,
                "expected": expected_exceedances,
                "actual_rate": observed_rate,
                "expected_rate": expected_rate,
            },
            "kupiec_test": {
                "test_statistic": lr_statistic,
                "p_value": kupiec_p_value,
                "critical_value_5pct": stats.chi2.ppf(0.95, df=1),
                "reject_null": kupiec_p_value < 0.05,
            },
            "independence_test": {
                "clustering_detected": clustering_detected,
                "transition_prob_01": pi_01,
                "transition_prob_11": pi_11,
            },
            "model_assessment": {
                "result": model_assessment,
                "detail": assessment_detail,
                "exceedance_ratio": observed_rate / expected_rate if expected_rate > 0 else 0,
            },
            "var_statistics": {
                "mean_var": np.mean(var_estimates),
                "std_var": np.std(var_estimates),
                "max_var": np.max(var_estimates),
                "min_var": np.min(var_estimates),
            },
            "worst_exceedances": sorted(
                [actual_returns[i] for i, exc in enumerate(exceedances) if exc]
            )[:5] if n_exceedances > 0 else [],
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

    # =========================================================================
    # REGIME-CONDITIONAL VAR (Phase 2: Risk Management Enhancement)
    # =========================================================================

    # Regime-specific parameters based on research
    REGIME_PARAMETERS = {
        VolatilityRegime.LOW: {
            "volatility_multiplier": 1.3,  # Complacency adjustment - actual vol often higher
            "correlation_floor": None,      # No floor in low vol
            "confidence_boost": 0.02,       # Use 97% instead of 95%
            "description": "Low volatility - potential complacency, tighten estimates",
        },
        VolatilityRegime.NORMAL: {
            "volatility_multiplier": 1.0,   # Standard parameters
            "correlation_floor": None,
            "confidence_boost": 0.0,        # Standard 95%
            "description": "Normal volatility - use standard VaR parameters",
        },
        VolatilityRegime.HIGH: {
            "volatility_multiplier": 1.2,   # Increased uncertainty
            "correlation_floor": 0.5,       # Correlations tend to increase in stress
            "confidence_boost": 0.01,       # Use 96%
            "description": "High volatility - increase VaR estimates",
        },
        VolatilityRegime.CRISIS: {
            "volatility_multiplier": 1.5,   # Significant increase
            "correlation_floor": 0.7,       # Correlation breakdown (all assets move together)
            "confidence_boost": 0.04,       # Use 99%
            "description": "Crisis conditions - maximum VaR adjustments",
        },
    }

    def detect_volatility_regime(
        self,
        vix_current: float | None = None,
        realized_vol: float | None = None,
    ) -> VolatilityRegime:
        """
        Detect current volatility regime.

        Uses VIX if available, otherwise falls back to realized volatility.

        Args:
            vix_current: Current VIX level (preferred)
            realized_vol: Realized portfolio volatility (fallback)

        Returns:
            VolatilityRegime classification
        """
        # Use VIX if available
        if vix_current is not None:
            if vix_current < 15:
                return VolatilityRegime.LOW
            elif vix_current < 20:
                return VolatilityRegime.NORMAL
            elif vix_current < 30:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.CRISIS

        # Fallback to realized volatility (annualized)
        if realized_vol is not None:
            # Map realized vol to VIX-equivalent
            # Rough mapping: VIX ~= annualized vol * 100
            vol_pct = realized_vol * 100 * np.sqrt(252)  # Annualize if daily

            if vol_pct < 12:
                return VolatilityRegime.LOW
            elif vol_pct < 18:
                return VolatilityRegime.NORMAL
            elif vol_pct < 25:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.CRISIS

        # Default to normal if no volatility info
        return VolatilityRegime.NORMAL

    def calculate_regime_conditional_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        current_regime: VolatilityRegime | None = None,
        vix_current: float | None = None,
        confidence_level: float | None = None,
        horizon_days: int | None = None,
        method: VaRMethod = VaRMethod.PARAMETRIC,
    ) -> RegimeConditionalVaRResult:
        """
        Calculate VaR adjusted for current volatility regime (Phase 2).

        Implements regime-conditional risk adjustments based on research:
        - Low volatility: Increase VaR estimates (complacency risk)
        - High volatility: Widen confidence, increase correlations
        - Crisis: Maximum adjustments, correlation floor

        Based on research findings:
        - Daniel & Moskowitz (2016): Momentum crash protection
        - VIX-based regime detection from RISK_ENHANCEMENTS.md
        - Correlation breakdown in crisis from CORRELATIONS_RISK.md

        Args:
            positions: Dictionary of symbol to position value
            portfolio_value: Total portfolio value
            current_regime: Override regime (auto-detect if None)
            vix_current: Current VIX level (for auto-detection)
            confidence_level: Base confidence level
            horizon_days: Time horizon
            method: VaR calculation method

        Returns:
            RegimeConditionalVaRResult with regime-adjusted VaR
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days

        # Detect regime if not provided
        if current_regime is None:
            # Try to estimate realized vol from returns
            realized_vol = None
            if self._returns_data:
                symbols = self._symbols
                weights = np.array([positions.get(s, 0) / portfolio_value for s in symbols])

                if len(symbols) > 0 and all(len(self._returns_data.get(s, [])) > 0 for s in symbols):
                    min_len = min(len(self._returns_data[s]) for s in symbols)
                    if min_len > 0:
                        returns_matrix = np.array([
                            self._returns_data[s][-min_len:] for s in symbols
                        ])
                        portfolio_returns = np.dot(weights, returns_matrix)
                        if len(portfolio_returns) > 0:
                            realized_vol = np.std(portfolio_returns)

            current_regime = self.detect_volatility_regime(vix_current, realized_vol)

        # Get regime parameters
        regime_params = self.REGIME_PARAMETERS.get(
            current_regime,
            self.REGIME_PARAMETERS[VolatilityRegime.NORMAL]
        )

        vol_multiplier = regime_params["volatility_multiplier"]
        corr_floor = regime_params["correlation_floor"]
        conf_boost = regime_params["confidence_boost"]

        # Adjust confidence level
        adjusted_confidence = min(0.99, confidence_level + conf_boost)

        # Calculate base VaR first
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

        # For regime-adjusted VaR, use stress_test_var with regime parameters
        if corr_floor is not None or vol_multiplier != 1.0:
            # Use stress test infrastructure with regime-specific parameters
            stressed_result = self.stress_test_var(
                positions,
                portfolio_value,
                volatility_multiplier=vol_multiplier,
                correlation_override=corr_floor,
            )
            regime_var = stressed_result.var_absolute

            # Recalculate with adjusted confidence if different
            if adjusted_confidence != confidence_level:
                # Scale VaR by ratio of z-scores
                z_base = stats.norm.ppf(confidence_level)
                z_adjusted = stats.norm.ppf(adjusted_confidence)
                regime_var = regime_var * (z_adjusted / z_base) if z_base > 0 else regime_var
        else:
            # Just scale base VaR
            regime_var = base_result.var_absolute * vol_multiplier

            # Apply confidence adjustment
            if adjusted_confidence != confidence_level:
                z_base = stats.norm.ppf(confidence_level)
                z_adjusted = stats.norm.ppf(adjusted_confidence)
                regime_var = regime_var * (z_adjusted / z_base) if z_base > 0 else regime_var

        # Calculate adjusted Expected Shortfall
        regime_es = None
        if base_result.expected_shortfall is not None:
            # ES typically scales similarly to VaR
            var_ratio = regime_var / base_result.var_absolute if base_result.var_absolute > 0 else 1.0
            regime_es = base_result.expected_shortfall * var_ratio

        logger.info(
            f"Regime-conditional VaR [{current_regime.value}]: "
            f"base=${base_result.var_absolute:,.0f} -> regime=${regime_var:,.0f} "
            f"(vol_mult={vol_multiplier}, corr_floor={corr_floor}, "
            f"conf={confidence_level:.0%}->{adjusted_confidence:.0%})"
        )

        return RegimeConditionalVaRResult(
            base_var=base_result,
            regime=current_regime,
            regime_adjusted_var=regime_var,
            regime_adjusted_es=regime_es,
            volatility_multiplier=vol_multiplier,
            correlation_floor=corr_floor,
            confidence_adjustment=conf_boost,
        )

    def get_regime_risk_parameters(
        self,
        regime: VolatilityRegime
    ) -> dict[str, Any]:
        """
        Get risk parameters for a given volatility regime.

        Useful for position sizing and risk limit adjustments.

        Args:
            regime: Volatility regime

        Returns:
            Dictionary of regime-specific parameters
        """
        params = self.REGIME_PARAMETERS.get(
            regime,
            self.REGIME_PARAMETERS[VolatilityRegime.NORMAL]
        )

        # Add derived parameters useful for trading decisions
        return {
            **params,
            "regime": regime.value,
            "position_size_multiplier": 1.0 / params["volatility_multiplier"],
            "max_leverage_reduction": (
                0.0 if regime == VolatilityRegime.LOW
                else 0.0 if regime == VolatilityRegime.NORMAL
                else 0.25 if regime == VolatilityRegime.HIGH
                else 0.50  # Crisis
            ),
            "new_positions_allowed": regime != VolatilityRegime.CRISIS,
            "stop_loss_multiplier": (
                0.8 if regime == VolatilityRegime.LOW  # Tighter stops
                else 1.0 if regime == VolatilityRegime.NORMAL
                else 1.3 if regime == VolatilityRegime.HIGH  # Wider stops
                else 1.5  # Crisis - very wide to avoid whipsaws
            ),
        }

    # =========================================================================
    # Phase 5.3: Cornish-Fisher VaR Adjustment
    # =========================================================================

    def calculate_cornish_fisher_var(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
        horizon_days: int | None = None,
    ) -> VaRResult:
        """
        Calculate Cornish-Fisher adjusted VaR (Phase 5.3).

        The Cornish-Fisher expansion adjusts the normal distribution quantile
        to account for skewness and kurtosis in the return distribution.

        Standard parametric VaR assumes returns are normally distributed,
        but financial returns typically have:
        - Negative skewness (larger losses than gains)
        - Excess kurtosis (fat tails, more extreme events)

        The Cornish-Fisher expansion modifies the z-score:
        z_cf = z + (z² - 1)S/6 + (z³ - 3z)(K-3)/24 - (2z³ - 5z)S²/36

        Where:
        - z = normal z-score for confidence level
        - S = skewness of returns
        - K = kurtosis of returns

        Research finding: Cornish-Fisher VaR can be 20-40% higher than
        normal VaR during periods of market stress.

        Args:
            positions: Dictionary of symbol -> position value
            portfolio_value: Total portfolio value
            confidence_level: Confidence level (default: from config)
            horizon_days: Time horizon in days (default: from config)

        Returns:
            VaRResult with Cornish-Fisher adjusted values
        """
        if not HAS_NUMPY or not HAS_SCIPY:
            logger.error("NumPy/SciPy not available for Cornish-Fisher VaR")
            return VaRResult(
                method=VaRMethod.PARAMETRIC,
                confidence_level=confidence_level or self._confidence_level,
                horizon_days=horizon_days or self._horizon_days,
                var_absolute=0.0,
                var_pct=0.0,
                details={"error": "dependencies_not_available"},
            )

        if confidence_level is None:
            confidence_level = self._confidence_level
        if horizon_days is None:
            horizon_days = self._horizon_days

        # First calculate standard parametric VaR to get portfolio statistics
        base_result = self.calculate_parametric_var(
            positions, portfolio_value, confidence_level, horizon_days
        )

        if "error" in base_result.details:
            return base_result

        # Get historical returns for the portfolio
        symbols = list(positions.keys())
        weights = np.array([positions[s] / portfolio_value for s in symbols])

        # Build portfolio returns from individual asset returns
        portfolio_returns = []
        for date_idx in range(len(self._returns_cache.get(symbols[0], []))):
            port_return = 0.0
            valid = True
            for i, symbol in enumerate(symbols):
                asset_returns = self._returns_cache.get(symbol, [])
                if date_idx < len(asset_returns):
                    port_return += weights[i] * asset_returns[date_idx]
                else:
                    valid = False
                    break
            if valid:
                portfolio_returns.append(port_return)

        if len(portfolio_returns) < 30:
            # Insufficient data for higher moments, return standard VaR
            logger.warning(
                f"Insufficient data for Cornish-Fisher ({len(portfolio_returns)} < 30), "
                "using standard parametric VaR"
            )
            base_result.details["cornish_fisher_note"] = "insufficient_data"
            return base_result

        returns_array = np.array(portfolio_returns)

        # Calculate higher moments
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array, fisher=True)  # Excess kurtosis

        # Normal z-score
        z = stats.norm.ppf(confidence_level)

        # Cornish-Fisher expansion
        # For VaR (measuring losses), we negate skewness because:
        # - Returns with negative skew → Losses with positive skew
        # - We want the upper tail of the loss distribution
        z_cf = self._cornish_fisher_quantile(z, -skewness, kurtosis)

        # Calculate adjusted VaR
        portfolio_std = base_result.details.get("portfolio_volatility", 0.0)
        portfolio_std_scaled = portfolio_std * np.sqrt(horizon_days)

        var_cf_pct = z_cf * portfolio_std_scaled
        var_cf_absolute = var_cf_pct * portfolio_value

        # Adjusted Expected Shortfall using Cornish-Fisher
        # Use trapezoidal integration for ES beyond CF-adjusted VaR
        # Note: we use -skewness for VaR/ES calculation (loss perspective)
        es_cf = self._calculate_cornish_fisher_es(
            z, -skewness, kurtosis, portfolio_std_scaled, portfolio_value, confidence_level
        )

        # Calculate adjustment factor for monitoring
        adjustment_factor = var_cf_absolute / base_result.var_absolute if base_result.var_absolute > 0 else 1.0

        logger.info(
            f"Cornish-Fisher VaR: normal=${base_result.var_absolute:,.0f} -> "
            f"CF=${var_cf_absolute:,.0f} (adjustment={adjustment_factor:.2f}x), "
            f"skew={skewness:.3f}, excess_kurt={kurtosis:.3f}"
        )

        return VaRResult(
            method=VaRMethod.PARAMETRIC,  # Still parametric, just with adjustment
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            var_absolute=var_cf_absolute,
            var_pct=var_cf_pct,
            expected_shortfall=es_cf,
            details={
                "method_variant": "cornish_fisher",
                "portfolio_volatility": portfolio_std,
                "z_score_normal": z,
                "z_score_cf": z_cf,
                "skewness": skewness,
                "excess_kurtosis": kurtosis,
                "adjustment_factor": adjustment_factor,
                "normal_var_absolute": base_result.var_absolute,
            }
        )

    def _cornish_fisher_quantile(
        self,
        z: float,
        skewness: float,
        excess_kurtosis: float,
    ) -> float:
        """
        Calculate the Cornish-Fisher adjusted quantile.

        The expansion corrects the normal quantile for non-normality:
        z_cf = z + (z²-1)S/6 + (z³-3z)(K-3)/24 - (2z³-5z)S²/36

        Args:
            z: Normal distribution quantile
            skewness: Sample skewness
            excess_kurtosis: Excess kurtosis (kurtosis - 3)

        Returns:
            Adjusted quantile incorporating skewness and kurtosis
        """
        S = skewness
        K = excess_kurtosis

        # Cornish-Fisher expansion terms
        term1 = (z ** 2 - 1) * S / 6
        term2 = (z ** 3 - 3 * z) * K / 24
        term3 = -(2 * z ** 3 - 5 * z) * (S ** 2) / 36

        z_cf = z + term1 + term2 + term3

        # Sanity check: CF adjustment should not reduce z below zero
        # or increase it by more than 3x (numerical stability)
        z_cf = max(z_cf, z * 0.5)  # At least 50% of normal
        z_cf = min(z_cf, z * 3.0)  # At most 3x normal

        return z_cf

    def _calculate_cornish_fisher_es(
        self,
        z: float,
        skewness: float,
        excess_kurtosis: float,
        portfolio_std_scaled: float,
        portfolio_value: float,
        confidence_level: float,
    ) -> float:
        """
        Calculate Expected Shortfall with Cornish-Fisher adjustment.

        Uses numerical integration over the tail to compute ES.

        Args:
            z: Normal z-score
            skewness: Skewness
            excess_kurtosis: Excess kurtosis
            portfolio_std_scaled: Scaled portfolio volatility
            portfolio_value: Portfolio value
            confidence_level: Confidence level

        Returns:
            Cornish-Fisher adjusted Expected Shortfall
        """
        # Simple approximation: use ratio of CF-VaR to normal VaR
        # and apply same ratio to ES
        z_cf = self._cornish_fisher_quantile(z, skewness, excess_kurtosis)

        # Normal ES
        es_normal_mult = stats.norm.pdf(z) / (1 - confidence_level)
        es_normal = es_normal_mult * portfolio_std_scaled * portfolio_value

        # Scale ES by same ratio as VaR
        ratio = z_cf / z if z > 0 else 1.0
        es_cf = es_normal * ratio

        return es_cf

    def get_cornish_fisher_adjustment_factor(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        confidence_level: float | None = None,
    ) -> dict[str, Any]:
        """
        Get the Cornish-Fisher adjustment factor without full VaR calculation.

        Useful for quick assessment of how much non-normality affects VaR.

        Args:
            positions: Portfolio positions
            portfolio_value: Portfolio value
            confidence_level: Confidence level

        Returns:
            Dictionary with skewness, kurtosis, and adjustment factor
        """
        if confidence_level is None:
            confidence_level = self._confidence_level

        symbols = list(positions.keys())
        weights = np.array([positions[s] / portfolio_value for s in symbols])

        # Build portfolio returns
        portfolio_returns = []
        for date_idx in range(len(self._returns_cache.get(symbols[0], []))):
            port_return = 0.0
            valid = True
            for i, symbol in enumerate(symbols):
                asset_returns = self._returns_cache.get(symbol, [])
                if date_idx < len(asset_returns):
                    port_return += weights[i] * asset_returns[date_idx]
                else:
                    valid = False
                    break
            if valid:
                portfolio_returns.append(port_return)

        if len(portfolio_returns) < 30:
            return {
                "skewness": 0.0,
                "excess_kurtosis": 0.0,
                "adjustment_factor": 1.0,
                "sample_size": len(portfolio_returns),
                "warning": "insufficient_data",
            }

        returns_array = np.array(portfolio_returns)
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array, fisher=True)

        z = stats.norm.ppf(confidence_level)
        # For VaR adjustment, use -skewness (loss perspective)
        z_cf = self._cornish_fisher_quantile(z, -skewness, kurtosis)

        return {
            "skewness": skewness,  # Report actual return skewness
            "excess_kurtosis": kurtosis,
            "z_normal": z,
            "z_cornish_fisher": z_cf,
            "adjustment_factor": z_cf / z if z > 0 else 1.0,
            "sample_size": len(portfolio_returns),
            "interpretation": self._interpret_cf_adjustment(skewness, kurtosis),
        }

    def _interpret_cf_adjustment(
        self,
        skewness: float,
        excess_kurtosis: float,
    ) -> str:
        """Provide interpretation of the Cornish-Fisher adjustment."""
        messages = []

        if skewness < -0.5:
            messages.append("Strong negative skew: large losses more likely than gains")
        elif skewness < -0.2:
            messages.append("Moderate negative skew: slight left tail risk")
        elif skewness > 0.5:
            messages.append("Positive skew: upside surprises more likely")

        if excess_kurtosis > 3:
            messages.append("Very fat tails: extreme events significantly more likely")
        elif excess_kurtosis > 1:
            messages.append("Fat tails: extreme events more likely than normal")
        elif excess_kurtosis < -0.5:
            messages.append("Thin tails: extreme events less likely than normal")

        if not messages:
            messages.append("Distribution close to normal")

        return "; ".join(messages)

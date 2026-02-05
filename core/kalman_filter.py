"""
Kalman Filter for Dynamic Hedge Ratio Estimation
=================================================

Implements adaptive hedge ratio estimation using the Kalman filter,
which is superior to static OLS regression for pairs trading because:

1. Time-varying hedge ratios capture relationship changes
2. Optimal noise filtering via state estimation
3. Confidence intervals via covariance matrix
4. Real-time updates without lookback window

Theory:
-------
The Kalman filter models the hedge ratio as a hidden state:
- State equation: beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)
- Observation: y_t = x_t * beta_t + v_t, v_t ~ N(0, R)

Where:
- beta_t is the time-varying hedge ratio
- y_t is asset A price
- x_t is asset B price
- Q is process noise (how much beta can change)
- R is measurement noise (spread volatility)

MATURITY: BETA
--------------
- [x] Single-variable Kalman filter
- [x] Multi-variable Kalman filter (hedge ratio + intercept)
- [x] Adaptive noise estimation
- [x] Confidence intervals
- [ ] Online hyperparameter tuning (TODO)

Production Readiness:
- Unit tests: Required
- Backtesting: Compare with static OLS
- Live testing: Monitor beta stability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Current state of the Kalman filter."""
    beta: float  # Current hedge ratio estimate
    beta_std: float  # Standard deviation of beta estimate
    intercept: float  # Current intercept (if using 2-state model)
    intercept_std: float  # Standard deviation of intercept
    spread_value: float  # Current spread value
    spread_std: float  # Spread standard deviation (R estimate)
    n_updates: int  # Number of observations processed
    last_prediction_error: float  # Latest prediction error (innovation)
    forecast_error_variance: float  # Forecast error variance


@dataclass
class KalmanResult:
    """Result from Kalman filter update."""
    hedge_ratio: float
    hedge_ratio_std: float
    intercept: float
    intercept_std: float
    spread: float
    zscore: float
    kalman_gain: float
    is_stable: bool
    stability_score: float  # 0-1, higher is more stable
    n_observations: int


class KalmanHedgeRatio:
    """
    Kalman filter for dynamic hedge ratio estimation.

    Uses the Kalman filter to estimate time-varying hedge ratios
    between two assets. Superior to static OLS because:

    1. Adapts to changing market conditions
    2. Provides confidence intervals via state covariance
    3. Optimal filtering of noise
    4. No lookback window required (recursive)

    Usage:
    ------
    ```python
    kf = KalmanHedgeRatio(delta=1e-4, ve=1e-3)

    for t in range(len(prices_a)):
        result = kf.update(prices_a[t], prices_b[t])
        hedge_ratio = result.hedge_ratio
        spread = prices_a[t] - hedge_ratio * prices_b[t]
    ```

    Parameters:
    -----------
    delta : float
        Process noise variance scaling. Higher = beta can change faster.
        Typical range: 1e-5 to 1e-3
        Default: 1e-4 (balanced between responsiveness and stability)

    ve : float
        Initial measurement noise estimate.
        Will be adaptively updated as more data arrives.
        Default: 1e-3

    include_intercept : bool
        If True, estimate intercept in addition to hedge ratio.
        Recommended for pairs that don't cross through origin.
        Default: True

    warmup_period : int
        Number of observations before filter is considered warmed up.
        Default: 30
    """

    def __init__(
        self,
        delta: float = 1e-4,
        ve: float = 1e-3,
        include_intercept: bool = True,
        warmup_period: int = 30,
        adaptive_noise: bool = True,
    ):
        self._delta = delta
        self._ve = ve
        self._include_intercept = include_intercept
        self._warmup_period = warmup_period
        self._adaptive_noise = adaptive_noise

        # State dimension: 1 for beta only, 2 for beta + intercept
        self._n_states = 2 if include_intercept else 1

        # Initialize state vector (beta, intercept) or just (beta)
        self._x = np.zeros(self._n_states)  # State estimate [beta, intercept]
        self._x[0] = 1.0  # Initial beta guess

        # Initialize state covariance matrix (uncertainty)
        self._P = np.eye(self._n_states)  # Large initial uncertainty

        # Process noise covariance (Q) - how much state can change
        self._Q = np.eye(self._n_states) * delta

        # Measurement noise variance (R) - spread volatility
        self._R = ve

        # Adaptive R estimation
        self._innovation_history: list[float] = []
        self._max_innovation_history = 100

        # Tracking
        self._n_updates = 0
        self._last_innovation = 0.0
        self._forecast_variance = 0.0

        # Spread statistics (for z-score)
        self._spread_mean = 0.0
        self._spread_var = 0.0
        self._spread_history: list[float] = []
        self._max_spread_history = 200

    def reset(self) -> None:
        """Reset filter to initial state."""
        self._x = np.zeros(self._n_states)
        self._x[0] = 1.0
        self._P = np.eye(self._n_states)
        self._R = self._ve
        self._n_updates = 0
        self._innovation_history = []
        self._spread_history = []
        self._spread_mean = 0.0
        self._spread_var = 0.0

    def update(self, price_a: float, price_b: float) -> KalmanResult:
        """
        Update Kalman filter with new price observations.

        Args:
            price_a: Price of asset A (dependent variable)
            price_b: Price of asset B (independent variable)

        Returns:
            KalmanResult with updated hedge ratio and diagnostics
        """
        # Build observation matrix H
        if self._include_intercept:
            H = np.array([[price_b, 1.0]])  # y = beta*x + intercept
        else:
            H = np.array([[price_b]])  # y = beta*x

        # Prediction step
        # State prediction: x_pred = x (random walk model)
        x_pred = self._x.copy()

        # Covariance prediction: P_pred = P + Q
        P_pred = self._P + self._Q

        # Innovation (measurement residual)
        # y = price_a, prediction = H @ x_pred
        y_pred = H @ x_pred
        innovation = price_a - y_pred[0]

        # Innovation covariance: S = H @ P_pred @ H' + R
        S = H @ P_pred @ H.T + self._R
        S_scalar = S[0, 0]

        # Kalman gain: K = P_pred @ H' @ S^-1
        if S_scalar > 1e-10:
            K = (P_pred @ H.T) / S_scalar
        else:
            K = np.zeros((self._n_states, 1))

        # State update: x = x_pred + K * innovation
        self._x = x_pred + (K * innovation).flatten()

        # Covariance update: P = (I - K @ H) @ P_pred
        self._P = (np.eye(self._n_states) - K @ H) @ P_pred

        # Ensure P remains symmetric positive definite
        self._P = (self._P + self._P.T) / 2
        min_var = 1e-8
        for i in range(self._n_states):
            if self._P[i, i] < min_var:
                self._P[i, i] = min_var

        # Store innovation for adaptive R
        self._innovation_history.append(innovation)
        if len(self._innovation_history) > self._max_innovation_history:
            self._innovation_history.pop(0)

        # Adaptive R estimation (measurement noise)
        if self._adaptive_noise and len(self._innovation_history) >= 20:
            # R should be approximately the variance of innovations
            innovations = np.array(self._innovation_history)
            self._R = np.var(innovations)
            # Add floor to prevent R from going to zero
            self._R = max(self._R, 1e-6)

        # Calculate spread and z-score
        beta = self._x[0]
        intercept = self._x[1] if self._include_intercept else 0.0
        spread = price_a - beta * price_b - intercept

        # Update spread statistics
        self._spread_history.append(spread)
        if len(self._spread_history) > self._max_spread_history:
            self._spread_history.pop(0)

        if len(self._spread_history) >= 5:
            spreads = np.array(self._spread_history)
            self._spread_mean = np.mean(spreads)
            self._spread_var = np.var(spreads)

        # Calculate z-score
        if self._spread_var > 1e-10:
            zscore = (spread - self._spread_mean) / np.sqrt(self._spread_var)
        else:
            zscore = 0.0

        # Update tracking
        self._n_updates += 1
        self._last_innovation = innovation
        self._forecast_variance = S_scalar

        # Calculate stability metrics
        beta_std = np.sqrt(self._P[0, 0])
        intercept_std = np.sqrt(self._P[1, 1]) if self._include_intercept else 0.0

        # Stability score based on:
        # 1. Coefficient of variation of beta
        # 2. Innovation behavior
        # 3. Number of observations
        is_warmed_up = self._n_updates >= self._warmup_period

        cv_beta = beta_std / abs(beta) if abs(beta) > 1e-8 else 1.0
        cv_score = max(0.0, 1.0 - cv_beta * 2)  # Lower CV = higher score

        warmup_score = min(1.0, self._n_updates / self._warmup_period)

        stability_score = cv_score * 0.7 + warmup_score * 0.3

        is_stable = is_warmed_up and cv_score > 0.5

        return KalmanResult(
            hedge_ratio=beta,
            hedge_ratio_std=beta_std,
            intercept=intercept,
            intercept_std=intercept_std,
            spread=spread,
            zscore=zscore,
            kalman_gain=K[0, 0] if K.size > 0 else 0.0,
            is_stable=is_stable,
            stability_score=stability_score,
            n_observations=self._n_updates,
        )

    def get_state(self) -> KalmanState:
        """Get current filter state."""
        beta = self._x[0]
        beta_std = np.sqrt(self._P[0, 0])
        intercept = self._x[1] if self._include_intercept else 0.0
        intercept_std = np.sqrt(self._P[1, 1]) if self._include_intercept else 0.0

        spread = self._spread_history[-1] if self._spread_history else 0.0
        spread_std = np.sqrt(self._spread_var) if self._spread_var > 0 else 0.0

        return KalmanState(
            beta=beta,
            beta_std=beta_std,
            intercept=intercept,
            intercept_std=intercept_std,
            spread_value=spread,
            spread_std=spread_std,
            n_updates=self._n_updates,
            last_prediction_error=self._last_innovation,
            forecast_error_variance=self._forecast_variance,
        )

    def get_hedge_ratio(self) -> float:
        """Get current hedge ratio estimate."""
        return self._x[0]

    def get_hedge_ratio_with_confidence(
        self,
        confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """
        Get hedge ratio with confidence interval.

        Args:
            confidence: Confidence level (0.95 = 95% CI)

        Returns:
            (hedge_ratio, lower_bound, upper_bound)
        """
        from scipy import stats as scipy_stats

        beta = self._x[0]
        beta_std = np.sqrt(self._P[0, 0])

        # Z-score for confidence level
        z = scipy_stats.norm.ppf((1 + confidence) / 2)

        lower = beta - z * beta_std
        upper = beta + z * beta_std

        return beta, lower, upper

    def get_spread(self, price_a: float, price_b: float) -> float:
        """
        Calculate spread using current hedge ratio.

        Args:
            price_a: Price of asset A
            price_b: Price of asset B

        Returns:
            Spread value
        """
        beta = self._x[0]
        intercept = self._x[1] if self._include_intercept else 0.0
        return price_a - beta * price_b - intercept

    def get_zscore(self, price_a: float, price_b: float) -> float:
        """
        Calculate z-score of spread using current hedge ratio.

        Args:
            price_a: Price of asset A
            price_b: Price of asset B

        Returns:
            Z-score of spread
        """
        spread = self.get_spread(price_a, price_b)

        if self._spread_var > 1e-10:
            return (spread - self._spread_mean) / np.sqrt(self._spread_var)
        return 0.0

    def process_series(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> list[KalmanResult]:
        """
        Process full price series and return results at each step.

        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B

        Returns:
            List of KalmanResult for each observation
        """
        if len(prices_a) != len(prices_b):
            raise ValueError("Price series must have same length")

        results = []
        for i in range(len(prices_a)):
            result = self.update(prices_a[i], prices_b[i])
            results.append(result)

        return results

    def get_spread_series(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process price series and return spread, hedge ratio, and z-score series.

        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B

        Returns:
            (spread_series, hedge_ratio_series, zscore_series)
        """
        results = self.process_series(prices_a, prices_b)

        spreads = np.array([r.spread for r in results])
        hedge_ratios = np.array([r.hedge_ratio for r in results])
        zscores = np.array([r.zscore for r in results])

        return spreads, hedge_ratios, zscores

    def compare_with_ols(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        ols_window: int = 60
    ) -> dict[str, Any]:
        """
        Compare Kalman filter hedge ratio with rolling OLS.

        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            ols_window: Window for rolling OLS

        Returns:
            Comparison metrics
        """
        # Reset and process with Kalman
        self.reset()
        kalman_results = self.process_series(prices_a, prices_b)
        kalman_betas = np.array([r.hedge_ratio for r in kalman_results])
        kalman_spreads = np.array([r.spread for r in kalman_results])

        # Calculate rolling OLS hedge ratios
        n = len(prices_a)
        ols_betas = np.full(n, np.nan)
        ols_spreads = np.full(n, np.nan)

        for i in range(ols_window, n):
            window_a = prices_a[i - ols_window:i]
            window_b = prices_b[i - ols_window:i]

            # OLS regression
            var_b = np.var(window_b)
            if var_b > 1e-10:
                cov = np.cov(window_a, window_b)[0, 1]
                beta = cov / var_b
                ols_betas[i] = beta
                ols_spreads[i] = prices_a[i] - beta * prices_b[i]

        # Calculate comparison metrics (only for warmed-up period)
        valid_idx = ~np.isnan(ols_betas)

        if not valid_idx.any():
            return {"error": "Insufficient data for comparison"}

        kalman_valid = kalman_betas[valid_idx]
        ols_valid = ols_betas[valid_idx]

        beta_diff = kalman_valid - ols_valid

        # Beta stability (lower std = more stable)
        kalman_beta_std = np.std(np.diff(kalman_valid))
        ols_beta_std = np.std(np.diff(ols_valid))

        # Spread variance (lower = better for trading)
        kalman_spread_var = np.var(kalman_spreads[valid_idx])
        ols_spread_var = np.var(ols_spreads[valid_idx])

        return {
            "kalman_beta_mean": float(np.mean(kalman_valid)),
            "ols_beta_mean": float(np.mean(ols_valid)),
            "beta_difference_mean": float(np.mean(beta_diff)),
            "beta_difference_std": float(np.std(beta_diff)),
            "kalman_beta_stability": float(kalman_beta_std),
            "ols_beta_stability": float(ols_beta_std),
            "stability_improvement_pct": float(
                (ols_beta_std - kalman_beta_std) / ols_beta_std * 100
                if ols_beta_std > 0 else 0
            ),
            "kalman_spread_variance": float(kalman_spread_var),
            "ols_spread_variance": float(ols_spread_var),
            "spread_variance_improvement_pct": float(
                (ols_spread_var - kalman_spread_var) / ols_spread_var * 100
                if ols_spread_var > 0 else 0
            ),
            "n_observations": int(np.sum(valid_idx)),
        }

    def get_status(self) -> dict[str, Any]:
        """Get filter status for monitoring."""
        state = self.get_state()

        return {
            "hedge_ratio": state.beta,
            "hedge_ratio_std": state.beta_std,
            "intercept": state.intercept,
            "spread": state.spread_value,
            "spread_std": state.spread_std,
            "n_observations": state.n_updates,
            "is_warmed_up": state.n_updates >= self._warmup_period,
            "measurement_noise_R": self._R,
            "process_noise_delta": self._delta,
            "state_covariance_trace": float(np.trace(self._P)),
        }


class MultiPairKalmanFilter:
    """
    Manages Kalman filters for multiple trading pairs.

    Provides a convenient interface for maintaining separate
    Kalman filters for each pair being tracked.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        ve: float = 1e-3,
        include_intercept: bool = True,
        warmup_period: int = 30,
    ):
        self._delta = delta
        self._ve = ve
        self._include_intercept = include_intercept
        self._warmup_period = warmup_period

        # Filters by pair key
        self._filters: dict[str, KalmanHedgeRatio] = {}

    def _get_pair_key(self, symbol_a: str, symbol_b: str) -> str:
        """Generate consistent key for a pair."""
        return f"{symbol_a}:{symbol_b}"

    def get_or_create_filter(
        self,
        symbol_a: str,
        symbol_b: str
    ) -> KalmanHedgeRatio:
        """Get existing filter or create new one for pair."""
        key = self._get_pair_key(symbol_a, symbol_b)

        if key not in self._filters:
            self._filters[key] = KalmanHedgeRatio(
                delta=self._delta,
                ve=self._ve,
                include_intercept=self._include_intercept,
                warmup_period=self._warmup_period,
            )
            logger.info(f"Created Kalman filter for pair {key}")

        return self._filters[key]

    def update(
        self,
        symbol_a: str,
        symbol_b: str,
        price_a: float,
        price_b: float
    ) -> KalmanResult:
        """
        Update filter for a pair with new prices.

        Args:
            symbol_a: Symbol of asset A
            symbol_b: Symbol of asset B
            price_a: Price of asset A
            price_b: Price of asset B

        Returns:
            KalmanResult with updated state
        """
        kf = self.get_or_create_filter(symbol_a, symbol_b)
        return kf.update(price_a, price_b)

    def get_hedge_ratio(self, symbol_a: str, symbol_b: str) -> float | None:
        """Get current hedge ratio for a pair."""
        key = self._get_pair_key(symbol_a, symbol_b)

        if key not in self._filters:
            return None

        return self._filters[key].get_hedge_ratio()

    def get_all_states(self) -> dict[str, KalmanState]:
        """Get states for all tracked pairs."""
        return {key: kf.get_state() for key, kf in self._filters.items()}

    def reset_pair(self, symbol_a: str, symbol_b: str) -> None:
        """Reset filter for a specific pair."""
        key = self._get_pair_key(symbol_a, symbol_b)

        if key in self._filters:
            self._filters[key].reset()
            logger.info(f"Reset Kalman filter for pair {key}")

    def reset_all(self) -> None:
        """Reset all filters."""
        for kf in self._filters.values():
            kf.reset()
        logger.info(f"Reset all {len(self._filters)} Kalman filters")

    def remove_pair(self, symbol_a: str, symbol_b: str) -> None:
        """Remove filter for a pair."""
        key = self._get_pair_key(symbol_a, symbol_b)

        if key in self._filters:
            del self._filters[key]
            logger.info(f"Removed Kalman filter for pair {key}")

    def get_status(self) -> dict[str, Any]:
        """Get status of all filters."""
        return {
            "n_pairs": len(self._filters),
            "pairs": list(self._filters.keys()),
            "filters": {
                key: kf.get_status() for key, kf in self._filters.items()
            },
        }


def create_kalman_filter(
    config: dict[str, Any] | None = None
) -> KalmanHedgeRatio:
    """
    Factory function to create a Kalman filter from config.

    Args:
        config: Configuration dictionary with optional keys:
            - delta: Process noise variance (default: 1e-4)
            - ve: Initial measurement noise (default: 1e-3)
            - include_intercept: Whether to estimate intercept (default: True)
            - warmup_period: Warmup period (default: 30)
            - adaptive_noise: Whether to adaptively estimate R (default: True)

    Returns:
        Configured KalmanHedgeRatio instance
    """
    config = config or {}

    return KalmanHedgeRatio(
        delta=config.get("delta", 1e-4),
        ve=config.get("ve", 1e-3),
        include_intercept=config.get("include_intercept", True),
        warmup_period=config.get("warmup_period", 30),
        adaptive_noise=config.get("adaptive_noise", True),
    )


def create_multi_pair_filter(
    config: dict[str, Any] | None = None
) -> MultiPairKalmanFilter:
    """
    Factory function to create a multi-pair Kalman filter manager.

    Args:
        config: Configuration dictionary (same as create_kalman_filter)

    Returns:
        Configured MultiPairKalmanFilter instance
    """
    config = config or {}

    return MultiPairKalmanFilter(
        delta=config.get("delta", 1e-4),
        ve=config.get("ve", 1e-3),
        include_intercept=config.get("include_intercept", True),
        warmup_period=config.get("warmup_period", 30),
    )

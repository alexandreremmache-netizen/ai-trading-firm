"""
Correlation Manager
===================

Manages correlation analysis for portfolio risk management.
Provides rolling correlation matrices, concentration metrics,
and regime change detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Market correlation regime classification."""
    LOW = "low"  # Average correlation < 0.3
    NORMAL = "normal"  # 0.3 <= correlation < 0.6
    HIGH = "high"  # 0.6 <= correlation < 0.8
    CRISIS = "crisis"  # correlation >= 0.8


class StressIndicator(Enum):
    """Market stress indicator type (#R7)."""
    VIX_SPIKE = "vix_spike"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN = "drawdown"
    EXTERNAL_SIGNAL = "external_signal"


@dataclass
class CorrelationAlert:
    """Alert for correlation regime change or threshold breach."""
    alert_id: str
    timestamp: datetime
    alert_type: str  # "regime_change", "threshold_breach", "concentration"
    severity: str  # "info", "warning", "critical"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class CorrelationSnapshot:
    """Point-in-time correlation analysis."""
    timestamp: datetime
    correlation_matrix: pd.DataFrame
    average_correlation: float
    max_pairwise_correlation: float
    min_pairwise_correlation: float
    herfindahl_index: float
    effective_n: float
    regime: CorrelationRegime
    eigenvalues: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "average_correlation": self.average_correlation,
            "max_pairwise_correlation": self.max_pairwise_correlation,
            "min_pairwise_correlation": self.min_pairwise_correlation,
            "herfindahl_index": self.herfindahl_index,
            "effective_n": self.effective_n,
            "regime": self.regime.value,
        }


class CorrelationManager:
    """
    Manages correlation analysis for the trading system.

    Features:
    - Rolling correlation matrix calculation
    - Herfindahl index (concentration) computation
    - Effective diversification count (effective N)
    - Correlation regime detection
    - Alert generation on regime changes
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize correlation manager.

        Args:
            config: Configuration dictionary with:
                - lookback_days: Rolling window size (default: 60)
                - max_pairwise_correlation: Alert threshold (default: 0.85)
                - min_history_days: Minimum data for calculation (default: 20)
                - regime_change_threshold: Sensitivity for regime detection (default: 0.15)
        """
        self._config = config or {}
        self._lookback_days = self._config.get("lookback_days", 60)
        self._max_pairwise_correlation = self._config.get("max_pairwise_correlation", 0.85)
        self._min_history_days = self._config.get("min_history_days", 20)
        self._regime_change_threshold = self._config.get("regime_change_threshold", 0.15)

        # State
        self._returns_history: dict[str, list[tuple[datetime, float]]] = {}
        self._last_snapshot: CorrelationSnapshot | None = None
        self._alert_history: list[CorrelationAlert] = []
        self._alert_counter = 0

        # Stress period handling (#R7)
        self._stress_mode = False
        self._stress_start_time: datetime | None = None
        self._stress_indicators: dict[StressIndicator, float] = {}
        self._stress_lookback_days = self._config.get("stress_lookback_days", 20)  # Shorter window during stress
        self._stress_decay_factor = self._config.get("stress_decay_factor", 0.90)  # Faster decay during stress
        self._vix_stress_threshold = self._config.get("vix_stress_threshold", 30.0)
        self._vol_spike_threshold = self._config.get("vol_spike_threshold", 2.0)  # 2x normal vol
        self._current_vix: float | None = None
        self._baseline_volatility: dict[str, float] = {}
        self._stress_correlation_cache: pd.DataFrame | None = None

        logger.info(
            f"CorrelationManager initialized: "
            f"lookback={self._lookback_days}d, stress_lookback={self._stress_lookback_days}d, "
            f"max_corr={self._max_pairwise_correlation}"
        )

    def add_returns(self, symbol: str, timestamp: datetime, returns: float) -> None:
        """
        Add returns observation for a symbol.

        Args:
            symbol: Instrument symbol
            timestamp: Observation timestamp
            returns: Period returns (e.g., daily returns)
        """
        if symbol not in self._returns_history:
            self._returns_history[symbol] = []

        self._returns_history[symbol].append((timestamp, returns))

        # Trim to lookback window
        cutoff = len(self._returns_history[symbol]) - self._lookback_days * 2
        if cutoff > 0:
            self._returns_history[symbol] = self._returns_history[symbol][cutoff:]

    def add_returns_batch(
        self,
        returns_df: pd.DataFrame,
        timestamp: datetime | None = None
    ) -> None:
        """
        Add batch of returns from a DataFrame.

        Args:
            returns_df: DataFrame with symbols as columns and returns as values
            timestamp: Optional timestamp (uses current time if not provided)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        for symbol in returns_df.columns:
            for idx, value in returns_df[symbol].dropna().items():
                if isinstance(idx, datetime):
                    ts = idx
                else:
                    ts = timestamp
                self.add_returns(symbol, ts, float(value))

    def calculate_correlation_matrix(
        self,
        symbols: list[str] | None = None,
        lookback_days: int | None = None,
        use_ewma: bool = False,
        ewma_decay: float | None = None
    ) -> pd.DataFrame | None:
        """
        Calculate rolling correlation matrix.

        Args:
            symbols: List of symbols to include (default: all available)
            lookback_days: Rolling window (default: configured lookback)
            use_ewma: If True, use EWMA (exponentially weighted) correlation (CORR-P1-1 fix)
            ewma_decay: Decay factor for EWMA (0 < decay < 1, lower = more weight on recent).
                       Default uses stress_decay_factor from config.

        Returns:
            Correlation matrix as DataFrame, or None if insufficient data
        """
        if lookback_days is None:
            lookback_days = self._lookback_days

        if symbols is None:
            symbols = list(self._returns_history.keys())

        if len(symbols) < 2:
            logger.warning("Need at least 2 symbols for correlation matrix")
            return None

        # Use EWMA correlation if requested (CORR-P1-1 fix)
        if use_ewma:
            decay = ewma_decay if ewma_decay is not None else self._stress_decay_factor
            return self._calculate_ewma_correlation_matrix(symbols, lookback_days, decay)

        # Build returns DataFrame
        returns_data = {}
        for symbol in symbols:
            if symbol in self._returns_history:
                history = self._returns_history[symbol]
                if len(history) >= self._min_history_days:
                    # Get last N observations
                    recent = history[-lookback_days:]
                    returns_data[symbol] = pd.Series(
                        [r[1] for r in recent],
                        index=[r[0] for r in recent]
                    )

        if len(returns_data) < 2:
            logger.warning("Insufficient data for correlation calculation")
            return None

        # Create DataFrame and calculate correlation
        returns_df = pd.DataFrame(returns_data)

        # Align dates and drop missing
        returns_df = returns_df.dropna(how="any")

        if len(returns_df) < self._min_history_days:
            logger.warning(
                f"Insufficient overlapping data: {len(returns_df)} < {self._min_history_days}"
            )
            return None

        return returns_df.corr()

    def calculate_snapshot(
        self,
        symbols: list[str] | None = None,
        weights: dict[str, float] | None = None
    ) -> CorrelationSnapshot | None:
        """
        Calculate complete correlation snapshot.

        Args:
            symbols: Symbols to analyze (default: all)
            weights: Portfolio weights by symbol (for weighted calculations)

        Returns:
            CorrelationSnapshot with all metrics, or None if insufficient data
        """
        corr_matrix = self.calculate_correlation_matrix(symbols)
        if corr_matrix is None:
            return None

        timestamp = datetime.now(timezone.utc)

        # Extract pairwise correlations (upper triangle, excluding diagonal)
        n = len(corr_matrix)
        pairwise_corrs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_corrs.append(corr_matrix.iloc[i, j])

        pairwise_corrs = np.array(pairwise_corrs)

        # Calculate metrics
        avg_corr = float(np.mean(pairwise_corrs)) if len(pairwise_corrs) > 0 else 0.0
        max_corr = float(np.max(pairwise_corrs)) if len(pairwise_corrs) > 0 else 0.0
        min_corr = float(np.min(pairwise_corrs)) if len(pairwise_corrs) > 0 else 0.0

        # Herfindahl Index (concentration)
        hhi = self._calculate_herfindahl(weights, list(corr_matrix.columns))

        # Effective N (diversification)
        effective_n = self._calculate_effective_n(corr_matrix, weights)

        # Eigenvalue analysis
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Eigenvalue calculation failed: {e}, falling back to identity matrix eigenvalues")
            # Fallback to identity matrix eigenvalues (all 1s)
            eigenvalues = np.ones(n)

        # Determine regime
        regime = self._classify_regime(avg_corr)

        snapshot = CorrelationSnapshot(
            timestamp=timestamp,
            correlation_matrix=corr_matrix,
            average_correlation=avg_corr,
            max_pairwise_correlation=max_corr,
            min_pairwise_correlation=min_corr,
            herfindahl_index=hhi,
            effective_n=effective_n,
            regime=regime,
            eigenvalues=eigenvalues,
        )

        # Check for alerts
        self._check_alerts(snapshot)

        # Store as last snapshot
        self._last_snapshot = snapshot

        return snapshot

    def _calculate_herfindahl(
        self,
        weights: dict[str, float] | None,
        symbols: list[str]
    ) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration.

        HHI = sum(w_i^2) where w_i are portfolio weights
        Range: 1/n (perfectly diversified) to 1 (fully concentrated)

        Args:
            weights: Portfolio weights by symbol
            symbols: List of symbols

        Returns:
            HHI value
        """
        if weights is None or len(weights) == 0:
            # Equal weights assumption
            n = len(symbols)
            return 1.0 / n if n > 0 else 1.0

        # Normalize weights
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight == 0:
            return 1.0

        normalized = {s: abs(weights.get(s, 0)) / total_weight for s in symbols}

        # Calculate HHI
        hhi = sum(w ** 2 for w in normalized.values())

        return float(hhi)

    def _calculate_effective_n(
        self,
        corr_matrix: pd.DataFrame,
        weights: dict[str, float] | None
    ) -> float:
        """
        Calculate effective number of independent bets.

        Uses eigenvalue decomposition of the correlation matrix for unweighted case.
        For weighted portfolio, uses the weighted effective N formula.

        Args:
            corr_matrix: Correlation matrix
            weights: Portfolio weights (if provided, uses weighted calculation)

        Returns:
            Effective N (1 = fully correlated, N = fully diversified)
        """
        n = len(corr_matrix)

        # Use weighted calculation if weights provided (PM-10 fix)
        if weights is not None and len(weights) > 0:
            return self._calculate_effective_n_weighted(corr_matrix, weights)

        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

            sum_eig = np.sum(eigenvalues)
            sum_eig_sq = np.sum(eigenvalues ** 2)

            if sum_eig_sq > 0:
                effective_n = (sum_eig ** 2) / sum_eig_sq
            else:
                effective_n = 1.0

            return float(effective_n)

        except Exception as e:
            logger.warning(f"Effective N calculation failed: {e}")
            return float(n)  # Return actual N as fallback

    def _calculate_effective_n_weighted(
        self,
        corr_matrix: pd.DataFrame,
        weights: dict[str, float]
    ) -> float:
        """
        Calculate effective N with dynamic portfolio weights (PM-10 fix).

        Uses the formula: Effective N = 1 / sum(w_i^2 * (1 + sum(w_j * corr_ij for j != i)))
        This accounts for both position concentration and correlation effects.

        For a portfolio with equal weights and zero correlations, this returns N.
        For a fully correlated portfolio, this approaches 1.

        Args:
            corr_matrix: Correlation matrix
            weights: Portfolio weights by symbol

        Returns:
            Weighted effective N (1 = fully correlated, N = fully diversified)
        """
        symbols = list(corr_matrix.columns)
        n = len(symbols)

        if n == 0:
            return 1.0

        # Build weight vector, using 0 for missing symbols
        w = np.array([weights.get(s, 0.0) for s in symbols])

        # Handle all-zero weights
        total_weight = np.sum(np.abs(w))
        if total_weight == 0:
            # Fall back to equal weights
            w = np.ones(n) / n
        else:
            # Normalize weights (use absolute values for long/short)
            w = np.abs(w) / total_weight

        try:
            corr_values = corr_matrix.values

            # Calculate weighted effective N
            # Formula: 1 / sum_i(w_i^2 * (1 + sum_j!=i(w_j * corr_ij)))
            # Simplified: 1 / (w^T * (I + diag(w)^{-1} * C * diag(w) - I) * w)
            # Which reduces to: 1 / (w^T * C * w) when normalized

            # Alternative clearer formula:
            # effective_n = 1 / sum_i(w_i * sum_j(w_j * corr_ij))
            # This is equivalent to 1 / (w^T * C * w)
            portfolio_correlation = np.dot(w, np.dot(corr_values, w))

            # Portfolio correlation should be between 0 and 1
            # When all correlations are 1, portfolio_correlation = 1
            # When all correlations are 0 (diagonal), portfolio_correlation = sum(w_i^2) = HHI
            portfolio_correlation = max(portfolio_correlation, 1e-10)

            # Effective N = 1 / portfolio_correlation
            # But this needs adjustment for the HHI baseline
            # Better formula: effective_n = HHI / portfolio_correlation * n
            hhi = np.sum(w ** 2)

            # The ratio tells us diversification benefit
            # For uncorrelated assets: portfolio_correlation = HHI
            # For fully correlated: portfolio_correlation = 1
            if portfolio_correlation > 0:
                # Effective N ranges from 1 (fully correlated) to 1/HHI (uncorrelated)
                # Scale to be comparable with eigenvalue-based effective N
                effective_n = hhi / portfolio_correlation * n if portfolio_correlation > hhi else n

                # Cap at actual number of positions
                effective_n = min(effective_n, n)
                effective_n = max(effective_n, 1.0)
            else:
                effective_n = float(n)

            return float(effective_n)

        except Exception as e:
            logger.warning(f"Weighted effective N calculation failed: {e}")
            return float(n)

    def _classify_regime(self, avg_correlation: float) -> CorrelationRegime:
        """
        Classify market correlation regime.

        Args:
            avg_correlation: Average pairwise correlation

        Returns:
            CorrelationRegime enum value
        """
        if avg_correlation >= 0.8:
            return CorrelationRegime.CRISIS
        elif avg_correlation >= 0.6:
            return CorrelationRegime.HIGH
        elif avg_correlation >= 0.3:
            return CorrelationRegime.NORMAL
        else:
            return CorrelationRegime.LOW

    def _check_alerts(self, snapshot: CorrelationSnapshot) -> None:
        """
        Check for alert conditions and generate alerts.

        Args:
            snapshot: Current correlation snapshot
        """
        # Check for regime change
        if self._last_snapshot is not None:
            if snapshot.regime != self._last_snapshot.regime:
                self._generate_alert(
                    alert_type="regime_change",
                    severity="warning" if snapshot.regime in [CorrelationRegime.HIGH, CorrelationRegime.CRISIS] else "info",
                    message=f"Correlation regime changed from {self._last_snapshot.regime.value} to {snapshot.regime.value}",
                    details={
                        "previous_regime": self._last_snapshot.regime.value,
                        "new_regime": snapshot.regime.value,
                        "previous_avg_corr": self._last_snapshot.average_correlation,
                        "new_avg_corr": snapshot.average_correlation,
                    }
                )

            # Check for significant correlation increase
            corr_change = snapshot.average_correlation - self._last_snapshot.average_correlation
            if corr_change > self._regime_change_threshold:
                self._generate_alert(
                    alert_type="correlation_spike",
                    severity="warning",
                    message=f"Average correlation increased by {corr_change:.2%}",
                    details={
                        "change": corr_change,
                        "threshold": self._regime_change_threshold,
                    }
                )

        # Check for max correlation threshold breach
        if snapshot.max_pairwise_correlation > self._max_pairwise_correlation:
            # Find the pair with highest correlation
            corr_matrix = snapshot.correlation_matrix
            max_pair = None
            max_val = 0

            for i, sym1 in enumerate(corr_matrix.columns):
                for j, sym2 in enumerate(corr_matrix.columns):
                    if i < j:
                        val = corr_matrix.iloc[i, j]
                        if val > max_val:
                            max_val = val
                            max_pair = (sym1, sym2)

            self._generate_alert(
                alert_type="threshold_breach",
                severity="warning",
                message=f"Pairwise correlation {max_val:.2f} exceeds threshold {self._max_pairwise_correlation}",
                details={
                    "pair": max_pair,
                    "correlation": max_val,
                    "threshold": self._max_pairwise_correlation,
                }
            )

        # Check for concentration (low effective N)
        min_effective_n = 2.0  # Minimum desired diversification
        if snapshot.effective_n < min_effective_n:
            self._generate_alert(
                alert_type="concentration",
                severity="warning",
                message=f"Low portfolio diversification: effective N = {snapshot.effective_n:.1f}",
                details={
                    "effective_n": snapshot.effective_n,
                    "min_effective_n": min_effective_n,
                    "herfindahl_index": snapshot.herfindahl_index,
                }
            )

    def _generate_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: dict[str, Any]
    ) -> CorrelationAlert:
        """
        Generate and store a correlation alert.

        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Human-readable message
            details: Additional details

        Returns:
            Generated CorrelationAlert
        """
        self._alert_counter += 1
        alert = CorrelationAlert(
            alert_id=f"CORR-{self._alert_counter:06d}",
            timestamp=datetime.now(timezone.utc),
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
        )

        self._alert_history.append(alert)

        # Keep only recent alerts
        max_alerts = 1000
        if len(self._alert_history) > max_alerts:
            self._alert_history = self._alert_history[-max_alerts:]

        logger.info(f"Correlation alert: [{severity.upper()}] {message}")

        return alert

    def get_pair_correlation(self, symbol1: str, symbol2: str) -> float | None:
        """
        Get correlation between two specific symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient, or None if not available
        """
        if self._last_snapshot is None:
            return None

        corr_matrix = self._last_snapshot.correlation_matrix

        if symbol1 not in corr_matrix.columns or symbol2 not in corr_matrix.columns:
            return None

        return float(corr_matrix.loc[symbol1, symbol2])

    def get_highly_correlated_pairs(
        self,
        threshold: float | None = None
    ) -> list[tuple[str, str, float]]:
        """
        Get list of highly correlated pairs.

        Args:
            threshold: Correlation threshold (default: configured max)

        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if threshold is None:
            threshold = self._max_pairwise_correlation

        if self._last_snapshot is None:
            return []

        corr_matrix = self._last_snapshot.correlation_matrix
        pairs = []

        for i, sym1 in enumerate(corr_matrix.columns):
            for j, sym2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) >= threshold:
                        pairs.append((sym1, sym2, float(corr)))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs

    def get_diversification_benefit(
        self,
        weights: dict[str, float],
        volatilities: dict[str, float]
    ) -> float | None:
        """
        Calculate diversification benefit.

        Diversification benefit = 1 - (portfolio_vol / weighted_avg_vol)

        Args:
            weights: Portfolio weights
            volatilities: Symbol volatilities

        Returns:
            Diversification benefit (0 = no benefit, 1 = maximum benefit)
        """
        if self._last_snapshot is None:
            return None

        corr_matrix = self._last_snapshot.correlation_matrix
        symbols = [s for s in weights.keys() if s in corr_matrix.columns]

        if len(symbols) < 2:
            return 0.0

        # Build weight and volatility vectors
        w = np.array([weights[s] for s in symbols])
        v = np.array([volatilities.get(s, 0.01) for s in symbols])

        # Weighted average volatility (undiversified)
        total_weight = np.sum(np.abs(w))
        if total_weight == 0:
            return 0.0

        w_norm = w / total_weight
        undiversified_vol = np.sum(np.abs(w_norm) * v)

        # Portfolio volatility (diversified)
        # Create covariance matrix from correlation and volatilities
        cov_matrix = np.zeros((len(symbols), len(symbols)))
        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                cov_matrix[i, j] = corr_matrix.loc[s1, s2] * v[i] * v[j]

        portfolio_variance = np.dot(w_norm, np.dot(cov_matrix, w_norm))
        portfolio_vol = np.sqrt(max(portfolio_variance, 0))

        if undiversified_vol == 0:
            return 0.0

        diversification_benefit = 1 - (portfolio_vol / undiversified_vol)

        return float(max(0, diversification_benefit))

    def get_recent_alerts(self, hours: int = 24) -> list[CorrelationAlert]:
        """
        Get recent correlation alerts.

        Args:
            hours: Lookback period in hours

        Returns:
            List of recent alerts
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)

        return [
            alert for alert in self._alert_history
            if alert.timestamp.timestamp() > cutoff
        ]

    def get_current_regime(self) -> CorrelationRegime | None:
        """Get current correlation regime."""
        if self._last_snapshot is None:
            return None
        return self._last_snapshot.regime

    def get_status(self) -> dict[str, Any]:
        """Get manager status for monitoring."""
        # Calculate stress correlation average if available (PM-12 exposure)
        stress_avg_corr = None
        if self._stress_correlation_cache is not None:
            upper_tri = self._stress_correlation_cache.values[
                np.triu_indices_from(self._stress_correlation_cache.values, k=1)
            ]
            if len(upper_tri) > 0:
                stress_avg_corr = float(np.mean(upper_tri))

        return {
            "symbols_tracked": len(self._returns_history),
            "lookback_days": self._lookback_days,
            "max_pairwise_correlation": self._max_pairwise_correlation,
            "current_regime": self._last_snapshot.regime.value if self._last_snapshot else None,
            "average_correlation": self._last_snapshot.average_correlation if self._last_snapshot else None,
            "effective_n": self._last_snapshot.effective_n if self._last_snapshot else None,
            "recent_alerts": len(self.get_recent_alerts(24)),
            # Stress mode status (#R7)
            "stress_mode": self._stress_mode,
            "stress_indicators": {k.value: v for k, v in self._stress_indicators.items()},
            "current_vix": self._current_vix,
            # Stress correlation info (PM-12 exposure)
            "stress_correlation_available": self._stress_correlation_cache is not None,
            "stress_average_correlation": stress_avg_corr,
        }

    # =========================================================================
    # STRESS PERIOD HANDLING (#R7)
    # =========================================================================

    def update_vix(self, vix_value: float) -> None:
        """
        Update VIX value for stress detection (#R7).

        Args:
            vix_value: Current VIX level
        """
        self._current_vix = vix_value

        if vix_value >= self._vix_stress_threshold:
            self._stress_indicators[StressIndicator.VIX_SPIKE] = vix_value
            if not self._stress_mode:
                self._enter_stress_mode(StressIndicator.VIX_SPIKE)
        elif StressIndicator.VIX_SPIKE in self._stress_indicators:
            del self._stress_indicators[StressIndicator.VIX_SPIKE]
            self._check_exit_stress_mode()

    def update_baseline_volatility(self, symbol: str, baseline_vol: float) -> None:
        """
        Update baseline volatility for stress detection (#R7).

        Args:
            symbol: Instrument symbol
            baseline_vol: Long-term average volatility
        """
        self._baseline_volatility[symbol] = baseline_vol

    def check_volatility_stress(self, symbol: str, current_vol: float) -> bool:
        """
        Check if current volatility indicates stress (#R7).

        Args:
            symbol: Instrument symbol
            current_vol: Current volatility

        Returns:
            True if volatility indicates stress
        """
        baseline = self._baseline_volatility.get(symbol)
        if baseline is None or baseline <= 0:
            return False

        vol_ratio = current_vol / baseline

        if vol_ratio >= self._vol_spike_threshold:
            self._stress_indicators[StressIndicator.VOLATILITY_SPIKE] = vol_ratio
            if not self._stress_mode:
                self._enter_stress_mode(StressIndicator.VOLATILITY_SPIKE)
            return True

        return False

    def signal_external_stress(self, stress_on: bool, reason: str = "") -> None:
        """
        Signal external stress condition (#R7).

        Called by risk agent or external monitoring systems.

        Args:
            stress_on: Whether stress mode should be active
            reason: Reason for stress signal
        """
        if stress_on:
            self._stress_indicators[StressIndicator.EXTERNAL_SIGNAL] = 1.0
            if not self._stress_mode:
                logger.warning(f"External stress signal received: {reason}")
                self._enter_stress_mode(StressIndicator.EXTERNAL_SIGNAL)
        else:
            if StressIndicator.EXTERNAL_SIGNAL in self._stress_indicators:
                del self._stress_indicators[StressIndicator.EXTERNAL_SIGNAL]
                self._check_exit_stress_mode()

    def _enter_stress_mode(self, trigger: StressIndicator) -> None:
        """
        Enter stress mode (#R7).

        In stress mode:
        - Use shorter lookback period
        - Use faster decay factor (more weight on recent data)
        - Trigger immediate correlation recalculation

        Args:
            trigger: The indicator that triggered stress mode
        """
        if self._stress_mode:
            return

        self._stress_mode = True
        self._stress_start_time = datetime.now(timezone.utc)
        self._stress_correlation_cache = None  # Invalidate cache

        logger.warning(
            f"ENTERING STRESS MODE: triggered by {trigger.value}. "
            f"Using lookback={self._stress_lookback_days}d, decay={self._stress_decay_factor}"
        )

        # Generate alert
        self._generate_alert(
            alert_type="stress_mode_enter",
            severity="critical",
            message=f"Entered stress mode due to {trigger.value}",
            details={
                "trigger": trigger.value,
                "indicators": {k.value: v for k, v in self._stress_indicators.items()},
                "stress_lookback": self._stress_lookback_days,
            }
        )

        # Immediately recalculate correlations with stress parameters
        self._recalculate_stress_correlations()

    def _check_exit_stress_mode(self) -> None:
        """Check if stress mode should be exited (#R7)."""
        if not self._stress_mode:
            return

        # Exit if no active stress indicators
        if len(self._stress_indicators) == 0:
            self._exit_stress_mode()
            return

        # Also check if current regime has normalized
        if self._last_snapshot and self._last_snapshot.regime in [CorrelationRegime.LOW, CorrelationRegime.NORMAL]:
            # Give a grace period before exiting
            if self._stress_start_time:
                stress_duration = (datetime.now(timezone.utc) - self._stress_start_time).total_seconds()
                # Stay in stress mode for at least 1 hour even if indicators clear
                if stress_duration < 3600:
                    return

            self._exit_stress_mode()

    def _exit_stress_mode(self) -> None:
        """Exit stress mode (#R7)."""
        if not self._stress_mode:
            return

        stress_duration = None
        if self._stress_start_time:
            stress_duration = (datetime.now(timezone.utc) - self._stress_start_time).total_seconds() / 3600

        self._stress_mode = False
        self._stress_start_time = None
        self._stress_correlation_cache = None

        logger.info(
            f"EXITING STRESS MODE after {stress_duration:.1f}h" if stress_duration else "EXITING STRESS MODE"
        )

        # Generate alert
        self._generate_alert(
            alert_type="stress_mode_exit",
            severity="info",
            message="Exited stress mode",
            details={"duration_hours": stress_duration}
        )

    def _recalculate_stress_correlations(self) -> None:
        """
        Recalculate correlations using stress parameters (#R7).

        Uses:
        - Shorter lookback period
        - EWMA with faster decay for more weight on recent data
        """
        symbols = list(self._returns_history.keys())
        if len(symbols) < 2:
            return

        # Use stress-mode parameters
        stress_matrix = self._calculate_ewma_correlation_matrix(
            symbols,
            lookback_days=self._stress_lookback_days,
            decay_factor=self._stress_decay_factor
        )

        if stress_matrix is not None:
            self._stress_correlation_cache = stress_matrix
            logger.info(
                f"Stress correlation matrix updated: "
                f"avg_corr={stress_matrix.values[np.triu_indices_from(stress_matrix.values, k=1)].mean():.2f}"
            )

    def _calculate_ewma_correlation_matrix(
        self,
        symbols: list[str],
        lookback_days: int,
        decay_factor: float
    ) -> pd.DataFrame | None:
        """
        Calculate EWMA-weighted correlation matrix (#R7).

        Uses exponential decay to give more weight to recent observations.

        Args:
            symbols: List of symbols
            lookback_days: Number of days to include
            decay_factor: EWMA decay factor (lower = more weight on recent)

        Returns:
            Correlation matrix as DataFrame
        """
        # Build returns DataFrame
        returns_data = {}
        for symbol in symbols:
            if symbol in self._returns_history:
                history = self._returns_history[symbol]
                if len(history) >= self._min_history_days:
                    recent = history[-lookback_days:]
                    returns_data[symbol] = pd.Series(
                        [r[1] for r in recent],
                        index=[r[0] for r in recent]
                    )

        if len(returns_data) < 2:
            return None

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna(how="any")

        if len(returns_df) < self._min_history_days:
            return None

        # Calculate EWMA weights
        n = len(returns_df)
        weights = np.array([(1 - decay_factor) * (decay_factor ** i) for i in range(n)])
        weights = weights[::-1]  # Most recent gets highest weight
        weights /= weights.sum()

        # Calculate weighted correlation
        n_symbols = len(returns_df.columns)
        corr_matrix = np.zeros((n_symbols, n_symbols))

        # Demean with weighted mean
        weighted_means = (returns_df.values * weights[:, np.newaxis]).sum(axis=0)
        centered = returns_df.values - weighted_means

        # Weighted std and covariance
        weighted_vars = (centered ** 2 * weights[:, np.newaxis]).sum(axis=0)
        weighted_stds = np.sqrt(weighted_vars)

        for i in range(n_symbols):
            for j in range(n_symbols):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    weighted_cov = (centered[:, i] * centered[:, j] * weights).sum()
                    if weighted_stds[i] > 0 and weighted_stds[j] > 0:
                        corr_matrix[i, j] = weighted_cov / (weighted_stds[i] * weighted_stds[j])
                    else:
                        corr_matrix[i, j] = 0.0

        return pd.DataFrame(
            corr_matrix,
            index=returns_df.columns,
            columns=returns_df.columns
        )

    def get_stress_adjusted_correlation_matrix(
        self,
        symbols: list[str] | None = None
    ) -> pd.DataFrame | None:
        """
        Get correlation matrix adjusted for current stress conditions (#R7).

        If in stress mode, returns stress-adjusted correlations with faster decay.
        Otherwise, returns the normal correlation matrix.

        Args:
            symbols: List of symbols to include

        Returns:
            Correlation matrix appropriate for current market conditions
        """
        if self._stress_mode and self._stress_correlation_cache is not None:
            if symbols is None:
                return self._stress_correlation_cache
            else:
                # Filter to requested symbols
                available = [s for s in symbols if s in self._stress_correlation_cache.columns]
                if len(available) >= 2:
                    return self._stress_correlation_cache.loc[available, available]

        # Use normal correlation matrix
        return self.calculate_correlation_matrix(symbols)

    def get_stressed_correlation_matrix(
        self,
        stress_multiplier: float = 1.5,
        correlation_floor: float = 0.4
    ) -> pd.DataFrame | None:
        """
        Generate stressed correlation matrix for scenario analysis (#R7).

        Simulates crisis conditions where correlations increase.

        Args:
            stress_multiplier: Factor to increase correlations
            correlation_floor: Minimum correlation in stress scenario

        Returns:
            Stressed correlation matrix
        """
        base_matrix = self.calculate_correlation_matrix()
        if base_matrix is None:
            return None

        # Apply stress transformation
        stressed = base_matrix.copy()

        for i in range(len(stressed)):
            for j in range(len(stressed)):
                if i != j:
                    # Increase correlation toward 1.0
                    base_corr = stressed.iloc[i, j]
                    # Stress increases positive correlations, makes negative less negative
                    stressed_corr = base_corr + (1.0 - abs(base_corr)) * (stress_multiplier - 1) / stress_multiplier

                    # Apply floor
                    if stressed_corr > 0:
                        stressed_corr = max(stressed_corr, correlation_floor)
                    else:
                        stressed_corr = min(stressed_corr, -correlation_floor)

                    # Cap at valid range
                    stressed.iloc[i, j] = max(-1.0, min(1.0, stressed_corr))

        return stressed

    def is_in_stress_mode(self) -> bool:
        """Check if currently in stress mode (#R7)."""
        return self._stress_mode

    def get_stress_duration_hours(self) -> float | None:
        """Get duration of current stress period in hours (#R7)."""
        if not self._stress_mode or self._stress_start_time is None:
            return None
        return (datetime.now(timezone.utc) - self._stress_start_time).total_seconds() / 3600

    def refresh_stress_correlations(self) -> pd.DataFrame | None:
        """
        Force refresh of stress correlation cache (PM-12 enhancement).

        Recalculates stress correlations using current data and stress parameters.
        Useful when you need updated stress correlations without triggering
        full stress mode.

        Returns:
            Updated stress correlation matrix, or None if insufficient data
        """
        self._recalculate_stress_correlations()
        return self._stress_correlation_cache

    def get_effective_n_weighted(
        self,
        weights: dict[str, float],
        symbols: list[str] | None = None
    ) -> float | None:
        """
        Calculate effective N with specific portfolio weights (PM-10 public API).

        This method allows computing the weighted effective N without
        calculating a full snapshot.

        Args:
            weights: Portfolio weights by symbol
            symbols: Symbols to include (default: all symbols in weights)

        Returns:
            Weighted effective N, or None if insufficient data
        """
        if symbols is None:
            symbols = list(weights.keys())

        corr_matrix = self.calculate_correlation_matrix(symbols)
        if corr_matrix is None:
            return None

        return self._calculate_effective_n_weighted(corr_matrix, weights)

    # =========================================================================
    # CORRELATION STABILITY METRICS (P2)
    # =========================================================================

    def calculate_correlation_stability(
        self,
        symbols: list[str] | None = None,
        window_sizes: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Calculate correlation stability metrics across different time windows (P2).

        Measures how stable correlations are over time by comparing
        correlation matrices computed with different lookback periods.

        Args:
            symbols: Symbols to analyze (default: all)
            window_sizes: List of window sizes to compare (default: [20, 40, 60])

        Returns:
            Dictionary with stability metrics including:
            - stability_score: Overall stability (0-1, higher = more stable)
            - window_divergence: Max divergence between windows
            - unstable_pairs: Pairs with highly variable correlations
        """
        if window_sizes is None:
            window_sizes = [20, 40, 60]

        if symbols is None:
            symbols = list(self._returns_history.keys())

        if len(symbols) < 2:
            return {
                "stability_score": 1.0,
                "window_divergence": 0.0,
                "unstable_pairs": [],
                "correlation_matrices": {},
            }

        # Calculate correlation matrices for each window
        matrices = {}
        for window in window_sizes:
            matrix = self.calculate_correlation_matrix(symbols, lookback_days=window)
            if matrix is not None:
                matrices[window] = matrix

        if len(matrices) < 2:
            return {
                "stability_score": 1.0,
                "window_divergence": 0.0,
                "unstable_pairs": [],
                "correlation_matrices": matrices,
            }

        # Compare matrices pairwise
        divergences = []
        pair_stability = {}

        window_list = sorted(matrices.keys())
        for i in range(len(window_list)):
            for j in range(i + 1, len(window_list)):
                w1, w2 = window_list[i], window_list[j]
                m1, m2 = matrices[w1], matrices[w2]

                # Calculate divergence for each pair
                common_symbols = list(set(m1.columns) & set(m2.columns))
                for sym1 in common_symbols:
                    for sym2 in common_symbols:
                        if sym1 < sym2:
                            corr1 = m1.loc[sym1, sym2]
                            corr2 = m2.loc[sym1, sym2]
                            diff = abs(corr1 - corr2)
                            divergences.append(diff)

                            pair_key = (sym1, sym2)
                            if pair_key not in pair_stability:
                                pair_stability[pair_key] = []
                            pair_stability[pair_key].append(diff)

        # Calculate overall stability
        if divergences:
            avg_divergence = float(np.mean(divergences))
            max_divergence = float(np.max(divergences))
            stability_score = max(0.0, 1.0 - avg_divergence)
        else:
            avg_divergence = 0.0
            max_divergence = 0.0
            stability_score = 1.0

        # Find unstable pairs (divergence > 0.3)
        unstable_pairs = []
        for pair, diffs in pair_stability.items():
            avg_diff = np.mean(diffs)
            if avg_diff > 0.3:
                unstable_pairs.append({
                    "pair": pair,
                    "avg_divergence": float(avg_diff),
                    "max_divergence": float(np.max(diffs)),
                })

        # Sort by divergence
        unstable_pairs.sort(key=lambda x: x["avg_divergence"], reverse=True)

        return {
            "stability_score": stability_score,
            "avg_divergence": avg_divergence,
            "window_divergence": max_divergence,
            "unstable_pairs": unstable_pairs[:10],  # Top 10 most unstable
            "windows_analyzed": window_list,
        }

    def check_correlation_breakdown(
        self,
        threshold_change: float = 0.25,
        lookback_comparison: int = 20
    ) -> list[CorrelationAlert]:
        """
        Check for correlation breakdown events (P2).

        Correlation breakdown occurs when previously stable correlations
        suddenly change significantly, often indicating regime change.

        Args:
            threshold_change: Minimum change to trigger alert (default: 0.25)
            lookback_comparison: Days to compare against (default: 20)

        Returns:
            List of correlation breakdown alerts
        """
        alerts = []

        symbols = list(self._returns_history.keys())
        if len(symbols) < 2:
            return alerts

        # Get current and historical correlation matrices
        current_matrix = self.calculate_correlation_matrix(symbols)
        if current_matrix is None:
            return alerts

        # Calculate historical matrix with offset
        historical_data = {}
        for symbol in symbols:
            if symbol in self._returns_history:
                history = self._returns_history[symbol]
                if len(history) > lookback_comparison + self._min_history_days:
                    # Get data ending lookback_comparison days ago
                    end_idx = len(history) - lookback_comparison
                    start_idx = max(0, end_idx - self._lookback_days)
                    historical_data[symbol] = history[start_idx:end_idx]

        if len(historical_data) < 2:
            return alerts

        # Build historical returns DataFrame
        returns_data = {}
        for symbol, history in historical_data.items():
            if len(history) >= self._min_history_days:
                returns_data[symbol] = pd.Series(
                    [r[1] for r in history],
                    index=[r[0] for r in history]
                )

        if len(returns_data) < 2:
            return alerts

        returns_df = pd.DataFrame(returns_data).dropna(how="any")
        if len(returns_df) < self._min_history_days:
            return alerts

        historical_matrix = returns_df.corr()

        # Compare matrices and find breakdowns
        common_symbols = list(set(current_matrix.columns) & set(historical_matrix.columns))

        for i, sym1 in enumerate(common_symbols):
            for sym2 in common_symbols[i+1:]:
                current_corr = current_matrix.loc[sym1, sym2]
                historical_corr = historical_matrix.loc[sym1, sym2]
                change = current_corr - historical_corr

                if abs(change) >= threshold_change:
                    severity = "critical" if abs(change) >= 0.4 else "warning"
                    direction = "increased" if change > 0 else "decreased"

                    alert = self._generate_alert(
                        alert_type="correlation_breakdown",
                        severity=severity,
                        message=(
                            f"Correlation between {sym1} and {sym2} {direction} "
                            f"by {abs(change):.2f} (from {historical_corr:.2f} to {current_corr:.2f})"
                        ),
                        details={
                            "pair": (sym1, sym2),
                            "current_correlation": float(current_corr),
                            "historical_correlation": float(historical_corr),
                            "change": float(change),
                            "lookback_days": lookback_comparison,
                        }
                    )
                    alerts.append(alert)

        return alerts

    def optimize_rolling_window(
        self,
        symbols: list[str] | None = None,
        test_windows: list[int] | None = None,
        metric: str = "stability"
    ) -> dict[str, Any]:
        """
        Optimize rolling window size for correlation calculation (P2).

        Tests different window sizes and recommends optimal based on:
        - stability: Minimize correlation estimate variance
        - responsiveness: Faster detection of regime changes
        - balanced: Trade-off between stability and responsiveness

        Args:
            symbols: Symbols to analyze
            test_windows: Window sizes to test (default: [20, 30, 40, 60, 90])
            metric: Optimization metric ("stability", "responsiveness", "balanced")

        Returns:
            Dictionary with optimal window and analysis
        """
        if test_windows is None:
            test_windows = [20, 30, 40, 60, 90]

        if symbols is None:
            symbols = list(self._returns_history.keys())

        if len(symbols) < 2:
            return {
                "optimal_window": self._lookback_days,
                "analysis": "Insufficient symbols for optimization",
            }

        results = []

        for window in test_windows:
            # Calculate matrices at multiple points in time
            matrices = []
            step = max(5, window // 4)

            for offset in range(0, min(60, window * 2), step):
                # Simulate historical calculation by using subset of data
                returns_data = {}
                for symbol in symbols:
                    if symbol in self._returns_history:
                        history = self._returns_history[symbol]
                        end_idx = len(history) - offset if offset > 0 else len(history)
                        start_idx = max(0, end_idx - window)
                        if end_idx - start_idx >= self._min_history_days:
                            subset = history[start_idx:end_idx]
                            returns_data[symbol] = pd.Series(
                                [r[1] for r in subset],
                                index=[r[0] for r in subset]
                            )

                if len(returns_data) >= 2:
                    df = pd.DataFrame(returns_data).dropna(how="any")
                    if len(df) >= self._min_history_days:
                        matrices.append(df.corr())

            if len(matrices) < 2:
                continue

            # Calculate stability (variance of correlations across time)
            all_corrs = []
            for m in matrices:
                upper_tri = m.values[np.triu_indices_from(m.values, k=1)]
                all_corrs.extend(upper_tri.tolist())

            if all_corrs:
                stability = 1.0 - np.std(all_corrs)
                responsiveness = 1.0 / window  # Shorter = more responsive
                balanced = 0.6 * stability + 0.4 * responsiveness * 60  # Normalize

                results.append({
                    "window": window,
                    "stability": float(stability),
                    "responsiveness": float(responsiveness),
                    "balanced": float(balanced),
                    "n_matrices": len(matrices),
                })

        if not results:
            return {
                "optimal_window": self._lookback_days,
                "analysis": "Could not compute metrics for any window",
            }

        # Select optimal based on metric
        if metric == "stability":
            optimal = max(results, key=lambda x: x["stability"])
        elif metric == "responsiveness":
            optimal = max(results, key=lambda x: x["responsiveness"])
        else:  # balanced
            optimal = max(results, key=lambda x: x["balanced"])

        return {
            "optimal_window": optimal["window"],
            "metric_used": metric,
            "optimal_score": optimal[metric],
            "all_results": results,
            "recommendation": (
                f"Optimal window is {optimal['window']} days for {metric} metric. "
                f"Stability={optimal['stability']:.3f}, Responsiveness={optimal['responsiveness']:.4f}"
            ),
        }

    def calculate_ewma_correlation_matrix(
        self,
        symbols: list[str] | None = None,
        lookback_days: int | None = None,
        decay_factor: float | None = None
    ) -> pd.DataFrame | None:
        """
        Public method to calculate EWMA correlation matrix (CORR-P1-1 public API).

        EWMA (Exponentially Weighted Moving Average) correlations give more
        weight to recent observations, which is useful for:
        - Detecting regime changes faster
        - Stress period analysis
        - Forward-looking risk estimation

        Args:
            symbols: List of symbols (default: all available)
            lookback_days: Window size (default: configured lookback)
            decay_factor: EWMA decay factor (0 < decay < 1).
                         Lower values = more weight on recent data.
                         Default: stress_decay_factor from config (typically 0.90)

        Returns:
            EWMA correlation matrix as DataFrame, or None if insufficient data

        Example:
            # Standard EWMA correlation
            ewma_corr = manager.calculate_ewma_correlation_matrix()

            # Faster decay for more responsive correlations
            fast_ewma = manager.calculate_ewma_correlation_matrix(decay_factor=0.85)
        """
        if lookback_days is None:
            lookback_days = self._lookback_days
        if decay_factor is None:
            decay_factor = self._stress_decay_factor
        if symbols is None:
            symbols = list(self._returns_history.keys())

        return self._calculate_ewma_correlation_matrix(symbols, lookback_days, decay_factor)

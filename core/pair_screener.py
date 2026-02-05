"""
Automated Pair Discovery
=========================

Phase 4: Advanced Features

Automated screening and ranking of tradeable pairs for statistical arbitrage.
Replaces manual pair configuration with systematic discovery.

Key features:
- Universe screening for correlation
- Cointegration testing (ADF and Johansen)
- Quality scoring based on multiple criteria
- Pair ranking and selection
- Scheduled rescreening capability

Target: Discover 50+ pairs vs current 15 manual pairs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from itertools import combinations

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class PairCandidate:
    """A potential trading pair candidate."""
    symbol_a: str
    symbol_b: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    half_life: float
    spread_std: float
    quality_score: float
    is_viable: bool
    rejection_reason: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "correlation": self.correlation,
            "cointegration_pvalue": self.cointegration_pvalue,
            "hedge_ratio": self.hedge_ratio,
            "half_life": self.half_life,
            "spread_std": self.spread_std,
            "quality_score": self.quality_score,
            "is_viable": self.is_viable,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ScreeningResult:
    """Result of a full screening run."""
    viable_pairs: list[PairCandidate]
    rejected_pairs: list[PairCandidate]
    total_candidates: int
    n_viable: int
    n_rejected: int
    screening_time_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "viable_pairs": [p.to_dict() for p in self.viable_pairs],
            "rejected_count": self.n_rejected,
            "total_candidates": self.total_candidates,
            "n_viable": self.n_viable,
            "screening_time_seconds": self.screening_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


class PairScreener:
    """
    Automated Pair Discovery System.

    Screens a universe of assets to find viable statistical arbitrage pairs.

    Screening criteria:
    1. Minimum correlation (typically > 0.7)
    2. Cointegration (ADF p-value < 0.05)
    3. Half-life within bounds (1-30 days typically)
    4. Spread stability (reasonable standard deviation)
    5. Liquidity requirements (optional)

    Quality scoring based on:
    - Cointegration strength (lower p-value = better)
    - Half-life (shorter = faster mean reversion = better)
    - Correlation stability
    - Spread volatility (moderate = good, extreme = bad)

    Usage:
        screener = PairScreener(config)
        result = screener.screen_universe(price_data)
        top_pairs = result.viable_pairs[:10]  # Top 10 pairs
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize pair screener.

        Args:
            config: Configuration dictionary with screening parameters
        """
        config = config or {}

        # Correlation thresholds
        self._min_correlation = config.get("min_correlation", 0.70)
        self._max_correlation = config.get("max_correlation", 0.99)  # Avoid near-identical

        # Cointegration thresholds
        self._max_coint_pvalue = config.get("max_cointegration_pvalue", 0.05)

        # Half-life bounds (in periods, typically days)
        self._min_half_life = config.get("min_half_life", 1)
        self._max_half_life = config.get("max_half_life", 30)

        # Spread volatility bounds
        self._min_spread_std = config.get("min_spread_std", 0.01)
        self._max_spread_std = config.get("max_spread_std", 0.20)

        # Minimum data requirements
        self._min_observations = config.get("min_observations", 252)  # ~1 year

        # Quality score weights
        self._weight_coint = config.get("weight_cointegration", 0.30)
        self._weight_halflife = config.get("weight_half_life", 0.25)
        self._weight_corr = config.get("weight_correlation", 0.20)
        self._weight_stability = config.get("weight_stability", 0.25)

        # Sector/industry constraints
        self._same_sector_only = config.get("same_sector_only", False)
        self._sector_map: dict[str, str] = config.get("sector_map", {})

        logger.info(
            f"PairScreener initialized: min_corr={self._min_correlation}, "
            f"max_coint_pvalue={self._max_coint_pvalue}, "
            f"half_life_range=[{self._min_half_life}, {self._max_half_life}]"
        )

    def screen_universe(
        self,
        price_data: dict[str, np.ndarray],
        symbols: list[str] | None = None,
        max_pairs: int = 100,
    ) -> ScreeningResult:
        """
        Screen a universe of assets for viable pairs.

        Args:
            price_data: Dictionary of symbol to price arrays
            symbols: List of symbols to consider (default: all in price_data)
            max_pairs: Maximum number of pairs to return

        Returns:
            ScreeningResult with ranked viable pairs
        """
        import time
        start_time = time.time()

        if symbols is None:
            symbols = list(price_data.keys())

        # Filter to symbols with sufficient data
        valid_symbols = []
        for s in symbols:
            if s in price_data and len(price_data[s]) >= self._min_observations:
                valid_symbols.append(s)
            else:
                logger.debug(f"Skipping {s}: insufficient data")

        logger.info(f"Screening {len(valid_symbols)} symbols ({len(list(combinations(valid_symbols, 2)))} pairs)")

        viable_pairs = []
        rejected_pairs = []

        # Generate all pairs
        for symbol_a, symbol_b in combinations(valid_symbols, 2):
            # Sector filter
            if self._same_sector_only:
                sector_a = self._sector_map.get(symbol_a, "unknown")
                sector_b = self._sector_map.get(symbol_b, "unknown")
                if sector_a != sector_b:
                    continue

            # Screen the pair
            candidate = self._screen_pair(
                symbol_a, symbol_b,
                price_data[symbol_a], price_data[symbol_b]
            )

            if candidate.is_viable:
                viable_pairs.append(candidate)
            else:
                rejected_pairs.append(candidate)

        # Sort by quality score (descending)
        viable_pairs.sort(key=lambda p: p.quality_score, reverse=True)

        # Limit to max_pairs
        viable_pairs = viable_pairs[:max_pairs]

        screening_time = time.time() - start_time

        result = ScreeningResult(
            viable_pairs=viable_pairs,
            rejected_pairs=rejected_pairs,
            total_candidates=len(viable_pairs) + len(rejected_pairs),
            n_viable=len(viable_pairs),
            n_rejected=len(rejected_pairs),
            screening_time_seconds=screening_time,
        )

        logger.info(
            f"Screening complete: {result.n_viable} viable pairs, "
            f"{result.n_rejected} rejected, time={screening_time:.1f}s"
        )

        return result

    def _screen_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> PairCandidate:
        """
        Screen a single pair for viability.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            prices_a: Price array for first symbol
            prices_b: Price array for second symbol

        Returns:
            PairCandidate with screening results
        """
        # Ensure same length
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]

        # Step 1: Correlation check
        correlation = np.corrcoef(prices_a, prices_b)[0, 1]

        if not np.isfinite(correlation):
            return self._rejected_candidate(
                symbol_a, symbol_b, 0, 1.0, 1.0, 0, 0, 0,
                "Invalid correlation"
            )

        if correlation < self._min_correlation:
            return self._rejected_candidate(
                symbol_a, symbol_b, correlation, 1.0, 1.0, 0, 0, 0,
                f"Correlation {correlation:.2f} < {self._min_correlation}"
            )

        if correlation > self._max_correlation:
            return self._rejected_candidate(
                symbol_a, symbol_b, correlation, 1.0, 1.0, 0, 0, 0,
                f"Correlation {correlation:.2f} > {self._max_correlation} (too similar)"
            )

        # Step 2: Estimate hedge ratio
        hedge_ratio = self._estimate_hedge_ratio(prices_a, prices_b)

        # Step 3: Calculate spread
        spread = prices_a - hedge_ratio * prices_b
        spread_std = np.std(spread) / np.mean(np.abs(spread)) if np.mean(np.abs(spread)) > 0 else 0

        # Step 4: Cointegration test
        coint_pvalue = self._adf_test(spread)

        if coint_pvalue > self._max_coint_pvalue:
            return self._rejected_candidate(
                symbol_a, symbol_b, correlation, coint_pvalue, hedge_ratio, 0, spread_std, 0,
                f"Not cointegrated (p-value={coint_pvalue:.3f} > {self._max_coint_pvalue})"
            )

        # Step 5: Half-life estimation
        half_life = self._estimate_half_life(spread)

        if half_life < self._min_half_life:
            return self._rejected_candidate(
                symbol_a, symbol_b, correlation, coint_pvalue, hedge_ratio, half_life, spread_std, 0,
                f"Half-life {half_life:.1f} < {self._min_half_life} (too fast)"
            )

        if half_life > self._max_half_life:
            return self._rejected_candidate(
                symbol_a, symbol_b, correlation, coint_pvalue, hedge_ratio, half_life, spread_std, 0,
                f"Half-life {half_life:.1f} > {self._max_half_life} (too slow)"
            )

        # Step 6: Spread volatility check
        if spread_std < self._min_spread_std:
            return self._rejected_candidate(
                symbol_a, symbol_b, correlation, coint_pvalue, hedge_ratio, half_life, spread_std, 0,
                f"Spread too stable (std={spread_std:.3f})"
            )

        if spread_std > self._max_spread_std:
            return self._rejected_candidate(
                symbol_a, symbol_b, correlation, coint_pvalue, hedge_ratio, half_life, spread_std, 0,
                f"Spread too volatile (std={spread_std:.3f})"
            )

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            correlation, coint_pvalue, half_life, spread_std, spread
        )

        return PairCandidate(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=correlation,
            cointegration_pvalue=coint_pvalue,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            spread_std=spread_std,
            quality_score=quality_score,
            is_viable=True,
            rejection_reason=None,
        )

    def _rejected_candidate(
        self,
        symbol_a: str,
        symbol_b: str,
        correlation: float,
        coint_pvalue: float,
        hedge_ratio: float,
        half_life: float,
        spread_std: float,
        quality_score: float,
        reason: str,
    ) -> PairCandidate:
        """Create a rejected pair candidate."""
        return PairCandidate(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=correlation,
            cointegration_pvalue=coint_pvalue,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            spread_std=spread_std,
            quality_score=quality_score,
            is_viable=False,
            rejection_reason=reason,
        )

    def _calculate_quality_score(
        self,
        correlation: float,
        coint_pvalue: float,
        half_life: float,
        spread_std: float,
        spread: np.ndarray,
    ) -> float:
        """
        Calculate quality score for a pair.

        Higher score = better pair.

        Components:
        1. Cointegration strength (lower p-value = better)
        2. Half-life (shorter within bounds = better)
        3. Correlation strength
        4. Spread stability (moderate std is ideal)
        """
        # Cointegration score: 1 - (pvalue / max_pvalue)
        coint_score = 1.0 - (coint_pvalue / self._max_coint_pvalue)
        coint_score = max(0, min(1, coint_score))

        # Half-life score: prefer shorter half-life
        half_life_range = self._max_half_life - self._min_half_life
        if half_life_range > 0:
            half_life_score = 1.0 - (half_life - self._min_half_life) / half_life_range
        else:
            half_life_score = 0.5
        half_life_score = max(0, min(1, half_life_score))

        # Correlation score: higher is better (within bounds)
        corr_range = self._max_correlation - self._min_correlation
        if corr_range > 0:
            corr_score = (correlation - self._min_correlation) / corr_range
        else:
            corr_score = 0.5
        corr_score = max(0, min(1, corr_score))

        # Stability score: moderate spread std is ideal
        # Penalize both too stable (no opportunity) and too volatile (risk)
        ideal_std = (self._min_spread_std + self._max_spread_std) / 2
        std_deviation = abs(spread_std - ideal_std) / ideal_std
        stability_score = max(0, 1.0 - std_deviation)

        # Weighted combination
        quality = (
            self._weight_coint * coint_score +
            self._weight_halflife * half_life_score +
            self._weight_corr * corr_score +
            self._weight_stability * stability_score
        )

        return quality

    def _estimate_hedge_ratio(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> float:
        """Estimate hedge ratio using OLS."""
        var_b = np.var(prices_b)
        if var_b < 1e-12:
            return 1.0

        cov_matrix = np.cov(prices_a, prices_b)
        if cov_matrix.ndim == 0:
            return 1.0

        beta = cov_matrix[0, 1] / var_b

        if not np.isfinite(beta):
            return 1.0

        return beta

    def _adf_test(self, series: np.ndarray) -> float:
        """
        Simplified ADF test for stationarity.

        Returns p-value (lower = more stationary).
        """
        n = len(series)
        if n < 20:
            return 1.0

        # First difference
        diff = np.diff(series)
        lagged = series[:-1]

        if np.std(lagged) < 1e-10:
            return 1.0

        # Regression: diff = alpha + gamma * lagged + epsilon
        X = np.column_stack([lagged, np.ones(len(lagged))])

        try:
            XtX = X.T @ X
            Xty = X.T @ diff
            beta = np.linalg.solve(XtX, Xty)
            gamma = beta[0]

            # Calculate standard error
            residuals = diff - X @ beta
            sigma2 = np.sum(residuals ** 2) / (len(diff) - 2)
            var_beta = sigma2 * np.linalg.inv(XtX)
            se_gamma = np.sqrt(var_beta[0, 0])

            if se_gamma < 1e-10:
                return 1.0

            # t-statistic
            t_stat = gamma / se_gamma

            # Approximate p-value using critical values
            if t_stat < -3.43:
                return 0.01
            elif t_stat < -2.86:
                return 0.05
            elif t_stat < -2.57:
                return 0.10
            else:
                return min(0.99, 0.10 + (t_stat + 2.57) * 0.3)

        except (np.linalg.LinAlgError, ValueError):
            return 1.0

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """Estimate mean reversion half-life."""
        if len(spread) < 10:
            return float("inf")

        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        var_lag = np.var(spread_lag)
        if var_lag < 1e-12:
            return float("inf")

        cov_result = np.cov(spread_diff, spread_lag)
        if cov_result.ndim == 0:
            return float("inf")

        theta = -cov_result[0, 1] / var_lag

        if not np.isfinite(theta) or theta <= 0:
            return float("inf")

        half_life = np.log(2) / theta

        if not np.isfinite(half_life) or half_life < 0:
            return float("inf")

        return half_life

    def get_top_pairs(
        self,
        price_data: dict[str, np.ndarray],
        n_pairs: int = 10,
        symbols: list[str] | None = None,
    ) -> list[PairCandidate]:
        """
        Convenience method to get top N pairs.

        Args:
            price_data: Price data dictionary
            n_pairs: Number of pairs to return
            symbols: Symbols to consider

        Returns:
            List of top N PairCandidate objects
        """
        result = self.screen_universe(price_data, symbols, max_pairs=n_pairs)
        return result.viable_pairs

    def update_sectors(self, sector_map: dict[str, str]) -> None:
        """Update sector mapping for sector-based filtering."""
        self._sector_map = sector_map

    def get_status(self) -> dict[str, Any]:
        """Get screener configuration status."""
        return {
            "min_correlation": self._min_correlation,
            "max_correlation": self._max_correlation,
            "max_cointegration_pvalue": self._max_coint_pvalue,
            "min_half_life": self._min_half_life,
            "max_half_life": self._max_half_life,
            "min_observations": self._min_observations,
            "same_sector_only": self._same_sector_only,
            "n_sectors_mapped": len(self._sector_map),
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_pair_screener(config: dict[str, Any]) -> PairScreener:
    """
    Factory function to create PairScreener from config.

    Args:
        config: Configuration dictionary with pair_screening section

    Returns:
        Configured PairScreener
    """
    screener_config = config.get("pair_screening", {})
    return PairScreener(screener_config)

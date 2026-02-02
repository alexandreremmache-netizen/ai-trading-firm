"""
Signal Validation Module
========================

Mean reversion signal validation (Issue #Q16).
Spread ratio validation for stat arb (Issue #Q17).

Features:
- Mean reversion signal quality checks
- Spread ratio statistical validation
- Signal confidence scoring
- Historical performance validation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Signal validation result."""
    VALID = "valid"
    WEAK = "weak"  # Marginally valid
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"


@dataclass
class MeanReversionValidation:
    """Mean reversion signal validation result (#Q16)."""
    symbol: str
    signal_direction: str  # 'LONG' or 'SHORT'
    signal_strength: float

    # Validation checks
    stationarity_test_passed: bool
    adf_statistic: float
    adf_pvalue: float

    half_life_days: float
    half_life_valid: bool  # True if within acceptable range

    current_zscore: float
    zscore_extreme: bool  # True if >2 or <-2

    hurst_exponent: float
    hurst_valid: bool  # True if <0.5 (mean reverting)

    # Confirmation
    support_resistance_confirmed: bool
    volume_confirmation: bool

    # Overall
    result: ValidationResult
    confidence: float
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'signal_direction': self.signal_direction,
            'signal_strength': self.signal_strength,
            'stationarity_test_passed': self.stationarity_test_passed,
            'adf_statistic': self.adf_statistic,
            'adf_pvalue': self.adf_pvalue,
            'half_life_days': self.half_life_days,
            'half_life_valid': self.half_life_valid,
            'current_zscore': self.current_zscore,
            'zscore_extreme': self.zscore_extreme,
            'hurst_exponent': self.hurst_exponent,
            'hurst_valid': self.hurst_valid,
            'result': self.result.value,
            'confidence': self.confidence,
            'rejection_reasons': self.rejection_reasons,
        }


@dataclass
class SpreadRatioValidation:
    """Spread ratio validation for stat arb (#Q17)."""
    symbol1: str
    symbol2: str

    # Spread statistics
    current_spread: float
    spread_mean: float
    spread_std: float
    current_zscore: float

    # Cointegration
    is_cointegrated: bool
    cointegration_pvalue: float

    # Hedge ratio
    hedge_ratio: float
    hedge_ratio_stable: bool
    hedge_ratio_std: float  # Rolling std of hedge ratio

    # Half-life
    spread_half_life_days: float
    half_life_acceptable: bool

    # Correlation
    correlation: float
    correlation_stable: bool

    # Beta neutrality
    is_beta_neutral: bool
    net_beta: float

    # Overall
    result: ValidationResult
    confidence: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'current_spread': self.current_spread,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'current_zscore': self.current_zscore,
            'is_cointegrated': self.is_cointegrated,
            'cointegration_pvalue': self.cointegration_pvalue,
            'hedge_ratio': self.hedge_ratio,
            'hedge_ratio_stable': self.hedge_ratio_stable,
            'spread_half_life_days': self.spread_half_life_days,
            'correlation': self.correlation,
            'is_beta_neutral': self.is_beta_neutral,
            'net_beta': self.net_beta,
            'result': self.result.value,
            'confidence': self.confidence,
            'warnings': self.warnings,
        }


class MeanReversionValidator:
    """
    Validates mean reversion signals (#Q16).

    Ensures signals have statistical support before trading.
    """

    def __init__(
        self,
        min_adf_confidence: float = 0.95,  # ADF test confidence level
        max_half_life_days: float = 30.0,  # Max acceptable half-life
        min_half_life_days: float = 1.0,  # Min half-life (avoid noise)
        max_hurst_exponent: float = 0.5,  # Mean reverting if <0.5
        zscore_entry_threshold: float = 2.0,  # Min z-score for entry
    ):
        self.min_adf_confidence = min_adf_confidence
        self.max_half_life = max_half_life_days
        self.min_half_life = min_half_life_days
        self.max_hurst = max_hurst_exponent
        self.zscore_threshold = zscore_entry_threshold

        # Historical data for validation
        self._price_history: dict[str, deque] = {}
        self._volume_history: dict[str, deque] = {}

    def update_price(self, symbol: str, price: float, volume: int = 0) -> None:
        """Update price history for a symbol."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=500)
            self._volume_history[symbol] = deque(maxlen=500)

        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)

    def validate_signal(
        self,
        symbol: str,
        direction: str,
        strength: float,
        current_price: float | None = None,
    ) -> MeanReversionValidation:
        """
        Validate a mean reversion signal.

        Args:
            symbol: Trading symbol
            direction: 'LONG' (expecting price to rise) or 'SHORT'
            strength: Signal strength (0-1)
            current_price: Optional current price

        Returns:
            MeanReversionValidation with detailed checks
        """
        prices = list(self._price_history.get(symbol, []))
        volumes = list(self._volume_history.get(symbol, []))

        rejection_reasons = []

        # Need minimum history
        if len(prices) < 50:
            return MeanReversionValidation(
                symbol=symbol,
                signal_direction=direction,
                signal_strength=strength,
                stationarity_test_passed=False,
                adf_statistic=0.0,
                adf_pvalue=1.0,
                half_life_days=0.0,
                half_life_valid=False,
                current_zscore=0.0,
                zscore_extreme=False,
                hurst_exponent=0.5,
                hurst_valid=False,
                support_resistance_confirmed=False,
                volume_confirmation=False,
                result=ValidationResult.INVALID,
                confidence=0.0,
                rejection_reasons=["Insufficient price history (need 50+ observations)"],
            )

        # ADF test for stationarity
        adf_stat, adf_pvalue = self._adf_test(prices)
        stationarity_passed = adf_pvalue < (1 - self.min_adf_confidence)

        if not stationarity_passed:
            rejection_reasons.append(f"ADF test failed (p={adf_pvalue:.4f})")

        # Half-life calculation
        half_life = self._estimate_half_life(prices)
        half_life_valid = self.min_half_life <= half_life <= self.max_half_life

        if not half_life_valid:
            rejection_reasons.append(f"Half-life out of range ({half_life:.1f} days)")

        # Current z-score
        mean_price = sum(prices) / len(prices)
        std_price = math.sqrt(sum((p - mean_price)**2 for p in prices) / len(prices))
        current = current_price if current_price else prices[-1]
        zscore = (current - mean_price) / std_price if std_price > 0 else 0

        zscore_extreme = abs(zscore) >= self.zscore_threshold

        # Check direction consistency
        if direction == "LONG" and zscore > 0:
            rejection_reasons.append("LONG signal but price above mean (z-score > 0)")
        elif direction == "SHORT" and zscore < 0:
            rejection_reasons.append("SHORT signal but price below mean (z-score < 0)")

        # Hurst exponent
        hurst = self._estimate_hurst(prices)
        hurst_valid = hurst < self.max_hurst

        if not hurst_valid:
            rejection_reasons.append(f"Hurst exponent too high ({hurst:.3f} >= {self.max_hurst})")

        # Support/resistance check
        support_resistance = self._check_support_resistance(prices, current, direction)

        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(volumes) if volumes else False

        # Calculate confidence
        confidence = self._calculate_confidence(
            stationarity_passed,
            half_life_valid,
            zscore_extreme,
            hurst_valid,
            support_resistance,
            volume_confirmed,
        )

        # Determine result
        if len(rejection_reasons) == 0:
            result = ValidationResult.VALID
        elif len(rejection_reasons) <= 2 and confidence >= 0.5:
            result = ValidationResult.WEAK
        elif confidence >= 0.4:
            result = ValidationResult.NEEDS_REVIEW
        else:
            result = ValidationResult.INVALID

        return MeanReversionValidation(
            symbol=symbol,
            signal_direction=direction,
            signal_strength=strength,
            stationarity_test_passed=stationarity_passed,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pvalue,
            half_life_days=half_life,
            half_life_valid=half_life_valid,
            current_zscore=zscore,
            zscore_extreme=zscore_extreme,
            hurst_exponent=hurst,
            hurst_valid=hurst_valid,
            support_resistance_confirmed=support_resistance,
            volume_confirmation=volume_confirmed,
            result=result,
            confidence=confidence,
            rejection_reasons=rejection_reasons,
        )

    def _adf_test(self, prices: list[float]) -> tuple[float, float]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Returns (test_statistic, p_value)
        """
        n = len(prices)
        if n < 20:
            return 0.0, 1.0

        # Calculate differences
        diff = [prices[i] - prices[i-1] for i in range(1, n)]

        # Lagged level
        lagged = prices[:-1]

        # Simple OLS regression: diff = alpha + beta * lagged + error
        mean_diff = sum(diff) / len(diff)
        mean_lagged = sum(lagged) / len(lagged)

        # Calculate beta (coefficient on lagged level)
        numerator = sum((lagged[i] - mean_lagged) * (diff[i] - mean_diff) for i in range(len(diff)))
        denominator = sum((l - mean_lagged)**2 for l in lagged)

        if denominator == 0:
            return 0.0, 1.0

        beta = numerator / denominator

        # Calculate residuals and standard error
        alpha = mean_diff - beta * mean_lagged
        residuals = [diff[i] - (alpha + beta * lagged[i]) for i in range(len(diff))]
        sse = sum(r**2 for r in residuals)
        mse = sse / (len(diff) - 2)
        se_beta = math.sqrt(mse / denominator) if denominator > 0 else 1.0

        # ADF statistic
        adf_stat = beta / se_beta if se_beta > 0 else 0

        # Approximate p-value using critical values
        # Critical values at 1%, 5%, 10% are approximately -3.43, -2.86, -2.57
        if adf_stat < -3.43:
            p_value = 0.01
        elif adf_stat < -2.86:
            p_value = 0.05
        elif adf_stat < -2.57:
            p_value = 0.10
        else:
            p_value = 0.50

        return adf_stat, p_value

    def _estimate_half_life(self, prices: list[float]) -> float:
        """Estimate mean reversion half-life using OLS."""
        n = len(prices)
        if n < 20:
            return float('inf')

        # Calculate log returns
        log_prices = [math.log(p) for p in prices if p > 0]
        if len(log_prices) < 20:
            return float('inf')

        # Mean
        mean_log = sum(log_prices) / len(log_prices)

        # Deviation from mean
        deviations = [lp - mean_log for lp in log_prices]

        # Regress change in deviation on lagged deviation
        changes = [deviations[i] - deviations[i-1] for i in range(1, len(deviations))]
        lagged_dev = deviations[:-1]

        mean_changes = sum(changes) / len(changes)
        mean_lagged = sum(lagged_dev) / len(lagged_dev)

        numerator = sum((lagged_dev[i] - mean_lagged) * (changes[i] - mean_changes) for i in range(len(changes)))
        denominator = sum((l - mean_lagged)**2 for l in lagged_dev)

        if denominator == 0:
            return float('inf')

        theta = numerator / denominator

        # Half-life = -ln(2) / theta
        if theta >= 0:
            return float('inf')

        half_life = -math.log(2) / theta

        return max(0.1, half_life)

    def _estimate_hurst(self, prices: list[float]) -> float:
        """
        Estimate Hurst exponent using R/S analysis.

        H < 0.5: mean reverting
        H = 0.5: random walk
        H > 0.5: trending
        """
        n = len(prices)
        if n < 20:
            return 0.5

        # Calculate returns
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, n) if prices[i-1] > 0]

        if len(returns) < 20:
            return 0.5

        # R/S analysis with different window sizes
        rs_values = []
        window_sizes = []

        for window in [10, 20, 40, 80]:
            if window > len(returns):
                break

            num_windows = len(returns) // window
            rs_for_window = []

            for i in range(num_windows):
                subset = returns[i * window:(i + 1) * window]
                mean_r = sum(subset) / len(subset)

                # Cumulative deviation
                cum_dev = []
                running = 0
                for r in subset:
                    running += r - mean_r
                    cum_dev.append(running)

                # Range
                R = max(cum_dev) - min(cum_dev)

                # Standard deviation
                S = math.sqrt(sum((r - mean_r)**2 for r in subset) / len(subset))

                if S > 0:
                    rs_for_window.append(R / S)

            if rs_for_window:
                rs_values.append(sum(rs_for_window) / len(rs_for_window))
                window_sizes.append(window)

        if len(rs_values) < 2:
            return 0.5

        # Fit log(R/S) = H * log(n) + c
        log_n = [math.log(w) for w in window_sizes]
        log_rs = [math.log(rs) for rs in rs_values if rs > 0]

        if len(log_rs) < 2:
            return 0.5

        mean_log_n = sum(log_n[:len(log_rs)]) / len(log_rs)
        mean_log_rs = sum(log_rs) / len(log_rs)

        numerator = sum((log_n[i] - mean_log_n) * (log_rs[i] - mean_log_rs) for i in range(len(log_rs)))
        denominator = sum((log_n[i] - mean_log_n)**2 for i in range(len(log_rs)))

        if denominator == 0:
            return 0.5

        hurst = numerator / denominator
        return max(0.0, min(1.0, hurst))

    def _check_support_resistance(
        self,
        prices: list[float],
        current: float,
        direction: str,
    ) -> bool:
        """Check if price is near support (LONG) or resistance (SHORT)."""
        recent = prices[-50:] if len(prices) >= 50 else prices

        recent_low = min(recent)
        recent_high = max(recent)
        price_range = recent_high - recent_low

        if price_range == 0:
            return False

        # Distance from extremes
        dist_from_low = (current - recent_low) / price_range
        dist_from_high = (recent_high - current) / price_range

        if direction == "LONG":
            # Near support (bottom 20% of range)
            return dist_from_low < 0.2
        else:
            # Near resistance (top 20% of range)
            return dist_from_high < 0.2

    def _check_volume_confirmation(self, volumes: list[int]) -> bool:
        """Check if volume confirms signal (above average)."""
        if len(volumes) < 20:
            return False

        avg_volume = sum(volumes[-20:]) / 20
        recent_volume = sum(volumes[-3:]) / 3

        return recent_volume > avg_volume * 1.2

    def _calculate_confidence(
        self,
        stationarity: bool,
        half_life: bool,
        zscore: bool,
        hurst: bool,
        support_resistance: bool,
        volume: bool,
    ) -> float:
        """Calculate overall confidence score."""
        weights = {
            'stationarity': 0.25,
            'half_life': 0.20,
            'zscore': 0.20,
            'hurst': 0.15,
            'support_resistance': 0.10,
            'volume': 0.10,
        }

        score = 0.0
        score += weights['stationarity'] if stationarity else 0
        score += weights['half_life'] if half_life else 0
        score += weights['zscore'] if zscore else 0
        score += weights['hurst'] if hurst else 0
        score += weights['support_resistance'] if support_resistance else 0
        score += weights['volume'] if volume else 0

        return round(score, 2)


class SpreadRatioValidator:
    """
    Validates spread ratios for stat arb (#Q17).

    Ensures pairs trading signals are statistically sound.
    """

    def __init__(
        self,
        min_cointegration_confidence: float = 0.95,
        max_half_life_days: float = 30.0,
        min_correlation: float = 0.6,
        hedge_ratio_stability_threshold: float = 0.2,  # Max std of rolling hedge ratio
        beta_neutrality_threshold: float = 0.1,  # Max net beta
    ):
        self.min_coint_confidence = min_cointegration_confidence
        self.max_half_life = max_half_life_days
        self.min_correlation = min_correlation
        self.hedge_stability_threshold = hedge_ratio_stability_threshold
        self.beta_threshold = beta_neutrality_threshold

        # Price histories
        self._prices: dict[str, deque] = {}

        # Beta values
        self._betas: dict[str, float] = {}

    def update_price(self, symbol: str, price: float) -> None:
        """Update price for a symbol."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=500)
        self._prices[symbol].append(price)

    def set_beta(self, symbol: str, beta: float) -> None:
        """Set market beta for a symbol."""
        self._betas[symbol] = beta

    def validate_spread(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
    ) -> SpreadRatioValidation:
        """
        Validate a pairs trading spread.

        Args:
            symbol1: Long leg symbol
            symbol2: Short leg symbol
            hedge_ratio: Number of symbol2 shares per symbol1 share

        Returns:
            SpreadRatioValidation with all checks
        """
        prices1 = list(self._prices.get(symbol1, []))
        prices2 = list(self._prices.get(symbol2, []))

        warnings = []

        min_len = min(len(prices1), len(prices2))

        if min_len < 60:
            return SpreadRatioValidation(
                symbol1=symbol1,
                symbol2=symbol2,
                current_spread=0.0,
                spread_mean=0.0,
                spread_std=0.0,
                current_zscore=0.0,
                is_cointegrated=False,
                cointegration_pvalue=1.0,
                hedge_ratio=hedge_ratio,
                hedge_ratio_stable=False,
                hedge_ratio_std=0.0,
                spread_half_life_days=float('inf'),
                half_life_acceptable=False,
                correlation=0.0,
                correlation_stable=False,
                is_beta_neutral=False,
                net_beta=0.0,
                result=ValidationResult.INVALID,
                confidence=0.0,
                warnings=["Insufficient price history (need 60+ observations)"],
            )

        # Align prices
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]

        # Calculate spread
        spread = [p1 - hedge_ratio * p2 for p1, p2 in zip(prices1, prices2)]
        current_spread = spread[-1]
        spread_mean = sum(spread) / len(spread)
        spread_std = math.sqrt(sum((s - spread_mean)**2 for s in spread) / len(spread))
        current_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        # Cointegration test (Engle-Granger)
        coint_stat, coint_pvalue = self._cointegration_test(prices1, prices2, hedge_ratio)
        is_cointegrated = coint_pvalue < (1 - self.min_coint_confidence)

        if not is_cointegrated:
            warnings.append(f"Cointegration test failed (p={coint_pvalue:.4f})")

        # Hedge ratio stability
        hedge_ratio_std = self._check_hedge_ratio_stability(prices1, prices2)
        hedge_stable = hedge_ratio_std < self.hedge_stability_threshold

        if not hedge_stable:
            warnings.append(f"Hedge ratio unstable (std={hedge_ratio_std:.3f})")

        # Half-life
        half_life = self._estimate_spread_half_life(spread)
        half_life_ok = half_life <= self.max_half_life

        if not half_life_ok:
            warnings.append(f"Spread half-life too long ({half_life:.1f} days)")

        # Correlation
        correlation = self._calculate_correlation(prices1, prices2)
        corr_stable = correlation >= self.min_correlation

        if not corr_stable:
            warnings.append(f"Correlation too low ({correlation:.3f})")

        # Beta neutrality
        beta1 = self._betas.get(symbol1, 1.0)
        beta2 = self._betas.get(symbol2, 1.0)
        net_beta = beta1 - hedge_ratio * beta2
        is_beta_neutral = abs(net_beta) < self.beta_threshold

        if not is_beta_neutral:
            warnings.append(f"Not beta neutral (net beta={net_beta:.3f})")

        # Calculate confidence
        checks_passed = sum([
            is_cointegrated,
            hedge_stable,
            half_life_ok,
            corr_stable,
            is_beta_neutral,
        ])
        confidence = checks_passed / 5

        # Determine result
        if len(warnings) == 0:
            result = ValidationResult.VALID
        elif len(warnings) <= 2 and confidence >= 0.6:
            result = ValidationResult.WEAK
        elif confidence >= 0.4:
            result = ValidationResult.NEEDS_REVIEW
        else:
            result = ValidationResult.INVALID

        return SpreadRatioValidation(
            symbol1=symbol1,
            symbol2=symbol2,
            current_spread=current_spread,
            spread_mean=spread_mean,
            spread_std=spread_std,
            current_zscore=current_zscore,
            is_cointegrated=is_cointegrated,
            cointegration_pvalue=coint_pvalue,
            hedge_ratio=hedge_ratio,
            hedge_ratio_stable=hedge_stable,
            hedge_ratio_std=hedge_ratio_std,
            spread_half_life_days=half_life,
            half_life_acceptable=half_life_ok,
            correlation=correlation,
            correlation_stable=corr_stable,
            is_beta_neutral=is_beta_neutral,
            net_beta=net_beta,
            result=result,
            confidence=confidence,
            warnings=warnings,
        )

    def _cointegration_test(
        self,
        prices1: list[float],
        prices2: list[float],
        hedge_ratio: float,
    ) -> tuple[float, float]:
        """Engle-Granger cointegration test."""
        # Calculate spread residuals
        spread = [p1 - hedge_ratio * p2 for p1, p2 in zip(prices1, prices2)]

        # ADF test on spread
        n = len(spread)
        if n < 20:
            return 0.0, 1.0

        # Calculate differences
        diff = [spread[i] - spread[i-1] for i in range(1, n)]
        lagged = spread[:-1]

        mean_diff = sum(diff) / len(diff)
        mean_lagged = sum(lagged) / len(lagged)

        numerator = sum((lagged[i] - mean_lagged) * (diff[i] - mean_diff) for i in range(len(diff)))
        denominator = sum((l - mean_lagged)**2 for l in lagged)

        if denominator == 0:
            return 0.0, 1.0

        beta = numerator / denominator

        # Calculate standard error
        alpha = mean_diff - beta * mean_lagged
        residuals = [diff[i] - (alpha + beta * lagged[i]) for i in range(len(diff))]
        sse = sum(r**2 for r in residuals)
        mse = sse / (len(diff) - 2)
        se_beta = math.sqrt(mse / denominator) if mse > 0 and denominator > 0 else 1.0

        adf_stat = beta / se_beta if se_beta > 0 else 0

        # Cointegration critical values (more stringent than ADF)
        # 1%: -3.90, 5%: -3.34, 10%: -3.04
        if adf_stat < -3.90:
            p_value = 0.01
        elif adf_stat < -3.34:
            p_value = 0.05
        elif adf_stat < -3.04:
            p_value = 0.10
        else:
            p_value = 0.50

        return adf_stat, p_value

    def _check_hedge_ratio_stability(
        self,
        prices1: list[float],
        prices2: list[float],
        window: int = 60,
    ) -> float:
        """Check stability of rolling hedge ratio."""
        if len(prices1) < window + 20:
            return 1.0  # High instability (bad)

        hedge_ratios = []

        for i in range(len(prices1) - window):
            p1_window = prices1[i:i + window]
            p2_window = prices2[i:i + window]

            # Simple linear regression for hedge ratio
            mean_p1 = sum(p1_window) / window
            mean_p2 = sum(p2_window) / window

            numerator = sum((p2_window[j] - mean_p2) * (p1_window[j] - mean_p1) for j in range(window))
            denominator = sum((p2_window[j] - mean_p2)**2 for j in range(window))

            if denominator > 0:
                hedge_ratios.append(numerator / denominator)

        if len(hedge_ratios) < 5:
            return 1.0

        mean_hr = sum(hedge_ratios) / len(hedge_ratios)
        std_hr = math.sqrt(sum((hr - mean_hr)**2 for hr in hedge_ratios) / len(hedge_ratios))

        return std_hr / abs(mean_hr) if mean_hr != 0 else 1.0

    def _estimate_spread_half_life(self, spread: list[float]) -> float:
        """Estimate half-life of spread mean reversion."""
        n = len(spread)
        if n < 20:
            return float('inf')

        mean_spread = sum(spread) / n
        deviations = [s - mean_spread for s in spread]

        # Regress change on lagged level
        changes = [deviations[i] - deviations[i-1] for i in range(1, n)]
        lagged = deviations[:-1]

        mean_changes = sum(changes) / len(changes)
        mean_lagged = sum(lagged) / len(lagged)

        numerator = sum((lagged[i] - mean_lagged) * (changes[i] - mean_changes) for i in range(len(changes)))
        denominator = sum((l - mean_lagged)**2 for l in lagged)

        if denominator == 0:
            return float('inf')

        theta = numerator / denominator

        if theta >= 0:
            return float('inf')

        return -math.log(2) / theta

    def _calculate_correlation(
        self,
        prices1: list[float],
        prices2: list[float],
    ) -> float:
        """Calculate Pearson correlation."""
        n = min(len(prices1), len(prices2))
        if n < 10:
            return 0.0

        mean1 = sum(prices1[-n:]) / n
        mean2 = sum(prices2[-n:]) / n

        numerator = sum((prices1[-n+i] - mean1) * (prices2[-n+i] - mean2) for i in range(n))
        denom1 = math.sqrt(sum((prices1[-n+i] - mean1)**2 for i in range(n)))
        denom2 = math.sqrt(sum((prices2[-n+i] - mean2)**2 for i in range(n)))

        if denom1 * denom2 == 0:
            return 0.0

        return numerator / (denom1 * denom2)

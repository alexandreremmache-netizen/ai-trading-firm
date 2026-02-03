"""
FX Correlation Module
=====================

Addresses issues:
- #X14: FX market depth not utilized
- #X15: No FX correlation regime switching

Features:
- FX market depth analysis
- Correlation regime detection and switching
- Cross-pair correlation tracking
- Regime-based strategy adjustment
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)


# =============================================================================
# FX PIP CALCULATION HELPERS (P0-6 Fix)
# =============================================================================

def get_pip_multiplier(pair: str) -> int:
    """
    Get pip multiplier for FX pair.

    For JPY pairs (USDJPY, EURJPY, etc.): 1 pip = 0.01, so multiplier = 100
    For other pairs (EURUSD, GBPUSD, etc.): 1 pip = 0.0001, so multiplier = 10000

    Args:
        pair: Currency pair (e.g., "USDJPY", "EURUSD")

    Returns:
        Pip multiplier (100 for JPY pairs, 10000 for others)
    """
    if "JPY" in pair.upper():
        return 100
    return 10000


class FXCorrelationRegime(str, Enum):
    """FX correlation regime states."""
    RISK_ON = "risk_on"  # USD weak, carry trade dominant
    RISK_OFF = "risk_off"  # USD strong, safe haven flows
    USD_BULL = "usd_bull"  # Broad USD strength
    USD_BEAR = "usd_bear"  # Broad USD weakness
    DIVERGENCE = "divergence"  # Low correlations, pair-specific
    CRISIS = "crisis"  # Extreme correlations, liquidity stress


@dataclass
class FXMarketDepth:
    """
    FX market depth analysis (#X14).

    Tracks order book depth for FX pairs.
    """
    pair: str
    timestamp: datetime
    bid_levels: list[tuple[float, float]]  # (price, size in millions)
    ask_levels: list[tuple[float, float]]
    total_bid_depth: float  # In millions
    total_ask_depth: float
    spread_pips: float
    depth_imbalance: float  # Positive = more bids

    @property
    def mid_price(self) -> float | None:
        """Calculate mid price."""
        if self.bid_levels and self.ask_levels:
            bid_price = self.bid_levels[0][0]
            ask_price = self.ask_levels[0][0]
            # Guard against invalid prices
            if bid_price > 0 and ask_price > 0:
                return (bid_price + ask_price) / 2
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "timestamp": self.timestamp.isoformat(),
            "bid_levels": len(self.bid_levels),
            "ask_levels": len(self.ask_levels),
            "total_bid_depth_mm": self.total_bid_depth,
            "total_ask_depth_mm": self.total_ask_depth,
            "spread_pips": self.spread_pips,
            "depth_imbalance": self.depth_imbalance,
        }


class FXMarketDepthAnalyzer:
    """
    Analyzes FX market depth (#X14).

    Uses depth data for execution optimization and market insight.
    """

    def __init__(self):
        self._depth_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def update_depth(self, depth: FXMarketDepth) -> None:
        """Update depth for a pair."""
        self._depth_history[depth.pair].append(depth)

    def get_current_depth(self, pair: str) -> FXMarketDepth | None:
        """Get most recent depth for pair."""
        history = self._depth_history.get(pair)
        if history:
            return history[-1]
        return None

    def estimate_market_impact(
        self,
        pair: str,
        size_millions: float,
        side: str,  # 'buy' or 'sell'
    ) -> dict | None:
        """
        Estimate market impact for a trade.

        Args:
            pair: FX pair
            size_millions: Trade size in millions
            side: 'buy' or 'sell'

        Returns:
            Dict with impact estimates
        """
        depth = self.get_current_depth(pair)
        if not depth:
            return None

        levels = depth.ask_levels if side == 'buy' else depth.bid_levels

        remaining_size = size_millions
        total_cost = 0.0
        levels_consumed = 0

        for price, level_size in levels:
            fill_at_level = min(remaining_size, level_size)
            total_cost += fill_at_level * price
            remaining_size -= fill_at_level
            levels_consumed += 1

            if remaining_size <= 0:
                break

        filled_size = size_millions - max(0, remaining_size)
        if filled_size > 0:
            avg_price = total_cost / filled_size
            best_price = levels[0][0] if levels else 0

            impact_pips = abs(avg_price - best_price) * get_pip_multiplier(pair)  # P0-6: Use proper multiplier for JPY pairs

            return {
                "pair": pair,
                "side": side,
                "size_millions": size_millions,
                "filled_millions": filled_size,
                "unfilled_millions": max(0, remaining_size),
                "levels_consumed": levels_consumed,
                "avg_fill_price": avg_price,
                "market_impact_pips": impact_pips,
                "can_fill_full": remaining_size <= 0,
            }

        return None

    def get_depth_quality(self, pair: str) -> dict:
        """Assess depth quality for a pair."""
        depth = self.get_current_depth(pair)
        if not depth:
            return {"pair": pair, "quality": "no_data"}

        quality_score = 0

        # Spread assessment
        if depth.spread_pips < 0.5:
            quality_score += 30
        elif depth.spread_pips < 1.0:
            quality_score += 20
        elif depth.spread_pips < 2.0:
            quality_score += 10

        # Depth assessment (total liquidity)
        total_depth = depth.total_bid_depth + depth.total_ask_depth
        if total_depth > 100:  # > $100M
            quality_score += 30
        elif total_depth > 50:
            quality_score += 20
        elif total_depth > 20:
            quality_score += 10

        # Balance assessment
        if abs(depth.depth_imbalance) < 0.2:
            quality_score += 20
        elif abs(depth.depth_imbalance) < 0.4:
            quality_score += 10

        # Levels assessment
        total_levels = len(depth.bid_levels) + len(depth.ask_levels)
        if total_levels >= 10:
            quality_score += 20
        elif total_levels >= 6:
            quality_score += 10

        return {
            "pair": pair,
            "quality_score": quality_score,
            "quality": "excellent" if quality_score >= 80 else
                       "good" if quality_score >= 60 else
                       "moderate" if quality_score >= 40 else "poor",
            "spread_pips": depth.spread_pips,
            "total_depth_mm": total_depth,
            "imbalance": depth.depth_imbalance,
        }


# =========================================================================
# FX CORRELATION REGIME SWITCHING (#X15)
# =========================================================================

@dataclass
class RegimeIndicators:
    """Indicators used for regime detection."""
    usd_index_change: float  # DXY change
    vix_level: float  # VIX level
    avg_g10_correlation: float  # Average correlation among G10 pairs
    carry_performance: float  # High-yield vs safe-haven performance
    spread_volatility: float  # Average spread volatility

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "usd_index_change": self.usd_index_change,
            "vix_level": self.vix_level,
            "avg_g10_correlation": self.avg_g10_correlation,
            "carry_performance": self.carry_performance,
            "spread_volatility": self.spread_volatility,
        }


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    current_regime: FXCorrelationRegime
    regime_probability: float
    regime_duration_days: int
    all_probabilities: dict[FXCorrelationRegime, float]
    indicators: RegimeIndicators
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current_regime": self.current_regime.value,
            "probability": self.regime_probability,
            "duration_days": self.regime_duration_days,
            "all_probabilities": {r.value: p for r, p in self.all_probabilities.items()},
            "indicators": self.indicators.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


class FXCorrelationRegimeDetector:
    """
    Detects FX correlation regime switches (#X15).

    Monitors cross-pair correlations and market indicators
    to identify regime changes.
    """

    # G10 currency pairs for correlation analysis
    G10_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD", "EURGBP",
    ]

    # High-yield (carry) pairs
    CARRY_PAIRS = ["AUDUSD", "NZDUSD", "USDMXN", "USDZAR"]

    # Safe-haven pairs
    SAFE_HAVEN_PAIRS = ["USDJPY", "USDCHF"]

    def __init__(
        self,
        correlation_window: int = 20,
        regime_change_threshold: float = 0.15,
    ):
        """
        Initialize regime detector.

        Args:
            correlation_window: Days for correlation calculation
            regime_change_threshold: Min probability change to trigger switch
        """
        self.correlation_window = correlation_window
        self.regime_change_threshold = regime_change_threshold

        # Return history
        self._returns: dict[str, deque] = defaultdict(lambda: deque(maxlen=correlation_window))
        self._vix_history: deque = deque(maxlen=correlation_window)
        self._dxy_history: deque = deque(maxlen=correlation_window)

        # Regime tracking
        self._current_regime = FXCorrelationRegime.DIVERGENCE
        self._regime_start: datetime | None = None
        self._regime_history: list[tuple[datetime, FXCorrelationRegime]] = []

    def update_return(self, pair: str, daily_return: float) -> None:
        """Update daily return for a pair."""
        self._returns[pair].append(daily_return)

    def update_indicators(
        self,
        vix: float | None = None,
        dxy_change: float | None = None,
    ) -> None:
        """Update market indicators."""
        if vix is not None:
            self._vix_history.append(vix)
        if dxy_change is not None:
            self._dxy_history.append(dxy_change)

    def detect_regime(self) -> RegimeState:
        """
        Detect current correlation regime.

        Returns:
            RegimeState with current regime and indicators
        """
        # Calculate indicators
        indicators = self._calculate_indicators()

        # Calculate regime probabilities
        probabilities = self._calculate_regime_probabilities(indicators)

        # Determine regime
        new_regime = max(probabilities.items(), key=lambda x: x[1])[0]
        regime_prob = probabilities[new_regime]

        # Check for regime change
        if new_regime != self._current_regime:
            if regime_prob - probabilities.get(self._current_regime, 0) > self.regime_change_threshold:
                self._regime_history.append((datetime.now(timezone.utc), self._current_regime))
                self._current_regime = new_regime
                self._regime_start = datetime.now(timezone.utc)
                logger.info(f"FX regime change detected: {new_regime.value}")

        # Calculate duration
        duration_days = 0
        if self._regime_start:
            duration_days = (datetime.now(timezone.utc) - self._regime_start).days

        return RegimeState(
            current_regime=self._current_regime,
            regime_probability=regime_prob,
            regime_duration_days=duration_days,
            all_probabilities=probabilities,
            indicators=indicators,
            timestamp=datetime.now(timezone.utc),
        )

    def _calculate_indicators(self) -> RegimeIndicators:
        """Calculate regime indicators."""
        # DXY change (5-day)
        if len(self._dxy_history) >= 5:
            usd_change = sum(list(self._dxy_history)[-5:])
        else:
            usd_change = 0.0

        # VIX level
        vix = self._vix_history[-1] if self._vix_history else 20.0

        # Average G10 correlation
        g10_corrs = []
        for i, pair1 in enumerate(self.G10_PAIRS):
            for pair2 in self.G10_PAIRS[i + 1:]:
                corr = self._calculate_correlation(pair1, pair2)
                if corr is not None:
                    g10_corrs.append(abs(corr))

        avg_corr = statistics.mean(g10_corrs) if g10_corrs else 0.5

        # Carry performance (high-yield vs safe-haven)
        carry_returns = []
        for pair in self.CARRY_PAIRS:
            if pair in self._returns and self._returns[pair]:
                carry_returns.append(sum(self._returns[pair]))

        safe_returns = []
        for pair in self.SAFE_HAVEN_PAIRS:
            if pair in self._returns and self._returns[pair]:
                safe_returns.append(sum(self._returns[pair]))

        carry_perf = 0.0
        if carry_returns and safe_returns:
            carry_perf = statistics.mean(carry_returns) - statistics.mean(safe_returns)

        # Spread volatility (simplified as return volatility)
        spread_vol = 0.0
        all_returns = []
        for returns in self._returns.values():
            all_returns.extend(returns)
        if len(all_returns) > 5:
            spread_vol = statistics.stdev(all_returns)

        return RegimeIndicators(
            usd_index_change=usd_change,
            vix_level=vix,
            avg_g10_correlation=avg_corr,
            carry_performance=carry_perf,
            spread_volatility=spread_vol,
        )

    def _calculate_correlation(self, pair1: str, pair2: str) -> float | None:
        """Calculate correlation between two pairs."""
        returns1 = list(self._returns.get(pair1, []))
        returns2 = list(self._returns.get(pair2, []))

        if len(returns1) < 10 or len(returns2) < 10:
            return None

        # Align lengths
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1[-min_len:]
        returns2 = returns2[-min_len:]

        # Pearson correlation
        mean1 = statistics.mean(returns1)
        mean2 = statistics.mean(returns2)

        numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(returns1, returns2))
        var1 = sum((r - mean1) ** 2 for r in returns1)
        var2 = sum((r - mean2) ** 2 for r in returns2)

        denominator = math.sqrt(var1 * var2)
        # Use threshold instead of exact zero to handle numerical precision
        if denominator < 1e-12:
            return 0.0

        # Clip return value to valid correlation range [-1.0, 1.0]
        correlation = numerator / denominator
        return max(-1.0, min(1.0, correlation))

    def get_correlation_matrix(self, pairs: list[str] | None = None) -> tuple[list[str], list[list[float]]]:
        """
        Get full correlation matrix for pairs (FX-P1-1 Fix).

        Returns a guaranteed symmetric correlation matrix.

        Args:
            pairs: List of pairs to include (defaults to G10_PAIRS)

        Returns:
            Tuple of (pair_names, correlation_matrix)
        """
        if pairs is None:
            pairs = self.G10_PAIRS

        n = len(pairs)
        corr_matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        # Calculate correlations
        for i in range(n):
            for j in range(i + 1, n):
                corr = self._calculate_correlation(pairs[i], pairs[j])
                if corr is not None:
                    corr_matrix[i][j] = corr
                    corr_matrix[j][i] = corr  # Ensure symmetry

        # FX-P1-1: Force symmetry after construction (defensive)
        # corr[i][j] = (corr[i][j] + corr[j][i]) / 2
        for i in range(n):
            for j in range(i + 1, n):
                avg = (corr_matrix[i][j] + corr_matrix[j][i]) / 2
                corr_matrix[i][j] = avg
                corr_matrix[j][i] = avg

        return pairs, corr_matrix

    def _calculate_regime_probabilities(
        self,
        indicators: RegimeIndicators,
    ) -> dict[FXCorrelationRegime, float]:
        """Calculate probability of each regime."""
        probabilities = {regime: 0.0 for regime in FXCorrelationRegime}

        # Risk-off: High VIX, high correlations, safe-haven outperforming
        if indicators.vix_level > 25 and indicators.avg_g10_correlation > 0.6:
            probabilities[FXCorrelationRegime.RISK_OFF] += 0.4
        if indicators.carry_performance < -0.01:  # Carry underperforming
            probabilities[FXCorrelationRegime.RISK_OFF] += 0.3

        # Risk-on: Low VIX, carry outperforming
        if indicators.vix_level < 20 and indicators.carry_performance > 0.01:
            probabilities[FXCorrelationRegime.RISK_ON] += 0.4
        if indicators.avg_g10_correlation < 0.4:
            probabilities[FXCorrelationRegime.RISK_ON] += 0.2

        # USD Bull: Strong DXY
        if indicators.usd_index_change > 0.02:
            probabilities[FXCorrelationRegime.USD_BULL] += 0.5

        # USD Bear: Weak DXY
        if indicators.usd_index_change < -0.02:
            probabilities[FXCorrelationRegime.USD_BEAR] += 0.5

        # Crisis: Very high VIX and correlations
        if indicators.vix_level > 35 and indicators.avg_g10_correlation > 0.75:
            probabilities[FXCorrelationRegime.CRISIS] = 0.6

        # Divergence: Low correlations
        if indicators.avg_g10_correlation < 0.35:
            probabilities[FXCorrelationRegime.DIVERGENCE] += 0.4

        # Normalize
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        else:
            probabilities[FXCorrelationRegime.DIVERGENCE] = 1.0

        return probabilities

    def get_strategy_adjustments(self) -> dict:
        """
        Get strategy adjustments based on current regime.

        Returns recommendations for position sizing, pair selection, etc.
        """
        regime = self._current_regime

        adjustments = {
            "regime": regime.value,
            "position_size_multiplier": 1.0,
            "preferred_pairs": [],
            "avoid_pairs": [],
            "strategy_recommendations": [],
        }

        if regime == FXCorrelationRegime.RISK_OFF:
            adjustments["position_size_multiplier"] = 0.7
            adjustments["preferred_pairs"] = ["USDJPY", "USDCHF"]
            adjustments["avoid_pairs"] = ["AUDUSD", "NZDUSD", "USDMXN"]
            adjustments["strategy_recommendations"] = [
                "Favor safe-haven currencies",
                "Reduce carry trade exposure",
                "Tighten stop losses",
            ]

        elif regime == FXCorrelationRegime.RISK_ON:
            adjustments["position_size_multiplier"] = 1.0
            adjustments["preferred_pairs"] = ["AUDUSD", "NZDUSD"]
            adjustments["avoid_pairs"] = []
            adjustments["strategy_recommendations"] = [
                "Carry trades favored",
                "Commodity currencies may outperform",
            ]

        elif regime == FXCorrelationRegime.USD_BULL:
            adjustments["position_size_multiplier"] = 1.0
            adjustments["preferred_pairs"] = ["USDJPY", "USDCAD", "USDCHF"]
            adjustments["strategy_recommendations"] = [
                "Favor long USD positions",
                "Consider shorting EUR, GBP",
            ]

        elif regime == FXCorrelationRegime.USD_BEAR:
            adjustments["position_size_multiplier"] = 1.0
            adjustments["preferred_pairs"] = ["EURUSD", "GBPUSD", "AUDUSD"]
            adjustments["strategy_recommendations"] = [
                "Favor short USD positions",
                "Commodity currencies may outperform",
            ]

        elif regime == FXCorrelationRegime.DIVERGENCE:
            adjustments["position_size_multiplier"] = 1.2
            adjustments["strategy_recommendations"] = [
                "Pair-specific analysis more important",
                "Technical signals more reliable",
                "Good environment for relative value",
            ]

        elif regime == FXCorrelationRegime.CRISIS:
            adjustments["position_size_multiplier"] = 0.5
            adjustments["preferred_pairs"] = ["USDJPY", "USDCHF"]
            adjustments["avoid_pairs"] = self.CARRY_PAIRS
            adjustments["strategy_recommendations"] = [
                "Reduce all positions",
                "Safe-haven only",
                "Monitor liquidity closely",
                "Wider stops due to volatility",
            ]

        return adjustments

    def get_regime_history(
        self,
        lookback_days: int = 90,
    ) -> list[dict]:
        """Get regime history for analysis."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        history = [
            {"timestamp": ts.isoformat(), "regime": regime.value}
            for ts, regime in self._regime_history
            if ts >= cutoff
        ]

        # Add current regime
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": self._current_regime.value,
        })

        return history


# =============================================================================
# ROLLING CORRELATION WINDOWS (P3 Enhancement)
# =============================================================================

@dataclass
class RollingCorrelationResult:
    """Result of rolling correlation analysis."""
    pair1: str
    pair2: str
    window_size: int
    correlations: list[float]
    timestamps: list[datetime]
    current_correlation: float | None
    trend: str  # "increasing", "decreasing", "stable"
    volatility: float  # Correlation volatility

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair1": self.pair1,
            "pair2": self.pair2,
            "window_size": self.window_size,
            "current_correlation": self.current_correlation,
            "trend": self.trend,
            "correlation_volatility": self.volatility,
            "num_observations": len(self.correlations),
        }


class RollingCorrelationAnalyzer:
    """
    Rolling correlation window analysis (P3 Enhancement).

    Tracks how correlations evolve over multiple time windows.
    """

    # Standard window sizes (in days)
    WINDOW_SIZES = [5, 10, 20, 60, 120]

    def __init__(self, max_history: int = 252):
        """
        Initialize rolling correlation analyzer.

        Args:
            max_history: Maximum number of daily returns to store
        """
        self.max_history = max_history
        self._returns: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._timestamps: deque = deque(maxlen=max_history)

    def update_returns(self, pair: str, daily_return: float, timestamp: datetime | None = None) -> None:
        """Update daily return for a pair."""
        self._returns[pair].append(daily_return)
        if timestamp and (not self._timestamps or timestamp != self._timestamps[-1]):
            self._timestamps.append(timestamp or datetime.now(timezone.utc))

    def calculate_rolling_correlation(
        self,
        pair1: str,
        pair2: str,
        window_size: int = 20,
    ) -> RollingCorrelationResult | None:
        """
        Calculate rolling correlation between two pairs.

        Args:
            pair1: First currency pair
            pair2: Second currency pair
            window_size: Rolling window size in days

        Returns:
            RollingCorrelationResult or None if insufficient data
        """
        returns1 = list(self._returns.get(pair1, []))
        returns2 = list(self._returns.get(pair2, []))

        if len(returns1) < window_size or len(returns2) < window_size:
            return None

        # Align lengths
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1[-min_len:]
        returns2 = returns2[-min_len:]

        # Calculate rolling correlations
        correlations = []
        timestamps = list(self._timestamps)[-min_len:]

        for i in range(window_size - 1, min_len):
            window1 = returns1[i - window_size + 1:i + 1]
            window2 = returns2[i - window_size + 1:i + 1]

            corr = self._pearson_correlation(window1, window2)
            if corr is not None:
                correlations.append(corr)

        if not correlations:
            return None

        # Determine trend
        trend = self._determine_trend(correlations)

        # Calculate correlation volatility
        volatility = statistics.stdev(correlations) if len(correlations) > 1 else 0.0

        return RollingCorrelationResult(
            pair1=pair1,
            pair2=pair2,
            window_size=window_size,
            correlations=correlations,
            timestamps=timestamps[window_size - 1:] if timestamps else [],
            current_correlation=correlations[-1] if correlations else None,
            trend=trend,
            volatility=volatility,
        )

    def calculate_multi_window_correlations(
        self,
        pair1: str,
        pair2: str,
        windows: list[int] | None = None,
    ) -> dict[int, RollingCorrelationResult | None]:
        """
        Calculate correlations across multiple window sizes.

        Args:
            pair1: First currency pair
            pair2: Second currency pair
            windows: List of window sizes (default: WINDOW_SIZES)

        Returns:
            Dict mapping window size to correlation result
        """
        windows = windows or self.WINDOW_SIZES

        return {
            window: self.calculate_rolling_correlation(pair1, pair2, window)
            for window in windows
        }

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float | None:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return None

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = math.sqrt(var_x * var_y)
        if denominator < 1e-12:
            return 0.0

        return max(-1.0, min(1.0, numerator / denominator))

    def _determine_trend(self, correlations: list[float]) -> str:
        """Determine trend from correlation series."""
        if len(correlations) < 5:
            return "stable"

        recent = correlations[-5:]
        older = correlations[-10:-5] if len(correlations) >= 10 else correlations[:5]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg

        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        return "stable"


# =============================================================================
# CORRELATION REGIME DETECTION (P3 Enhancement)
# =============================================================================

class CorrelationRegime(str, Enum):
    """Correlation regime states."""
    HIGH_CORRELATION = "high_correlation"  # All pairs moving together
    LOW_CORRELATION = "low_correlation"  # Pairs moving independently
    TRANSITIONING = "transitioning"  # Regime change in progress
    BREAKDOWN = "breakdown"  # Historical correlations breaking down


@dataclass
class CorrelationRegimeState:
    """Current correlation regime state."""
    regime: CorrelationRegime
    confidence: float
    avg_correlation: float
    correlation_dispersion: float  # Std dev of correlations
    regime_duration_days: int
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "avg_correlation": self.avg_correlation,
            "correlation_dispersion": self.correlation_dispersion,
            "regime_duration_days": self.regime_duration_days,
            "timestamp": self.timestamp.isoformat(),
        }


class CorrelationRegimeDetector:
    """
    Detects correlation regime changes (P3 Enhancement).

    Monitors overall correlation levels and dispersion to identify
    regime shifts that impact portfolio diversification.
    """

    # Thresholds for regime classification
    HIGH_CORR_THRESHOLD = 0.6
    LOW_CORR_THRESHOLD = 0.3
    DISPERSION_THRESHOLD = 0.2

    def __init__(self, lookback_days: int = 60):
        """
        Initialize correlation regime detector.

        Args:
            lookback_days: Days of history for regime detection
        """
        self.lookback_days = lookback_days
        self._correlation_history: deque = deque(maxlen=lookback_days)
        self._current_regime = CorrelationRegime.LOW_CORRELATION
        self._regime_start: datetime | None = None

    def update_correlations(self, correlation_matrix: list[list[float]]) -> None:
        """
        Update with new correlation matrix.

        Args:
            correlation_matrix: NxN correlation matrix
        """
        # Extract upper triangle (excluding diagonal)
        n = len(correlation_matrix)
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                correlations.append(correlation_matrix[i][j])

        if correlations:
            self._correlation_history.append({
                "timestamp": datetime.now(timezone.utc),
                "avg": statistics.mean(correlations),
                "std": statistics.stdev(correlations) if len(correlations) > 1 else 0.0,
                "correlations": correlations,
            })

    def detect_regime(self) -> CorrelationRegimeState:
        """
        Detect current correlation regime.

        Returns:
            CorrelationRegimeState with current regime and metrics
        """
        if not self._correlation_history:
            return CorrelationRegimeState(
                regime=CorrelationRegime.LOW_CORRELATION,
                confidence=0.5,
                avg_correlation=0.0,
                correlation_dispersion=0.0,
                regime_duration_days=0,
                timestamp=datetime.now(timezone.utc),
            )

        # Get recent correlations
        recent = list(self._correlation_history)[-10:]
        avg_correlations = [r["avg"] for r in recent]
        avg_dispersions = [r["std"] for r in recent]

        current_avg = statistics.mean(avg_correlations)
        current_dispersion = statistics.mean(avg_dispersions)

        # Determine regime
        new_regime, confidence = self._classify_regime(current_avg, current_dispersion)

        # Check for regime change
        if new_regime != self._current_regime:
            if confidence > 0.7:
                self._current_regime = new_regime
                self._regime_start = datetime.now(timezone.utc)
                logger.info(f"Correlation regime changed to: {new_regime.value}")

        # Calculate duration
        duration_days = 0
        if self._regime_start:
            duration_days = (datetime.now(timezone.utc) - self._regime_start).days

        return CorrelationRegimeState(
            regime=self._current_regime,
            confidence=confidence,
            avg_correlation=current_avg,
            correlation_dispersion=current_dispersion,
            regime_duration_days=duration_days,
            timestamp=datetime.now(timezone.utc),
        )

    def _classify_regime(
        self,
        avg_correlation: float,
        dispersion: float,
    ) -> tuple[CorrelationRegime, float]:
        """Classify correlation regime."""
        # High correlation regime
        if avg_correlation > self.HIGH_CORR_THRESHOLD:
            if dispersion < self.DISPERSION_THRESHOLD:
                return CorrelationRegime.HIGH_CORRELATION, 0.9
            else:
                return CorrelationRegime.TRANSITIONING, 0.6

        # Low correlation regime
        if avg_correlation < self.LOW_CORR_THRESHOLD:
            return CorrelationRegime.LOW_CORRELATION, 0.85

        # Check for breakdown (high dispersion with moderate avg)
        if dispersion > self.DISPERSION_THRESHOLD * 1.5:
            return CorrelationRegime.BREAKDOWN, 0.75

        # Transitioning
        return CorrelationRegime.TRANSITIONING, 0.5

    def get_diversification_effectiveness(self) -> float:
        """
        Calculate diversification effectiveness score.

        Returns:
            Score from 0-100 where higher = better diversification
        """
        if not self._correlation_history:
            return 50.0

        recent = self._correlation_history[-1]
        avg_corr = abs(recent["avg"])

        # Lower correlation = better diversification
        return max(0, (1 - avg_corr) * 100)


# =============================================================================
# CROSS-CURRENCY BASIS TRACKING (P3 Enhancement)
# =============================================================================

@dataclass
class CrossCurrencyBasis:
    """Cross-currency basis swap information."""
    pair: str  # e.g., "EURUSD"
    tenor: str  # e.g., "3M", "1Y"
    basis_bps: float  # Basis in basis points
    timestamp: datetime
    direction: str  # "bid" or "ask" for funding premium

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "tenor": self.tenor,
            "basis_bps": self.basis_bps,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
        }


class CrossCurrencyBasisTracker:
    """
    Tracks cross-currency basis swaps (P3 Enhancement).

    Monitors funding costs across currency pairs for:
    - Hedging cost estimation
    - Carry trade adjustments
    - Liquidity stress detection
    """

    # Standard tenors
    TENORS = ["1M", "3M", "6M", "1Y", "2Y", "5Y"]

    # Major basis pairs
    MAJOR_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF"]

    def __init__(self, history_days: int = 90):
        """
        Initialize basis tracker.

        Args:
            history_days: Days of history to maintain
        """
        self.history_days = history_days
        # Key: (pair, tenor) -> deque of CrossCurrencyBasis
        self._basis_history: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=history_days)
        )
        self._alerts: list[dict] = []

    def update_basis(self, basis: CrossCurrencyBasis) -> None:
        """Update basis for a pair/tenor."""
        key = (basis.pair, basis.tenor)
        self._basis_history[key].append(basis)

        # Check for alerts
        self._check_basis_alerts(basis)

    def get_current_basis(self, pair: str, tenor: str) -> CrossCurrencyBasis | None:
        """Get most recent basis for pair/tenor."""
        key = (pair, tenor)
        history = self._basis_history.get(key)
        if history:
            return history[-1]
        return None

    def get_basis_curve(self, pair: str) -> dict[str, float | None]:
        """
        Get full basis curve for a pair.

        Returns:
            Dict mapping tenor to basis (bps)
        """
        curve = {}
        for tenor in self.TENORS:
            basis = self.get_current_basis(pair, tenor)
            curve[tenor] = basis.basis_bps if basis else None
        return curve

    def get_hedging_cost(
        self,
        pair: str,
        tenor: str,
        notional_millions: float,
    ) -> dict | None:
        """
        Calculate hedging cost estimate.

        Args:
            pair: Currency pair
            tenor: Hedge tenor
            notional_millions: Notional in millions

        Returns:
            Dict with cost estimates
        """
        basis = self.get_current_basis(pair, tenor)
        if not basis:
            return None

        # Annual cost in basis points
        annual_cost_bps = abs(basis.basis_bps)

        # Adjust for tenor
        tenor_fraction = self._tenor_to_fraction(tenor)
        period_cost_bps = annual_cost_bps * tenor_fraction

        # Cost in currency
        period_cost_amount = notional_millions * 1_000_000 * period_cost_bps / 10_000

        return {
            "pair": pair,
            "tenor": tenor,
            "notional_millions": notional_millions,
            "basis_bps": basis.basis_bps,
            "annual_cost_bps": annual_cost_bps,
            "period_cost_bps": period_cost_bps,
            "period_cost_amount": period_cost_amount,
            "direction": basis.direction,
        }

    def get_basis_z_score(self, pair: str, tenor: str) -> float | None:
        """
        Calculate z-score of current basis vs history.

        Returns:
            Z-score (positive = wider than normal)
        """
        key = (pair, tenor)
        history = list(self._basis_history.get(key, []))

        if len(history) < 20:
            return None

        basis_values = [b.basis_bps for b in history]
        current = basis_values[-1]

        mean = statistics.mean(basis_values[:-1])
        std = statistics.stdev(basis_values[:-1])

        if std < 0.1:
            return 0.0

        return (current - mean) / std

    def detect_stress(self) -> dict:
        """
        Detect funding stress across currency pairs.

        Returns:
            Dict with stress indicators
        """
        stress_indicators = {
            "overall_stress": "normal",
            "stress_score": 0.0,
            "stressed_pairs": [],
            "widening_pairs": [],
        }

        stress_scores = []

        for pair in self.MAJOR_PAIRS:
            # Check 3M basis (most liquid tenor)
            z_score = self.get_basis_z_score(pair, "3M")

            if z_score is not None:
                stress_scores.append(abs(z_score))

                if z_score > 2.0:
                    stress_indicators["stressed_pairs"].append({
                        "pair": pair,
                        "z_score": z_score,
                    })
                elif z_score > 1.5:
                    stress_indicators["widening_pairs"].append({
                        "pair": pair,
                        "z_score": z_score,
                    })

        if stress_scores:
            avg_stress = statistics.mean(stress_scores)
            stress_indicators["stress_score"] = avg_stress

            if avg_stress > 2.0:
                stress_indicators["overall_stress"] = "high"
            elif avg_stress > 1.0:
                stress_indicators["overall_stress"] = "elevated"

        return stress_indicators

    def _tenor_to_fraction(self, tenor: str) -> float:
        """Convert tenor string to year fraction."""
        tenor_map = {
            "1M": 1/12,
            "3M": 0.25,
            "6M": 0.5,
            "1Y": 1.0,
            "2Y": 2.0,
            "5Y": 5.0,
        }
        return tenor_map.get(tenor, 1.0)

    def _check_basis_alerts(self, basis: CrossCurrencyBasis) -> None:
        """Check for alertable basis movements."""
        z_score = self.get_basis_z_score(basis.pair, basis.tenor)

        if z_score is not None and abs(z_score) > 2.5:
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "basis_stress",
                "pair": basis.pair,
                "tenor": basis.tenor,
                "basis_bps": basis.basis_bps,
                "z_score": z_score,
                "severity": "high" if abs(z_score) > 3.0 else "medium",
            }
            self._alerts.append(alert)
            logger.warning(f"Cross-currency basis alert: {basis.pair} {basis.tenor} z-score={z_score:.2f}")

    def get_alerts(self, since_hours: int = 24) -> list[dict]:
        """Get recent alerts."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        return [
            a for a in self._alerts
            if datetime.fromisoformat(a["timestamp"]) >= cutoff
        ]

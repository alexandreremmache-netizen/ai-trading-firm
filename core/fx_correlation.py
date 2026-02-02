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
            return (self.bid_levels[0][0] + self.ask_levels[0][0]) / 2
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

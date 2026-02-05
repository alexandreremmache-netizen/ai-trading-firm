"""
Signal Quality Score Module
===========================

Module for validating signal quality before CIO decisions.
Scores signals on 6 dimensions (0-100 total):
- Trend alignment (0-20)
- Volume confirmation (0-20)
- Volatility regime (0-15)
- Support/Resistance proximity (0-20)
- Timing quality (0-15)
- Confluence with other signals (0-10)

Signals below the minimum score threshold are rejected.
"""

from __future__ import annotations

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SignalQualityTier(Enum):
    """Signal quality tiers."""
    EXCELLENT = "excellent"    # > 85
    GOOD = "good"              # 70-85
    ACCEPTABLE = "acceptable"  # 55-70
    POOR = "poor"              # 40-55
    REJECT = "reject"          # < 40


@dataclass
class SignalQualityResult:
    """Complete quality analysis result."""
    total_score: float
    tier: SignalQualityTier
    is_valid: bool

    # Sub-scores
    trend_alignment: float
    volume_confirmation: float
    volatility_regime: float
    support_resistance: float
    timing: float
    confluence: float

    # Confidence boost to apply
    confidence_boost: float

    # Recommendation
    recommendation: str
    rejection_reasons: List[str] = field(default_factory=list)


class SignalQualityScorer:
    """
    Signal Quality Scorer for filtering low-quality signals.

    Integrates with CIO Agent to reject signals that don't meet
    minimum quality thresholds across multiple dimensions.
    """

    def __init__(
        self,
        min_total_score: float = 50.0,
        min_volume_score: float = 5.0,  # Lowered for warmup periods
        min_trend_score: float = 5.0,   # Lowered for warmup periods
        enable_logging: bool = True,
    ):
        self.min_total_score = min_total_score
        self.min_volume_score = min_volume_score
        self.min_trend_score = min_trend_score
        self.enable_logging = enable_logging

        # History for analysis (bounded)
        self._quality_history: deque[dict] = deque(maxlen=1000)

        # Statistics
        self._total_evaluated = 0
        self._total_rejected = 0

    def validate_signal(
        self,
        signal: Any,
        market_data: Dict[str, Any],
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
        other_signals: Optional[List[Any]] = None,
    ) -> SignalQualityResult:
        """
        Validate a signal and return its quality score.

        Args:
            signal: The SignalEvent to validate
            market_data: Market data dict with prices, volumes, atr, adx, volatility_regime
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
            other_signals: Other active signals for confluence check

        Returns:
            SignalQualityResult with complete quality assessment
        """
        support_levels = support_levels or []
        resistance_levels = resistance_levels or []
        other_signals = other_signals or []

        # Get direction from signal
        direction = getattr(signal, 'direction', None)
        if direction is not None:
            direction = direction.value if hasattr(direction, 'value') else str(direction)
        else:
            direction = "flat"

        # Get price from market data
        prices = market_data.get('prices', [])
        price = prices[-1] if prices else 0.0

        # Calculate sub-scores
        trend_score = self._calc_trend_score(direction, market_data)
        volume_score = self._calc_volume_score(market_data)
        vol_score = self._calc_volatility_score(direction, market_data)
        sr_score = self._calc_sr_score(price, direction, support_levels, resistance_levels)
        timing_score = self._calc_timing_score(direction, market_data)
        confluence_score = self._calc_confluence_score(direction, other_signals)

        # Total score
        total = trend_score + volume_score + vol_score + sr_score + timing_score + confluence_score

        # Check thresholds
        rejection_reasons = []

        if volume_score < self.min_volume_score:
            rejection_reasons.append(f"Volume score too low: {volume_score:.1f}/20")

        if trend_score < self.min_trend_score:
            rejection_reasons.append(f"Trend alignment weak: {trend_score:.1f}/20")

        if total < self.min_total_score:
            rejection_reasons.append(f"Total score too low: {total:.1f}/100")

        # Determine tier
        if total >= 85:
            tier = SignalQualityTier.EXCELLENT
        elif total >= 70:
            tier = SignalQualityTier.GOOD
        elif total >= 55:
            tier = SignalQualityTier.ACCEPTABLE
        elif total >= 40:
            tier = SignalQualityTier.POOR
        else:
            tier = SignalQualityTier.REJECT

        # Valid if acceptable tier and no rejection reasons
        is_valid = tier not in [SignalQualityTier.REJECT, SignalQualityTier.POOR] and not rejection_reasons

        # Confidence boost
        confidence_boost = self._calc_confidence_boost(total, tier)

        # Recommendation
        recommendations = {
            SignalQualityTier.EXCELLENT: "Strong entry recommended",
            SignalQualityTier.GOOD: "Entry recommended",
            SignalQualityTier.ACCEPTABLE: "Enter with caution",
            SignalQualityTier.POOR: "Avoid or reduce size",
            SignalQualityTier.REJECT: "REJECT - Poor quality signal",
        }
        recommendation = recommendations.get(tier, "Unknown")

        result = SignalQualityResult(
            total_score=total,
            tier=tier,
            is_valid=is_valid,
            trend_alignment=trend_score,
            volume_confirmation=volume_score,
            volatility_regime=vol_score,
            support_resistance=sr_score,
            timing=timing_score,
            confluence=confluence_score,
            confidence_boost=confidence_boost,
            recommendation=recommendation,
            rejection_reasons=rejection_reasons,
        )

        # Track statistics
        self._total_evaluated += 1
        if not is_valid:
            self._total_rejected += 1

        # Store for analysis
        self._quality_history.append({
            'symbol': getattr(signal, 'symbol', 'unknown'),
            'direction': direction,
            'score': total,
            'tier': tier.value,
            'is_valid': is_valid,
        })

        return result

    def _calc_trend_score(self, direction: str, market_data: Dict) -> float:
        """Score for trend alignment (0-20)."""
        prices = market_data.get('prices', [])
        adx = market_data.get('adx', 20)

        if len(prices) < 10:
            return 10.0  # Neutral score during warmup

        # Calculate SMAs
        sma_short = np.mean(prices[-min(10, len(prices)):])
        sma_long = np.mean(prices[-min(20, len(prices)):]) if len(prices) >= 20 else sma_short

        # Determine trend direction
        trend_direction = "long" if sma_short > sma_long else "short"
        aligned = (direction == trend_direction) or direction == "flat"

        # ADX factor (stronger trend = more weight)
        adx_factor = min(adx / 50.0, 1.0)

        if aligned:
            return min(15.0 + 5.0 * adx_factor, 20.0)
        else:
            return min(5.0 + 3.0 * adx_factor, 12.0)

    def _calc_volume_score(self, market_data: Dict) -> float:
        """Score for volume confirmation (0-20)."""
        volumes = market_data.get('volumes', [])

        if len(volumes) < 5:
            return 10.0  # Neutral score during warmup

        current_vol = volumes[-1] if volumes else 0
        avg_vol_recent = np.mean(volumes[-min(5, len(volumes)):])
        avg_vol_long = np.mean(volumes[-min(20, len(volumes)):]) if len(volumes) >= 20 else avg_vol_recent

        if avg_vol_long <= 0:
            return 10.0

        vol_ratio = current_vol / avg_vol_long
        increasing_vol = avg_vol_recent > avg_vol_long

        score = 10.0

        if vol_ratio > 2.0:
            score += 8.0
        elif vol_ratio > 1.5:
            score += 5.0
        elif vol_ratio > 1.0:
            score += 2.0
        else:
            score -= 3.0

        if increasing_vol:
            score += 2.0

        return max(0.0, min(score, 20.0))

    def _calc_volatility_score(self, direction: str, market_data: Dict) -> float:
        """Score for volatility regime fit (0-15)."""
        regime = market_data.get('volatility_regime', 'normal')

        # Normal volatility is best, high vol favors shorts, low vol is acceptable
        scores = {
            'low': 12.0,
            'normal': 15.0,
            'high': 13.0 if direction == 'short' else 8.0,
        }

        return scores.get(regime, 10.0)

    def _calc_sr_score(
        self, price: float, direction: str,
        support_levels: List[float], resistance_levels: List[float]
    ) -> float:
        """Score for support/resistance proximity (0-20)."""
        if price <= 0:
            return 10.0

        score = 10.0

        if direction == "long":
            # Near support is good for longs
            for support in support_levels:
                if support <= 0:
                    continue
                distance_pct = abs(price - support) / support * 100
                if distance_pct < 1.0:
                    score += 10.0
                    break
                elif distance_pct < 3.0:
                    score += 5.0
                    break

            # Near resistance is bad for longs
            for resistance in resistance_levels:
                if resistance <= 0:
                    continue
                distance_pct = abs(price - resistance) / resistance * 100
                if distance_pct < 2.0:
                    score -= 5.0
                    break

        elif direction == "short":
            # Near resistance is good for shorts
            for resistance in resistance_levels:
                if resistance <= 0:
                    continue
                distance_pct = abs(price - resistance) / resistance * 100
                if distance_pct < 1.0:
                    score += 10.0
                    break
                elif distance_pct < 3.0:
                    score += 5.0
                    break

            # Near support is bad for shorts
            for support in support_levels:
                if support <= 0:
                    continue
                distance_pct = abs(price - support) / support * 100
                if distance_pct < 2.0:
                    score -= 5.0
                    break

        return max(0.0, min(score, 20.0))

    def _calc_timing_score(self, direction: str, market_data: Dict) -> float:
        """Score for entry timing quality (0-15)."""
        prices = market_data.get('prices', [])

        if len(prices) < 5:
            return 10.0  # Neutral during warmup

        recent_return = (prices[-1] - prices[-5]) / prices[-5] * 100 if prices[-5] > 0 else 0

        score = 10.0

        if direction == "long":
            # Don't chase extended rallies
            if recent_return > 5.0:
                score -= 5.0
            elif recent_return > 3.0:
                score -= 2.0
            # Buying pullbacks is good
            elif recent_return < -3.0:
                score += 3.0
            elif recent_return < -5.0:
                score += 5.0

        elif direction == "short":
            # Don't chase extended selloffs
            if recent_return < -5.0:
                score -= 5.0
            elif recent_return < -3.0:
                score -= 2.0
            # Shorting bounces is good
            elif recent_return > 3.0:
                score += 3.0
            elif recent_return > 5.0:
                score += 5.0

        return max(0.0, min(score, 15.0))

    def _calc_confluence_score(self, direction: str, other_signals: List) -> float:
        """Score for confluence with other signals (0-10)."""
        if not other_signals:
            return 5.0  # Neutral if no other signals

        same_direction_count = 0
        opposite_count = 0

        for s in other_signals:
            sig_direction = getattr(s, 'direction', None)
            if sig_direction is not None:
                sig_direction = sig_direction.value if hasattr(sig_direction, 'value') else str(sig_direction)
            else:
                continue

            confidence = getattr(s, 'confidence', 0.5)
            if confidence < 0.5:
                continue  # Ignore low confidence signals

            if sig_direction == direction:
                same_direction_count += 1
            elif sig_direction not in [direction, "flat"]:
                opposite_count += 1

        score = 5.0

        if same_direction_count >= 2:
            score += 4.0
        elif same_direction_count >= 1:
            score += 2.0

        if opposite_count >= 1:
            score -= 3.0

        return max(0.0, min(score, 10.0))

    def _calc_confidence_boost(self, total_score: float, tier: SignalQualityTier) -> float:
        """Calculate confidence boost based on quality tier."""
        boosts = {
            SignalQualityTier.EXCELLENT: 0.15,
            SignalQualityTier.GOOD: 0.08,
            SignalQualityTier.ACCEPTABLE: 0.0,
            SignalQualityTier.POOR: -0.10,
            SignalQualityTier.REJECT: -0.25,
        }
        return boosts.get(tier, 0.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get quality scoring statistics."""
        if not self._quality_history:
            return {
                'total_evaluated': self._total_evaluated,
                'total_rejected': self._total_rejected,
                'rejection_rate_pct': 0.0,
            }

        scores = [h['score'] for h in self._quality_history]
        valid_count = sum(1 for h in self._quality_history if h['is_valid'])

        return {
            'total_evaluated': self._total_evaluated,
            'total_rejected': self._total_rejected,
            'rejection_rate_pct': (len(self._quality_history) - valid_count) / len(self._quality_history) * 100,
            'avg_score': float(np.mean(scores)),
            'min_score': float(min(scores)),
            'max_score': float(max(scores)),
            'tier_distribution': self._get_tier_distribution(),
        }

    def _get_tier_distribution(self) -> Dict[str, int]:
        """Get distribution of quality tiers."""
        distribution = {tier.value: 0 for tier in SignalQualityTier}
        for h in self._quality_history:
            tier = h.get('tier', 'unknown')
            if tier in distribution:
                distribution[tier] += 1
        return distribution


def create_signal_quality_scorer(
    config: Optional[Dict[str, Any]] = None
) -> SignalQualityScorer:
    """Factory function to create SignalQualityScorer from config."""
    config = config or {}
    return SignalQualityScorer(
        min_total_score=config.get('min_total_score', 50.0),
        min_volume_score=config.get('min_volume_score', 5.0),
        min_trend_score=config.get('min_trend_score', 5.0),
        enable_logging=config.get('enable_logging', True),
    )

"""
Regime Detection Module
=======================

Market regime detection for strategy switching (Issue #Q9).

Features:
- Volatility regime detection (low/normal/high/crisis)
- Trend regime identification (trending/mean-reverting/ranging)
- Correlation regime shifts
- Hidden Markov Model-based regime estimation
- VIX-based regime classification
- Multi-factor regime scoring
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""
    LOW = "low"  # VIX < 15 or realized vol < 10%
    NORMAL = "normal"  # VIX 15-20 or realized vol 10-20%
    HIGH = "high"  # VIX 20-30 or realized vol 20-30%
    CRISIS = "crisis"  # VIX > 30 or realized vol > 30%


class TrendRegime(str, Enum):
    """Trend regime classification."""
    STRONG_UPTREND = "strong_uptrend"  # ADX > 25, +DI > -DI
    WEAK_UPTREND = "weak_uptrend"  # ADX 15-25, +DI > -DI
    RANGING = "ranging"  # ADX < 15
    WEAK_DOWNTREND = "weak_downtrend"  # ADX 15-25, -DI > +DI
    STRONG_DOWNTREND = "strong_downtrend"  # ADX > 25, -DI > +DI


class MarketRegime(str, Enum):
    """Overall market regime."""
    RISK_ON = "risk_on"  # Low vol, uptrending, low correlation
    RISK_OFF = "risk_off"  # High vol, downtrending, high correlation
    TRANSITIONAL = "transitional"  # Mixed signals
    CRISIS = "crisis"  # Extreme vol, flight to safety


class CorrelationRegime(str, Enum):
    """Correlation regime classification."""
    DISPERSED = "dispersed"  # Low average correlation
    NORMAL = "normal"  # Normal correlation levels
    ELEVATED = "elevated"  # High correlation
    CRISIS = "crisis"  # All correlations approach 1


@dataclass
class RegimeState:
    """Current regime state."""
    timestamp: datetime
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    correlation_regime: CorrelationRegime
    market_regime: MarketRegime
    confidence: float  # 0-1
    metrics: dict = field(default_factory=dict)
    regime_duration_days: int = 0
    previous_regime: MarketRegime | None = None

    def is_favorable_for_momentum(self) -> bool:
        """Check if regime favors momentum strategies."""
        return (
            self.trend_regime in (TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND)
            and self.volatility_regime in (VolatilityRegime.LOW, VolatilityRegime.NORMAL)
        )

    def is_favorable_for_mean_reversion(self) -> bool:
        """Check if regime favors mean reversion strategies."""
        return (
            self.trend_regime == TrendRegime.RANGING
            and self.volatility_regime in (VolatilityRegime.NORMAL, VolatilityRegime.HIGH)
        )

    def is_risk_off(self) -> bool:
        """Check if market is in risk-off mode."""
        return self.market_regime in (MarketRegime.RISK_OFF, MarketRegime.CRISIS)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'volatility_regime': self.volatility_regime.value,
            'trend_regime': self.trend_regime.value,
            'correlation_regime': self.correlation_regime.value,
            'market_regime': self.market_regime.value,
            'confidence': self.confidence,
            'regime_duration_days': self.regime_duration_days,
            'metrics': self.metrics,
        }


@dataclass
class RegimeTransition:
    """Record of regime transition."""
    timestamp: datetime
    from_regime: MarketRegime
    to_regime: MarketRegime
    trigger: str
    metrics_snapshot: dict


class RegimeDetector:
    """
    Multi-factor regime detection engine.

    Combines volatility, trend, and correlation signals
    to identify the current market regime.
    """

    def __init__(
        self,
        volatility_lookback: int = 20,
        trend_lookback: int = 50,
        correlation_lookback: int = 60,
        regime_persistence: int = 5,  # Days before regime change confirmed
    ):
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.correlation_lookback = correlation_lookback
        self.regime_persistence = regime_persistence

        # Price history
        self._prices: dict[str, deque] = {}
        self._returns: dict[str, deque] = {}

        # State
        self._current_regime: RegimeState | None = None
        self._pending_regime: RegimeState | None = None
        self._pending_days: int = 0
        self._transitions: list[RegimeTransition] = []

        # VIX data
        self._vix_history: deque = deque(maxlen=252)

        # Thresholds
        self.vol_thresholds = {
            VolatilityRegime.LOW: 0.10,  # < 10%
            VolatilityRegime.NORMAL: 0.20,  # 10-20%
            VolatilityRegime.HIGH: 0.30,  # 20-30%
            VolatilityRegime.CRISIS: float('inf'),  # > 30%
        }

        self.correlation_thresholds = {
            CorrelationRegime.DISPERSED: 0.30,  # avg corr < 0.30
            CorrelationRegime.NORMAL: 0.50,  # 0.30-0.50
            CorrelationRegime.ELEVATED: 0.70,  # 0.50-0.70
            CorrelationRegime.CRISIS: float('inf'),  # > 0.70
        }

    def update_price(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Update price history for a symbol."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=max(self.trend_lookback, self.correlation_lookback) + 1)
            self._returns[symbol] = deque(maxlen=self.correlation_lookback)

        if self._prices[symbol]:
            prev_price = self._prices[symbol][-1]
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
                self._returns[symbol].append(ret)

        self._prices[symbol].append(price)

    def update_vix(self, vix_value: float) -> None:
        """Update VIX history."""
        self._vix_history.append(vix_value)

    def detect_regime(self) -> RegimeState:
        """
        Detect current market regime.

        Returns updated RegimeState.
        """
        timestamp = datetime.now(timezone.utc)
        metrics = {}

        # Detect individual regimes
        vol_regime, vol_metrics = self._detect_volatility_regime()
        trend_regime, trend_metrics = self._detect_trend_regime()
        corr_regime, corr_metrics = self._detect_correlation_regime()

        metrics.update(vol_metrics)
        metrics.update(trend_metrics)
        metrics.update(corr_metrics)

        # Combine into market regime
        market_regime = self._combine_regimes(vol_regime, trend_regime, corr_regime)
        confidence = self._calculate_confidence(vol_regime, trend_regime, corr_regime)

        new_regime = RegimeState(
            timestamp=timestamp,
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            correlation_regime=corr_regime,
            market_regime=market_regime,
            confidence=confidence,
            metrics=metrics,
        )

        # Apply regime persistence filter
        confirmed_regime = self._apply_persistence_filter(new_regime)

        return confirmed_regime

    def _detect_volatility_regime(self) -> tuple[VolatilityRegime, dict]:
        """Detect volatility regime."""
        metrics = {}

        # Use VIX if available
        if self._vix_history:
            current_vix = self._vix_history[-1]
            avg_vix = statistics.mean(self._vix_history)
            metrics['current_vix'] = current_vix
            metrics['avg_vix'] = avg_vix

            if current_vix < 15:
                vol_regime = VolatilityRegime.LOW
            elif current_vix < 20:
                vol_regime = VolatilityRegime.NORMAL
            elif current_vix < 30:
                vol_regime = VolatilityRegime.HIGH
            else:
                vol_regime = VolatilityRegime.CRISIS

            return vol_regime, metrics

        # Fall back to realized volatility
        realized_vols = []
        for symbol, returns in self._returns.items():
            if len(returns) >= self.volatility_lookback:
                recent = list(returns)[-self.volatility_lookback:]
                vol = statistics.stdev(recent) * math.sqrt(252)
                realized_vols.append(vol)

        if not realized_vols:
            return VolatilityRegime.NORMAL, metrics

        avg_vol = statistics.mean(realized_vols)
        metrics['realized_volatility'] = avg_vol

        for regime, threshold in self.vol_thresholds.items():
            if avg_vol < threshold:
                return regime, metrics

        return VolatilityRegime.CRISIS, metrics

    def _detect_trend_regime(self) -> tuple[TrendRegime, dict]:
        """Detect trend regime using ADX-like calculation."""
        metrics = {}

        # Need a primary market index
        symbols = list(self._prices.keys())
        if not symbols:
            return TrendRegime.RANGING, metrics

        # Use first symbol as proxy (ideally SPY or similar)
        primary = symbols[0]
        prices = list(self._prices[primary])

        if len(prices) < self.trend_lookback:
            return TrendRegime.RANGING, metrics

        # Calculate simplified trend strength and direction
        recent_prices = prices[-self.trend_lookback:]

        # Linear regression slope
        x_mean = (len(recent_prices) - 1) / 2
        y_mean = statistics.mean(recent_prices)

        numerator = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(recent_prices))
        denominator = sum((i - x_mean) ** 2 for i in range(len(recent_prices)))

        slope = numerator / denominator if denominator > 0 else 0

        # Normalize slope by price level
        normalized_slope = (slope / y_mean) * 100  # Percentage per period

        metrics['trend_slope'] = normalized_slope

        # Calculate ADX-like metric (simplified)
        returns = list(self._returns.get(primary, []))
        if len(returns) < 14:
            adx = 0
        else:
            # Directional movement
            recent_returns = returns[-14:]
            positive_moves = [max(r, 0) for r in recent_returns]
            negative_moves = [abs(min(r, 0)) for r in recent_returns]

            avg_positive = statistics.mean(positive_moves) if positive_moves else 0
            avg_negative = statistics.mean(negative_moves) if negative_moves else 0

            # Simplified ADX
            total_move = avg_positive + avg_negative
            if total_move > 0:
                di_diff = abs(avg_positive - avg_negative)
                adx = (di_diff / total_move) * 100
            else:
                adx = 0

        metrics['adx_approx'] = adx

        # Classify
        is_upward = normalized_slope > 0

        if adx > 25:
            trend_regime = TrendRegime.STRONG_UPTREND if is_upward else TrendRegime.STRONG_DOWNTREND
        elif adx > 15:
            trend_regime = TrendRegime.WEAK_UPTREND if is_upward else TrendRegime.WEAK_DOWNTREND
        else:
            trend_regime = TrendRegime.RANGING

        return trend_regime, metrics

    def _detect_correlation_regime(self) -> tuple[CorrelationRegime, dict]:
        """Detect correlation regime."""
        metrics = {}

        symbols = list(self._returns.keys())
        if len(symbols) < 2:
            return CorrelationRegime.NORMAL, metrics

        # Calculate pairwise correlations
        correlations = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                ret1 = list(self._returns[sym1])
                ret2 = list(self._returns[sym2])

                # Align lengths
                min_len = min(len(ret1), len(ret2), self.correlation_lookback)
                if min_len < 20:
                    continue

                ret1 = ret1[-min_len:]
                ret2 = ret2[-min_len:]

                corr = self._correlation(ret1, ret2)
                if corr is not None:
                    correlations.append(corr)

        if not correlations:
            return CorrelationRegime.NORMAL, metrics

        avg_corr = statistics.mean(correlations)
        max_corr = max(correlations)
        metrics['avg_correlation'] = avg_corr
        metrics['max_correlation'] = max_corr

        # Classify
        for regime, threshold in self.correlation_thresholds.items():
            if avg_corr < threshold:
                return regime, metrics

        return CorrelationRegime.CRISIS, metrics

    def _correlation(self, x: list[float], y: list[float]) -> float | None:
        """Calculate Pearson correlation."""
        if len(x) != len(y) or len(x) < 2:
            return None

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / (n - 1))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / (n - 1))

        if std_x == 0 or std_y == 0:
            return None

        return cov / (std_x * std_y)

    def _combine_regimes(
        self,
        vol_regime: VolatilityRegime,
        trend_regime: TrendRegime,
        corr_regime: CorrelationRegime,
    ) -> MarketRegime:
        """Combine individual regimes into overall market regime."""
        # Crisis check
        if vol_regime == VolatilityRegime.CRISIS or corr_regime == CorrelationRegime.CRISIS:
            return MarketRegime.CRISIS

        # Risk-off indicators
        risk_off_signals = 0
        if vol_regime == VolatilityRegime.HIGH:
            risk_off_signals += 1
        if trend_regime in (TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND):
            risk_off_signals += 1
        if corr_regime == CorrelationRegime.ELEVATED:
            risk_off_signals += 1

        # Risk-on indicators
        risk_on_signals = 0
        if vol_regime == VolatilityRegime.LOW:
            risk_on_signals += 1
        if trend_regime in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND):
            risk_on_signals += 1
        if corr_regime == CorrelationRegime.DISPERSED:
            risk_on_signals += 1

        # Determine overall regime
        if risk_off_signals >= 2:
            return MarketRegime.RISK_OFF
        elif risk_on_signals >= 2:
            return MarketRegime.RISK_ON
        else:
            return MarketRegime.TRANSITIONAL

    def _calculate_confidence(
        self,
        vol_regime: VolatilityRegime,
        trend_regime: TrendRegime,
        corr_regime: CorrelationRegime,
    ) -> float:
        """Calculate confidence in regime classification."""
        # Base confidence on signal agreement
        signals = []

        # Vol signal strength
        if vol_regime in (VolatilityRegime.LOW, VolatilityRegime.CRISIS):
            signals.append(1.0)  # Clear signal
        elif vol_regime == VolatilityRegime.NORMAL:
            signals.append(0.5)  # Neutral
        else:
            signals.append(0.8)  # Elevated but not extreme

        # Trend signal strength
        if trend_regime in (TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND):
            signals.append(1.0)
        elif trend_regime == TrendRegime.RANGING:
            signals.append(0.5)
        else:
            signals.append(0.7)

        # Correlation signal strength
        if corr_regime in (CorrelationRegime.DISPERSED, CorrelationRegime.CRISIS):
            signals.append(1.0)
        elif corr_regime == CorrelationRegime.NORMAL:
            signals.append(0.5)
        else:
            signals.append(0.8)

        return statistics.mean(signals)

    def _apply_persistence_filter(self, new_regime: RegimeState) -> RegimeState:
        """Apply persistence filter to avoid regime flip-flopping."""
        if self._current_regime is None:
            self._current_regime = new_regime
            return new_regime

        # Check if regime is changing
        if new_regime.market_regime != self._current_regime.market_regime:
            if self._pending_regime is None or self._pending_regime.market_regime != new_regime.market_regime:
                # Start counting
                self._pending_regime = new_regime
                self._pending_days = 1
            else:
                # Continue counting
                self._pending_days += 1

            # Check if persistence threshold met
            if self._pending_days >= self.regime_persistence:
                # Confirm regime change
                transition = RegimeTransition(
                    timestamp=new_regime.timestamp,
                    from_regime=self._current_regime.market_regime,
                    to_regime=new_regime.market_regime,
                    trigger=f"Persisted for {self._pending_days} days",
                    metrics_snapshot=new_regime.metrics,
                )
                self._transitions.append(transition)

                new_regime.previous_regime = self._current_regime.market_regime
                new_regime.regime_duration_days = 0
                self._current_regime = new_regime
                self._pending_regime = None
                self._pending_days = 0

                logger.info(
                    f"Regime change confirmed: {transition.from_regime.value} -> "
                    f"{transition.to_regime.value}"
                )

                return new_regime

            # Return current regime (not yet changed)
            self._current_regime.regime_duration_days += 1
            return self._current_regime

        # No change
        self._pending_regime = None
        self._pending_days = 0
        self._current_regime.regime_duration_days += 1
        self._current_regime.metrics = new_regime.metrics
        self._current_regime.timestamp = new_regime.timestamp

        return self._current_regime

    def get_transitions(self, limit: int = 20) -> list[RegimeTransition]:
        """Get recent regime transitions."""
        return self._transitions[-limit:]

    def get_strategy_weights(self, strategies: list[str]) -> dict[str, float]:
        """
        Get recommended strategy weights based on current regime.

        Args:
            strategies: List of strategy names

        Returns:
            Dictionary of strategy -> weight (0-1)
        """
        if not self._current_regime:
            return {s: 1.0 / len(strategies) for s in strategies}

        weights = {}
        regime = self._current_regime

        for strategy in strategies:
            strategy_lower = strategy.lower()

            # Momentum strategies
            if 'momentum' in strategy_lower or 'trend' in strategy_lower:
                if regime.is_favorable_for_momentum():
                    weights[strategy] = 1.0
                elif regime.market_regime == MarketRegime.CRISIS:
                    weights[strategy] = 0.2
                else:
                    weights[strategy] = 0.5

            # Mean reversion strategies
            elif 'mean_reversion' in strategy_lower or 'stat_arb' in strategy_lower:
                if regime.is_favorable_for_mean_reversion():
                    weights[strategy] = 1.0
                elif regime.market_regime == MarketRegime.CRISIS:
                    weights[strategy] = 0.3
                else:
                    weights[strategy] = 0.6

            # Volatility strategies
            elif 'vol' in strategy_lower or 'option' in strategy_lower:
                if regime.volatility_regime == VolatilityRegime.HIGH:
                    weights[strategy] = 1.0
                elif regime.volatility_regime == VolatilityRegime.CRISIS:
                    weights[strategy] = 0.8
                else:
                    weights[strategy] = 0.5

            # Market making
            elif 'market_making' in strategy_lower:
                if regime.volatility_regime == VolatilityRegime.LOW:
                    weights[strategy] = 1.0
                elif regime.volatility_regime == VolatilityRegime.CRISIS:
                    weights[strategy] = 0.1
                else:
                    weights[strategy] = 0.6

            # Default
            else:
                weights[strategy] = 0.5 if not regime.is_risk_off() else 0.3

        # Normalize if needed
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


@dataclass
class HMMRegimeModel:
    """
    Hidden Markov Model for regime detection.

    Simplified implementation without external dependencies.
    """

    n_states: int = 3  # Low vol, Normal, High vol
    transition_matrix: list[list[float]] = field(default_factory=list)
    emission_means: list[float] = field(default_factory=list)
    emission_stds: list[float] = field(default_factory=list)
    state_probs: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.transition_matrix:
            # Default transition matrix (sticky regimes)
            self.transition_matrix = [
                [0.95, 0.04, 0.01],  # Low vol tends to stay low
                [0.10, 0.80, 0.10],  # Normal can go either way
                [0.02, 0.08, 0.90],  # High vol tends to stay high
            ]

        if not self.emission_means:
            # Mean returns by regime
            self.emission_means = [0.0005, 0.0, -0.001]  # Daily returns

        if not self.emission_stds:
            # Vol by regime
            self.emission_stds = [0.008, 0.015, 0.030]  # Daily vol

        if not self.state_probs:
            # Initial state probabilities
            self.state_probs = [0.33, 0.34, 0.33]

    def forward_algorithm(self, observations: list[float]) -> list[list[float]]:
        """
        Run forward algorithm to get state probabilities.

        Returns matrix of state probabilities for each time step.
        """
        T = len(observations)
        alpha = [[0.0] * self.n_states for _ in range(T)]

        # Initialize
        for i in range(self.n_states):
            alpha[0][i] = self.state_probs[i] * self._emission_prob(observations[0], i)

        # Normalize
        total = sum(alpha[0])
        if total > 0:
            alpha[0] = [a / total for a in alpha[0]]

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                prob_sum = sum(
                    alpha[t-1][i] * self.transition_matrix[i][j]
                    for i in range(self.n_states)
                )
                alpha[t][j] = prob_sum * self._emission_prob(observations[t], j)

            # Normalize
            total = sum(alpha[t])
            if total > 0:
                alpha[t] = [a / total for a in alpha[t]]

        return alpha

    def _emission_prob(self, observation: float, state: int) -> float:
        """Calculate emission probability (Gaussian)."""
        mean = self.emission_means[state]
        std = self.emission_stds[state]

        if std == 0:
            return 1.0 if observation == mean else 0.0

        z = (observation - mean) / std
        return math.exp(-0.5 * z * z) / (std * math.sqrt(2 * math.pi))

    def predict_state(self, observations: list[float]) -> int:
        """Predict most likely current state."""
        if not observations:
            return 1  # Default to normal

        alpha = self.forward_algorithm(observations)
        last_probs = alpha[-1]

        return last_probs.index(max(last_probs))

    def get_state_probabilities(self, observations: list[float]) -> list[float]:
        """Get current state probabilities."""
        if not observations:
            return self.state_probs.copy()

        alpha = self.forward_algorithm(observations)
        return alpha[-1]


class RegimeAwareStrategyAllocator:
    """
    Allocates capital to strategies based on regime.

    Provides a unified interface for regime-based allocation.
    """

    def __init__(
        self,
        detector: RegimeDetector,
        strategy_configs: dict[str, dict] | None = None,
    ):
        self.detector = detector
        self.strategy_configs = strategy_configs or {}

        # Default regime preferences per strategy type
        self.default_preferences = {
            'momentum': {
                MarketRegime.RISK_ON: 1.0,
                MarketRegime.TRANSITIONAL: 0.6,
                MarketRegime.RISK_OFF: 0.3,
                MarketRegime.CRISIS: 0.1,
            },
            'mean_reversion': {
                MarketRegime.RISK_ON: 0.7,
                MarketRegime.TRANSITIONAL: 0.8,
                MarketRegime.RISK_OFF: 0.5,
                MarketRegime.CRISIS: 0.2,
            },
            'volatility': {
                MarketRegime.RISK_ON: 0.4,
                MarketRegime.TRANSITIONAL: 0.6,
                MarketRegime.RISK_OFF: 0.8,
                MarketRegime.CRISIS: 1.0,
            },
        }

    def get_allocation(
        self,
        strategies: dict[str, str],  # strategy_name -> strategy_type
        total_capital: float,
        base_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Get regime-adjusted capital allocation.

        Args:
            strategies: Map of strategy name to type
            total_capital: Total capital to allocate
            base_weights: Optional base weights (before regime adjustment)

        Returns:
            Dictionary of strategy -> allocated capital
        """
        regime_state = self.detector.detect_regime()
        current_regime = regime_state.market_regime

        # Start with base weights or equal
        if base_weights:
            weights = base_weights.copy()
        else:
            weights = {name: 1.0 / len(strategies) for name in strategies}

        # Apply regime adjustment
        for strategy_name, strategy_type in strategies.items():
            # Get preferences for this strategy type
            if strategy_name in self.strategy_configs:
                prefs = self.strategy_configs[strategy_name].get('regime_preferences', {})
            elif strategy_type in self.default_preferences:
                prefs = self.default_preferences[strategy_type]
            else:
                prefs = {}

            # Apply regime multiplier
            multiplier = prefs.get(current_regime, 0.5)
            weights[strategy_name] *= multiplier

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Convert to capital amounts
        allocation = {k: v * total_capital for k, v in weights.items()}

        logger.info(
            f"Regime-based allocation (regime={current_regime.value}, "
            f"confidence={regime_state.confidence:.2f}): {allocation}"
        )

        return allocation

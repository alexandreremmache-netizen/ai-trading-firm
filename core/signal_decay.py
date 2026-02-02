"""
Signal Decay/Half-Life Module
=============================

Models signal strength decay over time (Issue #Q18).

Features:
- Exponential decay modeling
- Half-life calculation and calibration
- Signal freshness scoring
- Multi-timeframe decay
- Historical decay analysis
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


class DecayModel(str, Enum):
    """Signal decay model types."""
    EXPONENTIAL = "exponential"  # Standard exponential decay
    LINEAR = "linear"  # Linear decay to zero
    STEP = "step"  # Immediate decay after threshold
    POWER = "power"  # Power law decay
    CUSTOM = "custom"  # User-defined


@dataclass
class SignalDecayConfig:
    """Configuration for signal decay."""
    model: DecayModel = DecayModel.EXPONENTIAL
    half_life_minutes: float = 60.0  # Signal loses half strength every hour
    min_strength: float = 0.1  # Minimum strength before signal is discarded
    max_age_minutes: float = 480.0  # 8 hours max age
    refresh_on_confirmation: bool = True  # Reset decay on confirming signal


@dataclass
class DecayedSignal:
    """Signal with decay tracking."""
    signal_id: str
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    original_strength: float  # Initial strength (0-1)
    current_strength: float  # After decay
    created_at: datetime
    last_updated: datetime
    source_strategy: str
    decay_config: SignalDecayConfig
    metadata: dict = field(default_factory=dict)

    @property
    def age_minutes(self) -> float:
        """Get signal age in minutes."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds() / 60

    @property
    def freshness_score(self) -> float:
        """Get freshness score (0-1, higher is fresher)."""
        return self.current_strength / self.original_strength if self.original_strength > 0 else 0

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        return (
            self.current_strength < self.decay_config.min_strength or
            self.age_minutes > self.decay_config.max_age_minutes
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'original_strength': self.original_strength,
            'current_strength': self.current_strength,
            'freshness_score': self.freshness_score,
            'age_minutes': self.age_minutes,
            'is_expired': self.is_expired,
            'source_strategy': self.source_strategy,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }


class SignalDecayCalculator:
    """
    Calculates signal decay based on various models (#Q18).

    Supports multiple decay models for different signal types.
    """

    def __init__(self, default_config: SignalDecayConfig | None = None):
        self.default_config = default_config or SignalDecayConfig()

    def calculate_decay(
        self,
        original_strength: float,
        age_minutes: float,
        config: SignalDecayConfig | None = None,
    ) -> float:
        """
        Calculate decayed signal strength.

        Args:
            original_strength: Initial signal strength (0-1)
            age_minutes: Signal age in minutes
            config: Decay configuration

        Returns:
            Decayed strength (0-1)
        """
        cfg = config or self.default_config

        if age_minutes <= 0:
            return original_strength

        if age_minutes > cfg.max_age_minutes:
            return 0.0

        if cfg.model == DecayModel.EXPONENTIAL:
            return self._exponential_decay(original_strength, age_minutes, cfg.half_life_minutes)

        elif cfg.model == DecayModel.LINEAR:
            return self._linear_decay(original_strength, age_minutes, cfg.max_age_minutes)

        elif cfg.model == DecayModel.STEP:
            return self._step_decay(original_strength, age_minutes, cfg.half_life_minutes)

        elif cfg.model == DecayModel.POWER:
            return self._power_decay(original_strength, age_minutes, cfg.half_life_minutes)

        else:
            return self._exponential_decay(original_strength, age_minutes, cfg.half_life_minutes)

    def _exponential_decay(
        self,
        strength: float,
        age_minutes: float,
        half_life_minutes: float,
    ) -> float:
        """
        Exponential decay: S(t) = S0 * exp(-λt)

        Where λ = ln(2) / half_life
        """
        if half_life_minutes <= 0:
            return 0.0

        decay_rate = math.log(2) / half_life_minutes
        return strength * math.exp(-decay_rate * age_minutes)

    def _linear_decay(
        self,
        strength: float,
        age_minutes: float,
        max_age_minutes: float,
    ) -> float:
        """Linear decay from strength to 0 over max_age."""
        if max_age_minutes <= 0:
            return 0.0

        remaining_fraction = max(0, 1 - (age_minutes / max_age_minutes))
        return strength * remaining_fraction

    def _step_decay(
        self,
        strength: float,
        age_minutes: float,
        threshold_minutes: float,
    ) -> float:
        """Step function: full strength until threshold, then zero."""
        return strength if age_minutes < threshold_minutes else 0.0

    def _power_decay(
        self,
        strength: float,
        age_minutes: float,
        half_life_minutes: float,
        power: float = 1.5,
    ) -> float:
        """Power law decay: S(t) = S0 / (1 + (t/τ)^α)"""
        if half_life_minutes <= 0:
            return 0.0

        return strength / (1 + (age_minutes / half_life_minutes) ** power)

    def estimate_half_life(
        self,
        signal_history: list[tuple[float, float]],  # [(age_minutes, strength), ...]
    ) -> float:
        """
        Estimate half-life from historical signal performance.

        Uses regression to fit exponential decay curve.

        Args:
            signal_history: List of (age, observed_strength) pairs

        Returns:
            Estimated half-life in minutes
        """
        if len(signal_history) < 3:
            return self.default_config.half_life_minutes

        # Convert to log space for linear regression
        # ln(S) = ln(S0) - λt
        valid_points = [(t, math.log(s)) for t, s in signal_history if s > 0]

        if len(valid_points) < 3:
            return self.default_config.half_life_minutes

        # Simple linear regression
        n = len(valid_points)
        sum_t = sum(p[0] for p in valid_points)
        sum_lns = sum(p[1] for p in valid_points)
        sum_t_lns = sum(p[0] * p[1] for p in valid_points)
        sum_t2 = sum(p[0] ** 2 for p in valid_points)

        # Slope = -λ
        denominator = n * sum_t2 - sum_t ** 2
        if denominator == 0:
            return self.default_config.half_life_minutes

        slope = (n * sum_t_lns - sum_t * sum_lns) / denominator
        decay_rate = -slope

        if decay_rate <= 0:
            return self.default_config.half_life_minutes

        # half_life = ln(2) / λ
        half_life = math.log(2) / decay_rate

        # Clamp to reasonable range
        return max(5.0, min(1440.0, half_life))  # 5 min to 24 hours


class SignalDecayManager:
    """
    Manages signal decay for multiple signals (#Q18).

    Tracks active signals and applies decay over time.
    """

    def __init__(
        self,
        calculator: SignalDecayCalculator | None = None,
        cleanup_interval_minutes: float = 5.0,
    ):
        self.calculator = calculator or SignalDecayCalculator()
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)

        # Active signals by symbol
        self._signals: dict[str, list[DecayedSignal]] = {}
        self._signal_counter = 0
        self._last_cleanup = datetime.now(timezone.utc)

        # Historical data for half-life calibration
        self._signal_performance: dict[str, deque] = {}  # strategy -> performance history

    def add_signal(
        self,
        symbol: str,
        direction: str,
        strength: float,
        source_strategy: str,
        config: SignalDecayConfig | None = None,
        metadata: dict | None = None,
    ) -> DecayedSignal:
        """
        Add a new signal to track.

        Args:
            symbol: Trading symbol
            direction: 'LONG', 'SHORT', or 'NEUTRAL'
            strength: Signal strength (0-1)
            source_strategy: Strategy that generated the signal
            config: Optional decay configuration
            metadata: Additional signal metadata

        Returns:
            The created DecayedSignal
        """
        now = datetime.now(timezone.utc)
        self._signal_counter += 1

        signal = DecayedSignal(
            signal_id=f"SIG_{now.strftime('%Y%m%d_%H%M%S')}_{self._signal_counter}",
            symbol=symbol,
            direction=direction,
            original_strength=min(1.0, max(0.0, strength)),
            current_strength=min(1.0, max(0.0, strength)),
            created_at=now,
            last_updated=now,
            source_strategy=source_strategy,
            decay_config=config or self.calculator.default_config,
            metadata=metadata or {},
        )

        if symbol not in self._signals:
            self._signals[symbol] = []

        self._signals[symbol].append(signal)

        logger.debug(
            f"Added signal {signal.signal_id}: {direction} {symbol} "
            f"strength={strength:.2f} from {source_strategy}"
        )

        return signal

    def update_signal(
        self,
        signal_id: str,
        new_strength: float | None = None,
        refresh: bool = False,
    ) -> DecayedSignal | None:
        """
        Update an existing signal.

        Args:
            signal_id: Signal to update
            new_strength: Optional new strength value
            refresh: If True, reset decay timer

        Returns:
            Updated signal or None if not found
        """
        for signals in self._signals.values():
            for signal in signals:
                if signal.signal_id == signal_id:
                    if new_strength is not None:
                        signal.current_strength = min(1.0, max(0.0, new_strength))

                    if refresh and signal.decay_config.refresh_on_confirmation:
                        signal.original_strength = signal.current_strength
                        signal.created_at = datetime.now(timezone.utc)

                    signal.last_updated = datetime.now(timezone.utc)
                    return signal

        return None

    def get_signal(self, signal_id: str) -> DecayedSignal | None:
        """Get a specific signal by ID."""
        for signals in self._signals.values():
            for signal in signals:
                if signal.signal_id == signal_id:
                    return signal
        return None

    def get_signals_for_symbol(
        self,
        symbol: str,
        include_expired: bool = False,
    ) -> list[DecayedSignal]:
        """Get all signals for a symbol."""
        self._apply_decay()

        signals = self._signals.get(symbol, [])

        if not include_expired:
            signals = [s for s in signals if not s.is_expired]

        return signals

    def get_aggregate_signal(self, symbol: str) -> dict:
        """
        Get aggregated signal for a symbol.

        Combines multiple signals with decay-weighted averaging.
        """
        signals = self.get_signals_for_symbol(symbol, include_expired=False)

        if not signals:
            return {
                'symbol': symbol,
                'direction': 'NEUTRAL',
                'strength': 0.0,
                'confidence': 0.0,
                'signal_count': 0,
            }

        # Separate by direction
        long_signals = [s for s in signals if s.direction == 'LONG']
        short_signals = [s for s in signals if s.direction == 'SHORT']

        long_strength = sum(s.current_strength for s in long_signals)
        short_strength = sum(s.current_strength for s in short_signals)

        net_strength = long_strength - short_strength

        if net_strength > 0:
            direction = 'LONG'
            strength = min(1.0, net_strength / len(signals))
        elif net_strength < 0:
            direction = 'SHORT'
            strength = min(1.0, abs(net_strength) / len(signals))
        else:
            direction = 'NEUTRAL'
            strength = 0.0

        # Confidence based on agreement
        total_strength = long_strength + short_strength
        confidence = abs(net_strength) / total_strength if total_strength > 0 else 0

        return {
            'symbol': symbol,
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'signal_count': len(signals),
            'long_signals': len(long_signals),
            'short_signals': len(short_signals),
            'avg_freshness': sum(s.freshness_score for s in signals) / len(signals),
        }

    def _apply_decay(self) -> None:
        """Apply decay to all signals."""
        now = datetime.now(timezone.utc)

        for symbol, signals in self._signals.items():
            for signal in signals:
                age_minutes = (now - signal.created_at).total_seconds() / 60
                signal.current_strength = self.calculator.calculate_decay(
                    signal.original_strength,
                    age_minutes,
                    signal.decay_config,
                )
                signal.last_updated = now

    def cleanup_expired(self) -> int:
        """Remove expired signals."""
        now = datetime.now(timezone.utc)

        if now - self._last_cleanup < self.cleanup_interval:
            return 0

        self._apply_decay()

        removed = 0
        for symbol in list(self._signals.keys()):
            before = len(self._signals[symbol])
            self._signals[symbol] = [s for s in self._signals[symbol] if not s.is_expired]
            removed += before - len(self._signals[symbol])

            if not self._signals[symbol]:
                del self._signals[symbol]

        self._last_cleanup = now

        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired signals")

        return removed

    def record_signal_performance(
        self,
        strategy: str,
        age_at_action: float,
        effectiveness: float,
    ) -> None:
        """
        Record signal performance for half-life calibration.

        Args:
            strategy: Strategy that generated the signal
            age_at_action: Age of signal when action was taken (minutes)
            effectiveness: How effective the signal was (0-1)
        """
        if strategy not in self._signal_performance:
            self._signal_performance[strategy] = deque(maxlen=100)

        self._signal_performance[strategy].append((age_at_action, effectiveness))

    def calibrate_half_life(self, strategy: str) -> float:
        """Calibrate half-life for a strategy based on historical performance."""
        history = self._signal_performance.get(strategy, [])

        if len(history) < 10:
            return self.calculator.default_config.half_life_minutes

        return self.calculator.estimate_half_life(list(history))

    def get_statistics(self) -> dict:
        """Get signal manager statistics."""
        self._apply_decay()

        all_signals = []
        for signals in self._signals.values():
            all_signals.extend(signals)

        active = [s for s in all_signals if not s.is_expired]
        expired = [s for s in all_signals if s.is_expired]

        return {
            'total_signals': len(all_signals),
            'active_signals': len(active),
            'expired_signals': len(expired),
            'symbols_tracked': len(self._signals),
            'avg_strength': sum(s.current_strength for s in active) / len(active) if active else 0,
            'avg_age_minutes': sum(s.age_minutes for s in active) / len(active) if active else 0,
            'by_direction': {
                'long': len([s for s in active if s.direction == 'LONG']),
                'short': len([s for s in active if s.direction == 'SHORT']),
                'neutral': len([s for s in active if s.direction == 'NEUTRAL']),
            },
        }

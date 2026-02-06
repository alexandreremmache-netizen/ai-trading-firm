"""
TTM Squeeze Volatility Strategy (Phase 6.3)
============================================

Volatility breakout strategy based on John Carter's TTM Squeeze indicator.

Key features:
- Bollinger Bands / Keltner Channel squeeze detection
- Momentum oscillator for direction
- Volatility expansion anticipation
- Position sizing based on squeeze duration

Research basis:
- Squeeze: BB inside KC = low volatility compression
- Release: BB outside KC = volatility expansion
- Longer squeeze = more powerful breakout potential
- Momentum direction determines trade direction

MATURITY: ALPHA
---------------
Status: New implementation
- [x] Squeeze detection (BB vs KC)
- [x] Momentum oscillator
- [x] Signal generation
- [ ] Integration with main system
- [ ] Backtesting validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class SqueezeState(Enum):
    """Squeeze state indicator."""
    SQUEEZE_ON = "squeeze_on"      # BB inside KC - compression
    SQUEEZE_OFF = "squeeze_off"    # BB outside KC - expansion
    SQUEEZE_FIRING = "firing"      # Just transitioned from on to off


class MomentumDirection(Enum):
    """Momentum direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SqueezeReading:
    """Current squeeze reading."""
    squeeze_state: SqueezeState
    momentum_value: float
    momentum_direction: MomentumDirection
    bars_in_squeeze: int
    bb_width: float
    kc_width: float
    squeeze_intensity: float  # How tight the squeeze is (0-1)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SqueezeSignal:
    """Trading signal from squeeze analysis."""
    symbol: str
    direction: str  # "LONG", "SHORT", "FLAT"
    signal_type: str  # "squeeze_fire", "momentum_continuation", "exit"
    strength: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    squeeze_bars: int
    momentum_value: float
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TTMSqueezeStrategy:
    """
    TTM Squeeze volatility breakout strategy (Phase 6.3).

    Identifies low volatility compression periods and trades the
    subsequent breakout in the direction of momentum.

    Configuration:
        bb_period: Bollinger Band period (default: 20)
        bb_std: Bollinger Band standard deviation (default: 2.0)
        kc_period: Keltner Channel period (default: 20)
        kc_atr_mult: Keltner Channel ATR multiplier (default: 1.5)
        momentum_period: Momentum lookback (default: 12)
        min_squeeze_bars: Minimum bars in squeeze for signal (default: 6)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize TTM Squeeze strategy."""
        config = config or {}

        # Bollinger Bands settings
        self._bb_period = config.get("bb_period", 20)
        self._bb_std = config.get("bb_std", 2.0)

        # Keltner Channel settings
        self._kc_period = config.get("kc_period", 20)
        self._kc_atr_mult = config.get("kc_atr_mult", 1.5)

        # Momentum settings
        self._momentum_period = config.get("momentum_period", 12)

        # Signal settings
        self._min_squeeze_bars = config.get("min_squeeze_bars", 6)
        self._stop_atr_mult = config.get("stop_atr_mult", 2.0)
        self._take_profit_atr_mult = config.get("take_profit_atr_mult", 3.0)

        # State tracking
        self._squeeze_history: dict[str, list[SqueezeState]] = {}
        self._last_readings: dict[str, SqueezeReading] = {}

        logger.info(
            f"TTMSqueezeStrategy initialized: "
            f"BB({self._bb_period}, {self._bb_std}), "
            f"KC({self._kc_period}, {self._kc_atr_mult})"
        )

    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int | None = None,
        std_mult: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Close prices
            period: SMA period
            std_mult: Standard deviation multiplier

        Returns:
            (upper_band, middle_band, lower_band)
        """
        period = period or self._bb_period
        std_mult = std_mult or self._bb_std

        if len(prices) < period:
            return np.array([]), np.array([]), np.array([])

        # Simple moving average
        middle = np.zeros(len(prices))
        upper = np.zeros(len(prices))
        lower = np.zeros(len(prices))

        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window, ddof=1)

            middle[i] = sma
            upper[i] = sma + std_mult * std
            lower[i] = sma - std_mult * std

        return upper, middle, lower

    def calculate_keltner_channel(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int | None = None,
        atr_mult: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Keltner Channel.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: EMA period
            atr_mult: ATR multiplier

        Returns:
            (upper_channel, middle_channel, lower_channel)
        """
        period = period or self._kc_period
        atr_mult = atr_mult or self._kc_atr_mult

        if len(close) < period:
            return np.array([]), np.array([]), np.array([])

        # EMA for middle line
        middle = self._calculate_ema(close, period)

        # ATR calculation
        atr = self._calculate_atr(high, low, close, period)

        upper = middle + atr_mult * atr
        lower = middle - atr_mult * atr

        return upper, middle, lower

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros(len(prices))
        multiplier = 2.0 / (period + 1)

        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Calculate Average True Range."""
        n = len(close)
        tr = np.zeros(n)

        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        # EMA of True Range
        atr = self._calculate_ema(tr, period)
        return atr

    def calculate_momentum(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int | None = None,
    ) -> np.ndarray:
        """
        Calculate TTM Squeeze momentum oscillator.

        John Carter's original TTM Squeeze momentum formula:
        Momentum = Donchian Midpoint - SMA(close)

        Where:
        - Donchian Midpoint = (Highest High + Lowest Low) / 2 over N periods
        - SMA = Simple Moving Average of close over N periods

        This measures price position relative to its average, weighted by
        the recent trading range. Positive = bullish, Negative = bearish.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period

        Returns:
            Momentum values (positive = bullish, negative = bearish)
        """
        period = period or self._momentum_period

        if len(close) < period:
            return np.zeros(len(close))

        momentum = np.zeros(len(close))

        for i in range(period - 1, len(close)):
            # Donchian channel midpoint (highest high + lowest low) / 2
            window_high = high[i - period + 1:i + 1]
            window_low = low[i - period + 1:i + 1]
            highest_high = np.max(window_high)
            lowest_low = np.min(window_low)
            donchian_midpoint = (highest_high + lowest_low) / 2

            # Simple Moving Average of close
            window_close = close[i - period + 1:i + 1]
            sma = np.mean(window_close)

            # TTM Squeeze momentum = close - (Donchian_midpoint + SMA) / 2
            equilibrium = (donchian_midpoint + sma) / 2
            momentum[i] = close[i] - equilibrium

        # FIX-32: Apply linear regression smoothing per Carter's original
        # Raw momentum is too noisy on 1-min bars. Linreg smooths while
        # preserving the slope (direction) better than a simple MA.
        smoothed = self._linreg_smooth(momentum, period)

        return smoothed

    def _linreg_smooth(self, values: np.ndarray, period: int) -> np.ndarray:
        """
        Apply linear regression smoothing to an array.

        For each point, fit a linear regression over the lookback period
        and use the endpoint value. This smooths noise while preserving trend.
        """
        result = np.zeros(len(values))
        for i in range(len(values)):
            if i < period - 1:
                result[i] = values[i]
                continue
            window = values[i - period + 1:i + 1]
            # Simple linear regression: y = a + b*x
            x = np.arange(period, dtype=float)
            x_mean = np.mean(x)
            y_mean = np.mean(window)
            denom = np.sum((x - x_mean) ** 2)
            if denom > 0:
                b = np.sum((x - x_mean) * (window - y_mean)) / denom
                a = y_mean - b * x_mean
                result[i] = a + b * (period - 1)  # Endpoint of regression
            else:
                result[i] = values[i]
        return result

    def detect_squeeze(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect squeeze condition (BB inside KC).

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            (squeeze_on, bb_width, kc_width)
            - squeeze_on: Boolean array (True = squeeze active)
            - bb_width: Bollinger Band width
            - kc_width: Keltner Channel width
        """
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(close)
        kc_upper, kc_mid, kc_lower = self.calculate_keltner_channel(high, low, close)

        if len(bb_upper) == 0 or len(kc_upper) == 0:
            return np.array([]), np.array([]), np.array([])

        # Squeeze is ON when BB is inside KC
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        # Width calculations
        bb_width = bb_upper - bb_lower
        kc_width = kc_upper - kc_lower

        return squeeze_on, bb_width, kc_width

    def analyze(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> SqueezeReading:
        """
        Analyze current squeeze state.

        Args:
            symbol: Instrument symbol
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            SqueezeReading with current analysis
        """
        squeeze_on, bb_width, kc_width = self.detect_squeeze(high, low, close)
        momentum = self.calculate_momentum(high, low, close)

        if len(squeeze_on) == 0:
            return SqueezeReading(
                squeeze_state=SqueezeState.SQUEEZE_OFF,
                momentum_value=0.0,
                momentum_direction=MomentumDirection.NEUTRAL,
                bars_in_squeeze=0,
                bb_width=0.0,
                kc_width=0.0,
                squeeze_intensity=0.0,
            )

        # Current values
        current_squeeze = squeeze_on[-1]
        current_momentum = momentum[-1]
        current_bb_width = bb_width[-1]
        current_kc_width = kc_width[-1]

        # Track squeeze history
        if symbol not in self._squeeze_history:
            self._squeeze_history[symbol] = []

        history = self._squeeze_history[symbol]

        # Determine squeeze state
        if current_squeeze:
            state = SqueezeState.SQUEEZE_ON
        else:
            # Check if just fired (was on, now off)
            if len(history) > 0 and history[-1] == SqueezeState.SQUEEZE_ON:
                state = SqueezeState.SQUEEZE_FIRING
            else:
                state = SqueezeState.SQUEEZE_OFF

        history.append(state)
        if len(history) > 100:
            history.pop(0)

        # Count bars in squeeze
        bars_in_squeeze = 0
        for h in reversed(history[:-1]):  # Exclude current
            if h == SqueezeState.SQUEEZE_ON:
                bars_in_squeeze += 1
            else:
                break

        # Determine momentum direction
        if abs(current_momentum) < 0.001:
            mom_direction = MomentumDirection.NEUTRAL
        elif current_momentum > 0:
            mom_direction = MomentumDirection.BULLISH
        else:
            mom_direction = MomentumDirection.BEARISH

        # Calculate squeeze intensity (how tight)
        if current_kc_width > 0:
            squeeze_intensity = max(0.0, 1.0 - (current_bb_width / current_kc_width))
        else:
            squeeze_intensity = 0.0

        reading = SqueezeReading(
            squeeze_state=state,
            momentum_value=current_momentum,
            momentum_direction=mom_direction,
            bars_in_squeeze=bars_in_squeeze,
            bb_width=current_bb_width,
            kc_width=current_kc_width,
            squeeze_intensity=squeeze_intensity,
        )

        self._last_readings[symbol] = reading
        return reading

    def generate_signal(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: float | None = None,
        current_position: str = "FLAT",
    ) -> SqueezeSignal | None:
        """
        Generate trading signal based on squeeze analysis.

        Args:
            symbol: Instrument symbol
            high: High prices
            low: Low prices
            close: Close prices
            atr: Current ATR (for stops)
            current_position: Current position ("FLAT", "LONG", "SHORT")

        Returns:
            SqueezeSignal if conditions met, None otherwise
        """
        reading = self.analyze(symbol, high, low, close)

        if len(close) == 0:
            return None

        current_price = close[-1]

        # Calculate ATR if not provided
        if atr is None:
            atr_arr = self._calculate_atr(high, low, close, 14)
            atr = atr_arr[-1] if len(atr_arr) > 0 else current_price * 0.02

        # Entry signal: Squeeze firing with sufficient buildup
        if current_position == "FLAT":
            if reading.squeeze_state == SqueezeState.SQUEEZE_FIRING:
                if reading.bars_in_squeeze >= self._min_squeeze_bars:
                    # Direction based on momentum
                    if reading.momentum_direction == MomentumDirection.BULLISH:
                        direction = "LONG"
                        stop_loss = current_price - self._stop_atr_mult * atr
                        take_profit = current_price + self._take_profit_atr_mult * atr
                    elif reading.momentum_direction == MomentumDirection.BEARISH:
                        direction = "SHORT"
                        stop_loss = current_price + self._stop_atr_mult * atr
                        take_profit = current_price - self._take_profit_atr_mult * atr
                    else:
                        return None  # No clear momentum direction

                    # Signal strength based on squeeze duration and intensity
                    strength = min(1.0, (reading.bars_in_squeeze / 20.0) + reading.squeeze_intensity * 0.5)

                    return SqueezeSignal(
                        symbol=symbol,
                        direction=direction,
                        signal_type="squeeze_fire",
                        strength=strength,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        squeeze_bars=reading.bars_in_squeeze,
                        momentum_value=reading.momentum_value,
                        rationale=(
                            f"Squeeze fired after {reading.bars_in_squeeze} bars, "
                            f"momentum={reading.momentum_value:.4f}, "
                            f"intensity={reading.squeeze_intensity:.2f}"
                        ),
                    )

        # Exit signal: Momentum reversal or squeeze back on
        elif current_position in ["LONG", "SHORT"]:
            should_exit = False
            exit_reason = ""

            # Exit if squeeze goes back on (volatility contracting again)
            if reading.squeeze_state == SqueezeState.SQUEEZE_ON:
                should_exit = True
                exit_reason = "Squeeze re-engaged, volatility contracting"

            # Exit on momentum reversal
            elif current_position == "LONG" and reading.momentum_direction == MomentumDirection.BEARISH:
                should_exit = True
                exit_reason = "Momentum turned bearish"
            elif current_position == "SHORT" and reading.momentum_direction == MomentumDirection.BULLISH:
                should_exit = True
                exit_reason = "Momentum turned bullish"

            if should_exit:
                return SqueezeSignal(
                    symbol=symbol,
                    direction="FLAT",
                    signal_type="exit",
                    strength=0.8,
                    entry_price=current_price,
                    stop_loss=0.0,
                    take_profit=0.0,
                    squeeze_bars=reading.bars_in_squeeze,
                    momentum_value=reading.momentum_value,
                    rationale=exit_reason,
                )

        return None

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        return {
            "bb_period": self._bb_period,
            "bb_std": self._bb_std,
            "kc_period": self._kc_period,
            "kc_atr_mult": self._kc_atr_mult,
            "momentum_period": self._momentum_period,
            "min_squeeze_bars": self._min_squeeze_bars,
            "tracked_symbols": len(self._last_readings),
            "readings": {
                symbol: {
                    "state": reading.squeeze_state.value,
                    "momentum": reading.momentum_direction.value,
                    "bars_in_squeeze": reading.bars_in_squeeze,
                }
                for symbol, reading in self._last_readings.items()
            },
        }


def create_ttm_squeeze_strategy(config: dict[str, Any] | None = None) -> TTMSqueezeStrategy:
    """Create TTMSqueezeStrategy instance."""
    return TTMSqueezeStrategy(config)

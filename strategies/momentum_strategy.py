"""
Momentum Strategy
=================

Implements momentum and trend-following logic.

TODO: This is a placeholder - implement actual momentum models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class MomentumSignal:
    """Momentum signal output."""
    symbol: str
    direction: str  # "long", "short", "flat"
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    indicators: dict[str, float]


class MomentumStrategy:
    """
    Momentum Strategy Implementation.

    Implements:
    1. Moving average crossovers
    2. RSI (Relative Strength Index)
    3. MACD (Moving Average Convergence Divergence)
    4. Rate of Change (ROC)

    TODO: Implement more sophisticated models:
    - ADX for trend strength
    - Ichimoku Cloud
    - Bollinger Bands
    - Volume-weighted indicators
    """

    def __init__(self, config: dict[str, Any]):
        self._fast_period = config.get("fast_period", 10)
        self._slow_period = config.get("slow_period", 30)
        self._signal_period = config.get("signal_period", 9)
        self._rsi_period = config.get("rsi_period", 14)
        self._rsi_overbought = config.get("rsi_overbought", 70)
        self._rsi_oversold = config.get("rsi_oversold", 30)

        # MACD settings (standard: 12, 26, 9)
        self._macd_fast = config.get("macd_fast_period", 12)
        self._macd_slow = config.get("macd_slow_period", 26)
        self._macd_signal = config.get("macd_signal_period", 9)

    def calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0
        return np.mean(prices[-period:])

    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate Exponential Moving Average (returns single value).
        """
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0

        alpha = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def calculate_ema_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate full EMA series for MACD signal line calculation.

        Returns array of EMA values, same length as input prices.
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        alpha = 2 / (period + 1)
        ema_series = np.zeros(len(prices))

        # Initialize with SMA for first 'period' values
        ema_series[:period] = np.nan
        ema_series[period - 1] = np.mean(prices[:period])

        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema_series[i] = alpha * prices[i] + (1 - alpha) * ema_series[i - 1]

        return ema_series

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index using Wilder's smoothing method.

        Wilder's smoothing is an exponential moving average with
        smoothing factor = 1/period (as opposed to 2/(period+1) for standard EMA).
        """
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period

        # Initialize with simple average of first 'period' values
        if len(gains) >= period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            # Apply Wilder's smoothing for remaining values
            for i in range(period, len(gains)):
                avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        else:
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

        if avg_loss < 1e-8:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        prices: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Calculate MACD, Signal line, and Histogram.

        MACD Line = Fast EMA (12) - Slow EMA (26)
        Signal Line = 9-period EMA of MACD Line
        Histogram = MACD Line - Signal Line

        Returns (macd, signal, histogram).
        """
        min_required = self._macd_slow + self._macd_signal
        if len(prices) < min_required:
            return 0.0, 0.0, 0.0

        # Calculate fast and slow EMA series
        fast_ema_series = self.calculate_ema_series(prices, self._macd_fast)
        slow_ema_series = self.calculate_ema_series(prices, self._macd_slow)

        # Calculate MACD line series
        macd_series = fast_ema_series - slow_ema_series

        # Remove NaN values for signal line calculation
        valid_macd = macd_series[~np.isnan(macd_series)]

        if len(valid_macd) < self._macd_signal:
            # Not enough data for signal line
            macd = valid_macd[-1] if len(valid_macd) > 0 else 0.0
            return macd, 0.0, macd

        # Calculate signal line (9-period EMA of MACD)
        signal_series = self.calculate_ema_series(valid_macd, self._macd_signal)

        # Get current values
        macd = valid_macd[-1]
        signal = signal_series[-1] if not np.isnan(signal_series[-1]) else 0.0
        histogram = macd - signal

        return macd, signal, histogram

    def calculate_roc(self, prices: np.ndarray, period: int = 10) -> float:
        """
        Calculate Rate of Change.
        """
        if len(prices) < period + 1:
            return 0.0

        return (prices[-1] - prices[-period - 1]) / prices[-period - 1] * 100

    def analyze(self, symbol: str, prices: np.ndarray) -> MomentumSignal:
        """
        Analyze price series and generate momentum signal.
        """
        if len(prices) < self._slow_period:
            return MomentumSignal(
                symbol=symbol,
                direction="flat",
                strength=0.0,
                confidence=0.0,
                indicators={},
            )

        # Calculate indicators
        fast_ma = self.calculate_sma(prices, self._fast_period)
        slow_ma = self.calculate_sma(prices, self._slow_period)
        rsi = self.calculate_rsi(prices, self._rsi_period)
        macd, macd_signal, macd_hist = self.calculate_macd(prices)
        roc = self.calculate_roc(prices)

        indicators = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "roc": roc,
        }

        # Score each indicator
        scores = []

        # MA crossover
        if fast_ma > slow_ma:
            scores.append(1)
        elif fast_ma < slow_ma:
            scores.append(-1)
        else:
            scores.append(0)

        # RSI
        if rsi < self._rsi_oversold:
            scores.append(1)  # Oversold = bullish
        elif rsi > self._rsi_overbought:
            scores.append(-1)  # Overbought = bearish
        else:
            scores.append(0)

        # MACD
        if macd > macd_signal:
            scores.append(1)
        elif macd < macd_signal:
            scores.append(-1)
        else:
            scores.append(0)

        # ROC
        if roc > 5:
            scores.append(1)
        elif roc < -5:
            scores.append(-1)
        else:
            scores.append(0)

        # Aggregate
        total_score = sum(scores)
        max_score = len(scores)

        strength = total_score / max_score

        if total_score > 1:
            direction = "long"
        elif total_score < -1:
            direction = "short"
        else:
            direction = "flat"

        # Confidence based on agreement
        agreement = sum(1 for s in scores if s == np.sign(total_score)) / len(scores)
        confidence = agreement

        return MomentumSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            indicators=indicators,
        )

"""
MACD-v Strategy
===============

Implements the MACD-v (MACD Volatility-Normalized) indicator strategy.

MACD-v normalizes the traditional MACD by ATR to account for volatility,
making signals comparable across different instruments and market conditions.

Formula:
    MACD-v = [(EMA(12) - EMA(26)) / ATR(26)] * 100

Interpretation:
    - Values oscillate around zero like traditional MACD
    - Overbought zone: +150 to +200 (instrument dependent)
    - Oversold zone: -150 to -200 (instrument dependent)
    - Normalized values allow cross-instrument comparison

Entry Rules:
    - LONG: MACD-v crosses above signal line from oversold zone
    - SHORT: MACD-v crosses below signal line from overbought zone

Exit Rules:
    - Opposite signal
    - Histogram momentum reversal
    - Stop-loss / Take-profit hit

MATURITY: ALPHA
---------------
Status: Initial implementation
- [x] MACD-v calculation
- [x] Signal line calculation
- [x] Histogram calculation
- [x] Zone detection (overbought/oversold)
- [x] Entry/exit signal generation
- [x] Stop-loss / Take-profit calculation
- [ ] Backtesting validation
- [ ] Live testing

References:
    - Alex Spiroglou's MACD-v indicator
    - Volatility-adjusted momentum analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class MACDvSignal:
    """MACD-v signal output."""
    symbol: str
    direction: str  # "long", "short", "flat", "exit_long", "exit_short"
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    indicators: dict[str, float]
    # Risk management
    stop_loss_price: float | None = None
    stop_loss_pct: float | None = None
    take_profit_price: float | None = None
    take_profit_pct: float | None = None
    # Strategy context
    macdv_value: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    zone: str = "neutral"  # "overbought", "oversold", "neutral"
    # Exit signal information
    is_exit_signal: bool = False
    exit_reason: str | None = None


class MACDvStrategy:
    """
    MACD-v (Volatility-Normalized MACD) Strategy Implementation.

    MACD-v improves on traditional MACD by normalizing with ATR,
    making the indicator comparable across instruments with different
    volatility profiles.

    Key improvements over standard MACD:
    1. Cross-instrument comparability
    2. Volatility-adjusted signal thresholds
    3. Clearer overbought/oversold zones
    4. Better trend strength assessment
    """

    def __init__(self, config: dict[str, Any]):
        # MACD parameters
        self._fast_period = config.get("fast_period", 12)
        self._slow_period = config.get("slow_period", 26)
        self._signal_period = config.get("signal_period", 9)

        # ATR period for normalization (typically same as slow EMA)
        self._atr_period = config.get("atr_period", 26)

        # Zone thresholds (volatility-normalized, so consistent across instruments)
        self._overbought_zone = config.get("overbought_zone", 150.0)
        self._oversold_zone = config.get("oversold_zone", -150.0)
        self._extreme_overbought = config.get("extreme_overbought", 200.0)
        self._extreme_oversold = config.get("extreme_oversold", -200.0)

        # Signal line cross sensitivity
        self._require_zone_exit = config.get("require_zone_exit", True)

        # Neutral zone filter (Charles H. Dow Award 2022 recommendation)
        # Signals in the -50 to +50 range are typically noise
        # Only trade when MACD-v shows clear directional bias
        self._neutral_zone_filter = config.get("neutral_zone_filter", True)
        self._neutral_zone_upper = config.get("neutral_zone_upper", 50.0)
        self._neutral_zone_lower = config.get("neutral_zone_lower", -50.0)

        # Ranging market detection (Spiroglou 2022)
        # MACD-v staying in neutral zone for 20-30+ bars indicates sideways/ranging market
        # Suppress all signals during ranging conditions to avoid whipsaws
        self._ranging_detection_enabled = config.get("ranging_detection_enabled", True)
        self._ranging_bar_threshold = config.get("ranging_bar_threshold", 25)
        self._bars_in_neutral: dict[str, int] = {}  # Track per symbol
        # FIX-36: Track last neutral update time to avoid per-tick counting
        self._last_neutral_update: dict[str, float] = {}  # symbol -> last update timestamp
        self._neutral_update_interval = config.get("neutral_update_interval", 60.0)  # 60s = 1 bar

        # Stop-loss and take-profit settings
        self._stop_loss_pct = config.get("stop_loss_pct", 2.0)  # 2%
        self._take_profit_pct = config.get("take_profit_pct", 4.0)  # 4% (2:1 R:R)
        self._use_atr_stop = config.get("use_atr_stop", True)
        self._stop_loss_atr_mult = config.get("stop_loss_atr_mult", 2.0)
        self._take_profit_atr_mult = config.get("take_profit_atr_mult", 4.0)

        # Histogram momentum settings
        self._histogram_momentum_bars = config.get("histogram_momentum_bars", 3)

        # Minimum confidence threshold
        self._min_confidence = config.get("min_confidence", 0.5)

    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average series.

        Args:
            prices: Price array
            period: EMA period

        Returns:
            Array of EMA values (NaN for insufficient data)
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(prices))

        # Initialize with SMA
        ema[:period] = np.nan
        ema[period - 1] = np.mean(prices[:period])

        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Calculate Average True Range series.

        True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)
        ATR = EMA of True Range

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: ATR period

        Returns:
            Array of ATR values
        """
        n = len(closes)
        if n < period + 1:
            return np.full(n, np.nan)

        # Calculate True Range
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]  # First bar: just high - low

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        alpha = 1.0 / period
        atr = np.zeros(n)
        atr[:period] = np.nan
        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        return atr

    def calculate_atr_from_closes(
        self,
        closes: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Estimate ATR from close prices only (when OHLC not available).

        Uses price changes as proxy for True Range.

        Args:
            closes: Close prices
            period: ATR period

        Returns:
            Array of estimated ATR values
        """
        n = len(closes)
        if n < period + 1:
            return np.full(n, np.nan)

        # Estimate TR from absolute price changes (scaled up)
        tr = np.zeros(n)
        tr[0] = closes[0] * 0.01  # Initial estimate: 1% of price

        for i in range(1, n):
            # Use absolute change, scaled by ~1.5 to approximate TR
            tr[i] = abs(closes[i] - closes[i - 1]) * 1.5

        # Minimum TR floor (0.1% of price)
        for i in range(n):
            tr[i] = max(tr[i], closes[i] * 0.001)

        # Smooth with Wilder's method
        alpha = 1.0 / period
        atr = np.zeros(n)
        atr[:period] = np.nan
        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        return atr

    def calculate_macdv(
        self,
        prices: np.ndarray,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD-v (volatility-normalized MACD).

        Formula: MACD-v = [(EMA(12) - EMA(26)) / ATR(26)] * 100

        Args:
            prices: Close prices
            highs: High prices (optional)
            lows: Low prices (optional)

        Returns:
            Tuple of (macdv_line, signal_line, histogram) arrays
        """
        n = len(prices)
        min_required = self._slow_period + self._signal_period

        if n < min_required:
            return (
                np.full(n, np.nan),
                np.full(n, np.nan),
                np.full(n, np.nan)
            )

        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, self._fast_period)
        slow_ema = self.calculate_ema(prices, self._slow_period)

        # Calculate ATR
        if highs is not None and lows is not None:
            atr = self.calculate_atr(highs, lows, prices, self._atr_period)
        else:
            atr = self.calculate_atr_from_closes(prices, self._atr_period)

        # Calculate MACD-v
        # Avoid division by zero
        macdv = np.zeros(n)
        for i in range(n):
            if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]) and not np.isnan(atr[i]) and atr[i] > 0:
                macdv[i] = ((fast_ema[i] - slow_ema[i]) / atr[i]) * 100
            else:
                macdv[i] = np.nan

        # Calculate signal line (EMA of MACD-v)
        # Extract valid MACD-v values for signal calculation
        valid_start = self._slow_period - 1
        if valid_start < 0:
            valid_start = 0

        signal_line = np.full(n, np.nan)

        # Find first valid index
        first_valid = -1
        for i in range(n):
            if not np.isnan(macdv[i]):
                first_valid = i
                break

        if first_valid >= 0 and n - first_valid >= self._signal_period:
            # Calculate EMA of MACD-v as signal line
            signal_alpha = 2.0 / (self._signal_period + 1)

            # Initialize signal line with SMA of first signal_period valid MACD-v values
            init_idx = first_valid + self._signal_period - 1
            if init_idx < n:
                signal_line[init_idx] = np.mean(macdv[first_valid:first_valid + self._signal_period])

                for i in range(init_idx + 1, n):
                    if not np.isnan(macdv[i]) and not np.isnan(signal_line[i - 1]):
                        signal_line[i] = signal_alpha * macdv[i] + (1 - signal_alpha) * signal_line[i - 1]

        # Calculate histogram
        histogram = macdv - signal_line

        return macdv, signal_line, histogram

    def get_zone(self, macdv_value: float) -> str:
        """
        Determine which zone the MACD-v value is in.

        Args:
            macdv_value: Current MACD-v value

        Returns:
            Zone name: "extreme_overbought", "overbought", "neutral",
                      "oversold", "extreme_oversold"
        """
        if np.isnan(macdv_value):
            return "neutral"

        if macdv_value >= self._extreme_overbought:
            return "extreme_overbought"
        elif macdv_value >= self._overbought_zone:
            return "overbought"
        elif macdv_value <= self._extreme_oversold:
            return "extreme_oversold"
        elif macdv_value <= self._oversold_zone:
            return "oversold"
        else:
            return "neutral"

    def check_histogram_momentum(
        self,
        histogram: np.ndarray,
        direction: str,
        bars: int | None = None
    ) -> tuple[bool, float]:
        """
        Check if histogram momentum confirms the signal direction.

        For LONG: histogram should be increasing (getting less negative or more positive)
        For SHORT: histogram should be decreasing (getting less positive or more negative)

        Args:
            histogram: Histogram array
            direction: Signal direction ("long" or "short")
            bars: Number of bars to check momentum

        Returns:
            Tuple of (momentum_confirms, momentum_strength)
        """
        if bars is None:
            bars = self._histogram_momentum_bars

        n = len(histogram)
        if n < bars + 1:
            return False, 0.0

        # Get recent histogram values
        recent = histogram[-(bars + 1):]

        # Remove NaN values
        recent = recent[~np.isnan(recent)]
        if len(recent) < 2:
            return False, 0.0

        # Calculate momentum (average change over period)
        changes = np.diff(recent)
        avg_change = np.mean(changes)

        # Normalize strength by recent histogram range
        hist_range = np.max(np.abs(recent))
        if hist_range > 0:
            momentum_strength = abs(avg_change) / hist_range
        else:
            momentum_strength = 0.0

        momentum_strength = min(1.0, momentum_strength)

        # Check if momentum confirms direction
        if direction == "long":
            confirms = avg_change > 0
        elif direction == "short":
            confirms = avg_change < 0
        else:
            confirms = False

        return confirms, momentum_strength

    def detect_crossover(
        self,
        macdv: np.ndarray,
        signal_line: np.ndarray,
        lookback: int = 2
    ) -> tuple[str, int]:
        """
        Detect MACD-v / signal line crossover.

        Args:
            macdv: MACD-v array
            signal_line: Signal line array
            lookback: How many bars back to check for crossover

        Returns:
            Tuple of (crossover_type, bars_ago) where:
            - crossover_type: "bullish", "bearish", or "none"
            - bars_ago: How many bars ago the crossover occurred
        """
        n = len(macdv)
        if n < lookback + 1:
            return "none", 0

        for i in range(1, lookback + 1):
            idx = n - i
            if idx < 1:
                break

            curr_macdv = macdv[idx]
            prev_macdv = macdv[idx - 1]
            curr_signal = signal_line[idx]
            prev_signal = signal_line[idx - 1]

            if np.isnan(curr_macdv) or np.isnan(prev_macdv) or np.isnan(curr_signal) or np.isnan(prev_signal):
                continue

            # Bullish crossover: MACD-v crosses above signal line
            if prev_macdv <= prev_signal and curr_macdv > curr_signal:
                return "bullish", i - 1

            # Bearish crossover: MACD-v crosses below signal line
            if prev_macdv >= prev_signal and curr_macdv < curr_signal:
                return "bearish", i - 1

        return "none", 0

    def calculate_stop_loss(
        self,
        current_price: float,
        direction: str,
        atr: float
    ) -> tuple[float | None, float | None]:
        """
        Calculate stop-loss price and percentage.

        Args:
            current_price: Entry price
            direction: "long" or "short"
            atr: Current ATR value

        Returns:
            Tuple of (stop_price, stop_pct)
        """
        if self._use_atr_stop and atr > 0 and not np.isnan(atr):
            stop_distance = atr * self._stop_loss_atr_mult
        else:
            stop_distance = current_price * (self._stop_loss_pct / 100)

        if direction == "long":
            stop_price = current_price - stop_distance
        elif direction == "short":
            stop_price = current_price + stop_distance
        else:
            return None, None

        stop_pct = (stop_distance / current_price) * 100
        return round(stop_price, 2), round(stop_pct, 2)

    def calculate_take_profit(
        self,
        current_price: float,
        direction: str,
        atr: float
    ) -> tuple[float | None, float | None]:
        """
        Calculate take-profit price and percentage.

        Args:
            current_price: Entry price
            direction: "long" or "short"
            atr: Current ATR value

        Returns:
            Tuple of (take_profit_price, take_profit_pct)
        """
        if self._use_atr_stop and atr > 0 and not np.isnan(atr):
            tp_distance = atr * self._take_profit_atr_mult
        else:
            tp_distance = current_price * (self._take_profit_pct / 100)

        if direction == "long":
            tp_price = current_price + tp_distance
        elif direction == "short":
            tp_price = current_price - tp_distance
        else:
            return None, None

        tp_pct = (tp_distance / current_price) * 100
        return round(tp_price, 2), round(tp_pct, 2)

    def analyze(
        self,
        symbol: str,
        prices: np.ndarray,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        timestamp: datetime | None = None,
        previous_zone: str | None = None
    ) -> MACDvSignal:
        """
        Analyze price series and generate MACD-v signal.

        Entry Logic:
        - LONG: MACD-v crosses above signal from oversold zone
        - SHORT: MACD-v crosses below signal from overbought zone

        Args:
            symbol: Trading symbol
            prices: Close price array
            highs: High prices (optional)
            lows: Low prices (optional)
            timestamp: Current timestamp
            previous_zone: Previous candle's zone (for zone exit detection)

        Returns:
            MACDvSignal with direction, strength, and indicators
        """
        n = len(prices)
        min_required = self._slow_period + self._signal_period

        # Insufficient data
        if n < min_required:
            return MACDvSignal(
                symbol=symbol,
                direction="flat",
                strength=0.0,
                confidence=0.0,
                indicators={},
                zone="neutral"
            )

        # Calculate MACD-v
        macdv, signal_line, histogram = self.calculate_macdv(prices, highs, lows)

        # Get current values
        curr_macdv = macdv[-1]
        curr_signal = signal_line[-1]
        curr_histogram = histogram[-1]

        if np.isnan(curr_macdv) or np.isnan(curr_signal):
            return MACDvSignal(
                symbol=symbol,
                direction="flat",
                strength=0.0,
                confidence=0.0,
                indicators={"macdv": 0.0, "signal": 0.0, "histogram": 0.0},
                zone="neutral"
            )

        # Get ATR for stop-loss calculation
        if highs is not None and lows is not None:
            atr = self.calculate_atr(highs, lows, prices, self._atr_period)
        else:
            atr = self.calculate_atr_from_closes(prices, self._atr_period)
        curr_atr = atr[-1] if not np.isnan(atr[-1]) else prices[-1] * 0.02

        # Determine current zone
        current_zone = self.get_zone(curr_macdv)

        # Detect crossover
        crossover_type, crossover_bars_ago = self.detect_crossover(macdv, signal_line)

        # Check histogram momentum
        histogram_momentum_long, momentum_strength_long = self.check_histogram_momentum(histogram, "long")
        histogram_momentum_short, momentum_strength_short = self.check_histogram_momentum(histogram, "short")

        # Build indicators dict
        indicators = {
            "macdv": round(curr_macdv, 2),
            "signal_line": round(curr_signal, 2),
            "histogram": round(curr_histogram, 2),
            "atr": round(curr_atr, 4),
            "overbought_zone": self._overbought_zone,
            "oversold_zone": self._oversold_zone,
            "crossover_type": crossover_type,
            "crossover_bars_ago": crossover_bars_ago,
            "histogram_momentum_long": histogram_momentum_long,
            "histogram_momentum_short": histogram_momentum_short,
        }

        # Signal generation logic
        direction = "flat"
        strength = 0.0
        confidence = 0.0

        # LONG signal conditions:
        # 1. Bullish crossover (MACD-v crosses above signal)
        # 2. Coming from oversold zone (or optionally, any zone)
        # 3. Histogram momentum confirming
        if crossover_type == "bullish":
            # Check zone condition
            zone_ok = True
            if self._require_zone_exit:
                zone_ok = previous_zone in ("oversold", "extreme_oversold") or current_zone in ("oversold", "extreme_oversold")

            if zone_ok:
                direction = "long"

                # Strength based on how oversold we were
                if current_zone == "extreme_oversold":
                    strength = 0.9
                elif current_zone == "oversold":
                    strength = 0.7
                else:
                    strength = 0.5

                # Confidence based on momentum confirmation
                base_confidence = 0.6
                if histogram_momentum_long:
                    confidence = base_confidence + 0.3 * momentum_strength_long
                else:
                    confidence = base_confidence * 0.7

        # SHORT signal conditions:
        # 1. Bearish crossover (MACD-v crosses below signal)
        # 2. Coming from overbought zone (or optionally, any zone)
        # 3. Histogram momentum confirming
        elif crossover_type == "bearish":
            zone_ok = True
            if self._require_zone_exit:
                zone_ok = previous_zone in ("overbought", "extreme_overbought") or current_zone in ("overbought", "extreme_overbought")

            if zone_ok:
                direction = "short"

                # Strength based on how overbought we were
                if current_zone == "extreme_overbought":
                    strength = -0.9
                elif current_zone == "overbought":
                    strength = -0.7
                else:
                    strength = -0.5

                # Confidence based on momentum confirmation
                base_confidence = 0.6
                if histogram_momentum_short:
                    confidence = base_confidence + 0.3 * momentum_strength_short
                else:
                    confidence = base_confidence * 0.7

        # Additional signal: Histogram momentum reversal (exit signal)
        # If we're in a position and histogram momentum reverses, consider exit
        if direction == "flat":
            # Check for exit signals based on histogram momentum reversal
            if curr_macdv > curr_signal and not histogram_momentum_long and curr_histogram < 0:
                # We're above signal but histogram is negative and not increasing
                # This could be a weakening long
                pass  # Exit signals handled at agent level with position tracking

        # Neutral zone filter (Charles H. Dow Award 2022)
        # Reject signals when MACD-v is in the noise zone (-50 to +50)
        if self._neutral_zone_filter and direction != "flat":
            if self._neutral_zone_lower < curr_macdv < self._neutral_zone_upper:
                logger.debug(
                    f"{symbol}: Signal rejected - MACD-v {curr_macdv:.1f} in neutral zone "
                    f"[{self._neutral_zone_lower}, {self._neutral_zone_upper}]"
                )
                direction = "flat"
                strength = 0.0
                confidence = 0.0

        # Ranging market detection (Spiroglou 2022)
        # Track how many bars MACD-v has been in the neutral zone
        # 25+ bars in neutral zone = ranging/sideways market = suppress all signals
        if self._ranging_detection_enabled:
            import time
            now = time.time()
            in_neutral = self._neutral_zone_lower < curr_macdv < self._neutral_zone_upper
            # FIX-36: Only update counter once per bar interval (60s default)
            # to avoid inflating count when called per-tick
            last_update = self._last_neutral_update.get(symbol, 0.0)
            if now - last_update >= self._neutral_update_interval:
                self._last_neutral_update[symbol] = now
                if in_neutral:
                    self._bars_in_neutral[symbol] = self._bars_in_neutral.get(symbol, 0) + 1
                else:
                    self._bars_in_neutral[symbol] = 0

            bars_neutral = self._bars_in_neutral.get(symbol, 0)
            if bars_neutral >= self._ranging_bar_threshold and direction != "flat":
                logger.debug(
                    f"{symbol}: Signal rejected - ranging market detected "
                    f"({bars_neutral} bars in neutral zone >= {self._ranging_bar_threshold})"
                )
                direction = "flat"
                strength = 0.0
                confidence = 0.0
                indicators["ranging_market"] = True
                indicators["bars_in_neutral"] = bars_neutral

        # Calculate stop-loss and take-profit for directional signals
        current_price = prices[-1]
        stop_loss_price = None
        stop_loss_pct = None
        take_profit_price = None
        take_profit_pct = None

        if direction in ("long", "short"):
            stop_loss_price, stop_loss_pct = self.calculate_stop_loss(
                current_price, direction, curr_atr
            )
            take_profit_price, take_profit_pct = self.calculate_take_profit(
                current_price, direction, curr_atr
            )

            indicators["stop_loss_price"] = stop_loss_price
            indicators["stop_loss_pct"] = stop_loss_pct
            indicators["take_profit_price"] = take_profit_price
            indicators["take_profit_pct"] = take_profit_pct

        return MACDvSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            indicators=indicators,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
            macdv_value=curr_macdv,
            signal_line=curr_signal,
            histogram=curr_histogram,
            zone=current_zone,
            is_exit_signal=False,
            exit_reason=None
        )

    def check_exit_conditions(
        self,
        macdv: float,
        signal_line: float,
        histogram: float,
        direction: str,
        histogram_history: np.ndarray
    ) -> tuple[bool, str | None]:
        """
        Check if exit conditions are met for an existing position.

        Exit conditions:
        1. Opposite crossover signal
        2. Histogram momentum reversal
        3. Extreme zone reached in opposite direction

        Args:
            macdv: Current MACD-v value
            signal_line: Current signal line value
            histogram: Current histogram value
            direction: Current position direction ("long" or "short")
            histogram_history: Recent histogram values

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Check opposite direction crossover
        if direction == "long" and macdv < signal_line:
            return True, "macdv_bearish_crossover"
        elif direction == "short" and macdv > signal_line:
            return True, "macdv_bullish_crossover"

        # Check histogram momentum reversal
        if direction == "long":
            momentum_confirms, _ = self.check_histogram_momentum(histogram_history, "long")
            if not momentum_confirms and histogram < 0:
                return True, "histogram_momentum_reversal"
        elif direction == "short":
            momentum_confirms, _ = self.check_histogram_momentum(histogram_history, "short")
            if not momentum_confirms and histogram > 0:
                return True, "histogram_momentum_reversal"

        # Check extreme zone in opposite direction
        zone = self.get_zone(macdv)
        if direction == "long" and zone in ("overbought", "extreme_overbought"):
            return True, "reached_overbought_zone"
        elif direction == "short" and zone in ("oversold", "extreme_oversold"):
            return True, "reached_oversold_zone"

        return False, None

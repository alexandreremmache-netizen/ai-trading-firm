"""
Technical Indicators Module
===========================

Comprehensive technical analysis indicators (Issues #Q13, #Q14, #Q15).

Features:
- ADX (Average Directional Index) trend strength
- Bollinger Bands with squeeze detection
- Volume-weighted indicators (VWAP, OBV, MFI)
- Additional momentum and trend indicators
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ADXResult:
    """ADX calculation result (Issue #Q13)."""
    adx: float  # Average Directional Index (0-100)
    plus_di: float  # +DI (positive directional indicator)
    minus_di: float  # -DI (negative directional indicator)
    dx: float  # Directional Index

    @property
    def trend_strength(self) -> str:
        """Classify trend strength."""
        if self.adx < 20:
            return "weak"
        elif self.adx < 25:
            return "emerging"
        elif self.adx < 50:
            return "strong"
        else:
            return "very_strong"

    @property
    def trend_direction(self) -> str:
        """Determine trend direction."""
        if self.plus_di > self.minus_di:
            return "bullish"
        elif self.minus_di > self.plus_di:
            return "bearish"
        else:
            return "neutral"


class ADXCalculator:
    """
    Average Directional Index calculator (Issue #Q13).

    ADX measures trend strength without regard to direction.
    Values above 25 indicate trending market.
    """

    def __init__(self, period: int = 14):
        self.period = period

        # History
        self._highs: deque = deque(maxlen=period + 1)
        self._lows: deque = deque(maxlen=period + 1)
        self._closes: deque = deque(maxlen=period + 1)

        # Smoothed values
        self._smoothed_plus_dm: float = 0.0
        self._smoothed_minus_dm: float = 0.0
        self._smoothed_tr: float = 0.0
        self._smoothed_dx: float = 0.0

        self._initialized: bool = False
        self._bar_count: int = 0

    def update(self, bar: OHLCV) -> ADXResult | None:
        """
        Update with new bar and return ADX result.

        Returns None until enough data is accumulated.
        """
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._closes.append(bar.close)
        self._bar_count += 1

        if len(self._highs) < 2:
            return None

        # Calculate True Range
        prev_close = self._closes[-2]
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close)
        )

        # Calculate Directional Movement
        prev_high = self._highs[-2]
        prev_low = self._lows[-2]

        up_move = bar.high - prev_high
        down_move = prev_low - bar.low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        # Smoothing (Wilder's smoothing)
        if not self._initialized:
            if self._bar_count < self.period + 1:
                # Accumulate initial values
                self._smoothed_tr += tr
                self._smoothed_plus_dm += plus_dm
                self._smoothed_minus_dm += minus_dm
                return None

            # First smoothed values
            self._initialized = True
        else:
            # Wilder's smoothing: current = prior - (prior/period) + current_value
            self._smoothed_tr = self._smoothed_tr - (self._smoothed_tr / self.period) + tr
            self._smoothed_plus_dm = self._smoothed_plus_dm - (self._smoothed_plus_dm / self.period) + plus_dm
            self._smoothed_minus_dm = self._smoothed_minus_dm - (self._smoothed_minus_dm / self.period) + minus_dm

        # Calculate +DI and -DI
        if self._smoothed_tr == 0:
            plus_di = 0
            minus_di = 0
        else:
            plus_di = 100 * self._smoothed_plus_dm / self._smoothed_tr
            minus_di = 100 * self._smoothed_minus_dm / self._smoothed_tr

        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum

        # Smooth DX to get ADX
        if self._smoothed_dx == 0:
            self._smoothed_dx = dx
        else:
            self._smoothed_dx = (self._smoothed_dx * (self.period - 1) + dx) / self.period

        return ADXResult(
            adx=self._smoothed_dx,
            plus_di=plus_di,
            minus_di=minus_di,
            dx=dx,
        )


@dataclass
class BollingerBandsResult:
    """Bollinger Bands result (Issue #Q14)."""
    upper: float
    middle: float  # SMA
    lower: float
    bandwidth: float  # (upper - lower) / middle
    percent_b: float  # (close - lower) / (upper - lower)

    @property
    def is_squeeze(self) -> bool:
        """Detect Bollinger squeeze (low volatility)."""
        return self.bandwidth < 0.05  # Bandwidth < 5%

    @property
    def position(self) -> str:
        """Price position within bands."""
        if self.percent_b > 1:
            return "above_upper"
        elif self.percent_b > 0.8:
            return "near_upper"
        elif self.percent_b < 0:
            return "below_lower"
        elif self.percent_b < 0.2:
            return "near_lower"
        else:
            return "middle"


class BollingerBandsCalculator:
    """
    Bollinger Bands calculator (Issue #Q14).

    Bands based on SMA and standard deviation.
    Useful for volatility and mean reversion signals.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

        self._closes: deque = deque(maxlen=period)

    def update(self, close: float) -> BollingerBandsResult | None:
        """
        Update with new close and return bands.

        Returns None until enough data.
        """
        self._closes.append(close)

        if len(self._closes) < self.period:
            return None

        closes = list(self._closes)
        sma = statistics.mean(closes)
        std = statistics.stdev(closes) if len(closes) > 1 else 0

        upper = sma + self.std_dev * std
        lower = sma - self.std_dev * std

        bandwidth = (upper - lower) / sma if sma > 0 else 0
        band_range = upper - lower
        percent_b = (close - lower) / band_range if band_range > 0 else 0.5

        return BollingerBandsResult(
            upper=upper,
            middle=sma,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b,
        )


@dataclass
class VWAPResult:
    """VWAP calculation result (Issue #Q15)."""
    vwap: float
    upper_band: float  # VWAP + 1 std dev
    lower_band: float  # VWAP - 1 std dev
    cumulative_volume: float
    deviation_pct: float  # Current price deviation from VWAP


class VWAPCalculator:
    """
    Volume Weighted Average Price calculator (Issue #Q15).

    Intraday benchmark for institutional execution.
    Resets daily.
    """

    def __init__(self, use_bands: bool = True, band_std: float = 1.0):
        self.use_bands = use_bands
        self.band_std = band_std

        self._cumulative_tp_vol: float = 0.0
        self._cumulative_volume: float = 0.0
        self._squared_deviations: list[float] = []
        self._last_date: datetime | None = None

    def reset(self) -> None:
        """Reset for new trading day."""
        self._cumulative_tp_vol = 0.0
        self._cumulative_volume = 0.0
        self._squared_deviations = []

    def update(self, bar: OHLCV) -> VWAPResult | None:
        """
        Update with new bar and return VWAP.

        Automatically resets on new day.
        """
        # Check for new day
        if self._last_date and bar.timestamp.date() != self._last_date.date():
            self.reset()
        self._last_date = bar.timestamp

        # Typical price
        typical_price = (bar.high + bar.low + bar.close) / 3

        # Accumulate
        self._cumulative_tp_vol += typical_price * bar.volume
        self._cumulative_volume += bar.volume

        if self._cumulative_volume == 0:
            return None

        vwap = self._cumulative_tp_vol / self._cumulative_volume

        # Track squared deviations for bands
        deviation = typical_price - vwap
        self._squared_deviations.append(deviation ** 2)

        # Calculate bands
        if self.use_bands and len(self._squared_deviations) > 1:
            variance = sum(self._squared_deviations) / len(self._squared_deviations)
            std_dev = math.sqrt(variance)
            upper = vwap + self.band_std * std_dev
            lower = vwap - self.band_std * std_dev
        else:
            upper = vwap
            lower = vwap

        deviation_pct = ((bar.close - vwap) / vwap * 100) if vwap > 0 else 0

        return VWAPResult(
            vwap=vwap,
            upper_band=upper,
            lower_band=lower,
            cumulative_volume=self._cumulative_volume,
            deviation_pct=deviation_pct,
        )


@dataclass
class OBVResult:
    """On Balance Volume result (Issue #Q15)."""
    obv: float
    obv_ema: float
    signal: str  # 'bullish', 'bearish', 'neutral'


class OBVCalculator:
    """
    On Balance Volume calculator (Issue #Q15).

    Cumulative volume indicator for trend confirmation.
    """

    def __init__(self, ema_period: int = 20):
        self.ema_period = ema_period

        self._obv: float = 0.0
        self._obv_ema: float = 0.0
        self._prev_close: float | None = None
        self._multiplier = 2 / (ema_period + 1)
        self._initialized: bool = False

    def update(self, close: float, volume: float) -> OBVResult | None:
        """
        Update with new close/volume and return OBV.
        """
        if self._prev_close is None:
            self._prev_close = close
            return None

        # Update OBV
        if close > self._prev_close:
            self._obv += volume
        elif close < self._prev_close:
            self._obv -= volume
        # else: OBV unchanged

        self._prev_close = close

        # Update EMA
        if not self._initialized:
            self._obv_ema = self._obv
            self._initialized = True
        else:
            self._obv_ema = (self._obv - self._obv_ema) * self._multiplier + self._obv_ema

        # Signal
        if self._obv > self._obv_ema * 1.02:
            signal = "bullish"
        elif self._obv < self._obv_ema * 0.98:
            signal = "bearish"
        else:
            signal = "neutral"

        return OBVResult(
            obv=self._obv,
            obv_ema=self._obv_ema,
            signal=signal,
        )


@dataclass
class MFIResult:
    """Money Flow Index result (Issue #Q15)."""
    mfi: float  # 0-100
    overbought: bool  # MFI > 80
    oversold: bool  # MFI < 20


class MFICalculator:
    """
    Money Flow Index calculator (Issue #Q15).

    Volume-weighted RSI that incorporates price and volume.
    """

    def __init__(self, period: int = 14):
        self.period = period

        self._typical_prices: deque = deque(maxlen=period + 1)
        self._volumes: deque = deque(maxlen=period + 1)

    def update(self, bar: OHLCV) -> MFIResult | None:
        """
        Update with new bar and return MFI.
        """
        typical_price = (bar.high + bar.low + bar.close) / 3
        self._typical_prices.append(typical_price)
        self._volumes.append(bar.volume)

        if len(self._typical_prices) < self.period + 1:
            return None

        # Calculate money flow for each period
        positive_flow = 0.0
        negative_flow = 0.0

        for i in range(1, len(self._typical_prices)):
            money_flow = self._typical_prices[i] * self._volumes[i]

            if self._typical_prices[i] > self._typical_prices[i-1]:
                positive_flow += money_flow
            elif self._typical_prices[i] < self._typical_prices[i-1]:
                negative_flow += money_flow

        # Calculate MFI
        if negative_flow == 0:
            mfi = 100
        else:
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))

        return MFIResult(
            mfi=mfi,
            overbought=mfi > 80,
            oversold=mfi < 20,
        )


@dataclass
class ATRResult:
    """Average True Range result."""
    atr: float
    atr_percent: float  # ATR as % of price


class ATRCalculator:
    """
    Average True Range calculator.

    Measures volatility.
    """

    def __init__(self, period: int = 14):
        self.period = period

        self._tr_values: deque = deque(maxlen=period)
        self._prev_close: float | None = None

    def update(self, bar: OHLCV) -> ATRResult | None:
        """
        Update with new bar and return ATR.
        """
        if self._prev_close is None:
            self._prev_close = bar.close
            return None

        # True Range
        tr = max(
            bar.high - bar.low,
            abs(bar.high - self._prev_close),
            abs(bar.low - self._prev_close)
        )
        self._tr_values.append(tr)
        self._prev_close = bar.close

        if len(self._tr_values) < self.period:
            return None

        atr = statistics.mean(self._tr_values)
        atr_percent = (atr / bar.close * 100) if bar.close > 0 else 0

        return ATRResult(atr=atr, atr_percent=atr_percent)


@dataclass
class StochasticResult:
    """Stochastic Oscillator result."""
    k: float  # Fast stochastic (0-100)
    d: float  # Slow stochastic (signal line)
    overbought: bool  # K > 80
    oversold: bool  # K < 20


class StochasticCalculator:
    """
    Stochastic Oscillator calculator.

    Momentum indicator comparing close to high-low range.
    """

    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period

        self._highs: deque = deque(maxlen=k_period)
        self._lows: deque = deque(maxlen=k_period)
        self._closes: deque = deque(maxlen=k_period)
        self._k_values: deque = deque(maxlen=d_period)

    def update(self, bar: OHLCV) -> StochasticResult | None:
        """
        Update with new bar and return Stochastic.
        """
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._closes.append(bar.close)

        if len(self._closes) < self.k_period:
            return None

        highest_high = max(self._highs)
        lowest_low = min(self._lows)

        # %K
        if highest_high == lowest_low:
            k = 50
        else:
            k = 100 * (bar.close - lowest_low) / (highest_high - lowest_low)

        self._k_values.append(k)

        # %D (SMA of %K)
        if len(self._k_values) < self.d_period:
            d = k
        else:
            d = statistics.mean(self._k_values)

        return StochasticResult(
            k=k,
            d=d,
            overbought=k > 80,
            oversold=k < 20,
        )


@dataclass
class KeltnerChannelResult:
    """Keltner Channel result."""
    upper: float
    middle: float  # EMA
    lower: float
    width_pct: float


class KeltnerChannelCalculator:
    """
    Keltner Channel calculator.

    EMA-based bands using ATR for width.
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 10, atr_mult: float = 2.0):
        self.ema_period = ema_period
        self.atr_mult = atr_mult

        self._ema: float | None = None
        self._ema_mult = 2 / (ema_period + 1)
        self._atr_calc = ATRCalculator(atr_period)

    def update(self, bar: OHLCV) -> KeltnerChannelResult | None:
        """
        Update with new bar and return Keltner Channel.
        """
        # Update EMA
        if self._ema is None:
            self._ema = bar.close
        else:
            self._ema = (bar.close - self._ema) * self._ema_mult + self._ema

        # Update ATR
        atr_result = self._atr_calc.update(bar)
        if atr_result is None:
            return None

        upper = self._ema + self.atr_mult * atr_result.atr
        lower = self._ema - self.atr_mult * atr_result.atr

        width_pct = ((upper - lower) / self._ema * 100) if self._ema > 0 else 0

        return KeltnerChannelResult(
            upper=upper,
            middle=self._ema,
            lower=lower,
            width_pct=width_pct,
        )


class TechnicalIndicatorSuite:
    """
    Comprehensive technical indicator suite.

    Combines multiple indicators for signal generation.
    """

    def __init__(
        self,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        mfi_period: int = 14,
        atr_period: int = 14,
    ):
        self.adx = ADXCalculator(adx_period)
        self.bollinger = BollingerBandsCalculator(bb_period, bb_std)
        self.vwap = VWAPCalculator()
        self.obv = OBVCalculator()
        self.mfi = MFICalculator(mfi_period)
        self.atr = ATRCalculator(atr_period)
        self.stochastic = StochasticCalculator()
        self.keltner = KeltnerChannelCalculator()

    def update(self, bar: OHLCV) -> dict[str, Any]:
        """
        Update all indicators with new bar.

        Returns dictionary of indicator results.
        """
        results = {}

        # Update each indicator
        adx_result = self.adx.update(bar)
        if adx_result:
            results['adx'] = {
                'value': adx_result.adx,
                'plus_di': adx_result.plus_di,
                'minus_di': adx_result.minus_di,
                'trend_strength': adx_result.trend_strength,
                'trend_direction': adx_result.trend_direction,
            }

        bb_result = self.bollinger.update(bar.close)
        if bb_result:
            results['bollinger'] = {
                'upper': bb_result.upper,
                'middle': bb_result.middle,
                'lower': bb_result.lower,
                'bandwidth': bb_result.bandwidth,
                'percent_b': bb_result.percent_b,
                'is_squeeze': bb_result.is_squeeze,
                'position': bb_result.position,
            }

        vwap_result = self.vwap.update(bar)
        if vwap_result:
            results['vwap'] = {
                'value': vwap_result.vwap,
                'upper': vwap_result.upper_band,
                'lower': vwap_result.lower_band,
                'deviation_pct': vwap_result.deviation_pct,
            }

        obv_result = self.obv.update(bar.close, bar.volume)
        if obv_result:
            results['obv'] = {
                'value': obv_result.obv,
                'ema': obv_result.obv_ema,
                'signal': obv_result.signal,
            }

        mfi_result = self.mfi.update(bar)
        if mfi_result:
            results['mfi'] = {
                'value': mfi_result.mfi,
                'overbought': mfi_result.overbought,
                'oversold': mfi_result.oversold,
            }

        atr_result = self.atr.update(bar)
        if atr_result:
            results['atr'] = {
                'value': atr_result.atr,
                'percent': atr_result.atr_percent,
            }

        stoch_result = self.stochastic.update(bar)
        if stoch_result:
            results['stochastic'] = {
                'k': stoch_result.k,
                'd': stoch_result.d,
                'overbought': stoch_result.overbought,
                'oversold': stoch_result.oversold,
            }

        keltner_result = self.keltner.update(bar)
        if keltner_result:
            results['keltner'] = {
                'upper': keltner_result.upper,
                'middle': keltner_result.middle,
                'lower': keltner_result.lower,
                'width_pct': keltner_result.width_pct,
            }

        return results

    def get_composite_signal(self, bar: OHLCV) -> dict[str, Any]:
        """
        Get composite trading signal from all indicators.

        Returns signal strength and direction.
        """
        indicators = self.update(bar)

        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # ADX trend
        if 'adx' in indicators:
            total_signals += 1
            if indicators['adx']['trend_direction'] == 'bullish' and indicators['adx']['value'] > 20:
                bullish_signals += 1
            elif indicators['adx']['trend_direction'] == 'bearish' and indicators['adx']['value'] > 20:
                bearish_signals += 1

        # Bollinger position
        if 'bollinger' in indicators:
            total_signals += 1
            pos = indicators['bollinger']['position']
            if pos == 'near_lower':
                bullish_signals += 1  # Mean reversion long
            elif pos == 'near_upper':
                bearish_signals += 1  # Mean reversion short

        # MFI
        if 'mfi' in indicators:
            total_signals += 1
            if indicators['mfi']['oversold']:
                bullish_signals += 1
            elif indicators['mfi']['overbought']:
                bearish_signals += 1

        # OBV
        if 'obv' in indicators:
            total_signals += 1
            if indicators['obv']['signal'] == 'bullish':
                bullish_signals += 1
            elif indicators['obv']['signal'] == 'bearish':
                bearish_signals += 1

        # Stochastic
        if 'stochastic' in indicators:
            total_signals += 1
            if indicators['stochastic']['oversold']:
                bullish_signals += 1
            elif indicators['stochastic']['overbought']:
                bearish_signals += 1

        # Calculate composite
        if total_signals == 0:
            return {'direction': 'neutral', 'strength': 0, 'confidence': 0}

        net_signal = bullish_signals - bearish_signals

        if net_signal > 0:
            direction = 'bullish'
            strength = bullish_signals / total_signals
        elif net_signal < 0:
            direction = 'bearish'
            strength = bearish_signals / total_signals
        else:
            direction = 'neutral'
            strength = 0

        return {
            'direction': direction,
            'strength': strength,
            'confidence': abs(net_signal) / total_signals,
            'bullish_count': bullish_signals,
            'bearish_count': bearish_signals,
            'total_indicators': total_signals,
            'indicators': indicators,
        }

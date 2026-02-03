"""
Momentum Strategy
=================

Implements momentum and trend-following logic.

MATURITY: BETA
--------------
Status: Core functionality implemented and tested
- [x] MA crossovers implemented
- [x] RSI calculation (Wilder's smoothing)
- [x] MACD with proper signal line
- [x] Stop-loss calculation (ATR-based)
- [x] ADX trend strength filter
- [x] Volatility-scaled position sizing
- [x] Event calendar blackout periods
- [ ] Volume-weighted indicators (TODO)
- [ ] Ichimoku Cloud (TODO)

Production Readiness:
- Unit tests: Partial coverage
- Backtesting: Not yet performed
- Live testing: Not yet performed

Use in production: WITH CAUTION - verify signals manually before trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# ISSUE_001: Event calendar for blackout periods
# FOMC meeting dates for 2024-2025 (should be updated annually)
FOMC_DATES = [
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
]

# Common high-impact economic events (blackout windows in hours before/after)
EVENT_BLACKOUT_WINDOWS = {
    "FOMC": {"before_hours": 24, "after_hours": 4},
    "NFP": {"before_hours": 12, "after_hours": 2},  # Non-Farm Payrolls
    "CPI": {"before_hours": 12, "after_hours": 2},
    "EARNINGS": {"before_hours": 24, "after_hours": 4},
}


@dataclass
class MomentumSignal:
    """Momentum signal output."""
    symbol: str
    direction: str  # "long", "short", "flat"
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    indicators: dict[str, float]
    # P1-13: Stop-loss levels to limit downside risk
    stop_loss_price: float | None = None  # Absolute price level for stop
    stop_loss_pct: float | None = None    # Percentage from entry
    # MM-P1-1: Volatility-scaled position size
    position_size_scalar: float = 1.0     # Multiply base size by this
    # MM-P1-2: Trend strength from ADX
    trend_strength: str = "unknown"       # "strong", "moderate", "weak", "unknown"
    # ISSUE_001: Blackout indicator
    blackout_active: bool = False         # True if in event blackout period
    blackout_reason: str | None = None    # Reason for blackout
    # P2: Divergence detection
    divergence_type: str | None = None    # "bullish", "bearish", None
    divergence_strength: float = 0.0      # 0 to 1
    # P2: Volume confirmation
    volume_confirmed: bool = False        # True if volume confirms signal
    volume_ratio: float = 1.0             # Current volume / average volume


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

        # P1-13: Stop-loss settings
        self._stop_loss_atr_multiplier = config.get("stop_loss_atr_multiplier", 2.0)
        self._stop_loss_pct = config.get("stop_loss_pct", 2.0)  # Default 2%
        self._use_atr_stop = config.get("use_atr_stop", True)  # ATR-based by default
        self._atr_period = config.get("atr_period", 14)

        # MM-P1-1: Volatility scaling settings
        self._vol_target = config.get("vol_target", 0.15)  # 15% annualized vol target
        self._vol_lookback = config.get("vol_lookback", 20)  # Days for realized vol calc
        self._min_position_scalar = config.get("min_position_scalar", 0.25)
        self._max_position_scalar = config.get("max_position_scalar", 2.0)

        # MM-P1-2: ADX settings
        self._adx_period = config.get("adx_period", 14)
        self._adx_strong_threshold = config.get("adx_strong_threshold", 25)
        self._adx_moderate_threshold = config.get("adx_moderate_threshold", 20)

        # ISSUE_001: Event calendar settings
        self._use_event_blackout = config.get("use_event_blackout", True)
        self._earnings_calendar: dict[str, list[str]] = config.get("earnings_calendar", {})
        self._custom_blackout_dates: list[str] = config.get("custom_blackout_dates", [])

        # P2: Divergence detection settings
        self._divergence_lookback = config.get("divergence_lookback", 14)
        self._divergence_min_swing = config.get("divergence_min_swing", 0.02)  # 2% min swing

        # P2: Volume confirmation settings
        self._volume_lookback = config.get("volume_confirmation_lookback", 20)
        self._volume_threshold = config.get("volume_threshold", 1.2)  # 20% above average

        # P2: Trend strength filter settings
        self._require_trend_confirmation = config.get("require_trend_confirmation", True)
        self._min_trend_strength_for_entry = config.get("min_trend_strength", "moderate")

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

        # Get current values - check array is not empty before accessing
        if len(valid_macd) == 0:
            return 0.0, 0.0, 0.0
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

    def calculate_atr(
        self,
        prices: np.ndarray,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        period: int = 14
    ) -> float:
        """
        P1-13: Calculate Average True Range for stop-loss placement.

        If highs/lows not provided, estimates from price changes.
        """
        if len(prices) < period + 1:
            return 0.0

        if highs is not None and lows is not None and len(highs) >= period:
            # Use actual high/low/close data
            true_ranges = []
            for i in range(1, min(len(prices), len(highs), len(lows))):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prices[i-1]),
                    abs(lows[i] - prices[i-1])
                )
                true_ranges.append(tr)

            if len(true_ranges) >= period:
                return np.mean(true_ranges[-period:])

        # Fallback: estimate from close prices
        # Use absolute daily changes as proxy for volatility
        changes = np.abs(np.diff(prices[-period-1:]))
        return np.mean(changes) * 1.5  # Scale up since we're missing high/low

    def calculate_stop_loss(
        self,
        current_price: float,
        direction: str,
        atr: float
    ) -> tuple[float, float]:
        """
        P1-13: Calculate stop-loss price and percentage.

        Args:
            current_price: Entry price
            direction: "long" or "short"
            atr: Average True Range

        Returns:
            (stop_price, stop_pct)
        """
        if self._use_atr_stop and atr > 0:
            stop_distance = atr * self._stop_loss_atr_multiplier
        else:
            stop_distance = current_price * (self._stop_loss_pct / 100)

        if direction == "long":
            stop_price = current_price - stop_distance
        elif direction == "short":
            stop_price = current_price + stop_distance
        else:
            return None, None

        stop_pct = (stop_distance / current_price) * 100
        return stop_price, stop_pct

    def is_blackout_period(
        self,
        symbol: str,
        timestamp: datetime | None = None
    ) -> tuple[bool, str | None]:
        """
        ISSUE_001: Check if we're in an event blackout period.

        Args:
            symbol: Stock symbol to check
            timestamp: Time to check (defaults to now)

        Returns:
            (is_blackout, reason) - True if in blackout, with reason string
        """
        if not self._use_event_blackout:
            return False, None

        if timestamp is None:
            timestamp = datetime.now()

        # Check FOMC dates
        fomc_window = EVENT_BLACKOUT_WINDOWS["FOMC"]
        for fomc_date_str in FOMC_DATES:
            fomc_date = datetime.strptime(fomc_date_str, "%Y-%m-%d")
            # Set to 2pm ET (typical FOMC announcement time)
            fomc_time = fomc_date.replace(hour=14, minute=0)

            blackout_start = fomc_time - timedelta(hours=fomc_window["before_hours"])
            blackout_end = fomc_time + timedelta(hours=fomc_window["after_hours"])

            if blackout_start <= timestamp <= blackout_end:
                return True, f"FOMC meeting on {fomc_date_str}"

        # Check custom blackout dates
        for blackout_date_str in self._custom_blackout_dates:
            try:
                blackout_date = datetime.strptime(blackout_date_str, "%Y-%m-%d")
                if blackout_date.date() == timestamp.date():
                    return True, f"Custom blackout on {blackout_date_str}"
            except ValueError:
                logger.warning(f"Invalid blackout date format: {blackout_date_str}")

        # Check earnings calendar for this symbol
        if symbol in self._earnings_calendar:
            earnings_window = EVENT_BLACKOUT_WINDOWS["EARNINGS"]
            for earnings_date_str in self._earnings_calendar[symbol]:
                try:
                    earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
                    # Assume earnings after market close
                    earnings_time = earnings_date.replace(hour=16, minute=0)

                    blackout_start = earnings_time - timedelta(hours=earnings_window["before_hours"])
                    blackout_end = earnings_time + timedelta(hours=earnings_window["after_hours"])

                    if blackout_start <= timestamp <= blackout_end:
                        return True, f"Earnings for {symbol} on {earnings_date_str}"
                except ValueError:
                    logger.warning(f"Invalid earnings date format for {symbol}: {earnings_date_str}")

        return False, None

    def calculate_realized_volatility(
        self,
        prices: np.ndarray,
        lookback: int | None = None
    ) -> float:
        """
        MM-P1-1: Calculate annualized realized volatility.

        Args:
            prices: Price series
            lookback: Number of days (defaults to _vol_lookback)

        Returns:
            Annualized volatility (e.g., 0.20 for 20%)
        """
        if lookback is None:
            lookback = self._vol_lookback

        if len(prices) < lookback + 1:
            return 0.0

        # Calculate log returns
        recent_prices = prices[-lookback - 1:]
        log_returns = np.log(recent_prices[1:] / recent_prices[:-1])

        # Daily volatility
        daily_vol = np.std(log_returns)

        # Annualize (assuming 252 trading days)
        annualized_vol = daily_vol * np.sqrt(252)

        return annualized_vol

    def calculate_position_size_scalar(
        self,
        realized_vol: float
    ) -> float:
        """
        MM-P1-1: Calculate position size scalar based on volatility targeting.

        Position size is scaled inversely to volatility to maintain
        consistent risk across different vol regimes.

        Args:
            realized_vol: Annualized realized volatility

        Returns:
            Scalar to multiply base position size by (clamped to min/max)
        """
        if realized_vol <= 0:
            return 1.0

        # Scale position inversely to volatility
        # Higher vol = smaller position, lower vol = larger position
        raw_scalar = self._vol_target / realized_vol

        # Clamp to reasonable bounds
        scalar = max(self._min_position_scalar, min(self._max_position_scalar, raw_scalar))

        return scalar

    def calculate_adx(
        self,
        prices: np.ndarray,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        period: int | None = None
    ) -> tuple[float, str]:
        """
        MM-P1-2: Calculate Average Directional Index for trend strength.

        Args:
            prices: Close prices
            highs: High prices (optional, estimated if not provided)
            lows: Low prices (optional, estimated if not provided)
            period: ADX period (defaults to _adx_period)

        Returns:
            (adx_value, trend_strength) where trend_strength is
            "strong", "moderate", "weak", or "unknown"
        """
        if period is None:
            period = self._adx_period

        min_required = period * 2 + 1
        if len(prices) < min_required:
            return 0.0, "unknown"

        # Estimate highs/lows if not provided
        if highs is None or lows is None:
            # Use rolling max/min as proxy
            highs = np.zeros(len(prices))
            lows = np.zeros(len(prices))
            for i in range(1, len(prices)):
                # Estimate high/low from price movement
                change = prices[i] - prices[i-1]
                if change > 0:
                    highs[i] = prices[i]
                    lows[i] = prices[i-1]
                else:
                    highs[i] = prices[i-1]
                    lows[i] = prices[i]
            highs[0] = prices[0]
            lows[0] = prices[0]

        n = len(prices)

        # Calculate True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - prices[i-1]),
                abs(lows[i] - prices[i-1])
            )

        # Calculate +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth with Wilder's method (alpha = 1/period)
        alpha = 1.0 / period

        atr = np.zeros(n)
        smooth_plus_dm = np.zeros(n)
        smooth_minus_dm = np.zeros(n)

        # Initialize with simple average
        atr[period] = np.mean(tr[1:period+1])
        smooth_plus_dm[period] = np.mean(plus_dm[1:period+1])
        smooth_minus_dm[period] = np.mean(minus_dm[1:period+1])

        # Apply Wilder's smoothing
        for i in range(period + 1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            smooth_plus_dm[i] = alpha * plus_dm[i] + (1 - alpha) * smooth_plus_dm[i-1]
            smooth_minus_dm[i] = alpha * minus_dm[i] + (1 - alpha) * smooth_minus_dm[i-1]

        # Calculate +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * smooth_plus_dm[i] / atr[i]
                minus_di[i] = 100 * smooth_minus_dm[i] / atr[i]

        # Calculate DX
        dx = np.zeros(n)
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # Calculate ADX (smoothed DX)
        if n < period * 2:
            return 0.0, "unknown"

        adx = np.mean(dx[period:period*2])  # Initial ADX
        for i in range(period * 2, n):
            adx = alpha * dx[i] + (1 - alpha) * adx

        # Determine trend strength
        if adx >= self._adx_strong_threshold:
            trend_strength = "strong"
        elif adx >= self._adx_moderate_threshold:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"

        return adx, trend_strength

    def detect_divergence(
        self,
        prices: np.ndarray,
        rsi_values: np.ndarray | None = None,
        lookback: int | None = None
    ) -> tuple[str | None, float]:
        """
        P2: Detect price/RSI divergence.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high

        Args:
            prices: Price series
            rsi_values: Pre-calculated RSI series (optional, will calculate if None)
            lookback: Lookback period for swing detection

        Returns:
            (divergence_type, strength) - "bullish", "bearish", or None
        """
        if lookback is None:
            lookback = self._divergence_lookback

        if len(prices) < lookback + self._rsi_period:
            return None, 0.0

        # Calculate RSI series if not provided
        if rsi_values is None:
            rsi_values = self._calculate_rsi_series(prices)

        if len(rsi_values) < lookback:
            return None, 0.0

        # Get recent price and RSI windows
        recent_prices = prices[-lookback:]
        recent_rsi = rsi_values[-lookback:]

        # Find swing highs and lows
        price_highs, price_lows = self._find_swing_points(recent_prices)
        rsi_highs, rsi_lows = self._find_swing_points(recent_rsi)

        # Check for bullish divergence (price lower low, RSI higher low)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            last_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            last_rsi_low = rsi_lows[-1]
            prev_rsi_low = rsi_lows[-2]

            price_lower = recent_prices[last_price_low] < recent_prices[prev_price_low]
            rsi_higher = recent_rsi[last_rsi_low] > recent_rsi[prev_rsi_low]

            if price_lower and rsi_higher:
                # Calculate divergence strength
                price_diff = (recent_prices[prev_price_low] - recent_prices[last_price_low]) / recent_prices[prev_price_low]
                rsi_diff = (recent_rsi[last_rsi_low] - recent_rsi[prev_rsi_low]) / 100
                strength = min(1.0, (price_diff + rsi_diff) / 0.1)  # Normalize to 0-1
                return "bullish", strength

        # Check for bearish divergence (price higher high, RSI lower high)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            last_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            last_rsi_high = rsi_highs[-1]
            prev_rsi_high = rsi_highs[-2]

            price_higher = recent_prices[last_price_high] > recent_prices[prev_price_high]
            rsi_lower = recent_rsi[last_rsi_high] < recent_rsi[prev_rsi_high]

            if price_higher and rsi_lower:
                price_diff = (recent_prices[last_price_high] - recent_prices[prev_price_high]) / recent_prices[prev_price_high]
                rsi_diff = (recent_rsi[prev_rsi_high] - recent_rsi[last_rsi_high]) / 100
                strength = min(1.0, (price_diff + rsi_diff) / 0.1)
                return "bearish", strength

        return None, 0.0

    def _calculate_rsi_series(self, prices: np.ndarray) -> np.ndarray:
        """Calculate RSI series for divergence detection."""
        if len(prices) < self._rsi_period + 1:
            return np.array([50.0])

        rsi_series = []
        for i in range(self._rsi_period + 1, len(prices) + 1):
            rsi = self.calculate_rsi(prices[:i], self._rsi_period)
            rsi_series.append(rsi)

        return np.array(rsi_series)

    def _find_swing_points(
        self,
        series: np.ndarray,
        window: int = 3
    ) -> tuple[list[int], list[int]]:
        """
        Find swing high and low indices in a series.

        Args:
            series: Data series
            window: Number of bars on each side to confirm swing

        Returns:
            (high_indices, low_indices)
        """
        highs = []
        lows = []

        for i in range(window, len(series) - window):
            is_high = True
            is_low = True

            for j in range(1, window + 1):
                if series[i] <= series[i - j] or series[i] <= series[i + j]:
                    is_high = False
                if series[i] >= series[i - j] or series[i] >= series[i + j]:
                    is_low = False

            if is_high:
                highs.append(i)
            if is_low:
                lows.append(i)

        return highs, lows

    def calculate_volume_confirmation(
        self,
        volumes: np.ndarray,
        direction: str,
        lookback: int | None = None
    ) -> tuple[bool, float]:
        """
        P2: Check if volume confirms the signal direction.

        For long signals: volume should be above average on up moves
        For short signals: volume should be above average on down moves

        Args:
            volumes: Volume series
            direction: Signal direction ("long", "short", "flat")
            lookback: Period for average volume calculation

        Returns:
            (is_confirmed, volume_ratio)
        """
        if lookback is None:
            lookback = self._volume_lookback

        if len(volumes) < lookback + 1:
            return False, 1.0

        # Calculate average volume
        avg_volume = np.mean(volumes[-lookback-1:-1])
        if avg_volume <= 0:
            return False, 1.0

        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume

        # Volume should be above threshold for confirmation
        is_confirmed = volume_ratio >= self._volume_threshold

        # For flat direction, no confirmation needed
        if direction == "flat":
            return True, volume_ratio

        return is_confirmed, volume_ratio

    def filter_by_trend_strength(
        self,
        direction: str,
        trend_strength: str,
        adx_value: float
    ) -> tuple[str, float]:
        """
        P2: Filter signals based on trend strength.

        Weak trends should not generate entry signals.
        Moderate/strong trends pass through.

        Args:
            direction: Original signal direction
            trend_strength: ADX-based trend strength
            adx_value: Raw ADX value

        Returns:
            (filtered_direction, confidence_adjustment)
        """
        if not self._require_trend_confirmation:
            return direction, 1.0

        if direction == "flat":
            return direction, 1.0

        min_strength = self._min_trend_strength_for_entry

        # Map strength levels to numeric values for comparison
        strength_levels = {"weak": 0, "moderate": 1, "strong": 2, "unknown": -1}
        current_level = strength_levels.get(trend_strength, -1)
        required_level = strength_levels.get(min_strength, 1)

        if current_level < required_level:
            # Trend too weak - suppress signal
            logger.debug(
                f"Signal {direction} suppressed: trend_strength={trend_strength} "
                f"< required={min_strength} (ADX={adx_value:.1f})"
            )
            return "flat", 0.5

        # Adjust confidence based on trend strength
        confidence_adjustment = 0.7 + (current_level * 0.15)  # 0.7, 0.85, 1.0

        return direction, confidence_adjustment

    def analyze(self, symbol: str, prices: np.ndarray, timestamp: datetime | None = None, volumes: np.ndarray | None = None) -> MomentumSignal:
        """
        Analyze price series and generate momentum signal.

        Args:
            symbol: Trading symbol
            prices: Price series (numpy array)
            timestamp: Current timestamp for blackout checking (defaults to now)
            volumes: Volume series (optional, for volume confirmation)

        Returns:
            MomentumSignal with direction, strength, and risk parameters
        """
        # ISSUE_001: Check for event blackout period FIRST
        blackout_active, blackout_reason = self.is_blackout_period(symbol, timestamp)

        if len(prices) < self._slow_period:
            return MomentumSignal(
                symbol=symbol,
                direction="flat",
                strength=0.0,
                confidence=0.0,
                indicators={},
                blackout_active=blackout_active,
                blackout_reason=blackout_reason,
                divergence_type=None,
                divergence_strength=0.0,
                volume_confirmed=False,
                volume_ratio=1.0,
            )

        # Calculate indicators
        fast_ma = self.calculate_sma(prices, self._fast_period)
        slow_ma = self.calculate_sma(prices, self._slow_period)
        rsi = self.calculate_rsi(prices, self._rsi_period)
        macd, macd_signal, macd_hist = self.calculate_macd(prices)
        roc = self.calculate_roc(prices)

        # MM-P1-2: Calculate ADX for trend strength
        adx, trend_strength = self.calculate_adx(prices)

        # MM-P1-1: Calculate realized volatility and position scalar
        realized_vol = self.calculate_realized_volatility(prices)
        position_size_scalar = self.calculate_position_size_scalar(realized_vol)

        # P2: Detect divergence (price vs RSI)
        divergence_type, divergence_strength = self.detect_divergence(prices)

        # P2: Volume confirmation
        volume_confirmed = False
        volume_ratio = 1.0
        if volumes is not None and len(volumes) > 0:
            volume_confirmed, volume_ratio = self.calculate_volume_confirmation(
                volumes, "long" if sum(1 for s in [1, 0, 0, 0] if s > 0) > 0 else "short"
            )

        indicators = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "roc": roc,
            "adx": adx,
            "realized_vol": realized_vol,
            "position_size_scalar": position_size_scalar,
            "divergence_type": divergence_type,
            "divergence_strength": divergence_strength,
            "volume_confirmed": volume_confirmed,
            "volume_ratio": volume_ratio,
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

        # P2: Apply divergence signals
        # Bullish divergence can strengthen long signals or weaken short signals
        # Bearish divergence can strengthen short signals or weaken long signals
        if divergence_type == "bullish" and divergence_strength > 0.3:
            if direction == "short":
                # Bullish divergence contradicts short - reduce strength
                strength *= (1.0 - divergence_strength * 0.5)
            elif direction == "flat" and divergence_strength > 0.5:
                # Strong bullish divergence - consider long
                direction = "long"
                strength = divergence_strength * 0.5
        elif divergence_type == "bearish" and divergence_strength > 0.3:
            if direction == "long":
                # Bearish divergence contradicts long - reduce strength
                strength *= (1.0 - divergence_strength * 0.5)
            elif direction == "flat" and divergence_strength > 0.5:
                # Strong bearish divergence - consider short
                direction = "short"
                strength = -divergence_strength * 0.5

        # P2: Apply trend strength filter
        direction, trend_confidence_adj = self.filter_by_trend_strength(
            direction, trend_strength, adx
        )

        # P2: Volume confirmation affects signal (re-calculate now that we know direction)
        if volumes is not None and len(volumes) > 0:
            volume_confirmed, volume_ratio = self.calculate_volume_confirmation(
                volumes, direction
            )

        # ISSUE_001: Force flat direction during blackout periods
        if blackout_active:
            logger.info(f"Blackout active for {symbol}: {blackout_reason} - suppressing signal")
            direction = "flat"
            # Keep strength/confidence for informational purposes but signal is suppressed

        # MM-P1-2: Reduce confidence if trend is weak (ADX < threshold)
        # Strong trends warrant higher conviction signals
        confidence_adjustment = 1.0
        if trend_strength == "weak":
            confidence_adjustment = 0.7
        elif trend_strength == "moderate":
            confidence_adjustment = 0.85

        # Confidence based on agreement, adjusted for trend strength
        agreement = sum(1 for s in scores if s == np.sign(total_score)) / len(scores)
        confidence = agreement * confidence_adjustment * trend_confidence_adj

        # P2: Volume confirmation boosts or reduces confidence
        if volumes is not None and direction != "flat":
            if volume_confirmed:
                confidence = min(1.0, confidence * 1.1)  # 10% boost
            else:
                confidence *= 0.85  # 15% reduction without volume confirmation

        # P1-13: Calculate stop-loss if we have a directional signal
        stop_loss_price = None
        stop_loss_pct = None
        if direction != "flat":
            current_price = prices[-1]
            atr = self.calculate_atr(prices, period=self._atr_period)
            stop_loss_price, stop_loss_pct = self.calculate_stop_loss(
                current_price, direction, atr
            )
            indicators["atr"] = atr
            indicators["stop_loss_price"] = stop_loss_price
            indicators["stop_loss_pct"] = stop_loss_pct

        return MomentumSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            indicators=indicators,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            position_size_scalar=position_size_scalar,
            trend_strength=trend_strength,
            blackout_active=blackout_active,
            blackout_reason=blackout_reason,
            divergence_type=divergence_type,
            divergence_strength=divergence_strength,
            volume_confirmed=volume_confirmed,
            volume_ratio=volume_ratio,
        )

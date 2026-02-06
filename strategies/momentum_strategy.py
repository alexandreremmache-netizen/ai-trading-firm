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
- [x] 52-Week High/Low signal (P3 implementation)
- [x] Dual Momentum - Antonacci (P3 implementation)
- [x] Cross-Sectional Momentum (P3 implementation)
- [x] Multi-Timeframe Analysis (P4 implementation)
- [x] Volume-weighted indicators (see core/volume_indicators.py)
- [x] Ichimoku Cloud (see strategies/ichimoku_strategy.py)

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
    direction: str  # "long", "short", "flat", "exit_long", "exit_short"
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    indicators: dict[str, float]
    # P1-13: Stop-loss levels to limit downside risk
    stop_loss_price: float | None = None  # Absolute price level for stop
    stop_loss_pct: float | None = None    # Percentage from entry
    # RISK-001: Take-profit levels
    take_profit_price: float | None = None  # Absolute price level for TP
    take_profit_pct: float | None = None    # Percentage from entry
    # RISK-002: Maximum holding period
    max_holding_bars: int = 20              # Exit after this many bars
    # RISK-003: Strategy-level risk limit
    strategy_max_loss_pct: float = 5.0      # Max loss per strategy (%)
    # MM-P1-1: Volatility-scaled position size
    position_size_scalar: float = 1.0     # Multiply base size by this
    # MM-P1-2: Trend strength from ADX
    trend_strength: str = "unknown"       # "strong", "moderate", "weak", "unknown"
    # RISK-004: Market regime
    market_regime: str = "unknown"        # "trending", "ranging", "volatile", "unknown"
    regime_suitable: bool = True          # False if regime unfavorable for momentum
    # ISSUE_001: Blackout indicator
    blackout_active: bool = False         # True if in event blackout period
    blackout_reason: str | None = None    # Reason for blackout
    # P2: Divergence detection
    divergence_type: str | None = None    # "bullish", "bearish", None
    divergence_strength: float = 0.0      # 0 to 1
    # P2: Volume confirmation
    volume_confirmed: bool = False        # True if volume confirms signal
    volume_ratio: float = 1.0             # Current volume / average volume
    # RISK-005: Exit signal information
    is_exit_signal: bool = False          # True if this is an exit signal
    exit_reason: str | None = None        # Reason for exit signal
    # P3: 52-Week High/Low signal
    week52_signal: float = 0.0            # 1 near high, -1 near low, 0 neutral
    week52_proximity: float = 0.5         # 0-1 position within 52-week range


class MomentumStrategy:
    """
    Momentum Strategy Implementation.

    Implements:
    1. Moving average crossovers
    2. RSI (Relative Strength Index)
    3. MACD (Moving Average Convergence Divergence)
    4. Rate of Change (ROC)
    5. ADX for trend strength
    6. 52-Week High/Low signal
    7. Multi-timeframe momentum analysis

    Related modules:
    - Volume indicators: core/volume_indicators.py (VWMA, VWAP, OBV, etc.)
    - Ichimoku Cloud: strategies/ichimoku_strategy.py
    - Bollinger Bands: core/technical_indicators.py
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

        # QUICK WIN #2: Asset-specific RSI overrides
        self._rsi_overrides = config.get("rsi_overrides", {})

        # QUICK WIN #5: Risk-Reward filter
        self._min_risk_reward_ratio = config.get("min_risk_reward_ratio", 2.0)

        # QUICK WIN #6: Triple confirmation filter
        self._use_triple_confirmation = config.get("use_triple_confirmation", True)
        self._min_confirmations = config.get("min_confirmations", 2)

        # QUICK WIN #10: Strict ADX filter enforcement
        self._require_strong_trend = config.get("require_strong_trend", True)
        self._suppress_weak_trend = config.get("suppress_weak_trend", True)

        # RISK-001: Take-profit settings
        self._take_profit_atr_multiplier = config.get("take_profit_atr_multiplier", 3.0)
        self._take_profit_pct = config.get("take_profit_pct", 4.0)  # Default 4%
        self._use_atr_take_profit = config.get("use_atr_take_profit", True)

        # RISK-002: Maximum holding period
        self._max_holding_bars = config.get("max_holding_bars", 20)  # Exit after N bars

        # RISK-003: Strategy-level risk limit
        self._strategy_max_loss_pct = config.get("strategy_max_loss_pct", 5.0)  # 5% max loss

        # RISK-004: Regime detection settings
        self._ranging_adx_threshold = config.get("ranging_adx_threshold", 20)  # ADX below = ranging
        self._volatile_regime_vol_multiplier = config.get("volatile_regime_vol_multiplier", 1.5)
        self._suppress_in_ranging = config.get("suppress_in_ranging", True)  # Don't trade in ranging

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

    def calculate_52week_signal(
        self,
        prices: np.ndarray,
        lookback: int = 252,
        high_threshold: float = 0.90,
        low_threshold: float = 0.10
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate proximity to 52-week high/low for momentum signal.

        Research finding: Stocks near 52-week highs outperform by 0.5%/month.
        This is known as the "52-week high effect" documented by George & Hwang (2004).

        The intuition is that psychological anchoring to 52-week highs causes
        underreaction to positive news, leading to predictable outperformance.

        Args:
            prices: Price series (at least 252 trading days recommended)
            lookback: Number of bars for high/low calculation (252 = ~1 year)
            high_threshold: Proximity to high required for bullish signal (0.90 = within 10%)
            low_threshold: Proximity to low required for bearish signal (0.10 = within 10%)

        Returns:
            (signal, details) where:
            - signal: 1 if near 52-week high, -1 if near 52-week low, 0 otherwise
            - details: dict with 52w_high, 52w_low, current_price, proximity_to_high
        """
        if len(prices) < 20:  # Minimum required data
            return 0.0, {
                "52w_high": 0.0,
                "52w_low": 0.0,
                "current_price": prices[-1] if len(prices) > 0 else 0.0,
                "proximity_to_high": 0.0,
                "proximity_to_low": 0.0,
                "signal_reason": "insufficient_data"
            }

        # Use available data up to lookback period
        effective_lookback = min(lookback, len(prices))
        lookback_prices = prices[-effective_lookback:]

        high_52w = np.max(lookback_prices)
        low_52w = np.min(lookback_prices)
        current_price = prices[-1]

        # Calculate proximity: 1.0 means at 52w high, 0.0 means at 52w low
        price_range = high_52w - low_52w
        if price_range > 0:
            proximity_to_high = (current_price - low_52w) / price_range
        else:
            proximity_to_high = 0.5  # No range = neutral

        proximity_to_low = 1.0 - proximity_to_high

        # Determine signal based on proximity thresholds
        signal = 0.0
        signal_reason = "neutral_zone"

        if proximity_to_high >= high_threshold:
            # Near 52-week high - bullish momentum signal
            # Stocks breaking to new highs tend to continue higher
            signal = 1.0
            signal_reason = "near_52w_high"
        elif proximity_to_low >= (1.0 - low_threshold):
            # Near 52-week low - bearish momentum signal
            # Stocks near lows often continue lower (momentum effect)
            signal = -1.0
            signal_reason = "near_52w_low"

        details = {
            "52w_high": high_52w,
            "52w_low": low_52w,
            "current_price": current_price,
            "proximity_to_high": proximity_to_high,
            "proximity_to_low": proximity_to_low,
            "lookback_days": effective_lookback,
            "signal_reason": signal_reason
        }

        return signal, details

    def calculate_cross_sectional_momentum(
        self,
        price_data: dict[str, np.ndarray],
        lookback_long: int = 252,
        lookback_short: int = 21,
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
        skip_recent: int = 21
    ) -> dict[str, dict[str, Any]]:
        """
        Calculate cross-sectional momentum rankings.

        Cross-sectional momentum (Jegadeesh & Titman, 1993) ranks assets
        by past returns and goes long winners, short losers.

        The standard approach uses 12-month returns with 1-month skip
        (to avoid short-term reversal effect).

        Research finding: Top decile outperforms bottom by ~1%/month historically.

        Args:
            price_data: Dict of {symbol: price_array} for all assets
            lookback_long: Long lookback period (252 = 12 months)
            lookback_short: Short lookback period (21 = 1 month) to skip
            top_pct: Percentage of top performers to go long (0.2 = top 20%)
            bottom_pct: Percentage of bottom performers to short (0.2 = bottom 20%)
            skip_recent: Skip most recent N days (reversal avoidance)

        Returns:
            Dict of {symbol: {signal, momentum_score, rank, percentile, bucket}}
        """
        results = {}

        if len(price_data) < 3:
            return results

        # Calculate momentum score for each asset
        momentum_scores = {}
        for symbol, prices in price_data.items():
            if len(prices) < lookback_long + skip_recent + 1:
                continue

            # Standard momentum: return over lookback period, skipping recent days
            # This avoids short-term reversal effect
            end_idx = -skip_recent if skip_recent > 0 else len(prices)
            start_idx = end_idx - lookback_long

            if isinstance(end_idx, int) and end_idx < 0:
                end_price = prices[end_idx]
            else:
                end_price = prices[-skip_recent - 1] if skip_recent > 0 else prices[-1]

            start_price = prices[start_idx]

            if start_price <= 0:
                continue

            # Calculate momentum as simple return
            momentum = (end_price - start_price) / start_price

            momentum_scores[symbol] = {
                "momentum_score": momentum,
                "start_price": start_price,
                "end_price": end_price,
                "lookback_days": lookback_long,
                "skip_days": skip_recent,
            }

        if len(momentum_scores) < 3:
            return results

        # Rank by momentum score
        sorted_symbols = sorted(
            momentum_scores.keys(),
            key=lambda s: momentum_scores[s]["momentum_score"],
            reverse=True
        )

        n_assets = len(sorted_symbols)
        n_long = max(1, int(n_assets * top_pct))
        n_short = max(1, int(n_assets * bottom_pct))

        # Generate signals based on ranking
        for rank, symbol in enumerate(sorted_symbols, 1):
            scores = momentum_scores[symbol]
            percentile = 1.0 - (rank - 1) / n_assets

            # Determine bucket
            if rank <= n_long:
                bucket = "long"
                signal = 1.0
                # Weight by rank within long bucket
                signal_strength = 1.0 - (rank - 1) / n_long * 0.3  # 0.7 to 1.0
            elif rank > n_assets - n_short:
                bucket = "short"
                signal = -1.0
                # Weight by rank within short bucket
                rank_from_bottom = n_assets - rank + 1
                signal_strength = 1.0 - (rank_from_bottom - 1) / n_short * 0.3
            else:
                bucket = "neutral"
                signal = 0.0
                signal_strength = 0.0

            results[symbol] = {
                "signal": signal,
                "signal_strength": signal_strength,
                "momentum_score": scores["momentum_score"],
                "rank": rank,
                "percentile": percentile,
                "bucket": bucket,
                "n_assets": n_assets,
                "is_winner": bucket == "long",
                "is_loser": bucket == "short",
            }

        return results

    def calculate_momentum_factor(
        self,
        price_data: dict[str, np.ndarray],
        lookback: int = 252,
        skip_recent: int = 21,
        weighting: str = "equal"
    ) -> dict[str, Any]:
        """
        Calculate momentum factor portfolio weights.

        This constructs a classic momentum factor (WML - Winners Minus Losers)
        as used in academic research.

        Args:
            price_data: Dict of {symbol: price_array}
            lookback: Lookback period for momentum calculation
            skip_recent: Skip recent N days (reversal avoidance)
            weighting: "equal" for equal-weight, "momentum" for momentum-weighted

        Returns:
            Dict with long/short weights and factor characteristics
        """
        cs_momentum = self.calculate_cross_sectional_momentum(
            price_data,
            lookback_long=lookback,
            skip_recent=skip_recent
        )

        if not cs_momentum:
            return {"long_weights": {}, "short_weights": {}, "error": "Insufficient data"}

        # Separate long and short buckets
        long_assets = {s: data for s, data in cs_momentum.items() if data["bucket"] == "long"}
        short_assets = {s: data for s, data in cs_momentum.items() if data["bucket"] == "short"}

        # Calculate weights
        if weighting == "equal":
            n_long = len(long_assets)
            n_short = len(short_assets)
            long_weights = {s: 1.0 / n_long if n_long > 0 else 0 for s in long_assets}
            short_weights = {s: 1.0 / n_short if n_short > 0 else 0 for s in short_assets}
        else:  # momentum-weighted
            # Weight by momentum score (stronger momentum = higher weight)
            total_long_mom = sum(d["momentum_score"] for d in long_assets.values())
            total_short_mom = abs(sum(d["momentum_score"] for d in short_assets.values()))

            if total_long_mom > 0:
                long_weights = {
                    s: data["momentum_score"] / total_long_mom
                    for s, data in long_assets.items()
                }
            else:
                long_weights = {s: 1.0 / len(long_assets) for s in long_assets}

            if total_short_mom > 0:
                short_weights = {
                    s: abs(data["momentum_score"]) / total_short_mom
                    for s, data in short_assets.items()
                }
            else:
                short_weights = {s: 1.0 / len(short_assets) for s in short_assets}

        # Calculate factor characteristics
        avg_long_momentum = (
            np.mean([d["momentum_score"] for d in long_assets.values()])
            if long_assets else 0
        )
        avg_short_momentum = (
            np.mean([d["momentum_score"] for d in short_assets.values()])
            if short_assets else 0
        )

        return {
            "long_weights": long_weights,
            "short_weights": short_weights,
            "n_long": len(long_assets),
            "n_short": len(short_assets),
            "avg_long_momentum": avg_long_momentum,
            "avg_short_momentum": avg_short_momentum,
            "spread": avg_long_momentum - avg_short_momentum,
            "weighting": weighting,
        }

    def multi_timeframe_momentum(
        self,
        prices_daily: np.ndarray,
        prices_weekly: np.ndarray | None = None,
        prices_monthly: np.ndarray | None = None,
        daily_roc_period: int = 20,
        weekly_roc_period: int = 4,
        monthly_roc_period: int = 1,
    ) -> dict[str, Any]:
        """
        Multi-timeframe momentum alignment analysis.

        Signals are strongest when all timeframes agree on direction.
        This reduces false signals from short-term noise.

        Research finding: Multi-timeframe alignment improves win rate
        by 15-25% compared to single timeframe signals.

        Args:
            prices_daily: Daily price array
            prices_weekly: Weekly price array (optional, will aggregate from daily)
            prices_monthly: Monthly price array (optional, will aggregate from daily)
            daily_roc_period: ROC period for daily (default: 20 days)
            weekly_roc_period: ROC period for weekly (default: 4 weeks)
            monthly_roc_period: ROC period for monthly (default: 1 month)

        Returns:
            Dict with signal direction, strength, and timeframe breakdown
        """
        # Aggregate to weekly/monthly if not provided
        if prices_weekly is None and len(prices_daily) >= 5:
            prices_weekly = self._aggregate_to_weekly(prices_daily)
        if prices_monthly is None and len(prices_daily) >= 20:
            prices_monthly = self._aggregate_to_monthly(prices_daily)

        # Calculate trend for each timeframe
        trends = {}

        # Daily trend
        if len(prices_daily) >= daily_roc_period + 1:
            daily_roc = self.calculate_roc(prices_daily, daily_roc_period)
            daily_trend = 1 if daily_roc > 0 else -1 if daily_roc < 0 else 0
            trends["daily"] = {"trend": daily_trend, "roc": daily_roc}
        else:
            trends["daily"] = {"trend": 0, "roc": 0.0}

        # Weekly trend
        if prices_weekly is not None and len(prices_weekly) >= weekly_roc_period + 1:
            weekly_roc = self.calculate_roc(prices_weekly, weekly_roc_period)
            weekly_trend = 1 if weekly_roc > 0 else -1 if weekly_roc < 0 else 0
            trends["weekly"] = {"trend": weekly_trend, "roc": weekly_roc}
        else:
            trends["weekly"] = {"trend": 0, "roc": 0.0}

        # Monthly trend
        if prices_monthly is not None and len(prices_monthly) >= monthly_roc_period + 1:
            monthly_roc = self.calculate_roc(prices_monthly, monthly_roc_period)
            monthly_trend = 1 if monthly_roc > 0 else -1 if monthly_roc < 0 else 0
            trends["monthly"] = {"trend": monthly_trend, "roc": monthly_roc}
        else:
            trends["monthly"] = {"trend": 0, "roc": 0.0}

        # Calculate alignment score
        alignment = (
            trends["daily"]["trend"] +
            trends["weekly"]["trend"] +
            trends["monthly"]["trend"]
        )

        # Determine signal
        if alignment == 3:
            direction = "long"
            strength = 1.0
            alignment_desc = "strong_bullish"
        elif alignment == -3:
            direction = "short"
            strength = 1.0
            alignment_desc = "strong_bearish"
        elif alignment == 2:
            direction = "long"
            strength = 0.67
            alignment_desc = "moderate_bullish"
        elif alignment == -2:
            direction = "short"
            strength = 0.67
            alignment_desc = "moderate_bearish"
        elif abs(alignment) == 1:
            direction = "long" if alignment > 0 else "short"
            strength = 0.33
            alignment_desc = "weak_signal"
        else:
            direction = "flat"
            strength = 0.0
            alignment_desc = "conflicting"

        return {
            "direction": direction,
            "strength": strength,
            "alignment_score": alignment,
            "alignment_desc": alignment_desc,
            "timeframes": trends,
            "is_aligned": abs(alignment) >= 2,
            "fully_aligned": abs(alignment) == 3,
        }

    def _aggregate_to_weekly(self, daily_prices: np.ndarray) -> np.ndarray:
        """Aggregate daily prices to weekly (last price of each week)."""
        if len(daily_prices) < 5:
            return daily_prices

        # Take every 5th price (approximate weekly)
        weekly = []
        for i in range(4, len(daily_prices), 5):
            weekly.append(daily_prices[i])

        return np.array(weekly) if weekly else daily_prices[-1:]

    def _aggregate_to_monthly(self, daily_prices: np.ndarray) -> np.ndarray:
        """Aggregate daily prices to monthly (last price of each month)."""
        if len(daily_prices) < 20:
            return daily_prices

        # Take every ~21 prices (approximate monthly)
        monthly = []
        for i in range(20, len(daily_prices), 21):
            monthly.append(daily_prices[i])

        return np.array(monthly) if monthly else daily_prices[-1:]

    def analyze_with_mtf(
        self,
        symbol: str,
        prices_daily: np.ndarray,
        prices_weekly: np.ndarray | None = None,
        prices_monthly: np.ndarray | None = None,
        timestamp: datetime | None = None,
        volumes: np.ndarray | None = None,
        require_mtf_alignment: bool = True,
        min_alignment: int = 2,
    ) -> MomentumSignal:
        """
        Enhanced analyze() with multi-timeframe confirmation.

        This wraps the standard analyze() method and adds MTF alignment
        as an additional filter/confirmation.

        Args:
            symbol: Trading symbol
            prices_daily: Daily price series
            prices_weekly: Weekly prices (optional)
            prices_monthly: Monthly prices (optional)
            timestamp: Current timestamp
            volumes: Volume series
            require_mtf_alignment: If True, suppress signals without alignment
            min_alignment: Minimum alignment score required (2 = 2 of 3 agree)

        Returns:
            MomentumSignal with MTF information in indicators
        """
        # Get base signal from standard analysis
        base_signal = self.analyze(symbol, prices_daily, timestamp, volumes)

        # Calculate MTF alignment
        mtf_result = self.multi_timeframe_momentum(
            prices_daily, prices_weekly, prices_monthly
        )

        # Add MTF info to indicators
        base_signal.indicators["mtf_alignment_score"] = mtf_result["alignment_score"]
        base_signal.indicators["mtf_aligned"] = mtf_result["is_aligned"]
        base_signal.indicators["mtf_fully_aligned"] = mtf_result["fully_aligned"]
        base_signal.indicators["mtf_direction"] = mtf_result["direction"]

        # Apply MTF filter if required
        if require_mtf_alignment and base_signal.direction != "flat":
            alignment = abs(mtf_result["alignment_score"])

            if alignment < min_alignment:
                # Insufficient alignment - suppress signal
                logger.info(
                    f"MTF filter: {symbol} signal {base_signal.direction} suppressed "
                    f"(alignment={alignment} < required={min_alignment})"
                )
                base_signal = MomentumSignal(
                    symbol=symbol,
                    direction="flat",
                    strength=0.0,
                    confidence=base_signal.confidence * 0.5,
                    indicators=base_signal.indicators,
                    blackout_active=base_signal.blackout_active,
                    blackout_reason=base_signal.blackout_reason,
                )
            elif mtf_result["direction"] != base_signal.direction:
                # MTF disagrees with base signal - reduce confidence
                base_signal.confidence *= 0.7
                base_signal.indicators["mtf_disagreement"] = True
            else:
                # MTF confirms signal - boost confidence
                if mtf_result["fully_aligned"]:
                    base_signal.confidence = min(1.0, base_signal.confidence * 1.2)
                    base_signal.strength = min(1.0, base_signal.strength * 1.1)

        return base_signal

    def calculate_dual_momentum(
        self,
        asset_returns: dict[str, np.ndarray],
        risk_free_rate: float = 0.04,
        lookback: int = 252,
        absolute_threshold: float = 0.0
    ) -> dict[str, dict[str, Any]]:
        """
        Calculate Dual Momentum signals (Gary Antonacci method).

        Dual Momentum combines:
        1. Relative momentum: comparing assets against each other
        2. Absolute momentum: comparing asset against risk-free rate

        Research finding: Dual momentum achieves 15-17% CAGR vs 10% buy-and-hold
        with lower drawdowns (Antonacci, 2014).

        The strategy:
        - Buy asset with highest relative momentum
        - BUT only if absolute momentum > 0 (above risk-free rate)
        - If absolute momentum < 0, go to cash/bonds

        Args:
            asset_returns: Dict of {symbol: returns_array} for each asset
            risk_free_rate: Annual risk-free rate (default 4%)
            lookback: Lookback period in trading days (252 = 1 year)
            absolute_threshold: Minimum excess return over risk-free to invest

        Returns:
            Dict of {symbol: {signal, relative_momentum, absolute_momentum, rank}}
        """
        results = {}

        if not asset_returns:
            return results

        # Convert annual risk-free rate to period return
        risk_free_period = risk_free_rate * (lookback / 252)

        # Calculate momentum for each asset
        momentum_scores = {}
        for symbol, returns in asset_returns.items():
            if len(returns) < lookback:
                effective_lookback = len(returns)
            else:
                effective_lookback = lookback

            if effective_lookback < 20:  # Minimum required
                momentum_scores[symbol] = {
                    "relative_momentum": 0.0,
                    "absolute_momentum": 0.0,
                    "total_return": 0.0,
                }
                continue

            # Calculate total return over lookback period
            recent_returns = returns[-effective_lookback:]
            total_return = np.prod(1 + recent_returns) - 1

            # Absolute momentum: excess return over risk-free rate
            absolute_momentum = total_return - risk_free_period

            momentum_scores[symbol] = {
                "relative_momentum": total_return,  # Will be ranked later
                "absolute_momentum": absolute_momentum,
                "total_return": total_return,
            }

        # Rank by relative momentum (highest = rank 1)
        sorted_symbols = sorted(
            momentum_scores.keys(),
            key=lambda s: momentum_scores[s]["total_return"],
            reverse=True
        )

        for rank, symbol in enumerate(sorted_symbols, 1):
            momentum_scores[symbol]["rank"] = rank

        # Generate signals
        for symbol, scores in momentum_scores.items():
            # Determine signal based on dual momentum rules
            if scores["rank"] == 1:
                # Best relative momentum
                if scores["absolute_momentum"] > absolute_threshold:
                    # Positive absolute momentum - invest
                    signal = 1.0
                    signal_reason = "best_relative_positive_absolute"
                else:
                    # Negative absolute momentum - go to cash
                    signal = 0.0
                    signal_reason = "best_relative_but_negative_absolute"
            else:
                # Not the best relative momentum
                if scores["absolute_momentum"] > absolute_threshold:
                    # Still has positive absolute momentum
                    signal = 0.5 / scores["rank"]  # Weighted by rank
                    signal_reason = f"rank_{scores['rank']}_positive_absolute"
                else:
                    signal = 0.0
                    signal_reason = "not_best_and_negative_absolute"

            results[symbol] = {
                "signal": signal,
                "relative_momentum": scores["total_return"],
                "absolute_momentum": scores["absolute_momentum"],
                "rank": scores["rank"],
                "signal_reason": signal_reason,
                "is_best_asset": scores["rank"] == 1,
                "invest_recommendation": signal > 0,
            }

        return results

    def calculate_absolute_momentum(
        self,
        prices: np.ndarray,
        lookback: int = 252,
        risk_free_rate: float = 0.04
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate absolute (time-series) momentum for single asset.

        Absolute momentum compares asset's return to a threshold
        (typically risk-free rate). Positive = invest, negative = cash.

        Args:
            prices: Price series
            lookback: Lookback period (252 = 1 year)
            risk_free_rate: Annual risk-free rate

        Returns:
            (signal, details) where signal is 1 (invest), 0 (cash), or -1 (short)
        """
        if len(prices) < 20:
            return 0.0, {
                "total_return": 0.0,
                "excess_return": 0.0,
                "risk_free_hurdle": 0.0,
                "signal_reason": "insufficient_data"
            }

        effective_lookback = min(lookback, len(prices) - 1)
        start_price = prices[-(effective_lookback + 1)]
        end_price = prices[-1]

        total_return = (end_price - start_price) / start_price
        risk_free_period = risk_free_rate * (effective_lookback / 252)
        excess_return = total_return - risk_free_period

        # Generate signal
        if excess_return > 0.02:  # 2% buffer above risk-free
            signal = 1.0
            signal_reason = "positive_excess_return"
        elif excess_return < -0.02:  # Significantly negative
            signal = -1.0
            signal_reason = "negative_excess_return"
        else:
            signal = 0.0
            signal_reason = "near_risk_free_threshold"

        details = {
            "total_return": total_return,
            "excess_return": excess_return,
            "risk_free_hurdle": risk_free_period,
            "lookback_days": effective_lookback,
            "signal_reason": signal_reason,
            "annualized_return": total_return * (252 / effective_lookback),
        }

        return signal, details

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

    def calculate_take_profit(
        self,
        current_price: float,
        direction: str,
        atr: float
    ) -> tuple[float, float]:
        """
        RISK-001: Calculate take-profit price and percentage.

        Args:
            current_price: Entry price
            direction: "long" or "short"
            atr: Average True Range

        Returns:
            (take_profit_price, take_profit_pct)
        """
        if self._use_atr_take_profit and atr > 0:
            tp_distance = atr * self._take_profit_atr_multiplier
        else:
            tp_distance = current_price * (self._take_profit_pct / 100)

        if direction == "long":
            tp_price = current_price + tp_distance
        elif direction == "short":
            tp_price = current_price - tp_distance
        else:
            return None, None

        tp_pct = (tp_distance / current_price) * 100
        return tp_price, tp_pct

    def detect_market_regime(
        self,
        prices: np.ndarray,
        adx: float,
        realized_vol: float
    ) -> tuple[str, bool]:
        """
        RISK-004: Detect market regime and suitability for momentum trading.

        Momentum strategies work best in trending markets and poorly in ranging markets.

        Args:
            prices: Price series
            adx: Current ADX value
            realized_vol: Current realized volatility

        Returns:
            (regime, is_suitable_for_momentum)
        """
        # Determine regime based on ADX and volatility
        # ADX < 20 = ranging/choppy
        # ADX 20-25 = weak trend
        # ADX > 25 = strong trend
        # High vol with low ADX = volatile/choppy

        base_vol = self._vol_target  # Use vol target as baseline
        vol_ratio = realized_vol / base_vol if base_vol > 0 else 1.0

        if adx < self._ranging_adx_threshold:
            if vol_ratio > self._volatile_regime_vol_multiplier:
                regime = "volatile"
                suitable = False  # High vol + no trend = bad for momentum
            else:
                regime = "ranging"
                suitable = not self._suppress_in_ranging  # Ranging = bad for momentum
        elif adx < 25:
            regime = "weak_trend"
            suitable = True  # Weak trend can still work
        else:
            if vol_ratio > self._volatile_regime_vol_multiplier:
                regime = "volatile_trending"
                suitable = True  # Trending even with high vol can work
            else:
                regime = "trending"
                suitable = True  # Ideal for momentum

        return regime, suitable

    def check_exit_conditions(
        self,
        current_price: float,
        entry_price: float,
        direction: str,
        bars_held: int,
        stop_loss_price: float,
        take_profit_price: float,
        rsi: float
    ) -> tuple[bool, str | None]:
        """
        RISK-005: Check if exit conditions are met for an existing position.

        Args:
            current_price: Current market price
            entry_price: Original entry price
            direction: Position direction ("long" or "short")
            bars_held: Number of bars position has been held
            stop_loss_price: Stop-loss price level
            take_profit_price: Take-profit price level
            rsi: Current RSI value

        Returns:
            (should_exit, reason)
        """
        # Check max holding period
        if bars_held >= self._max_holding_bars:
            return True, "max_holding_period_reached"

        # Check stop-loss
        if direction == "long" and current_price <= stop_loss_price:
            return True, "stop_loss_triggered"
        elif direction == "short" and current_price >= stop_loss_price:
            return True, "stop_loss_triggered"

        # Check take-profit
        if direction == "long" and current_price >= take_profit_price:
            return True, "take_profit_triggered"
        elif direction == "short" and current_price <= take_profit_price:
            return True, "take_profit_triggered"

        # Check RSI reversal (exit long on overbought, exit short on oversold)
        if direction == "long" and rsi > self._rsi_overbought:
            return True, "rsi_overbought_exit"
        elif direction == "short" and rsi < self._rsi_oversold:
            return True, "rsi_oversold_exit"

        return False, None

    def generate_exit_signal(
        self,
        symbol: str,
        direction: str,
        exit_reason: str,
        current_price: float,
        indicators: dict[str, float]
    ) -> MomentumSignal:
        """
        RISK-005: Generate an exit signal for an existing position.

        Args:
            symbol: Trading symbol
            direction: Current position direction ("long" or "short")
            exit_reason: Reason for exit
            current_price: Current market price
            indicators: Current indicator values

        Returns:
            MomentumSignal with exit direction
        """
        exit_direction = "exit_long" if direction == "long" else "exit_short"

        return MomentumSignal(
            symbol=symbol,
            direction=exit_direction,
            strength=0.0,
            confidence=1.0,  # High confidence for exit signals
            indicators=indicators,
            stop_loss_price=None,
            stop_loss_pct=None,
            take_profit_price=None,
            take_profit_pct=None,
            max_holding_bars=self._max_holding_bars,
            strategy_max_loss_pct=self._strategy_max_loss_pct,
            position_size_scalar=0.0,  # No new position
            trend_strength="unknown",
            market_regime="unknown",
            regime_suitable=True,
            blackout_active=False,
            blackout_reason=None,
            divergence_type=None,
            divergence_strength=0.0,
            volume_confirmed=False,
            volume_ratio=1.0,
            is_exit_signal=True,
            exit_reason=exit_reason,
        )

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

    # =========================================================================
    # QUICK WIN #2: Get asset-specific RSI thresholds
    # =========================================================================
    def _get_rsi_thresholds(self, symbol: str) -> tuple[float, float]:
        """
        QUICK WIN #2: Get asset-specific RSI thresholds.

        High volatility assets (MNQ, NG, GBP) use wider bands to reduce false signals.
        Gold uses mid-band entries (40-60 zone).
        """
        if symbol in self._rsi_overrides:
            override = self._rsi_overrides[symbol]
            return (
                override.get("overbought", self._rsi_overbought),
                override.get("oversold", self._rsi_oversold)
            )
        return (self._rsi_overbought, self._rsi_oversold)

    # =========================================================================
    # QUICK WIN #5: Validate Risk-Reward ratio
    # =========================================================================
    def _validate_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        min_rr_ratio: float | None = None
    ) -> bool:
        """
        QUICK WIN #5: Validate that trade has acceptable risk-reward ratio.

        Args:
            entry_price: Proposed entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            min_rr_ratio: Minimum required R:R (default from config)

        Returns:
            True if trade meets minimum R:R requirement
        """
        if min_rr_ratio is None:
            min_rr_ratio = self._min_risk_reward_ratio

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk <= 0:
            return False

        rr_ratio = reward / risk

        if rr_ratio < min_rr_ratio:
            logger.info(
                f"Signal rejected: R:R ratio {rr_ratio:.2f} < minimum {min_rr_ratio}"
            )
            return False

        return True

    # =========================================================================
    # QUICK WIN #6: Triple confirmation filter
    # =========================================================================
    def _check_triple_confirmation(
        self,
        rsi: float,
        macd_histogram: float,
        stoch_k: float | None,
        stoch_d: float | None,
        direction: str
    ) -> tuple[bool, float]:
        """
        QUICK WIN #6: Check if RSI, MACD, and Stochastic all confirm the signal.

        Triple confirmation strategies achieve 52-73% win rates vs 45-55% for
        single indicator (from research).

        Returns:
            (is_confirmed, confidence_boost)
        """
        if not self._use_triple_confirmation:
            return True, 1.0

        confirmations = 0

        # Use RSI mid-point as approximation if no stochastic
        if stoch_k is None:
            stoch_k = rsi
        if stoch_d is None:
            stoch_d = rsi - 5 if direction == "long" else rsi + 5

        if direction == "long":
            # RSI: Above 50 (momentum confirmation)
            if rsi > 50:
                confirmations += 1
            # MACD: Positive histogram
            if macd_histogram > 0:
                confirmations += 1
            # Stochastic: %K > %D and not overbought
            if stoch_k > stoch_d and stoch_k < 80:
                confirmations += 1

        elif direction == "short":
            # RSI: Below 50
            if rsi < 50:
                confirmations += 1
            # MACD: Negative histogram
            if macd_histogram < 0:
                confirmations += 1
            # Stochastic: %K < %D and not oversold
            if stoch_k < stoch_d and stoch_k > 20:
                confirmations += 1

        # Require minimum confirmations (default: 2 of 3)
        is_confirmed = confirmations >= self._min_confirmations
        confidence_boost = confirmations / 3.0  # 0.33, 0.67, or 1.0

        if not is_confirmed:
            logger.info(
                f"Triple confirmation failed: {confirmations}/3 for {direction}, "
                f"RSI={rsi:.1f}, MACD_hist={macd_histogram:.4f}"
            )

        return is_confirmed, confidence_boost

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

        # P3: Calculate 52-week high/low signal
        week52_signal, week52_details = self.calculate_52week_signal(prices)

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
            # P3: 52-week high/low indicators
            "week52_signal": week52_signal,
            "week52_high": week52_details["52w_high"],
            "week52_low": week52_details["52w_low"],
            "week52_proximity": week52_details["proximity_to_high"],
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

        # QUICK WIN #2: Get asset-specific RSI thresholds
        rsi_overbought, rsi_oversold = self._get_rsi_thresholds(symbol)

        # RSI with asset-specific thresholds
        if rsi < rsi_oversold:
            scores.append(1)  # Oversold = bullish
        elif rsi > rsi_overbought:
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

        # P3: 52-week high/low (research: stocks near 52w highs outperform by 0.5%/month)
        if week52_signal > 0.5:
            scores.append(1)
        elif week52_signal < -0.5:
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

        # Initialize confidence before conditional adjustments
        confidence = 0.5  # Base confidence

        # QUICK WIN #6: Apply triple confirmation filter
        if direction != "flat":
            is_confirmed, conf_boost = self._check_triple_confirmation(
                rsi=rsi,
                macd_histogram=macd_hist,
                stoch_k=None,  # Stochastic not calculated yet, use RSI approximation
                stoch_d=None,
                direction=direction
            )
            if not is_confirmed:
                # Reduce strength but don't completely suppress
                strength *= 0.5
                confidence *= 0.7
            else:
                # Boost confidence when confirmed
                confidence = min(1.0, confidence * (1 + conf_boost * 0.2))

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

        # FIX-31: Merge agreement-based confidence with prior adjustments
        # (triple confirmation, volume) instead of overwriting
        agreement = sum(1 for s in scores if s == np.sign(total_score)) / len(scores)
        agreement_confidence = agreement * confidence_adjustment * trend_confidence_adj
        # Blend: 50% prior confidence (from triple confirmation etc.) + 50% agreement
        confidence = (confidence + agreement_confidence) / 2

        # P2: Volume confirmation boosts or reduces confidence
        if volumes is not None and direction != "flat":
            if volume_confirmed:
                confidence = min(1.0, confidence * 1.1)  # 10% boost
            else:
                confidence *= 0.85  # 15% reduction without volume confirmation

        # RISK-004: Detect market regime and check suitability
        market_regime, regime_suitable = self.detect_market_regime(
            prices, adx, realized_vol
        )

        # Suppress signals in unfavorable regimes
        if not regime_suitable and direction != "flat":
            logger.info(
                f"Signal {direction} suppressed for {symbol}: "
                f"market_regime={market_regime} not suitable for momentum"
            )
            direction = "flat"
            strength = 0.0

        # P1-13: Calculate stop-loss if we have a directional signal
        stop_loss_price = None
        stop_loss_pct = None
        take_profit_price = None
        take_profit_pct = None
        if direction != "flat":
            current_price = prices[-1]
            atr = self.calculate_atr(prices, period=self._atr_period)
            stop_loss_price, stop_loss_pct = self.calculate_stop_loss(
                current_price, direction, atr
            )
            # RISK-001: Calculate take-profit
            take_profit_price, take_profit_pct = self.calculate_take_profit(
                current_price, direction, atr
            )
            indicators["atr"] = atr
            indicators["stop_loss_price"] = stop_loss_price
            indicators["stop_loss_pct"] = stop_loss_pct
            indicators["take_profit_price"] = take_profit_price
            indicators["take_profit_pct"] = take_profit_pct

            # QUICK WIN #5: Validate Risk-Reward ratio
            if stop_loss_price and take_profit_price:
                if not self._validate_risk_reward(current_price, stop_loss_price, take_profit_price):
                    # R:R too low - suppress signal
                    direction = "flat"
                    strength = 0.0
                    indicators["rr_rejected"] = True

        # Add regime info to indicators
        indicators["market_regime"] = market_regime
        indicators["regime_suitable"] = regime_suitable

        return MomentumSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            indicators=indicators,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
            max_holding_bars=self._max_holding_bars,
            strategy_max_loss_pct=self._strategy_max_loss_pct,
            position_size_scalar=position_size_scalar,
            trend_strength=trend_strength,
            market_regime=market_regime,
            regime_suitable=regime_suitable,
            blackout_active=blackout_active,
            blackout_reason=blackout_reason,
            divergence_type=divergence_type,
            divergence_strength=divergence_strength,
            volume_confirmed=volume_confirmed,
            volume_ratio=volume_ratio,
            is_exit_signal=False,
            exit_reason=None,
            week52_signal=week52_signal,
            week52_proximity=week52_details["proximity_to_high"],
        )

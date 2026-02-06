"""
Mean Reversion Single-Asset Strategy (Phase 6.5)
=================================================

Single-asset mean reversion strategy using multiple technical indicators.

Key features:
- RSI-based oversold/overbought detection
- Bollinger Band mean reversion
- Z-score based entry/exit
- Volume confirmation
- Multi-timeframe analysis

Research basis:
- RSI extremes (<30, >70) often precede reversions
- BB touches with volume divergence = higher probability
- Mean reversion works best in range-bound environments
- Time-based exits reduce drawdown

MATURITY: ALPHA
---------------
Status: New implementation
- [x] RSI calculation
- [x] BB mean reversion signals
- [x] Z-score analysis
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


class MarketRegime(Enum):
    """Market regime for mean reversion applicability."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"  # Best for mean reversion
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


class SignalType(Enum):
    """Mean reversion signal types."""
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    BB_LOWER_TOUCH = "bb_lower"
    BB_UPPER_TOUCH = "bb_upper"
    ZSCORE_EXTREME = "zscore"
    COMBINED = "combined"


@dataclass
class MeanReversionState:
    """Current state of mean reversion indicators."""
    rsi: float
    zscore: float
    bb_position: float  # -1 to 1 (position within BB)
    regime: MarketRegime
    volatility_ratio: float  # Current vol / avg vol
    momentum: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MeanReversionSignal:
    """Mean reversion trading signal."""
    symbol: str
    direction: str  # "LONG", "SHORT", "FLAT"
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    rsi: float
    zscore: float
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MeanReversionStrategy:
    """
    Single-asset mean reversion strategy (Phase 6.5).

    Uses RSI, Bollinger Bands, and z-score to identify
    oversold/overbought conditions for mean reversion trades.

    Configuration:
        rsi_period: RSI calculation period (default: 14)
        rsi_oversold: RSI oversold threshold (default: 30)
        rsi_overbought: RSI overbought threshold (default: 70)
        bb_period: Bollinger Band period (default: 20)
        bb_std: Bollinger Band std mult (default: 2.0)
        zscore_threshold: Z-score for extreme detection (default: 2.0)
        lookback_period: Lookback for statistics (default: 50)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize mean reversion strategy."""
        config = config or {}

        # RSI settings - Connors RSI(2) configuration for mean reversion
        # RSI(2) is more responsive than RSI(14) for short-term reversions
        # Extreme thresholds (5/95) required for high-probability entries
        # Reference: Connors Research "Short Term Trading Strategies That Work"
        self._rsi_period = config.get("rsi_period", 2)
        self._rsi_oversold = config.get("rsi_oversold", 5)
        self._rsi_overbought = config.get("rsi_overbought", 95)

        # Bollinger Band settings
        self._bb_period = config.get("bb_period", 20)
        self._bb_std = config.get("bb_std", 2.0)

        # Z-score settings
        self._zscore_threshold = config.get("zscore_threshold", 2.0)
        self._lookback_period = config.get("lookback_period", 50)

        # Risk settings
        self._stop_atr_mult = config.get("stop_atr_mult", 2.0)
        # Take profit: 2.5 ATR allows full reversion capture (was 1.5)
        # Mean reversion often overshoots the mean - capture more upside
        self._take_profit_atr_mult = config.get("take_profit_atr_mult", 2.5)
        self._max_holding_bars = config.get("max_holding_bars", 20)

        # Regime detection
        self._trend_threshold = config.get("trend_threshold", 0.6)
        self._vol_lookback = config.get("vol_lookback", 20)

        # State tracking
        self._states: dict[str, MeanReversionState] = {}

        logger.info(
            f"MeanReversionStrategy initialized: "
            f"RSI({self._rsi_period}, {self._rsi_oversold}/{self._rsi_overbought}), "
            f"BB({self._bb_period}, {self._bb_std})"
        )

    def calculate_rsi(self, prices: np.ndarray, period: int | None = None) -> np.ndarray:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Close prices
            period: RSI period

        Returns:
            RSI values (0-100)
        """
        period = period or self._rsi_period

        if len(prices) < period + 1:
            return np.zeros(len(prices))

        # Price changes
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))

        rsi = np.zeros(len(prices))

        # Initial averages (SMA)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        else:
            rsi[period] = 100

        # Subsequent values (EMA-style smoothing)
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100

        return rsi

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
            (upper, middle, lower) bands
        """
        period = period or self._bb_period
        std_mult = std_mult or self._bb_std

        if len(prices) < period:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))

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

    def calculate_zscore(self, prices: np.ndarray, lookback: int | None = None) -> float:
        """
        Calculate z-score of current price.

        Args:
            prices: Price series
            lookback: Lookback period

        Returns:
            Z-score of current price
        """
        lookback = lookback or self._lookback_period

        if len(prices) < lookback:
            return 0.0

        window = prices[-lookback:]
        mean = np.mean(window)
        std = np.std(window, ddof=1)

        if std < 1e-10:
            return 0.0

        return (prices[-1] - mean) / std

    def calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate current ATR value."""
        if len(close) < period + 1:
            return 0.0

        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]

        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        return np.mean(tr[-period:])

    def detect_regime(
        self,
        prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            prices: Close prices
            high: High prices
            low: Low prices

        Returns:
            MarketRegime classification
        """
        if len(prices) < self._lookback_period:
            return MarketRegime.RANGE_BOUND

        # Check trend using linear regression
        x = np.arange(self._lookback_period)
        y = prices[-self._lookback_period:]
        slope, _ = np.polyfit(x, y, 1)

        # Normalize slope by price level
        normalized_slope = slope / np.mean(y) * 100

        # Check volatility
        current_vol = np.std(prices[-self._vol_lookback:])
        avg_vol = np.std(prices[-self._lookback_period:])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Classify regime
        if vol_ratio > 1.5:
            return MarketRegime.HIGH_VOL
        elif vol_ratio < 0.5:
            return MarketRegime.LOW_VOL
        elif normalized_slope > self._trend_threshold:
            return MarketRegime.TRENDING_UP
        elif normalized_slope < -self._trend_threshold:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGE_BOUND

    def get_bb_position(
        self,
        price: float,
        bb_upper: float,
        bb_lower: float,
    ) -> float:
        """
        Get position within Bollinger Bands (-1 to 1).

        Returns:
            -1 at lower band, 0 at middle, +1 at upper band
        """
        if bb_upper == bb_lower:
            return 0.0

        middle = (bb_upper + bb_lower) / 2
        half_width = (bb_upper - bb_lower) / 2

        position = (price - middle) / half_width
        return max(-1.5, min(1.5, position))

    def analyze(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> MeanReversionState:
        """
        Analyze current mean reversion state.

        Args:
            symbol: Instrument symbol
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            MeanReversionState with analysis
        """
        # Calculate indicators
        rsi = self.calculate_rsi(close)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        zscore = self.calculate_zscore(close)
        regime = self.detect_regime(close, high, low)

        # Current values
        current_rsi = rsi[-1] if len(rsi) > 0 else 50.0
        current_bb_upper = bb_upper[-1] if len(bb_upper) > 0 else close[-1]
        current_bb_lower = bb_lower[-1] if len(bb_lower) > 0 else close[-1]

        bb_position = self.get_bb_position(close[-1], current_bb_upper, current_bb_lower)

        # Volatility ratio
        if len(close) >= self._lookback_period:
            current_vol = np.std(close[-self._vol_lookback:])
            avg_vol = np.std(close[-self._lookback_period:])
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # Momentum (rate of change)
        if len(close) >= 10:
            momentum = (close[-1] - close[-10]) / close[-10] * 100
        else:
            momentum = 0.0

        state = MeanReversionState(
            rsi=current_rsi,
            zscore=zscore,
            bb_position=bb_position,
            regime=regime,
            volatility_ratio=vol_ratio,
            momentum=momentum,
        )

        self._states[symbol] = state
        return state

    def generate_signal(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray | None = None,
        current_position: str = "FLAT",
    ) -> MeanReversionSignal | None:
        """
        Generate mean reversion trading signal.

        Args:
            symbol: Instrument symbol
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data (optional)
            current_position: Current position

        Returns:
            MeanReversionSignal if conditions met
        """
        if len(close) < self._lookback_period:
            return None

        state = self.analyze(symbol, high, low, close)
        current_price = close[-1]
        atr = self.calculate_atr(high, low, close)

        if atr == 0:
            atr = current_price * 0.02

        # Check regime suitability
        if state.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Mean reversion less effective in strong trends
            regime_penalty = 0.5
        else:
            regime_penalty = 1.0

        # Count signals
        signals_triggered = []
        direction = None
        strength = 0.0

        # RSI signals
        if state.rsi < self._rsi_oversold:
            signals_triggered.append(SignalType.RSI_OVERSOLD)
            direction = "LONG"
            strength += (self._rsi_oversold - state.rsi) / self._rsi_oversold

        elif state.rsi > self._rsi_overbought:
            signals_triggered.append(SignalType.RSI_OVERBOUGHT)
            direction = "SHORT"
            strength += (state.rsi - self._rsi_overbought) / (100 - self._rsi_overbought)

        # Bollinger Band signals
        if state.bb_position < -0.9:
            signals_triggered.append(SignalType.BB_LOWER_TOUCH)
            if direction is None:
                direction = "LONG"
            elif direction == "LONG":
                strength += 0.3
            else:
                direction = None  # Conflicting signals

        elif state.bb_position > 0.9:
            signals_triggered.append(SignalType.BB_UPPER_TOUCH)
            if direction is None:
                direction = "SHORT"
            elif direction == "SHORT":
                strength += 0.3
            else:
                direction = None

        # Z-score signals
        if state.zscore < -self._zscore_threshold:
            signals_triggered.append(SignalType.ZSCORE_EXTREME)
            if direction is None:
                direction = "LONG"
            elif direction == "LONG":
                strength += 0.3
            else:
                direction = None

        elif state.zscore > self._zscore_threshold:
            signals_triggered.append(SignalType.ZSCORE_EXTREME)
            if direction is None:
                direction = "SHORT"
            elif direction == "SHORT":
                strength += 0.3
            else:
                direction = None

        # GAP FILTER: After a large overnight gap (>3%), suppress signals
        # that go WITH the gap (i.e., don't short after gap down, don't long after gap up).
        # Mean reversion after a gap should FADE the move, not chase it.
        if len(close) >= 20 and direction is not None:
            # Estimate "prior session" price as mean of first 10 bars
            prior_price = np.mean(close[:10])
            gap_pct = (current_price - prior_price) / prior_price if prior_price > 0 else 0
            if abs(gap_pct) > 0.03:  # >3% gap detected
                if gap_pct < -0.03 and direction == "SHORT":
                    # Gap DOWN but signal says SHORT - suppress (should fade the gap = LONG)
                    logger.info(
                        f"MeanReversion GAP FILTER: {symbol} gapped down {gap_pct*100:.1f}%, "
                        f"suppressing SHORT signal (should fade gap = LONG)"
                    )
                    direction = None
                elif gap_pct > 0.03 and direction == "LONG":
                    # Gap UP but signal says LONG - suppress (should fade the gap = SHORT)
                    logger.info(
                        f"MeanReversion GAP FILTER: {symbol} gapped up {gap_pct*100:.1f}%, "
                        f"suppressing LONG signal (should fade gap = SHORT)"
                    )
                    direction = None

        # No clear signal
        if direction is None or len(signals_triggered) < 1:
            # Check for exit signals
            if current_position != "FLAT":
                # Exit if RSI returns to neutral
                if current_position == "LONG" and state.rsi > 50:
                    return MeanReversionSignal(
                        symbol=symbol,
                        direction="FLAT",
                        signal_type=SignalType.RSI_OVERSOLD,
                        strength=0.6,
                        entry_price=current_price,
                        stop_loss=0.0,
                        take_profit=0.0,
                        rsi=state.rsi,
                        zscore=state.zscore,
                        rationale="RSI returned to neutral, taking profit",
                    )
                elif current_position == "SHORT" and state.rsi < 50:
                    return MeanReversionSignal(
                        symbol=symbol,
                        direction="FLAT",
                        signal_type=SignalType.RSI_OVERBOUGHT,
                        strength=0.6,
                        entry_price=current_price,
                        stop_loss=0.0,
                        take_profit=0.0,
                        rsi=state.rsi,
                        zscore=state.zscore,
                        rationale="RSI returned to neutral, taking profit",
                    )
            return None

        # Entry signals only when flat
        if current_position != "FLAT":
            return None

        # Apply regime penalty
        strength = min(1.0, strength * regime_penalty)

        # Require minimum strength
        if strength < 0.3:
            return None

        # Determine signal type
        if len(signals_triggered) >= 2:
            signal_type = SignalType.COMBINED
        else:
            signal_type = signals_triggered[0]

        # Calculate stops and targets
        if direction == "LONG":
            stop_loss = current_price - self._stop_atr_mult * atr
            take_profit = current_price + self._take_profit_atr_mult * atr
        else:  # SHORT
            stop_loss = current_price + self._stop_atr_mult * atr
            take_profit = current_price - self._take_profit_atr_mult * atr

        return MeanReversionSignal(
            symbol=symbol,
            direction=direction,
            signal_type=signal_type,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rsi=state.rsi,
            zscore=state.zscore,
            rationale=(
                f"{signal_type.value}: RSI={state.rsi:.1f}, "
                f"z={state.zscore:.2f}, BB_pos={state.bb_position:.2f}, "
                f"regime={state.regime.value}"
            ),
        )

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        return {
            "rsi_period": self._rsi_period,
            "rsi_thresholds": (self._rsi_oversold, self._rsi_overbought),
            "bb_period": self._bb_period,
            "bb_std": self._bb_std,
            "zscore_threshold": self._zscore_threshold,
            "tracked_symbols": len(self._states),
            "states": {
                symbol: {
                    "rsi": state.rsi,
                    "zscore": state.zscore,
                    "regime": state.regime.value,
                }
                for symbol, state in self._states.items()
            },
        }


def create_mean_reversion_strategy(config: dict[str, Any] | None = None) -> MeanReversionStrategy:
    """Create MeanReversionStrategy instance."""
    return MeanReversionStrategy(config)

"""
Momentum / Trend Following Agent
================================

Generates signals based on price momentum and trend indicators.
Implements moving average crossovers, RSI, and breakout detection.

Responsibility: Momentum/trend signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from core.agent_base import SignalAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    MarketDataEvent,
    SignalEvent,
    SignalDirection,
)
from core.demand_zones import DemandZoneDetector, CandleStick
from core.session_checker import filter_signal_by_session, is_optimal_trading_time

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class MomentumState:
    """State for tracking momentum indicators per symbol."""
    symbol: str
    prices: deque  # Rolling price window
    fast_ma: float = 0.0
    slow_ma: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    last_signal: SignalDirection = SignalDirection.FLAT
    gains: deque = field(default_factory=lambda: deque(maxlen=14))
    losses: deque = field(default_factory=lambda: deque(maxlen=14))
    # EMA state for MACD calculation
    fast_ema: float = 0.0  # 12-period EMA
    slow_ema: float = 0.0  # 26-period EMA
    macd_history: deque = field(default_factory=lambda: deque(maxlen=50))
    signal_ema: float = 0.0  # 9-period EMA of MACD
    ema_initialized: bool = False
    # ADX (Average Directional Index) state for trend strength
    adx: float = 0.0  # Current ADX value
    plus_dm_ema: float = 0.0  # +DM exponential moving average
    minus_dm_ema: float = 0.0  # -DM exponential moving average
    tr_ema: float = 0.0  # True Range exponential moving average
    adx_ema: float = 0.0  # ADX smoothing EMA
    adx_initialized: bool = False
    highs: deque = field(default_factory=lambda: deque(maxlen=50))  # High prices for ADX
    lows: deque = field(default_factory=lambda: deque(maxlen=50))  # Low prices for ADX
    # Demand/Supply zone tracking (MoonDev-inspired)
    zone_detector: DemandZoneDetector | None = None
    at_demand_zone: bool = False
    at_supply_zone: bool = False
    zone_bias: str = "neutral"  # "bullish", "bearish", "neutral"
    # OHLCV for zone detection
    candle_open: float = 0.0
    candle_high: float = 0.0
    candle_low: float = 0.0
    candle_volume: int = 0
    last_candle_time: int = 0  # Unix timestamp for candle aggregation


class MomentumAgent(SignalAgent):
    """
    Momentum / Trend Following Agent.

    Implements multiple momentum indicators:
    1. Moving Average Crossover (fast/slow MA)
    2. RSI (Relative Strength Index)
    3. MACD (Moving Average Convergence Divergence)

    Signal output:
    - Trend direction (long/short/flat)
    - Signal strength based on indicator confluence
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._fast_period = config.parameters.get("fast_period", 10)
        # Slow MA period: 50 (industry standard for medium-term trends)
        # Research: 50/200 "Golden Cross" is institutional standard
        self._slow_period = config.parameters.get("slow_period", 50)
        self._signal_period = config.parameters.get("signal_period", 9)
        self._rsi_period = config.parameters.get("rsi_period", 14)
        self._rsi_overbought = config.parameters.get("rsi_overbought", 70)
        self._rsi_oversold = config.parameters.get("rsi_oversold", 30)

        # ADX (Average Directional Index) configuration - Wilder's trend strength
        # ADX > 25 indicates a trending market (Wilder 1978)
        # Only generate trend-following signals when ADX confirms trend strength
        self._adx_filter_enabled = config.parameters.get("adx_filter_enabled", True)
        self._adx_period = config.parameters.get("adx_period", 14)  # Standard Wilder period
        self._adx_threshold = config.parameters.get("adx_threshold", 25)  # Min ADX for signals

        # Demand Zone configuration (MoonDev-inspired)
        self._demand_zones_enabled = config.parameters.get("demand_zones_enabled", True)
        self._zone_lookback = config.parameters.get("zone_lookback", 50)
        self._zone_touch_threshold = config.parameters.get("zone_touch_threshold", 0.02)
        self._candle_period_seconds = config.parameters.get("candle_period_seconds", 300)  # 5-min candles

        # Phase 2: RSI Trend Filter Mode
        self._rsi_trend_filter_enabled = config.parameters.get("rsi_trend_filter_enabled", True)
        self._rsi_trend_zone = config.parameters.get("rsi_trend_zone", (40, 60))
        self._rsi_filter_mode = config.parameters.get("rsi_filter_mode", "adaptive")
        # Modes: "trend" (only trade with trend), "reversal" (contrarian), "adaptive" (auto-switch)

        # Minimum confidence threshold for signal generation
        self._min_confidence = config.parameters.get("min_confidence", 0.75)

        # State per symbol
        self._symbols: dict[str, MomentumState] = {}
        self._max_lookback = max(self._slow_period, self._rsi_period, self._adx_period * 2) * 2

    async def initialize(self) -> None:
        """Initialize momentum tracking."""
        logger.info(
            f"MomentumAgent initializing with MA({self._fast_period}/{self._slow_period}), "
            f"RSI({self._rsi_period}), ADX={self._adx_filter_enabled}(>{self._adx_threshold}), "
            f"DemandZones={self._demand_zones_enabled}"
        )

    async def process_event(self, event: Event) -> None:
        """Process market data and generate momentum signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        price = event.mid

        if price <= 0:
            return

        # Get or create symbol state
        if symbol not in self._symbols:
            self._symbols[symbol] = MomentumState(
                symbol=symbol,
                prices=deque(maxlen=self._max_lookback),
                zone_detector=DemandZoneDetector(
                    lookback=self._zone_lookback,
                    zone_touch_threshold=self._zone_touch_threshold,
                ) if self._demand_zones_enabled else None,
            )

        state = self._symbols[symbol]

        # Update price
        prev_price = state.prices[-1] if state.prices else price
        state.prices.append(price)

        # Update high/low for ADX calculation
        high = event.high if event.high > 0 else price
        low = event.low if event.low > 0 else price
        state.highs.append(high)
        state.lows.append(low)

        # Update demand zone detector with candle data
        if self._demand_zones_enabled and state.zone_detector:
            self._update_zone_candle(state, event)

        # Update RSI components
        change = price - prev_price
        if change > 0:
            state.gains.append(change)
            state.losses.append(0)
        else:
            state.gains.append(0)
            state.losses.append(abs(change))

        # Calculate indicators and generate signal
        if len(state.prices) >= self._slow_period:
            signal = self._generate_momentum_signal(state)
            if signal:
                # Filter weak signals by confidence threshold
                if signal.confidence < self._min_confidence:
                    logger.debug(
                        f"Signal filtered: confidence {signal.confidence:.2f} < threshold {self._min_confidence}"
                    )
                    return
                await self._event_bus.publish_signal(signal)
                self._audit_logger.log_event(signal)

    def _generate_momentum_signal(self, state: MomentumState) -> SignalEvent | None:
        """
        Generate momentum signal based on multiple indicators.

        Implements:
        - Moving Average Crossover
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - ADX trend strength filter (Wilder, ADX > 25)
        """
        prices = np.array(list(state.prices))

        # Calculate Moving Averages
        state.fast_ma = np.mean(prices[-self._fast_period:])
        state.slow_ma = np.mean(prices[-self._slow_period:])

        # Calculate RSI
        state.rsi = self._calculate_rsi(state)

        # Calculate MACD (using proper EMA)
        current_price = prices[-1]
        state.macd, state.macd_signal = self._calculate_macd(state, current_price)

        # Calculate ADX for trend strength (Wilder 1978)
        adx = self._calculate_adx(state)

        # Determine signal from indicator confluence
        signal_direction, strength, rationale = self._evaluate_indicators(state)

        # Apply ADX trend strength filter
        # Only generate trend-following signals when ADX > threshold
        if self._adx_filter_enabled and adx > 0:
            if adx < self._adx_threshold:
                # Weak trend - filter out trend-following signals
                logger.debug(
                    f"ADX filter: {state.symbol} ADX={adx:.1f} < {self._adx_threshold} - "
                    f"signal suppressed (weak trend)"
                )
                return None
            else:
                # Strong trend confirmed - add ADX to rationale
                rationale = f"{rationale} | ADX={adx:.1f} (strong trend)"

        # Phase 2: Apply RSI trend filter
        if self._rsi_trend_filter_enabled:
            signal_direction, strength, filter_note = self._apply_rsi_trend_filter(
                signal_direction, strength, state
            )
            if filter_note:
                rationale = f"{rationale} | {filter_note}"

        # Only signal on direction change
        if signal_direction == state.last_signal:
            return None

        state.last_signal = signal_direction

        if signal_direction == SignalDirection.FLAT:
            return None

        # QUICK WIN #3: Check trading session quality
        # Reduce or suppress signals during sub-optimal sessions
        is_optimal, session_reason = is_optimal_trading_time(
            state.symbol,
            asset_class=self._get_asset_class(state.symbol),
            config=self._config.parameters if hasattr(self._config, 'parameters') else {}
        )
        if not is_optimal and "Avoided" in session_reason:
            logger.info(
                f"Signal for {state.symbol} suppressed due to session: {session_reason}"
            )
            return None
        elif not is_optimal:
            # Reduce strength for sub-optimal sessions
            strength = strength * 0.5
            rationale = f"{rationale} [Session: {session_reason}]"

        # Build data sources tuple
        data_sources = ["ib_market_data", "momentum_indicator"]
        if self._demand_zones_enabled and (state.at_demand_zone or state.at_supply_zone):
            data_sources.extend(["demand_zone", "supply_zone", "price_level"])

        # Calculate stop-loss and target price based on ATR (2% default if ATR not available)
        current_price = prices[-1]
        atr = self._calculate_atr(state) if hasattr(self, '_calculate_atr') else current_price * 0.02
        stop_loss_distance = max(atr * 2.0, current_price * 0.02)  # 2x ATR or 2% minimum
        target_distance = stop_loss_distance * 2.0  # 2:1 reward/risk ratio

        if signal_direction == SignalDirection.LONG:
            stop_loss = round(current_price - stop_loss_distance, 2)
            target_price = round(current_price + target_distance, 2)
        else:  # SHORT
            stop_loss = round(current_price + stop_loss_distance, 2)
            target_price = round(current_price - target_distance, 2)

        return SignalEvent(
            source_agent=self.name,
            strategy_name="momentum_trend",
            symbol=state.symbol,
            direction=signal_direction,
            strength=strength,
            confidence=self._calculate_confidence(state),
            rationale=rationale,
            data_sources=tuple(data_sources),
            target_price=target_price,
            stop_loss=stop_loss,
        )

    def _calculate_rsi(self, state: MomentumState) -> float:
        """
        Calculate Relative Strength Index (RSI).

        RSI is a momentum oscillator that measures the speed and magnitude
        of price movements. It oscillates between 0 and 100.

        Formula:
            RS = Average Gain / Average Loss (over N periods)
            RSI = 100 - (100 / (1 + RS))

        Interpretation:
            - RSI > 70: Overbought (potential reversal down)
            - RSI < 30: Oversold (potential reversal up)
            - RSI = 50: Neutral

        Mathematical properties:
            - When avg_gain >> avg_loss: RS -> infinity, RSI -> 100
            - When avg_gain << avg_loss: RS -> 0, RSI -> 0
            - When avg_gain = avg_loss: RS = 1, RSI = 50

        Args:
            state: MomentumState containing gains and losses deques

        Returns:
            RSI value between 0 and 100
        """
        if len(state.gains) < self._rsi_period:
            return 50.0  # Neutral RSI if insufficient data

        avg_gain = np.mean(list(state.gains))
        avg_loss = np.mean(list(state.losses))

        # Handle edge case: no losses = maximum RSI
        if avg_loss < 1e-8:
            return 100.0

        # RS = Relative Strength = ratio of average gain to average loss
        rs = avg_gain / avg_loss

        # RSI formula: transforms RS to 0-100 scale
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, state: MomentumState, price: float) -> tuple[float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence) indicator.

        MACD is a trend-following momentum indicator that shows the relationship
        between two exponential moving averages (EMAs) of prices.

        Components:
            MACD Line = 12-period EMA - 26-period EMA
            Signal Line = 9-period EMA of MACD Line
            Histogram = MACD Line - Signal Line (not returned here)

        EMA (Exponential Moving Average) formula:
            EMA_t = price * k + EMA_{t-1} * (1 - k)
            where k = 2 / (period + 1) is the smoothing factor

        Trading signals:
            - MACD > Signal: Bullish (momentum increasing)
            - MACD < Signal: Bearish (momentum decreasing)
            - MACD crosses above Signal: Buy signal
            - MACD crosses below Signal: Sell signal

        Interpretation:
            - When fast EMA > slow EMA: Recent prices are higher (uptrend)
            - MACD oscillates around zero
            - Large positive/negative values indicate strong trends
            - Zero crossovers indicate trend changes

        Args:
            state: MomentumState containing price history and EMA values
            price: Current price to update EMAs

        Returns:
            Tuple of (MACD value, Signal line value)
        """
        # EMA smoothing factors
        fast_k = 2.0 / (12 + 1)  # 12-period: k = 0.1538
        slow_k = 2.0 / (26 + 1)  # 26-period: k = 0.0741
        signal_k = 2.0 / (9 + 1)  # 9-period: k = 0.2

        prices = list(state.prices)

        # Initialize EMAs with SMA if not yet initialized
        if not state.ema_initialized and len(prices) >= 26:
            state.fast_ema = np.mean(prices[-12:])
            state.slow_ema = np.mean(prices[-26:])
            state.ema_initialized = True
            # Initialize MACD and signal
            macd = state.fast_ema - state.slow_ema
            state.macd_history.append(macd)
            state.signal_ema = macd
            return macd, macd

        if not state.ema_initialized:
            return 0.0, 0.0

        # Update EMAs with new price
        state.fast_ema = price * fast_k + state.fast_ema * (1 - fast_k)
        state.slow_ema = price * slow_k + state.slow_ema * (1 - slow_k)

        # Calculate MACD
        macd = state.fast_ema - state.slow_ema
        state.macd_history.append(macd)

        # Calculate signal line (9-period EMA of MACD)
        state.signal_ema = macd * signal_k + state.signal_ema * (1 - signal_k)

        return macd, state.signal_ema

    def _calculate_atr(self, state: MomentumState, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility-based stop-loss.

        ATR measures market volatility by decomposing the entire range of an asset
        price for a period. Used for setting stop-loss distances.

        True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
        ATR = Moving average of True Range over N periods

        Args:
            state: MomentumState containing price history
            period: ATR calculation period (default 14)

        Returns:
            ATR value (or 2% of current price if insufficient data)
        """
        prices = list(state.prices)
        if len(prices) < period + 1:
            # Not enough data, return 2% of current price as fallback
            return prices[-1] * 0.02 if prices else 0.0

        # Calculate True Range approximation using price changes
        # (In a full implementation, we'd use High, Low, Close)
        true_ranges = []
        for i in range(1, min(period + 1, len(prices))):
            tr = abs(prices[-i] - prices[-i - 1])
            true_ranges.append(tr)

        if not true_ranges:
            return prices[-1] * 0.02

        return sum(true_ranges) / len(true_ranges)

    def _calculate_adx(self, state: MomentumState) -> float:
        """
        Calculate Average Directional Index (ADX) for trend strength.

        ADX was developed by J. Welles Wilder Jr. (1978) to measure trend strength.
        It does NOT indicate trend direction, only the strength of the trend.

        ADX Interpretation (Wilder):
            - ADX < 20: No trend / weak trend
            - ADX 20-25: Emerging trend
            - ADX > 25: Strong trend (trade with trend)
            - ADX > 50: Very strong trend
            - ADX > 75: Extremely strong trend

        Calculation:
            1. True Range (TR) = max(H-L, |H-Prev_C|, |L-Prev_C|)
            2. +DM = H - Prev_H if (H - Prev_H) > (Prev_L - L) else 0
            3. -DM = Prev_L - L if (Prev_L - L) > (H - Prev_H) else 0
            4. +DI = 100 * EMA(+DM) / EMA(TR)
            5. -DI = 100 * EMA(-DM) / EMA(TR)
            6. DX = 100 * |+DI - -DI| / (+DI + -DI)
            7. ADX = EMA(DX) over N periods

        Args:
            state: MomentumState containing highs, lows, and prices

        Returns:
            ADX value (0-100)
        """
        prices = list(state.prices)
        highs = list(state.highs)
        lows = list(state.lows)

        if len(prices) < self._adx_period + 1 or len(highs) < 2 or len(lows) < 2:
            return 0.0

        # Smoothing factor for Wilder's EMA (1/N instead of 2/(N+1))
        alpha = 1.0 / self._adx_period

        # Calculate current +DM, -DM, and TR
        curr_high = highs[-1]
        prev_high = highs[-2]
        curr_low = lows[-1]
        prev_low = lows[-2]
        prev_close = prices[-2] if len(prices) > 1 else prices[-1]

        # True Range
        tr = max(
            curr_high - curr_low,
            abs(curr_high - prev_close),
            abs(curr_low - prev_close)
        )

        # Directional Movement
        up_move = curr_high - prev_high
        down_move = prev_low - curr_low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        # Initialize EMAs with first values
        if not state.adx_initialized and len(prices) >= self._adx_period:
            # Initialize with average of first N periods
            state.tr_ema = tr
            state.plus_dm_ema = plus_dm
            state.minus_dm_ema = minus_dm
            state.adx_ema = 25.0  # Start at threshold
            state.adx_initialized = True
        elif not state.adx_initialized:
            return 0.0

        # Update Wilder smoothing EMAs
        state.tr_ema = state.tr_ema + alpha * (tr - state.tr_ema)
        state.plus_dm_ema = state.plus_dm_ema + alpha * (plus_dm - state.plus_dm_ema)
        state.minus_dm_ema = state.minus_dm_ema + alpha * (minus_dm - state.minus_dm_ema)

        # Calculate +DI and -DI
        if state.tr_ema > 0:
            plus_di = 100.0 * state.plus_dm_ema / state.tr_ema
            minus_di = 100.0 * state.minus_dm_ema / state.tr_ema
        else:
            plus_di = 0.0
            minus_di = 0.0

        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            dx = 0.0

        # Update ADX with Wilder smoothing
        state.adx_ema = state.adx_ema + alpha * (dx - state.adx_ema)
        state.adx = state.adx_ema

        return state.adx

    def _evaluate_indicators(
        self,
        state: MomentumState,
    ) -> tuple[SignalDirection, float, str]:
        """
        Evaluate indicator confluence for signal generation.

        Returns (direction, strength, rationale)
        """
        bullish_signals = 0
        bearish_signals = 0
        reasons = []

        # MA Crossover
        if state.fast_ma > state.slow_ma:
            bullish_signals += 1
            reasons.append(f"MA crossover bullish (fast={state.fast_ma:.2f} > slow={state.slow_ma:.2f})")
        elif state.fast_ma < state.slow_ma:
            bearish_signals += 1
            reasons.append(f"MA crossover bearish (fast={state.fast_ma:.2f} < slow={state.slow_ma:.2f})")

        # RSI
        if state.rsi < self._rsi_oversold:
            bullish_signals += 1
            reasons.append(f"RSI oversold ({state.rsi:.1f} < {self._rsi_oversold})")
        elif state.rsi > self._rsi_overbought:
            bearish_signals += 1
            reasons.append(f"RSI overbought ({state.rsi:.1f} > {self._rsi_overbought})")

        # MACD
        if state.macd > state.macd_signal:
            bullish_signals += 1
            reasons.append(f"MACD bullish ({state.macd:.4f} > signal {state.macd_signal:.4f})")
        elif state.macd < state.macd_signal:
            bearish_signals += 1
            reasons.append(f"MACD bearish ({state.macd:.4f} < signal {state.macd_signal:.4f})")

        # Demand/Supply Zone Analysis (MoonDev-inspired)
        if self._demand_zones_enabled:
            if state.at_demand_zone:
                bullish_signals += 1
                reasons.append(f"At demand zone (support, bias={state.zone_bias})")
            elif state.at_supply_zone:
                bearish_signals += 1
                reasons.append(f"At supply zone (resistance, bias={state.zone_bias})")

            # Zone bias can add partial weight
            if state.zone_bias == "bullish" and not state.at_supply_zone:
                bullish_signals += 0.5
                reasons.append("Zone structure bullish")
            elif state.zone_bias == "bearish" and not state.at_demand_zone:
                bearish_signals += 0.5
                reasons.append("Zone structure bearish")

        # Determine direction and strength (now using 4 indicators instead of 3)
        max_signals = 4 if self._demand_zones_enabled else 3
        total_signals = bullish_signals + bearish_signals

        if total_signals == 0:
            return SignalDirection.FLAT, 0.0, "No clear momentum signal"

        if bullish_signals > bearish_signals:
            strength = bullish_signals / max_signals  # Normalize to [0, 1]
            return SignalDirection.LONG, strength, " | ".join(reasons)
        elif bearish_signals > bullish_signals:
            strength = bearish_signals / max_signals
            return SignalDirection.SHORT, -strength, " | ".join(reasons)
        else:
            return SignalDirection.FLAT, 0.0, "Mixed signals - no clear direction"

    def _calculate_confidence(self, state: MomentumState) -> float:
        """
        Calculate confidence based on indicator agreement.

        TODO: Add more factors:
        - Volume confirmation
        - Trend strength (ADX)
        - Historical accuracy
        """
        # Count agreeing indicators
        agreement = 0

        ma_bullish = state.fast_ma > state.slow_ma
        rsi_bullish = state.rsi < 50
        macd_bullish = state.macd > state.macd_signal

        if ma_bullish == rsi_bullish:
            agreement += 1
        if ma_bullish == macd_bullish:
            agreement += 1
        if rsi_bullish == macd_bullish:
            agreement += 1

        # Base confidence + agreement bonus
        confidence = 0.4 + (agreement * 0.15)

        # Boost confidence if at a zone that confirms direction
        if state.at_demand_zone and ma_bullish:
            confidence += 0.1
        elif state.at_supply_zone and not ma_bullish:
            confidence += 0.1

        return min(0.9, confidence)

    def _update_zone_candle(self, state: MomentumState, event: MarketDataEvent) -> None:
        """
        Update demand zone detector with candle data.

        Aggregates tick data into candles for zone detection.
        """
        from datetime import datetime, timezone
        import time

        current_time = int(time.time())
        candle_boundary = current_time - (current_time % self._candle_period_seconds)

        price = event.mid or event.last
        if price <= 0:
            return

        # Check if we need to start a new candle
        if state.last_candle_time != candle_boundary:
            # Complete previous candle if we have one
            if state.last_candle_time > 0 and state.candle_open > 0:
                candle = CandleStick(
                    timestamp=datetime.fromtimestamp(state.last_candle_time, tz=timezone.utc),
                    open=state.candle_open,
                    high=state.candle_high,
                    low=state.candle_low,
                    close=price,  # Previous price was the close
                    volume=state.candle_volume,
                )
                state.zone_detector.add_candle(candle)

                # Update zone analysis
                self._update_zone_analysis(state, price)

            # Start new candle
            state.last_candle_time = candle_boundary
            state.candle_open = price
            state.candle_high = price
            state.candle_low = price
            state.candle_volume = event.volume
        else:
            # Update current candle
            state.candle_high = max(state.candle_high, event.high if event.high > 0 else price)
            state.candle_low = min(state.candle_low, event.low if event.low > 0 else price)
            state.candle_volume += event.volume

    def _update_zone_analysis(self, state: MomentumState, current_price: float) -> None:
        """Update zone analysis for the symbol."""
        if not state.zone_detector:
            return

        analysis = state.zone_detector.get_zone_analysis(current_price)

        state.at_demand_zone = analysis["at_demand_zone"]
        state.at_supply_zone = analysis["at_supply_zone"]
        state.zone_bias = analysis["bias"]

        if state.at_demand_zone:
            logger.debug(f"Momentum: {state.symbol} at demand zone (bias={state.zone_bias})")
        elif state.at_supply_zone:
            logger.debug(f"Momentum: {state.symbol} at supply zone (bias={state.zone_bias})")

    def get_zone_status(self, symbol: str) -> dict | None:
        """Get demand/supply zone status for a symbol."""
        if symbol not in self._symbols:
            return None

        state = self._symbols[symbol]
        if not state.zone_detector:
            return None

        current_price = state.prices[-1] if state.prices else 0
        return state.zone_detector.get_zone_analysis(current_price)

    def _get_asset_class(self, symbol: str) -> str:
        """
        QUICK WIN #3: Determine asset class from symbol.

        Used for session-based filtering.
        """
        symbol_upper = symbol.upper()

        # Forex pairs
        if len(symbol) == 6 and symbol_upper.isalpha():
            return "forex"
        if symbol_upper in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]:
            return "forex"

        # Index futures
        if symbol_upper in ["ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K"]:
            return "futures"

        # Energy commodities
        if symbol_upper in ["CL", "MCL", "NG", "RB", "HO"]:
            return "commodities"

        # Metals
        if symbol_upper in ["GC", "MGC", "SI", "SIL", "HG", "PL"]:
            return "commodities"

        # Agriculture
        if symbol_upper in ["ZC", "ZW", "ZS", "ZM", "ZL"]:
            return "commodities"

        # Bonds
        if symbol_upper in ["ZB", "ZN", "ZF"]:
            return "futures"

        # Default to equity
        return "equity"

    # =========================================================================
    # Phase 2: RSI Trend Filter Mode
    # =========================================================================

    def _get_rsi_filter_mode(self, rsi: float) -> str:
        """
        Determine RSI filter mode based on current RSI level.

        Phase 2: RSI Trend Filter Mode
        - Trend Zone (40-60): Only trade with MA trend direction
        - Reversal Zone (<30 or >70): Contrarian signals allowed
        - Transition Zone: Reduced signal strength

        Args:
            rsi: Current RSI value

        Returns:
            Filter mode: "trend", "reversal", or "transition"
        """
        if self._rsi_filter_mode != "adaptive":
            return self._rsi_filter_mode

        low_zone, high_zone = self._rsi_trend_zone

        if rsi <= self._rsi_oversold or rsi >= self._rsi_overbought:
            return "reversal"
        elif low_zone <= rsi <= high_zone:
            return "trend"
        else:
            return "transition"

    def _apply_rsi_trend_filter(
        self,
        direction: SignalDirection,
        strength: float,
        state: MomentumState,
    ) -> tuple[SignalDirection, float, str]:
        """
        Apply RSI trend filter to the signal.

        Phase 2: RSI Trend Filter Mode
        Based on research showing RSI works differently in trending vs ranging markets:
        - In trending markets (RSI 40-60): Follow the trend, ignore RSI extremes
        - In extreme RSI: Look for reversals

        Args:
            direction: Proposed signal direction
            strength: Signal strength
            state: MomentumState with RSI and MA data

        Returns:
            (adjusted_direction, adjusted_strength, filter_note)
        """
        if not self._rsi_trend_filter_enabled:
            return direction, strength, ""

        rsi = state.rsi
        filter_mode = self._get_rsi_filter_mode(rsi)

        ma_bullish = state.fast_ma > state.slow_ma

        if filter_mode == "trend":
            # In trend zone: only trade with MA direction
            if direction == SignalDirection.LONG and not ma_bullish:
                return SignalDirection.FLAT, 0.0, "RSI_FILTER: Long blocked (MA bearish in trend zone)"
            elif direction == SignalDirection.SHORT and ma_bullish:
                return SignalDirection.FLAT, 0.0, "RSI_FILTER: Short blocked (MA bullish in trend zone)"
            else:
                # Signal aligned with trend, boost slightly
                return direction, min(1.0, strength * 1.1), "RSI_FILTER: Aligned with trend"

        elif filter_mode == "reversal":
            # In extreme RSI zone: contrarian signals more valid
            if rsi <= self._rsi_oversold and direction == SignalDirection.LONG:
                # Oversold + Long = good reversal setup
                return direction, min(1.0, strength * 1.2), f"RSI_FILTER: Oversold reversal ({rsi:.1f})"
            elif rsi >= self._rsi_overbought and direction == SignalDirection.SHORT:
                # Overbought + Short = good reversal setup
                return direction, min(1.0, strength * 1.2), f"RSI_FILTER: Overbought reversal ({rsi:.1f})"
            elif rsi <= self._rsi_oversold and direction == SignalDirection.SHORT:
                # Shorting into oversold - risky
                return direction, strength * 0.5, f"RSI_FILTER: Caution shorting oversold ({rsi:.1f})"
            elif rsi >= self._rsi_overbought and direction == SignalDirection.LONG:
                # Buying into overbought - risky
                return direction, strength * 0.5, f"RSI_FILTER: Caution buying overbought ({rsi:.1f})"

        else:  # transition
            # In transition zone: reduce all signals slightly
            return direction, strength * 0.8, f"RSI_FILTER: Transition zone ({rsi:.1f})"

        return direction, strength, ""

    def get_rsi_filter_status(self, symbol: str) -> dict | None:
        """Get RSI filter status for a symbol."""
        if symbol not in self._symbols:
            return None

        state = self._symbols[symbol]
        filter_mode = self._get_rsi_filter_mode(state.rsi)

        return {
            "enabled": self._rsi_trend_filter_enabled,
            "rsi": state.rsi,
            "filter_mode": filter_mode,
            "ma_bullish": state.fast_ma > state.slow_ma,
            "trend_zone": self._rsi_trend_zone,
            "overbought": self._rsi_overbought,
            "oversold": self._rsi_oversold,
        }

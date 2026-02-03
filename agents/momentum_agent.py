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
        self._slow_period = config.parameters.get("slow_period", 30)
        self._signal_period = config.parameters.get("signal_period", 9)
        self._rsi_period = config.parameters.get("rsi_period", 14)
        self._rsi_overbought = config.parameters.get("rsi_overbought", 70)
        self._rsi_oversold = config.parameters.get("rsi_oversold", 30)

        # State per symbol
        self._symbols: dict[str, MomentumState] = {}
        self._max_lookback = max(self._slow_period, self._rsi_period) * 2

    async def initialize(self) -> None:
        """Initialize momentum tracking."""
        logger.info(
            f"MomentumAgent initializing with MA({self._fast_period}/{self._slow_period}), "
            f"RSI({self._rsi_period})"
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
            )

        state = self._symbols[symbol]

        # Update price
        prev_price = state.prices[-1] if state.prices else price
        state.prices.append(price)

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
                await self._event_bus.publish_signal(signal)
                self._audit_logger.log_event(signal)

    def _generate_momentum_signal(self, state: MomentumState) -> SignalEvent | None:
        """
        Generate momentum signal based on multiple indicators.

        TODO: Implement more sophisticated momentum models:
        - ADX for trend strength
        - Bollinger Bands for volatility-adjusted signals
        - Volume confirmation
        - Multiple timeframe analysis
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

        # Determine signal from indicator confluence
        signal_direction, strength, rationale = self._evaluate_indicators(state)

        # Only signal on direction change
        if signal_direction == state.last_signal:
            return None

        state.last_signal = signal_direction

        if signal_direction == SignalDirection.FLAT:
            return None

        return SignalEvent(
            source_agent=self.name,
            strategy_name="momentum_trend",
            symbol=state.symbol,
            direction=signal_direction,
            strength=strength,
            confidence=self._calculate_confidence(state),
            rationale=rationale,
            data_sources=(state.symbol, "IB_market_data"),
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

        # Determine direction and strength
        total_signals = bullish_signals + bearish_signals

        if total_signals == 0:
            return SignalDirection.FLAT, 0.0, "No clear momentum signal"

        if bullish_signals > bearish_signals:
            strength = bullish_signals / 3.0  # Normalize to [0, 1]
            return SignalDirection.LONG, strength, " | ".join(reasons)
        elif bearish_signals > bullish_signals:
            strength = bearish_signals / 3.0
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

        return min(0.9, confidence)

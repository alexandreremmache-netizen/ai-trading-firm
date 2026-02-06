"""
TTM Squeeze Volatility Agent
============================

Generates signals based on TTM Squeeze indicator (Bollinger Bands inside Keltner Channels).
Identifies volatility compression and breakout opportunities.

Responsibility: TTM Squeeze signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from strategies.ttm_squeeze_strategy import (
    TTMSqueezeStrategy,
    create_ttm_squeeze_strategy,
    SqueezeState,
    MomentumDirection,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class SymbolState:
    """State tracking for a symbol."""
    prices: deque = field(default_factory=lambda: deque(maxlen=200))
    highs: deque = field(default_factory=lambda: deque(maxlen=200))
    lows: deque = field(default_factory=lambda: deque(maxlen=200))
    last_signal: SignalDirection = SignalDirection.FLAT
    in_squeeze: bool = False
    squeeze_bars: int = 0


class TTMSqueezeAgent(SignalAgent):
    """
    TTM Squeeze Volatility Agent.

    Implements TTM Squeeze strategy:
    1. Detects volatility squeeze (BB inside KC)
    2. Monitors momentum direction during squeeze
    3. Signals on squeeze release with momentum confirmation

    Signal output:
    - Direction based on momentum at squeeze release
    - Confidence based on squeeze duration and momentum strength
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._bb_length = config.parameters.get("bb_length", 20)
        self._bb_mult = config.parameters.get("bb_mult", 2.0)
        self._kc_length = config.parameters.get("kc_length", 20)
        self._kc_mult = config.parameters.get("kc_mult", 1.5)
        self._mom_length = config.parameters.get("mom_length", 12)
        self._min_squeeze_bars = config.parameters.get("min_squeeze_bars", 6)

        # FIX-13: Strategy expects bb_period/kc_period/kc_atr_mult (not bb_length/kc_length/kc_mult)
        self._strategy = create_ttm_squeeze_strategy({
            "bb_period": self._bb_length,
            "bb_mult": self._bb_mult,
            "kc_period": self._kc_length,
            "kc_atr_mult": self._kc_mult,
            "mom_length": self._mom_length,
        })

        # State tracking per symbol
        self._states: dict[str, SymbolState] = {}

        logger.info(
            f"TTMSqueezeAgent initialized with BB({self._bb_length},{self._bb_mult}), "
            f"KC({self._kc_length},{self._kc_mult}), min_squeeze={self._min_squeeze_bars}"
        )

    async def initialize(self) -> None:
        """Initialize squeeze tracking."""
        logger.info(f"TTMSqueezeAgent ready: BB({self._bb_length}), KC({self._kc_length})")

    async def _emit_warmup_heartbeat(self, symbol: str, reason: str) -> None:
        """Emit FLAT heartbeat signal during warmup to participate in barrier sync."""
        signal = SignalEvent(
            source_agent=self.name,
            symbol=symbol,
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.0,
            rationale=f"Warmup: {reason}",
            data_sources=("warmup",),
        )
        await self._event_bus.publish_signal(signal)

    def _get_state(self, symbol: str) -> SymbolState:
        """Get or create state for symbol."""
        if symbol not in self._states:
            self._states[symbol] = SymbolState()
        return self._states[symbol]

    async def process_event(self, event: Event) -> None:
        """Process market data and check for squeeze signals."""
        if event.event_type == EventType.MARKET_DATA:
            await self._process_market_data(event)

    async def _process_market_data(self, event: MarketDataEvent) -> None:
        """Process market data and check for squeeze signals."""
        symbol = event.symbol
        price = event.last
        high = getattr(event, 'high', price) or price
        low = getattr(event, 'low', price) or price
        timestamp = event.timestamp

        if price is None or price <= 0:
            return

        state = self._get_state(symbol)

        # Update price history
        state.prices.append(price)
        state.highs.append(high)
        state.lows.append(low)

        # Need enough data
        min_length = max(self._bb_length, self._kc_length, self._mom_length) + 10
        if len(state.prices) < min_length:
            await self._emit_warmup_heartbeat(
                symbol,
                f"Collecting data ({len(state.prices)}/{min_length} bars)"
            )
            return

        # Analyze squeeze state
        close = np.array(state.prices)
        high = np.array(state.highs)
        low = np.array(state.lows)

        # Strategy expects: analyze(symbol, high, low, close)
        squeeze_reading = self._strategy.analyze(
            symbol=symbol,
            high=high,
            low=low,
            close=close,
        )

        # Convert SqueezeReading to dict format expected by agent
        squeeze_result = {
            "squeeze_state": squeeze_reading.squeeze_state,
            "momentum_direction": squeeze_reading.momentum_direction,
            "momentum_value": squeeze_reading.momentum_value,
            "squeeze_ratio": 1.0 - squeeze_reading.squeeze_intensity if squeeze_reading.squeeze_intensity else 1.0,
        }

        if squeeze_result is None:
            await self._emit_warmup_heartbeat(symbol, "No squeeze data")
            return

        squeeze_state = squeeze_result.get("squeeze_state", SqueezeState.SQUEEZE_OFF)
        momentum_dir = squeeze_result.get("momentum_direction", MomentumDirection.NEUTRAL)
        momentum_value = squeeze_result.get("momentum_value", 0.0)
        squeeze_ratio = squeeze_result.get("squeeze_ratio", 1.0)

        signal_emitted = False

        # Track squeeze duration
        if squeeze_state == SqueezeState.SQUEEZE_ON:
            if not state.in_squeeze:
                state.in_squeeze = True
                state.squeeze_bars = 1
            else:
                state.squeeze_bars += 1
        elif squeeze_state in (SqueezeState.SQUEEZE_OFF, SqueezeState.SQUEEZE_FIRING):
            # FIX-12: SQUEEZE_FIRING = just transitioned from ON to OFF (the release event)
            # This is THE signal - BB just broke outside KC after being compressed
            was_in_squeeze = state.in_squeeze
            squeeze_duration = state.squeeze_bars
            state.in_squeeze = False
            state.squeeze_bars = 0

            # Signal on squeeze release if conditions met
            if was_in_squeeze and squeeze_duration >= self._min_squeeze_bars:
                await self._generate_squeeze_signal(
                    symbol=symbol,
                    price=price,
                    momentum_dir=momentum_dir,
                    momentum_value=momentum_value,
                    squeeze_duration=squeeze_duration,
                    squeeze_ratio=squeeze_ratio,
                    state=state,
                    timestamp=timestamp,
                )
                signal_emitted = True
        else:
            # NO_SQUEEZE - reset
            state.in_squeeze = False
            state.squeeze_bars = 0

        # Emit heartbeat if no real signal was generated
        if not signal_emitted:
            await self._emit_warmup_heartbeat(symbol, f"Squeeze state: {squeeze_state.name}")

    async def _generate_squeeze_signal(
        self,
        symbol: str,
        price: float,
        momentum_dir: MomentumDirection,
        momentum_value: float,
        squeeze_duration: int,
        squeeze_ratio: float,
        state: SymbolState,
        timestamp: datetime,
    ) -> None:
        """Generate signal on squeeze release."""
        # Determine direction from momentum
        if momentum_dir == MomentumDirection.BULLISH:
            direction = SignalDirection.LONG
        elif momentum_dir == MomentumDirection.BEARISH:
            direction = SignalDirection.SHORT
        else:
            return  # No clear direction

        # Skip if same as last signal
        if state.last_signal == direction:
            return

        state.last_signal = direction

        # Calculate confidence
        # Base confidence from momentum strength
        confidence = min(0.9, abs(momentum_value) / 10.0 + 0.3)

        # Bonus for longer squeeze duration (more compression = bigger move expected)
        duration_bonus = min(0.2, squeeze_duration / 20.0 * 0.2)
        confidence += duration_bonus

        # Adjust for squeeze ratio (tighter squeeze = better)
        if squeeze_ratio < 0.8:
            confidence += 0.1

        confidence = min(0.95, confidence)

        # Build rationale
        rationale_parts = [
            "TTM Squeeze release",
            f"Duration: {squeeze_duration} bars",
            f"Momentum: {momentum_value:.2f} ({momentum_dir.value})",
            f"Squeeze ratio: {squeeze_ratio:.2f}",
        ]

        # Calculate ATR for stops
        prices = np.array(state.prices)
        highs = np.array(state.highs)
        lows = np.array(state.lows)

        # FIX-33: Proper 3-component True Range calculation
        h = highs[-14:]
        l = lows[-14:]
        c = prices[-14:]
        if len(c) > 1:
            prev_c = np.concatenate([[c[0]], c[:-1]])
            tr = np.maximum(
                h - l,
                np.maximum(np.abs(h - prev_c), np.abs(l - prev_c))
            )
            atr = np.mean(tr)
        else:
            atr = price * 0.02

        if direction == SignalDirection.LONG:
            stop_loss = price - (atr * 2.0)
            take_profit = price + (atr * 4.0)
        else:
            stop_loss = price + (atr * 2.0)
            take_profit = price - (atr * 4.0)

        # Create signal
        signal = SignalEvent(
            source_agent=self.name,
            symbol=symbol,
            direction=direction,
            strength=confidence,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
            data_sources=("ttm_squeeze", "bollinger", "keltner", "momentum"),
            stop_loss=stop_loss,
            target_price=take_profit,
        )

        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

        logger.info(
            f"TTMSqueezeAgent signal: {symbol} {direction.value} "
            f"squeeze_bars={squeeze_duration} confidence={confidence:.2f}"
        )

    def get_status(self) -> dict:
        """Get agent status for monitoring."""
        base_status = super().get_status()

        squeeze_status = {}
        for symbol, state in self._states.items():
            squeeze_status[symbol] = {
                "in_squeeze": state.in_squeeze,
                "squeeze_bars": state.squeeze_bars,
                "last_signal": state.last_signal.value,
                "data_points": len(state.prices),
            }

        base_status.update({
            "bb_length": self._bb_length,
            "bb_mult": self._bb_mult,
            "kc_length": self._kc_length,
            "kc_mult": self._kc_mult,
            "min_squeeze_bars": self._min_squeeze_bars,
            "tracked_symbols": len(self._states),
            "squeeze_status": squeeze_status,
        })
        return base_status

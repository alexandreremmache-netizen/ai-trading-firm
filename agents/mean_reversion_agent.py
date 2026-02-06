"""
Mean Reversion Trading Agent
============================

Generates signals based on single-asset mean reversion strategies.
Uses RSI extremes, Bollinger Band touches, and Z-score analysis.

Responsibility: Mean reversion signal generation ONLY.
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
from strategies.mean_reversion_strategy import (
    MeanReversionStrategy,
    create_mean_reversion_strategy,
    MarketRegime,
    SignalType,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class SymbolState:
    """State tracking for mean reversion analysis."""
    prices: deque = field(default_factory=lambda: deque(maxlen=100))
    last_signal: SignalDirection = SignalDirection.FLAT
    last_signal_time: datetime | None = None
    signal_cooldown_minutes: int = 30


class MeanReversionAgent(SignalAgent):
    """
    Mean Reversion Trading Agent.

    Implements mean reversion strategies:
    1. RSI extreme reversals (oversold/overbought)
    2. Bollinger Band mean reversion (touch outer bands)
    3. Z-score based entries (>2 std from mean)
    4. Regime filtering (avoid trending markets)

    Signal output:
    - Counter-trend entries at extremes
    - Confidence based on multiple indicator confirmation
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        # FIX-19: Connors RSI(2) defaults, not RSI(14)
        self._rsi_period = config.parameters.get("rsi_period", 2)
        self._rsi_oversold = config.parameters.get("rsi_oversold", 5)
        self._rsi_overbought = config.parameters.get("rsi_overbought", 95)
        self._bb_period = config.parameters.get("bb_period", 20)
        self._bb_std = config.parameters.get("bb_std", 2.0)
        self._zscore_threshold = config.parameters.get("zscore_threshold", 2.0)
        self._min_indicators = config.parameters.get("min_indicators", 2)
        self._cooldown_minutes = config.parameters.get("cooldown_minutes", 30)

        # Create strategy
        self._strategy = create_mean_reversion_strategy({
            "rsi_period": self._rsi_period,
            "rsi_oversold": self._rsi_oversold,
            "rsi_overbought": self._rsi_overbought,
            "bb_period": self._bb_period,
            "bb_std": self._bb_std,
            "zscore_threshold": self._zscore_threshold,
        })

        # State tracking per symbol
        self._states: dict[str, SymbolState] = {}

        logger.info(
            f"MeanReversionAgent initialized with RSI({self._rsi_period}), "
            f"BB({self._bb_period},{self._bb_std}), z_thresh={self._zscore_threshold}"
        )

    async def initialize(self) -> None:
        """Initialize mean reversion tracking."""
        logger.info(f"MeanReversionAgent ready: RSI({self._rsi_period}), BB({self._bb_period})")

    async def _emit_warmup_heartbeat(self, symbol: str, reason: str) -> None:
        """Emit FLAT heartbeat signal to participate in barrier sync."""
        signal = SignalEvent(
            source_agent=self.name,
            symbol=symbol,
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.0,
            rationale=f"Heartbeat: {reason}",
            data_sources=("heartbeat",),
        )
        await self._event_bus.publish_signal(signal)

    def _get_state(self, symbol: str) -> SymbolState:
        """Get or create state for symbol."""
        if symbol not in self._states:
            self._states[symbol] = SymbolState(
                signal_cooldown_minutes=self._cooldown_minutes
            )
        return self._states[symbol]

    async def process_event(self, event: Event) -> None:
        """Process market data for mean reversion signals."""
        if event.event_type == EventType.MARKET_DATA:
            await self._process_market_data(event)

    async def _process_market_data(self, event: MarketDataEvent) -> None:
        """Process market data for mean reversion signals."""
        symbol = event.symbol
        price = event.last
        timestamp = event.timestamp

        if price is None or price <= 0:
            return

        state = self._get_state(symbol)

        # Update price history
        state.prices.append(price)

        # Need enough data
        min_length = max(self._rsi_period, self._bb_period) + 5
        if len(state.prices) < min_length:
            await self._emit_warmup_heartbeat(symbol, f"Collecting data ({len(state.prices)}/{min_length})")
            return

        # Check cooldown
        if state.last_signal_time is not None:
            elapsed = (timestamp - state.last_signal_time).total_seconds() / 60
            if elapsed < state.signal_cooldown_minutes:
                await self._emit_warmup_heartbeat(symbol, "Cooldown")
                return

        # Analyze for mean reversion opportunities
        # Strategy expects: analyze(symbol, high, low, close)
        # We only have close prices, so use close for high/low as approximation
        close = np.array(state.prices)
        # For mean reversion with only close prices, use close as approximation
        high = close
        low = close

        # Get full analysis state from strategy
        analysis_state = self._strategy.analyze(
            symbol=symbol,
            high=high,
            low=low,
            close=close,
        )

        # Convert MeanReversionState to dict format
        # Build analysis dict from state
        analysis = {
            "regime": analysis_state.regime,
            "rsi": analysis_state.rsi,
            "zscore": analysis_state.zscore,
            "bb_position": analysis_state.bb_position,
            "atr": price * 0.015,  # Approximate ATR
        }

        # Collect signals based on indicator values
        signals = []
        if analysis_state.rsi < self._rsi_oversold:
            signals.append({"direction": "long", "type": "rsi", "strength": (self._rsi_oversold - analysis_state.rsi) / self._rsi_oversold, "value": analysis_state.rsi})
        elif analysis_state.rsi > self._rsi_overbought:
            signals.append({"direction": "short", "type": "rsi", "strength": (analysis_state.rsi - self._rsi_overbought) / (100 - self._rsi_overbought), "value": analysis_state.rsi})

        if analysis_state.bb_position < -0.9:
            signals.append({"direction": "long", "type": "bollinger", "strength": 0.5, "value": analysis_state.bb_position})
        elif analysis_state.bb_position > 0.9:
            signals.append({"direction": "short", "type": "bollinger", "strength": 0.5, "value": analysis_state.bb_position})

        if analysis_state.zscore < -self._zscore_threshold:
            signals.append({"direction": "long", "type": "zscore", "strength": 0.5, "value": analysis_state.zscore})
        elif analysis_state.zscore > self._zscore_threshold:
            signals.append({"direction": "short", "type": "zscore", "strength": 0.5, "value": analysis_state.zscore})

        analysis["signals"] = signals

        if analysis is None:
            return

        # Check regime - avoid trending markets
        regime = analysis.get("regime", MarketRegime.RANGE_BOUND)
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            await self._emit_warmup_heartbeat(symbol, f"Trending regime: {regime.name}")
            return  # Skip mean reversion in trending markets

        # Collect signals
        signals = analysis.get("signals", [])
        if len(signals) < self._min_indicators:
            await self._emit_warmup_heartbeat(symbol, "Insufficient indicators")
            return  # Need multiple confirmations

        # Determine direction from signals
        long_signals = [s for s in signals if s.get("direction") == "long"]
        short_signals = [s for s in signals if s.get("direction") == "short"]

        if len(long_signals) >= self._min_indicators:
            direction = SignalDirection.LONG
            active_signals = long_signals
        elif len(short_signals) >= self._min_indicators:
            direction = SignalDirection.SHORT
            active_signals = short_signals
        else:
            await self._emit_warmup_heartbeat(symbol, "No consensus")
            return  # No consensus

        # Skip same direction as last signal
        if state.last_signal == direction:
            await self._emit_warmup_heartbeat(symbol, "Same direction")
            return

        # Calculate confidence from signal strengths
        avg_strength = np.mean([s.get("strength", 0.5) for s in active_signals])
        num_confirmations = len(active_signals)

        # Base confidence + bonus for more confirmations
        confidence = min(0.9, avg_strength + (num_confirmations - self._min_indicators) * 0.1)

        # Reduce confidence in volatile regimes
        if regime == MarketRegime.HIGH_VOL:
            confidence *= 0.8

        # Build rationale
        rationale_parts = [f"Mean reversion ({num_confirmations} confirmations)"]

        for sig in active_signals[:3]:  # Limit to top 3
            sig_type = sig.get("type", "unknown")
            sig_value = sig.get("value", 0)
            if sig_type == "rsi":
                rationale_parts.append(f"RSI: {sig_value:.1f}")
            elif sig_type == "bollinger":
                rationale_parts.append(f"BB: {sig_value:.2f}σ")
            elif sig_type == "zscore":
                rationale_parts.append(f"Z-score: {sig_value:.2f}")

        rationale_parts.append(f"Regime: {regime.value}")

        # Calculate stops
        atr = analysis.get("atr", price * 0.015)
        if direction == SignalDirection.LONG:
            stop_loss = price - (atr * 2.0)
            target_price = price + (atr * 3.0)  # Tighter target for mean reversion
        else:
            stop_loss = price + (atr * 2.0)
            target_price = price - (atr * 3.0)

        # Update state
        state.last_signal = direction
        state.last_signal_time = timestamp

        # Create signal
        signal = SignalEvent(
            source_agent=self.name,
            symbol=symbol,
            direction=direction,
            strength=confidence,
            confidence=confidence,
            rationale=" | ".join(rationale_parts) + f" | Regime: {regime.value} | Confirmations: {num_confirmations}",
            data_sources=("rsi", "bollinger_bands", "zscore", "regime_detection"),
            stop_loss=stop_loss,
            target_price=target_price,
        )

        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

        logger.info(
            f"MeanReversionAgent signal: {symbol} {direction.value} "
            f"confirmations={num_confirmations} confidence={confidence:.2f} regime={regime.value}"
        )

    def get_status(self) -> dict:
        """Get agent status for monitoring."""
        base_status = super().get_status()

        symbol_status = {}
        for symbol, state in self._states.items():
            symbol_status[symbol] = {
                "data_points": len(state.prices),
                "last_signal": state.last_signal.value,
                "last_signal_time": state.last_signal_time.isoformat() if state.last_signal_time else None,
            }

        base_status.update({
            "rsi_period": self._rsi_period,
            "rsi_oversold": self._rsi_oversold,
            "rsi_overbought": self._rsi_overbought,
            "bb_period": self._bb_period,
            "bb_std": self._bb_std,
            "zscore_threshold": self._zscore_threshold,
            "min_indicators": self._min_indicators,
            "tracked_symbols": len(self._states),
            "symbol_status": symbol_status,
        })
        return base_status

"""
Session-Based Trading Agent
===========================

Generates signals based on trading session patterns and opening range breakouts.
Implements session-specific strategies for Asian, London, NY sessions.

Responsibility: Session-based signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from collections import deque
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
from strategies.session_strategy import (
    SessionStrategy,
    create_session_strategy,
    SessionWindow,
    SESSION_WINDOWS,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


class SessionAgent(SignalAgent):
    """
    Session-Based Trading Agent.

    Implements trading strategies based on:
    1. Opening Range Breakouts (ORB)
    2. Session momentum patterns
    3. Session-specific volume analysis
    4. Overlap session opportunities (London-NY)

    Signal output:
    - Direction based on breakout/momentum
    - Confidence based on volume confirmation and session quality
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._orb_minutes = config.parameters.get("orb_minutes", 30)
        self._breakout_threshold = config.parameters.get("breakout_threshold", 0.002)
        self._volume_confirmation = config.parameters.get("volume_confirmation", True)
        # Volume ratio: 1.5x minimum per industry research (was 1.2x, too low)
        # Research shows 1.5x minimum, 2.0x optimal for breakout confirmation
        self._min_volume_ratio = config.parameters.get("min_volume_ratio", 1.5)
        self._allowed_sessions = config.parameters.get(
            "allowed_sessions", ["london", "new_york", "london_ny_overlap"]
        )

        # Create session strategy
        self._strategy = create_session_strategy({
            "opening_range_minutes": self._orb_minutes,
            "breakout_threshold_pct": self._breakout_threshold * 100,
            "volume_confirmation": self._volume_confirmation,
            "min_volume_ratio": self._min_volume_ratio,
        })

        # State tracking per symbol
        self._last_signals: dict[str, SignalDirection] = {}

        # Price history tracking per symbol (needed for strategy)
        from collections import deque
        self._price_history: dict[str, deque] = {}
        self._timestamp_history: dict[str, deque] = {}
        self._volume_history: dict[str, deque] = {}
        self._history_maxlen = 100  # Keep last 100 bars

        logger.info(
            f"SessionAgent initialized with ORB={self._orb_minutes}min, "
            f"threshold={self._breakout_threshold:.2%}, sessions={self._allowed_sessions}"
        )

    async def initialize(self) -> None:
        """Initialize session tracking."""
        logger.info(
            f"SessionAgent ready: ORB={self._orb_minutes}min, "
            f"sessions={self._allowed_sessions}"
        )

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

    async def process_event(self, event: Event) -> None:
        """Process market data and generate session-based signals."""
        if event.event_type == EventType.MARKET_DATA:
            await self._process_market_data(event)

    async def _process_market_data(self, event: MarketDataEvent) -> None:
        """Process market data and generate session-based signals."""
        symbol = event.symbol
        price = event.last
        volume = getattr(event, 'volume', 0) or 0
        timestamp = event.timestamp

        if price is None or price <= 0:
            return

        # Check if current session is allowed
        current_session = self._get_current_session(timestamp)
        if current_session not in self._allowed_sessions:
            await self._emit_warmup_heartbeat(symbol, f"Session {current_session} not active")
            return

        # Initialize history for new symbols
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._history_maxlen)
            self._timestamp_history[symbol] = deque(maxlen=self._history_maxlen)
            self._volume_history[symbol] = deque(maxlen=self._history_maxlen)

        # Get high/low for opening range calculation
        high = getattr(event, 'high', price) or price
        low = getattr(event, 'low', price) or price

        # Add to history
        self._price_history[symbol].append(price)
        self._timestamp_history[symbol].append(timestamp)
        self._volume_history[symbol].append(volume)

        # Need enough history for analysis
        if len(self._price_history[symbol]) < 10:
            await self._emit_warmup_heartbeat(
                symbol,
                f"Collecting data ({len(self._price_history[symbol])}/10 bars)"
            )
            return

        # Convert to numpy arrays for strategy
        prices = np.array(self._price_history[symbol])
        timestamps_list = list(self._timestamp_history[symbol])
        volumes = np.array(self._volume_history[symbol])

        # Calculate ATR approximation (using price range as proxy)
        price_range = np.max(prices[-14:]) - np.min(prices[-14:]) if len(prices) >= 14 else price * 0.02
        atr = price_range / 14 if price_range > 0 else price * 0.02

        avg_volume = np.mean(volumes) if len(volumes) > 0 else volume

        # Strategy expects: analyze(symbol, prices, timestamps, atr, volume, avg_volume, bar_minutes)
        analysis = self._strategy.analyze(
            symbol=symbol,
            prices=prices,
            timestamps=timestamps_list,
            atr=atr,
            volume=volume,
            avg_volume=avg_volume,
            bar_minutes=1,
        )

        if analysis is None:
            return

        # Strategy returns SessionSignal dataclass with direction as "LONG", "SHORT", "NEUTRAL"
        signal_direction = analysis.direction  # "LONG", "SHORT", "NEUTRAL"
        strength = analysis.strength
        stop_loss_price = analysis.stop_loss
        take_profit_price = analysis.take_profit
        volume_confirmed = analysis.volume_confirmed
        signal_type = analysis.signal_type  # "breakout", "momentum", etc.
        rationale = analysis.rationale

        # Convert to SignalDirection enum
        if signal_direction == "LONG":
            direction = SignalDirection.LONG
        elif signal_direction == "SHORT":
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.FLAT

        # Calculate confidence
        confidence = strength
        rationale_parts = [rationale]

        # Volume confirmation bonus
        if volume_confirmed:
            confidence = min(1.0, confidence + 0.1)
            rationale_parts.append("Volume confirmed")

        # Add session info to rationale
        rationale_parts.append(f"Session: {current_session}")

        # Skip if no meaningful signal or below threshold
        if direction == SignalDirection.FLAT or confidence < 0.3:
            return

        # Skip if same signal as last time (avoid spam)
        if self._last_signals.get(symbol) == direction:
            return

        self._last_signals[symbol] = direction

        # Use stop loss and take profit from strategy signal (already calculated)
        stop_loss = stop_loss_price
        take_profit = take_profit_price

        # Create signal event
        signal = SignalEvent(
            source_agent=self.name,
            symbol=symbol,
            direction=direction,
            strength=confidence,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
            data_sources=("session_strategy", signal_type, "volume"),
            stop_loss=stop_loss,
            target_price=take_profit,
        )

        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

        logger.info(
            f"SessionAgent signal: {symbol} {direction.value} "
            f"confidence={confidence:.2f} session={current_session}"
        )

    def _get_current_session(self, timestamp: datetime) -> str:
        """Determine current trading session."""
        # Ensure UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        hour = timestamp.hour

        # Session times in UTC
        # Asian: 00:00-08:00 UTC (Tokyo 09:00-17:00 JST)
        # London: 08:00-16:00 UTC
        # NY: 13:00-21:00 UTC (adjusted for overlap)
        # London-NY Overlap: 13:00-16:00 UTC

        if 13 <= hour < 16:
            return "london_ny_overlap"
        elif 8 <= hour < 16:
            return "london"
        elif 13 <= hour < 21:
            return "new_york"
        elif 0 <= hour < 8:
            return "asian"
        else:
            return "after_hours"

    def get_status(self) -> dict:
        """Get agent status for monitoring."""
        base_status = super().get_status()
        base_status.update({
            "orb_minutes": self._orb_minutes,
            "breakout_threshold": self._breakout_threshold,
            "allowed_sessions": self._allowed_sessions,
            "tracked_symbols": len(self._last_signals),
            "strategy_state": self._strategy.get_status() if hasattr(self._strategy, 'get_status') else {},
        })
        return base_status

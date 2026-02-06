"""
Event-Driven Trading Agent
==========================

Generates signals based on economic calendar events (FOMC, NFP, CPI).
Implements pre/post event positioning strategies.

Responsibility: Event-driven signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from core.agent_base import SignalAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    MarketDataEvent,
    SignalEvent,
    SignalDirection,
)
from strategies.event_driven_strategy import (
    EventDrivenStrategy,
    create_event_driven_strategy,
    EventType as EconEventType,
    EventImpact,
    EconomicEvent,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


class EventDrivenAgent(SignalAgent):
    """
    Event-Driven Trading Agent.

    Implements trading around economic events:
    1. FOMC - Federal Reserve rate decisions
    2. NFP - Non-Farm Payrolls
    3. CPI - Consumer Price Index
    4. GDP - Gross Domestic Product

    Signal output:
    - Pre-event positioning (volatility plays)
    - Post-event momentum following
    - Surprise-based directional trades
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        # Pre-event window: 24h for FOMC (was 2h, too short for proper positioning)
        # Research: institutional positioning starts 24-48h before major Fed events
        self._pre_event_hours = config.parameters.get("pre_event_hours", 24)
        self._post_event_hours = config.parameters.get("post_event_hours", 4)
        # Min surprise: 3.0 std devs for significant moves (was 1.0, too sensitive)
        # Research: only >2.5 std surprise generates sustained directional moves
        self._min_surprise_std = config.parameters.get("min_surprise_std", 3.0)
        self._tracked_events = config.parameters.get("tracked_events", [
            "FOMC", "NFP", "CPI", "GDP", "RETAIL_SALES", "ISM_PMI"
        ])

        # Create strategy
        self._strategy = create_event_driven_strategy({
            "pre_event_hours": self._pre_event_hours,
            "post_event_hours": self._post_event_hours,
            "min_surprise_std": self._min_surprise_std,
        })

        # State tracking
        self._last_signals: dict[str, SignalDirection] = {}
        self._active_events: dict[str, EconomicEvent] = {}

        # Symbol to event type mapping (which symbols react to which events)
        self._symbol_event_sensitivity = config.parameters.get("symbol_sensitivity", {
            "ES": ["FOMC", "NFP", "CPI", "GDP"],
            "MES": ["FOMC", "NFP", "CPI", "GDP"],
            "NQ": ["FOMC", "NFP", "CPI"],
            "MNQ": ["FOMC", "NFP", "CPI"],
            "GC": ["FOMC", "CPI"],
            "MGC": ["FOMC", "CPI"],
            "EURUSD": ["FOMC", "NFP", "CPI", "GDP"],
            "USDJPY": ["FOMC", "NFP"],
            "CL": ["NFP", "GDP", "RETAIL_SALES"],
            "MCL": ["NFP", "GDP", "RETAIL_SALES"],
        })

        logger.info(
            f"EventDrivenAgent initialized tracking {self._tracked_events}, "
            f"pre={self._pre_event_hours}h, post={self._post_event_hours}h"
        )

    async def initialize(self) -> None:
        """Initialize event tracking."""
        logger.info(f"EventDrivenAgent ready: tracking {self._tracked_events}")

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
        """Process market data and generate event-driven signals."""
        if event.event_type == EventType.MARKET_DATA:
            await self._process_market_data(event)

    async def _process_market_data(self, event: MarketDataEvent) -> None:
        """Process market data around economic events."""
        symbol = event.symbol
        price = event.last
        timestamp = event.timestamp

        if price is None or price <= 0:
            return

        # Check which events this symbol is sensitive to
        sensitive_events = self._symbol_event_sensitivity.get(symbol, [])
        if not sensitive_events:
            return

        # Get next upcoming event from strategy
        next_event = self._strategy.get_next_event(timestamp)

        # If no event in strategy, emit heartbeat and return
        if next_event is None:
            await self._emit_warmup_heartbeat(symbol, "No economic events scheduled")
            return

        # Check if this symbol is sensitive to this event
        event_type_str = next_event.event_type.value.upper()
        if event_type_str not in sensitive_events:
            return

        # Get event window
        window = self._strategy.get_event_window(timestamp, next_event)

        # Map EventWindow enum to string
        from strategies.event_driven_strategy import EventWindow
        window_type_map = {
            EventWindow.PRE_EVENT: "pre",
            EventWindow.DURING_EVENT: "during",
            EventWindow.POST_EVENT_EARLY: "post",
            EventWindow.POST_EVENT_LATE: "post_late",
            EventWindow.OUTSIDE_WINDOW: None,
        }
        window_type = window_type_map.get(window, None)

        if window_type is None:
            return

        event_name = next_event.event_type.value.upper()
        time_to_event = (next_event.timestamp - timestamp).total_seconds() / 3600
        surprise = 0.0  # No actual data available yet

        # Build analysis dict for compatibility
        analysis = {
            "event_window": {
                "type": window_type,
                "event_name": event_name,
                "time_to_event_hours": time_to_event,
                "surprise_std": surprise,
            },
            "atr": price * 0.015,  # Approximate ATR
        }

        direction = SignalDirection.FLAT
        confidence = 0.0
        rationale_parts = []

        if window_type == "pre":
            # Pre-event: reduce exposure or position for vol expansion
            signal_result = self._get_pre_event_signal(
                symbol=symbol,
                event_name=event_name,
                time_to_event=time_to_event,
            )
            if signal_result:
                direction = signal_result["direction"]
                confidence = signal_result["confidence"]
                rationale_parts = signal_result["rationale"]

        elif window_type == "post":
            # Post-event: momentum following based on surprise
            signal_result = self._get_post_event_signal(
                symbol=symbol,
                event_name=event_name,
                surprise=surprise,
                price=price,
                analysis=analysis,
            )
            if signal_result:
                direction = signal_result["direction"]
                confidence = signal_result["confidence"]
                rationale_parts = signal_result["rationale"]

        if direction == SignalDirection.FLAT or confidence < 0.3:
            return

        # Skip duplicate signals
        signal_key = f"{symbol}_{event_name}_{window_type}"
        if self._last_signals.get(signal_key) == direction:
            return

        self._last_signals[signal_key] = direction

        # Calculate stops
        atr = analysis.get("atr", price * 0.015)
        if direction == SignalDirection.LONG:
            stop_loss = price - (atr * 2.5)
            take_profit = price + (atr * 5.0)
        else:
            stop_loss = price + (atr * 2.5)
            take_profit = price - (atr * 5.0)

        # Create signal
        signal = SignalEvent(
            source_agent=self.name,
            symbol=symbol,
            direction=direction,
            strength=confidence,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
            data_sources=("economic_calendar", "event_surprise", event_name.lower()),
            stop_loss=stop_loss,
            target_price=take_profit,
        )

        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

        logger.info(
            f"EventDrivenAgent signal: {symbol} {direction.value} "
            f"event={event_name} window={window_type} confidence={confidence:.2f}"
        )

    def _get_pre_event_signal(
        self,
        symbol: str,
        event_name: str,
        time_to_event: float,
    ) -> dict | None:
        """Generate pre-event signal (typically reduce risk or prepare for vol)."""
        # For high-impact events close to release, reduce positions
        if time_to_event <= 1.0 and event_name in ["FOMC", "NFP"]:
            return {
                "direction": SignalDirection.FLAT,
                "confidence": 0.7,
                "rationale": [
                    f"Pre-event caution: {event_name}",
                    f"{time_to_event:.1f}h to release",
                    "Reducing exposure before high-impact event",
                ],
            }

        # Slight bullish bias ahead of expected positive events
        if event_name in ["GDP", "RETAIL_SALES"] and 1.0 < time_to_event <= 2.0:
            return {
                "direction": SignalDirection.LONG,
                "confidence": 0.4,
                "rationale": [
                    f"Pre-event positioning: {event_name}",
                    f"{time_to_event:.1f}h to release",
                    "Mild bullish bias ahead of growth data",
                ],
            }

        return None

    def _get_post_event_signal(
        self,
        symbol: str,
        event_name: str,
        surprise: float,
        price: float,
        analysis: dict,
    ) -> dict | None:
        """Generate post-event signal based on surprise magnitude."""
        if abs(surprise) < self._min_surprise_std:
            return None

        # Determine direction based on surprise
        # Positive surprise (better than expected) = bullish for equities
        # Negative surprise = bearish
        if surprise > 0:
            direction = SignalDirection.LONG
            surprise_desc = "positive"
        else:
            direction = SignalDirection.SHORT
            surprise_desc = "negative"

        # Special handling for rate-sensitive events
        if event_name == "FOMC":
            # Higher rates = bearish for equities
            if surprise > 0:  # More hawkish than expected
                direction = SignalDirection.SHORT
                surprise_desc = "hawkish"
            else:
                direction = SignalDirection.LONG
                surprise_desc = "dovish"

        # CPI: Higher inflation = rate hike expectations = bearish
        if event_name == "CPI" and surprise > 0:
            direction = SignalDirection.SHORT
            surprise_desc = "hot inflation"

        # Confidence scales with surprise magnitude
        confidence = min(0.85, 0.4 + abs(surprise) * 0.15)

        rationale = [
            f"Post-{event_name} momentum",
            f"Surprise: {surprise:.1f} std ({surprise_desc})",
            f"Following event-driven move",
        ]

        return {
            "direction": direction,
            "confidence": confidence,
            "rationale": rationale,
        }

    def register_event(self, event: EconomicEvent) -> None:
        """Register an upcoming economic event."""
        self._strategy.add_event(event)
        logger.info(f"Registered event: {event.event_type.value} at {event.timestamp}")

    def get_status(self) -> dict:
        """Get agent status for monitoring."""
        base_status = super().get_status()
        base_status.update({
            "tracked_events": self._tracked_events,
            "pre_event_hours": self._pre_event_hours,
            "post_event_hours": self._post_event_hours,
            "min_surprise_std": self._min_surprise_std,
            "active_signals": len(self._last_signals),
            "strategy_state": self._strategy.get_status() if hasattr(self._strategy, 'get_status') else {},
        })
        return base_status

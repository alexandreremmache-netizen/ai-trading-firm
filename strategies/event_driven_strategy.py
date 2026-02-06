"""
Event-Driven Strategy (Phase 6.4)
==================================

Strategy for trading around major economic events (FOMC, NFP, CPI, etc.).

Key features:
- Economic calendar integration
- Pre-event positioning (straddles/vol buying)
- Post-event momentum capture
- Risk management around events

Research basis:
- FOMC: Typically 30bp daily vol expansion
- NFP: Strong short-term momentum if surprise
- CPI: Inflation-sensitive assets react 15-30 min
- Event windows: Pre (-30min), During, Post (+2hr)

MATURITY: ALPHA
---------------
Status: New implementation
- [x] Event definitions
- [x] Event window detection
- [x] Signal generation
- [ ] Calendar feed integration
- [ ] Backtesting validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Economic event types."""
    FOMC = "fomc"              # Federal Reserve meeting
    NFP = "nfp"                # Non-Farm Payrolls
    CPI = "cpi"                # Consumer Price Index
    GDP = "gdp"                # GDP release
    PPI = "ppi"                # Producer Price Index
    RETAIL_SALES = "retail"   # Retail Sales
    JOBLESS_CLAIMS = "claims" # Weekly jobless claims
    ISM_MFG = "ism_mfg"       # ISM Manufacturing
    ISM_SVC = "ism_svc"       # ISM Services
    ECB = "ecb"               # European Central Bank
    BOJ = "boj"               # Bank of Japan
    BOE = "boe"               # Bank of England
    EARNINGS = "earnings"     # Corporate earnings
    OTHER = "other"


class EventImpact(Enum):
    """Expected event impact level."""
    HIGH = "high"       # Major market mover
    MEDIUM = "medium"   # Moderate impact
    LOW = "low"         # Minor impact


class EventWindow(Enum):
    """Position in event timeline."""
    PRE_EVENT = "pre"          # Before event (positioning)
    DURING_EVENT = "during"    # Event happening
    POST_EVENT_EARLY = "post_early"    # 0-30 min after
    POST_EVENT_LATE = "post_late"      # 30min-2hr after
    OUTSIDE_WINDOW = "outside"  # No event nearby


@dataclass
class EconomicEvent:
    """Economic event definition."""
    event_type: EventType
    timestamp: datetime
    impact: EventImpact
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    currency: str = "USD"
    description: str = ""


@dataclass
class EventAnalysis:
    """Analysis of event and market reaction."""
    event: EconomicEvent
    window: EventWindow
    surprise: float | None  # actual - forecast
    surprise_std: float | None  # surprise in standard deviations
    vol_expansion: float  # Current vol / normal vol
    direction_bias: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EventSignal:
    """Trading signal from event analysis."""
    symbol: str
    event_type: EventType
    direction: str  # "LONG", "SHORT", "FLAT"
    signal_type: str  # "pre_event_vol", "post_event_momentum", "fade", "exit"
    strength: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    event_window: EventWindow
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# EVENT IMPACT DEFINITIONS
# =============================================================================

EVENT_CHARACTERISTICS = {
    EventType.FOMC: {
        "typical_vol_mult": 1.5,
        "reaction_minutes": 120,
        "pre_window_minutes": 1440,  # 24 hours (Phase 11 correction)
        "sensitive_assets": ["ES", "NQ", "ZN", "ZB", "DX", "GC", "EURUSD"],
        "surprise_threshold_std": 0.5,
    },
    EventType.NFP: {
        "typical_vol_mult": 2.0,
        "reaction_minutes": 60,
        "pre_window_minutes": 30,
        "sensitive_assets": ["ES", "NQ", "DX", "EURUSD", "USDJPY", "GC"],
        "surprise_threshold_std": 1.0,
    },
    EventType.CPI: {
        "typical_vol_mult": 1.8,
        "reaction_minutes": 45,
        "pre_window_minutes": 30,
        "sensitive_assets": ["ES", "NQ", "ZN", "ZB", "TLT", "GC", "TIPS"],
        "surprise_threshold_std": 0.5,
    },
    EventType.GDP: {
        "typical_vol_mult": 1.3,
        "reaction_minutes": 60,
        "pre_window_minutes": 30,
        "sensitive_assets": ["ES", "NQ", "DX"],
        "surprise_threshold_std": 0.5,
    },
    EventType.ECB: {
        "typical_vol_mult": 1.4,
        "reaction_minutes": 90,
        "pre_window_minutes": 45,
        "sensitive_assets": ["EURUSD", "DAX", "BUND"],
        "surprise_threshold_std": 0.5,
    },
}


class EventDrivenStrategy:
    """
    Event-driven trading strategy (Phase 6.4).

    Trades around major economic events using pre-event positioning
    and post-event momentum capture.

    Configuration:
        vol_entry_threshold: Vol expansion threshold for vol trades (default: 0.7)
        momentum_threshold: Min momentum for post-event entry (default: 0.5)
        stop_atr_mult: ATR multiplier for stops (default: 2.0)
        take_profit_atr_mult: ATR multiplier for targets (default: 3.0)
        max_pre_event_position: Max position before event (default: 0.5)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize event-driven strategy."""
        config = config or {}

        # Signal thresholds
        self._vol_entry_threshold = config.get("vol_entry_threshold", 0.7)
        self._momentum_threshold = config.get("momentum_threshold", 0.5)
        self._surprise_threshold = config.get("surprise_threshold", 1.0)

        # Risk settings
        self._stop_atr_mult = config.get("stop_atr_mult", 2.0)
        self._take_profit_atr_mult = config.get("take_profit_atr_mult", 3.0)
        self._max_pre_event_position = config.get("max_pre_event_position", 0.5)

        # State tracking
        self._upcoming_events: list[EconomicEvent] = []
        self._recent_analyses: dict[str, EventAnalysis] = {}
        self._historical_surprises: dict[EventType, list[float]] = {}

        logger.info(
            f"EventDrivenStrategy initialized: "
            f"vol_threshold={self._vol_entry_threshold}, "
            f"momentum_threshold={self._momentum_threshold}"
        )

    def add_event(self, event: EconomicEvent) -> None:
        """Add upcoming event to calendar."""
        self._upcoming_events.append(event)
        self._upcoming_events.sort(key=lambda e: e.timestamp)
        logger.info(f"Added event: {event.event_type.value} at {event.timestamp}")

    def get_next_event(
        self,
        current_time: datetime,
        event_type: EventType | None = None,
    ) -> EconomicEvent | None:
        """Get next upcoming event."""
        for event in self._upcoming_events:
            if event.timestamp > current_time:
                if event_type is None or event.event_type == event_type:
                    return event
        return None

    def get_event_window(
        self,
        current_time: datetime,
        event: EconomicEvent,
    ) -> EventWindow:
        """
        Determine current position in event window.

        Args:
            current_time: Current timestamp
            event: The event to check

        Returns:
            EventWindow indicating position relative to event
        """
        chars = EVENT_CHARACTERISTICS.get(event.event_type, {})
        pre_window = chars.get("pre_window_minutes", 30)
        reaction_window = chars.get("reaction_minutes", 60)

        time_diff = (event.timestamp - current_time).total_seconds() / 60

        if time_diff > pre_window:
            return EventWindow.OUTSIDE_WINDOW
        elif time_diff > 0:
            return EventWindow.PRE_EVENT
        elif time_diff > -5:  # Within 5 min of event
            return EventWindow.DURING_EVENT
        elif time_diff > -30:
            return EventWindow.POST_EVENT_EARLY
        elif time_diff > -reaction_window:
            return EventWindow.POST_EVENT_LATE
        else:
            return EventWindow.OUTSIDE_WINDOW

    def calculate_surprise(
        self,
        event: EconomicEvent,
    ) -> tuple[float | None, float | None]:
        """
        Calculate event surprise in absolute and standardized terms.

        Returns:
            (surprise, surprise_std) - absolute and standardized surprise
        """
        if event.actual is None or event.forecast is None:
            return None, None

        surprise = event.actual - event.forecast

        # Get historical surprise std
        history = self._historical_surprises.get(event.event_type, [])
        if len(history) >= 5:
            std = np.std(history)
            if std > 0:
                surprise_std = surprise / std
            else:
                surprise_std = 0.0
        else:
            # Default standardization based on forecast
            if event.forecast != 0:
                surprise_std = surprise / (abs(event.forecast) * 0.1)
            else:
                surprise_std = 0.0

        return surprise, surprise_std

    def analyze_event(
        self,
        event: EconomicEvent,
        current_time: datetime,
        current_vol: float,
        normal_vol: float,
    ) -> EventAnalysis:
        """
        Analyze event and market reaction.

        Args:
            event: The economic event
            current_time: Current timestamp
            current_vol: Current volatility (e.g., realized or implied)
            normal_vol: Normal volatility level

        Returns:
            EventAnalysis with complete analysis
        """
        window = self.get_event_window(current_time, event)
        surprise, surprise_std = self.calculate_surprise(event)

        # Vol expansion ratio
        vol_expansion = current_vol / normal_vol if normal_vol > 0 else 1.0

        # Determine direction bias
        if surprise_std is not None:
            if surprise_std > self._surprise_threshold:
                # Positive surprise
                if event.event_type in [EventType.NFP, EventType.GDP, EventType.ISM_MFG]:
                    direction_bias = "bullish"  # Better economy = stocks up
                elif event.event_type == EventType.CPI:
                    direction_bias = "bearish"  # Higher inflation = bonds down, stocks uncertain
                else:
                    direction_bias = "neutral"
            elif surprise_std < -self._surprise_threshold:
                # Negative surprise
                if event.event_type in [EventType.NFP, EventType.GDP, EventType.ISM_MFG]:
                    direction_bias = "bearish"
                elif event.event_type == EventType.CPI:
                    direction_bias = "bullish"  # Lower inflation = dovish
                else:
                    direction_bias = "neutral"
            else:
                direction_bias = "neutral"
        else:
            direction_bias = "neutral"

        # Confidence based on surprise magnitude and vol confirmation
        confidence = 0.0
        if surprise_std is not None:
            confidence = min(1.0, abs(surprise_std) / 2.0)
            if vol_expansion > 1.2:
                confidence *= 1.2
            confidence = min(1.0, confidence)

        analysis = EventAnalysis(
            event=event,
            window=window,
            surprise=surprise,
            surprise_std=surprise_std,
            vol_expansion=vol_expansion,
            direction_bias=direction_bias,
            confidence=confidence,
        )

        self._recent_analyses[event.event_type.value] = analysis
        return analysis

    def generate_signal(
        self,
        symbol: str,
        event: EconomicEvent,
        current_time: datetime,
        current_price: float,
        current_vol: float,
        normal_vol: float,
        atr: float,
        price_change_since_event: float = 0.0,
        current_position: str = "FLAT",
    ) -> EventSignal | None:
        """
        Generate trading signal based on event analysis.

        Args:
            symbol: Instrument symbol
            event: Economic event
            current_time: Current timestamp
            current_price: Current price
            current_vol: Current volatility
            normal_vol: Normal volatility
            atr: Average True Range
            price_change_since_event: Price change since event (for post-event)
            current_position: Current position

        Returns:
            EventSignal if conditions met, None otherwise
        """
        # Check if symbol is sensitive to this event
        chars = EVENT_CHARACTERISTICS.get(event.event_type, {})
        sensitive = chars.get("sensitive_assets", [])
        if sensitive and symbol not in sensitive:
            return None

        analysis = self.analyze_event(event, current_time, current_vol, normal_vol)

        # Pre-event: Consider vol positioning
        if analysis.window == EventWindow.PRE_EVENT:
            if current_position == "FLAT":
                vol_ratio = current_vol / normal_vol if normal_vol > 0 else 1.0

                # If vol is suppressed before event, expect expansion
                if vol_ratio < self._vol_entry_threshold:
                    # For now, signal to be cautious (vol strategy would buy straddles)
                    return EventSignal(
                        symbol=symbol,
                        event_type=event.event_type,
                        direction="FLAT",  # No directional bias pre-event
                        signal_type="pre_event_vol",
                        strength=0.3,
                        entry_price=current_price,
                        stop_loss=0.0,
                        take_profit=0.0,
                        event_window=analysis.window,
                        rationale=(
                            f"Pre-event vol suppressed ({vol_ratio:.2f}x normal), "
                            f"event: {event.event_type.value} in "
                            f"{(event.timestamp - current_time).total_seconds() / 60:.0f}min"
                        ),
                    )

        # Post-event early: Momentum capture
        elif analysis.window == EventWindow.POST_EVENT_EARLY:
            if current_position == "FLAT":
                if analysis.confidence >= self._momentum_threshold:
                    if analysis.direction_bias == "bullish":
                        direction = "LONG"
                        stop_loss = current_price - self._stop_atr_mult * atr
                        take_profit = current_price + self._take_profit_atr_mult * atr
                    elif analysis.direction_bias == "bearish":
                        direction = "SHORT"
                        stop_loss = current_price + self._stop_atr_mult * atr
                        take_profit = current_price - self._take_profit_atr_mult * atr
                    else:
                        return None

                    return EventSignal(
                        symbol=symbol,
                        event_type=event.event_type,
                        direction=direction,
                        signal_type="post_event_momentum",
                        strength=analysis.confidence,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        event_window=analysis.window,
                        rationale=(
                            f"Post-event momentum: {event.event_type.value} "
                            f"surprise={analysis.surprise_std:.2f}std, "
                            f"vol_exp={analysis.vol_expansion:.2f}x"
                        ),
                    )

        # Post-event late: Consider fading overreaction
        elif analysis.window == EventWindow.POST_EVENT_LATE:
            if current_position != "FLAT":
                # Consider exiting as reaction fades
                return EventSignal(
                    symbol=symbol,
                    event_type=event.event_type,
                    direction="FLAT",
                    signal_type="exit",
                    strength=0.6,
                    entry_price=current_price,
                    stop_loss=0.0,
                    take_profit=0.0,
                    event_window=analysis.window,
                    rationale="Event reaction window ending, consider exit",
                )

        return None

    def update_historical_surprise(
        self,
        event_type: EventType,
        surprise: float,
    ) -> None:
        """Update historical surprise data for standardization."""
        if event_type not in self._historical_surprises:
            self._historical_surprises[event_type] = []

        self._historical_surprises[event_type].append(surprise)

        # Keep last 20 surprises
        if len(self._historical_surprises[event_type]) > 20:
            self._historical_surprises[event_type].pop(0)

    def clear_past_events(self, current_time: datetime) -> int:
        """Remove events that are past their reaction window."""
        initial_count = len(self._upcoming_events)

        self._upcoming_events = [
            e for e in self._upcoming_events
            if self.get_event_window(current_time, e) != EventWindow.OUTSIDE_WINDOW
            or e.timestamp > current_time
        ]

        removed = initial_count - len(self._upcoming_events)
        if removed > 0:
            logger.debug(f"Cleared {removed} past events")

        return removed

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        return {
            "vol_entry_threshold": self._vol_entry_threshold,
            "momentum_threshold": self._momentum_threshold,
            "surprise_threshold": self._surprise_threshold,
            "upcoming_events": len(self._upcoming_events),
            "events": [
                {
                    "type": e.event_type.value,
                    "timestamp": e.timestamp.isoformat(),
                    "impact": e.impact.value,
                }
                for e in self._upcoming_events[:5]
            ],
            "recent_analyses": {
                k: {
                    "window": v.window.value,
                    "direction_bias": v.direction_bias,
                    "confidence": v.confidence,
                }
                for k, v in self._recent_analyses.items()
            },
        }


def create_event_driven_strategy(config: dict[str, Any] | None = None) -> EventDrivenStrategy:
    """Create EventDrivenStrategy instance."""
    return EventDrivenStrategy(config)

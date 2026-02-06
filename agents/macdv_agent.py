"""
MACD-v Agent
============

Signal agent implementing the MACD-v (Volatility-Normalized MACD) strategy.

MACD-v normalizes traditional MACD by ATR, making it comparable across
different instruments regardless of their volatility profiles.

Formula:
    MACD-v = [(EMA(12) - EMA(26)) / ATR(26)] * 100

Responsibility: MACD-v signal generation ONLY.
Does NOT make trading decisions or send orders.

Entry Rules:
    - LONG: MACD-v crosses above signal line from oversold zone
    - SHORT: MACD-v crosses below signal line from overbought zone

Exit Rules:
    - Opposite crossover signal
    - Histogram momentum reversal
    - Stop-loss / Take-profit hit

MATURITY: ALPHA
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
from core.session_checker import filter_signal_by_session, is_optimal_trading_time
from strategies.macdv_strategy import MACDvStrategy, MACDvSignal

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class MACDvState:
    """State for tracking MACD-v indicators per symbol."""
    symbol: str
    prices: deque  # Rolling close prices
    highs: deque   # Rolling high prices
    lows: deque    # Rolling low prices
    # MACD-v indicator values
    macdv: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    previous_zone: str = "neutral"
    current_zone: str = "neutral"
    # Last signal tracking
    last_signal: SignalDirection = SignalDirection.FLAT
    last_crossover: str = "none"
    # Histogram history for momentum
    histogram_history: deque = field(default_factory=lambda: deque(maxlen=10))


class MACDvAgent(SignalAgent):
    """
    MACD-v Signal Agent.

    Implements the MACD-v (Volatility-Normalized MACD) indicator strategy.
    MACD-v improves on traditional MACD by normalizing with ATR, making
    signals comparable across instruments with different volatility profiles.

    Key Features:
    1. Volatility-normalized signals (comparable across instruments)
    2. Clear overbought/oversold zones (+/- 150-200)
    3. Signal line crossovers with zone confirmation
    4. Histogram momentum confirmation

    Signal Output:
    - Direction: LONG, SHORT, or FLAT
    - Confidence based on zone depth and histogram momentum
    - Stop-loss and take-profit using ATR-based calculation
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Strategy parameters from config
        strategy_config = {
            "fast_period": config.parameters.get("fast_period", 12),
            "slow_period": config.parameters.get("slow_period", 26),
            "signal_period": config.parameters.get("signal_period", 9),
            "atr_period": config.parameters.get("atr_period", 26),
            "overbought_zone": config.parameters.get("overbought_zone", 150.0),
            "oversold_zone": config.parameters.get("oversold_zone", -150.0),
            "extreme_overbought": config.parameters.get("extreme_overbought", 200.0),
            "extreme_oversold": config.parameters.get("extreme_oversold", -200.0),
            "require_zone_exit": config.parameters.get("require_zone_exit", True),
            "stop_loss_pct": config.parameters.get("stop_loss_pct", 2.0),
            "take_profit_pct": config.parameters.get("take_profit_pct", 4.0),
            "use_atr_stop": config.parameters.get("use_atr_stop", True),
            "stop_loss_atr_mult": config.parameters.get("stop_loss_atr_mult", 2.0),
            "take_profit_atr_mult": config.parameters.get("take_profit_atr_mult", 4.0),
            "histogram_momentum_bars": config.parameters.get("histogram_momentum_bars", 3),
            "min_confidence": config.parameters.get("min_confidence", 0.5),
        }

        # Initialize strategy
        self._strategy = MACDvStrategy(strategy_config)

        # Minimum confidence threshold for signal emission
        # FIX-20: 0.75 blocked most signals (strategy base confidence is 0.6)
        self._min_confidence = config.parameters.get("min_confidence", 0.50)

        # State per symbol
        self._symbols: dict[str, MACDvState] = {}

        # Lookback requirements
        self._slow_period = strategy_config["slow_period"]
        self._signal_period = strategy_config["signal_period"]
        self._max_lookback = self._slow_period + self._signal_period + 50  # Buffer

    async def initialize(self) -> None:
        """Initialize MACD-v tracking."""
        logger.info(
            f"MACDvAgent initializing with EMA({self._strategy._fast_period}/{self._strategy._slow_period}), "
            f"Signal({self._strategy._signal_period}), "
            f"Zones(OB={self._strategy._overbought_zone}/OS={self._strategy._oversold_zone})"
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
        """Process market data and generate MACD-v signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        price = event.last or event.mid

        if price <= 0:
            return

        # Get high/low prices (use price if not available)
        high = event.high if event.high > 0 else price
        low = event.low if event.low > 0 else price

        # Get or create symbol state
        if symbol not in self._symbols:
            self._symbols[symbol] = MACDvState(
                symbol=symbol,
                prices=deque(maxlen=self._max_lookback),
                highs=deque(maxlen=self._max_lookback),
                lows=deque(maxlen=self._max_lookback),
            )

        state = self._symbols[symbol]

        # Update price history
        state.prices.append(price)
        state.highs.append(high)
        state.lows.append(low)

        # Check if we have enough data
        min_required = self._slow_period + self._signal_period
        if len(state.prices) < min_required:
            await self._emit_warmup_heartbeat(
                symbol,
                f"Collecting data ({len(state.prices)}/{min_required} bars)"
            )
            return

        # Generate signal using strategy
        prices_array = np.array(list(state.prices))
        highs_array = np.array(list(state.highs))
        lows_array = np.array(list(state.lows))

        macdv_signal = self._strategy.analyze(
            symbol=symbol,
            prices=prices_array,
            highs=highs_array,
            lows=lows_array,
            previous_zone=state.previous_zone
        )

        # Update state
        state.macdv = macdv_signal.macdv_value
        state.signal_line = macdv_signal.signal_line
        state.histogram = macdv_signal.histogram
        state.previous_zone = state.current_zone
        state.current_zone = macdv_signal.zone
        state.histogram_history.append(macdv_signal.histogram)

        # Convert signal direction
        if macdv_signal.direction == "long":
            signal_direction = SignalDirection.LONG
        elif macdv_signal.direction == "short":
            signal_direction = SignalDirection.SHORT
        else:
            signal_direction = SignalDirection.FLAT

        # Only emit on direction change
        if signal_direction == state.last_signal:
            await self._emit_warmup_heartbeat(symbol, "No direction change")
            return

        state.last_signal = signal_direction

        # Skip FLAT signals
        if signal_direction == SignalDirection.FLAT:
            await self._emit_warmup_heartbeat(symbol, "FLAT signal")
            return

        # Apply confidence threshold
        if macdv_signal.confidence < self._min_confidence:
            logger.debug(
                f"MACD-v signal filtered: confidence {macdv_signal.confidence:.2f} < "
                f"threshold {self._min_confidence}"
            )
            await self._emit_warmup_heartbeat(symbol, "Low confidence")
            return

        # Check trading session quality
        is_optimal, session_reason = is_optimal_trading_time(
            symbol,
            asset_class=self._get_asset_class(symbol),
            config=self._config.parameters if hasattr(self._config, 'parameters') else {}
        )

        strength = macdv_signal.strength
        confidence = macdv_signal.confidence

        if not is_optimal and "Avoided" in session_reason:
            logger.info(
                f"MACD-v signal for {symbol} suppressed due to session: {session_reason}"
            )
            await self._emit_warmup_heartbeat(symbol, f"Session: {session_reason}")
            return
        elif not is_optimal:
            # Reduce strength for sub-optimal sessions
            strength = strength * 0.5
            confidence = confidence * 0.8

        # Build rationale
        rationale = self._build_rationale(macdv_signal, session_reason if not is_optimal else None)

        # Build data sources
        data_sources = ("ib_market_data", "macdv_indicator", "atr")

        # Create and emit signal event
        signal_event = SignalEvent(
            source_agent=self.name,
            strategy_name="macdv",
            symbol=symbol,
            direction=signal_direction,
            strength=strength,
            confidence=confidence,
            rationale=rationale,
            data_sources=data_sources,
            target_price=macdv_signal.take_profit_price,
            stop_loss=macdv_signal.stop_loss_price,
        )

        await self._event_bus.publish_signal(signal_event)
        self._audit_logger.log_event(signal_event)

        logger.info(
            f"MACD-v signal: {symbol} {signal_direction.value} | "
            f"MACD-v={macdv_signal.macdv_value:.1f} | "
            f"Zone={macdv_signal.zone} | "
            f"Confidence={confidence:.2f}"
        )

    def _build_rationale(
        self,
        signal: MACDvSignal,
        session_note: str | None = None
    ) -> str:
        """Build rationale string for the signal."""
        parts = []

        # Crossover info
        crossover = signal.indicators.get("crossover_type", "none")
        if crossover == "bullish":
            parts.append(f"Bullish crossover (MACD-v > signal)")
        elif crossover == "bearish":
            parts.append(f"Bearish crossover (MACD-v < signal)")

        # Zone info
        parts.append(f"Zone: {signal.zone}")

        # MACD-v values
        parts.append(
            f"MACD-v={signal.macdv_value:.1f}, "
            f"Signal={signal.signal_line:.1f}, "
            f"Hist={signal.histogram:.1f}"
        )

        # Momentum confirmation
        if signal.direction == "long" and signal.indicators.get("histogram_momentum_long"):
            parts.append("Histogram momentum confirmed")
        elif signal.direction == "short" and signal.indicators.get("histogram_momentum_short"):
            parts.append("Histogram momentum confirmed")

        # Session note
        if session_note:
            parts.append(f"[Session: {session_note}]")

        return " | ".join(parts)

    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol for session filtering."""
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

    def get_indicator_status(self, symbol: str) -> dict | None:
        """
        Get current MACD-v indicator status for a symbol.

        Returns:
            Dict with MACD-v values and zone info, or None if not tracking.
        """
        if symbol not in self._symbols:
            return None

        state = self._symbols[symbol]

        return {
            "symbol": symbol,
            "macdv": state.macdv,
            "signal_line": state.signal_line,
            "histogram": state.histogram,
            "current_zone": state.current_zone,
            "previous_zone": state.previous_zone,
            "last_signal": state.last_signal.value,
            "data_points": len(state.prices),
            "overbought_zone": self._strategy._overbought_zone,
            "oversold_zone": self._strategy._oversold_zone,
        }

    def get_all_tracked_symbols(self) -> list[str]:
        """Get list of all symbols being tracked."""
        return list(self._symbols.keys())

"""
Index Spread Trading Agent
==========================

Generates signals based on index spread/pairs trading strategies.
Implements MES/MNQ spread trading with z-score based entries.

Responsibility: Index spread signal generation ONLY.
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
from strategies.index_spread_strategy import (
    IndexSpreadStrategy,
    create_index_spread_strategy,
    SpreadState,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class PairState:
    """State for tracking a spread pair."""
    leg1_symbol: str
    leg2_symbol: str
    leg1_price: float = 0.0
    leg2_price: float = 0.0
    leg1_updated: datetime | None = None
    leg2_updated: datetime | None = None
    last_signal: SignalDirection = SignalDirection.FLAT
    # Price history for strategy analysis
    leg1_prices: deque = field(default_factory=lambda: deque(maxlen=100))
    leg2_prices: deque = field(default_factory=lambda: deque(maxlen=100))

    def is_ready(self, max_age_seconds: float = 5.0) -> bool:
        """Check if both legs have recent prices."""
        if self.leg1_updated is None or self.leg2_updated is None:
            return False
        now = datetime.now(timezone.utc)
        leg1_age = (now - self.leg1_updated).total_seconds()
        leg2_age = (now - self.leg2_updated).total_seconds()
        return leg1_age <= max_age_seconds and leg2_age <= max_age_seconds


class IndexSpreadAgent(SignalAgent):
    """
    Index Spread Trading Agent.

    Implements pairs/spread trading strategies:
    1. MES/MNQ spread (S&P 500 vs Nasdaq 100 futures)
    2. Z-score based mean reversion entries
    3. Dynamic hedge ratio estimation
    4. Half-life based holding period

    Signal output:
    - Long spread (long leg1, short leg2) or Short spread
    - Confidence based on z-score magnitude and half-life
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._entry_zscore = config.parameters.get("entry_zscore", 2.0)
        self._exit_zscore = config.parameters.get("exit_zscore", 0.5)
        self._lookback = config.parameters.get("lookback", 60)
        self._min_half_life = config.parameters.get("min_half_life", 1)
        self._max_half_life = config.parameters.get("max_half_life", 30)

        # Define spread pairs
        self._pairs = config.parameters.get("pairs", [
            {"leg1": "MES", "leg2": "MNQ", "name": "ES_NQ_spread"},
            {"leg1": "MYM", "leg2": "MNQ", "name": "YM_NQ_spread"},
        ])

        # FIX-23: Strategy expects zscore_entry/zscore_exit/lookback_days (not entry_zscore/exit_zscore/lookback)
        self._strategy = create_index_spread_strategy({
            "zscore_entry": self._entry_zscore,
            "zscore_exit": self._exit_zscore,
            "lookback_days": self._lookback,
            "min_half_life": self._min_half_life,
            "max_half_life": self._max_half_life,
        })

        # State tracking per pair
        self._pair_states: dict[str, PairState] = {}
        for pair in self._pairs:
            name = pair["name"]
            self._pair_states[name] = PairState(
                leg1_symbol=pair["leg1"],
                leg2_symbol=pair["leg2"],
            )

        # Symbol to pair mapping
        self._symbol_to_pairs: dict[str, list[str]] = {}
        for pair in self._pairs:
            for leg in [pair["leg1"], pair["leg2"]]:
                if leg not in self._symbol_to_pairs:
                    self._symbol_to_pairs[leg] = []
                self._symbol_to_pairs[leg].append(pair["name"])

        logger.info(
            f"IndexSpreadAgent initialized with {len(self._pairs)} pairs, "
            f"entry_z={self._entry_zscore}, exit_z={self._exit_zscore}"
        )

    async def initialize(self) -> None:
        """Initialize spread tracking."""
        logger.info(f"IndexSpreadAgent ready: {len(self._pairs)} pairs")

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
        """Process market data and generate spread signals."""
        if event.event_type == EventType.MARKET_DATA:
            await self._process_market_data(event)

    async def _process_market_data(self, event: MarketDataEvent) -> None:
        """Process market data and update spread signals."""
        symbol = event.symbol
        price = event.last
        timestamp = event.timestamp

        if price is None or price <= 0:
            return

        # Check if this symbol is part of any tracked pairs
        if symbol not in self._symbol_to_pairs:
            return

        # Update all pairs containing this symbol
        for pair_name in self._symbol_to_pairs[symbol]:
            pair_state = self._pair_states[pair_name]

            # Update the appropriate leg and price history
            if symbol == pair_state.leg1_symbol:
                pair_state.leg1_price = price
                pair_state.leg1_updated = timestamp
                pair_state.leg1_prices.append(price)
            elif symbol == pair_state.leg2_symbol:
                pair_state.leg2_price = price
                pair_state.leg2_updated = timestamp
                pair_state.leg2_prices.append(price)

            # Check if we can generate a signal
            if pair_state.is_ready():
                await self._analyze_spread(pair_name, pair_state, timestamp)

    async def _analyze_spread(
        self,
        pair_name: str,
        pair_state: PairState,
        timestamp: datetime,
    ) -> None:
        """Analyze spread and generate signal if appropriate."""
        # Need enough price history for analysis
        min_history = max(20, self._lookback)
        if len(pair_state.leg1_prices) < min_history or len(pair_state.leg2_prices) < min_history:
            await self._emit_warmup_heartbeat(
                pair_state.leg1_symbol,
                f"Collecting data ({len(pair_state.leg1_prices)}/{min_history} bars)"
            )
            return

        # Convert to numpy arrays
        prices_a = np.array(pair_state.leg1_prices)
        prices_b = np.array(pair_state.leg2_prices)

        # Use the smaller array size for both
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]

        # Get current position status (simplified - assume FLAT)
        current_position = "FLAT"
        if pair_state.last_signal == SignalDirection.LONG:
            current_position = "LONG_SPREAD"
        elif pair_state.last_signal == SignalDirection.SHORT:
            current_position = "SHORT_SPREAD"

        # Strategy expects: generate_signal(spread_name, prices_a, prices_b, current_position)
        # First get spread name from pair_name (e.g., "ES_NQ_spread" -> "ES_NQ")
        spread_name = pair_name.replace("_spread", "").upper()
        if spread_name not in ["MES_MNQ", "ES_NQ", "MES_MYM", "MNQ_M2K"]:
            # Use leg symbols to construct spread name
            spread_name = f"{pair_state.leg1_symbol}_{pair_state.leg2_symbol}"

        signal_result = self._strategy.generate_signal(
            spread_name=spread_name,
            prices_a=prices_a,
            prices_b=prices_b,
            current_position=current_position,
        )

        if signal_result is None:
            return

        # SpreadSignal has: direction, strength, hedge_ratio, entry_zscore, target_zscore, etc.
        direction = signal_result.direction.lower() if signal_result.direction else "flat"
        zscore = signal_result.entry_zscore
        half_life = 0  # Not directly available, would need to call analyze_spread
        hedge_ratio = signal_result.hedge_ratio

        # Convert to SignalDirection
        if direction == "long_spread":
            signal_direction = SignalDirection.LONG
        elif direction == "short_spread":
            signal_direction = SignalDirection.SHORT
        elif direction in ("exit", "flat"):  # FIX-26: Strategy can return "flat" for exits
            signal_direction = SignalDirection.FLAT
        else:
            return

        # Skip if same signal as last time
        if pair_state.last_signal == signal_direction and signal_direction != SignalDirection.FLAT:
            return

        pair_state.last_signal = signal_direction

        # Calculate confidence based on z-score magnitude
        confidence = min(0.95, abs(zscore) / (self._entry_zscore * 1.5))

        # Adjust for half-life quality (prefer shorter half-life)
        if half_life > 0:
            half_life_factor = 1.0 - (half_life / self._max_half_life) * 0.3
            confidence *= max(0.5, half_life_factor)

        # Build rationale
        rationale_parts = [
            f"Spread: {pair_name}",
            f"Z-score: {zscore:.2f}",
            f"Half-life: {half_life:.1f} days",
            f"Hedge ratio: {hedge_ratio:.4f}",
        ]

        if signal_direction == SignalDirection.LONG:
            rationale_parts.append(f"Long {pair_state.leg1_symbol}, Short {pair_state.leg2_symbol}")
        elif signal_direction == SignalDirection.SHORT:
            rationale_parts.append(f"Short {pair_state.leg1_symbol}, Long {pair_state.leg2_symbol}")

        # Calculate stop loss (based on price with 2% SL, 4% TP for 2:1 R:R)
        if signal_direction == SignalDirection.LONG:
            stop_loss = pair_state.leg1_price * 0.98  # 2% stop on leg1
            take_profit = pair_state.leg1_price * 1.04  # 4% target
        elif signal_direction == SignalDirection.SHORT:
            stop_loss = pair_state.leg1_price * 1.02
            take_profit = pair_state.leg1_price * 0.96
        else:
            stop_loss = None
            take_profit = None

        # Create signal event (use leg1 symbol as primary)
        signal = SignalEvent(
            source_agent=self.name,
            symbol=pair_state.leg1_symbol,
            direction=signal_direction,
            strength=confidence,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
            data_sources=("index_spread_strategy", "zscore", "cointegration"),
            stop_loss=stop_loss,
            target_price=take_profit,
        )

        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

        logger.info(
            f"IndexSpreadAgent signal: {pair_name} {signal_direction.value} "
            f"z={zscore:.2f} confidence={confidence:.2f}"
        )

    def get_status(self) -> dict:
        """Get agent status for monitoring."""
        base_status = super().get_status()

        pair_status = {}
        for name, state in self._pair_states.items():
            pair_status[name] = {
                "leg1": state.leg1_symbol,
                "leg2": state.leg2_symbol,
                "leg1_price": state.leg1_price,
                "leg2_price": state.leg2_price,
                "last_signal": state.last_signal.value,
                "ready": state.is_ready(),
            }

        base_status.update({
            "entry_zscore": self._entry_zscore,
            "exit_zscore": self._exit_zscore,
            "lookback": self._lookback,
            "pairs": pair_status,
            "strategy_state": self._strategy.get_status() if hasattr(self._strategy, 'get_status') else {},
        })
        return base_status

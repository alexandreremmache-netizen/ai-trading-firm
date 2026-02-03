"""
Market Making Agent
===================

Generates signals for market making strategy (IB-compatible latency).
Provides liquidity by quoting bid/ask spreads.

NOTE: This is NOT High-Frequency Trading (HFT).
- Quote refresh rate is IB-compatible (>100ms)
- No nanosecond-level strategies
- Focus on spread capture, not speed

Responsibility: Market making signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
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

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class MarketMakingState:
    """State for market making per symbol."""
    symbol: str
    bid_prices: deque
    ask_prices: deque
    mid_prices: deque
    spreads: deque
    volumes: deque
    inventory: int = 0
    last_quote_time: datetime | None = None
    fair_value: float = 0.0
    volatility: float = 0.0


class MarketMakingAgent(SignalAgent):
    """
    Market Making Agent.

    Implements passive market making strategy:
    1. Calculate fair value from order book
    2. Estimate short-term volatility
    3. Set bid/ask quotes around fair value
    4. Manage inventory risk

    NOT HFT - operates at IB-compatible latencies (100ms+).
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._spread_bps = config.parameters.get("spread_bps", 10)  # Minimum spread
        self._max_inventory = config.parameters.get("max_inventory", 1000)
        self._quote_refresh_ms = config.parameters.get("quote_refresh_ms", 1000)  # IB compatible

        # State per symbol
        self._symbols: dict[str, MarketMakingState] = {}
        self._lookback = 100  # Rolling window size

    async def initialize(self) -> None:
        """Initialize market making state."""
        logger.info(
            f"MarketMakingAgent initializing with spread={self._spread_bps}bps, "
            f"max_inventory={self._max_inventory}, refresh={self._quote_refresh_ms}ms"
        )

        # Validate we're not doing HFT
        if self._quote_refresh_ms < 100:
            logger.warning("Quote refresh < 100ms may not be IB compatible. Adjusting to 100ms.")
            self._quote_refresh_ms = 100

    async def process_event(self, event: Event) -> None:
        """Process market data and generate market making signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol

        # Skip if insufficient data
        if event.bid <= 0 or event.ask <= 0:
            return

        # Get or create symbol state
        if symbol not in self._symbols:
            self._symbols[symbol] = MarketMakingState(
                symbol=symbol,
                bid_prices=deque(maxlen=self._lookback),
                ask_prices=deque(maxlen=self._lookback),
                mid_prices=deque(maxlen=self._lookback),
                spreads=deque(maxlen=self._lookback),
                volumes=deque(maxlen=self._lookback),
            )

        state = self._symbols[symbol]

        # Update state
        state.bid_prices.append(event.bid)
        state.ask_prices.append(event.ask)
        state.mid_prices.append(event.mid)
        state.spreads.append(event.spread)
        state.volumes.append(event.volume)

        # Check quote refresh rate (not HFT)
        now = datetime.now(timezone.utc)
        if state.last_quote_time:
            elapsed_ms = (now - state.last_quote_time).total_seconds() * 1000
            if elapsed_ms < self._quote_refresh_ms:
                return  # Rate limiting

        state.last_quote_time = now

        # Generate signal if we have enough data
        if len(state.mid_prices) >= 20:
            signal = self._generate_mm_signal(state, event)
            if signal:
                await self._event_bus.publish_signal(signal)
                self._audit_logger.log_event(signal)

    def _generate_mm_signal(
        self,
        state: MarketMakingState,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Generate market making signal.

        TODO: Implement proper market making model:
        - Order flow imbalance
        - Inventory optimization (Avellaneda-Stoikov)
        - Adverse selection risk
        - Optimal spread calculation
        """
        # Calculate fair value (simple mid for now)
        state.fair_value = market_data.mid

        # Calculate realized volatility
        mid_array = np.array(list(state.mid_prices))
        if len(mid_array) >= 2:
            returns = np.diff(np.log(mid_array))
            state.volatility = np.std(returns) * np.sqrt(252 * 390)  # Annualized

        # Calculate optimal spread based on volatility and inventory
        optimal_spread = self._calculate_optimal_spread(state, market_data)

        # Check if current spread is attractive
        current_spread_bps = (market_data.spread / market_data.mid) * 10000

        # Generate signal based on spread opportunity
        if current_spread_bps > optimal_spread * 1.5:
            # Wide spread - opportunity to provide liquidity
            return self._generate_liquidity_signal(state, market_data, optimal_spread)

        # Inventory management signal
        if abs(state.inventory) > self._max_inventory * 0.8:
            return self._generate_inventory_signal(state, market_data)

        return None

    def _calculate_optimal_spread(
        self,
        state: MarketMakingState,
        market_data: MarketDataEvent,
    ) -> float:
        """
        Calculate optimal spread in basis points.

        TODO: Implement Avellaneda-Stoikov optimal market making:
        - s* = γσ²T + (2/γ)ln(1 + γ/k)
        - Where γ is risk aversion, σ is volatility, k is order arrival rate
        """
        # Base spread from configuration
        base_spread = self._spread_bps

        # Volatility adjustment
        vol_adjustment = state.volatility * 10  # Scale factor

        # Inventory adjustment (wider spread when inventory is high)
        inventory_pct = abs(state.inventory) / self._max_inventory
        inventory_adjustment = inventory_pct * 5

        optimal = base_spread + vol_adjustment + inventory_adjustment

        return min(100, max(self._spread_bps, optimal))  # Cap at 100 bps

    def _generate_liquidity_signal(
        self,
        state: MarketMakingState,
        market_data: MarketDataEvent,
        optimal_spread: float,
    ) -> SignalEvent:
        """Generate signal to provide liquidity."""
        # Determine quote direction based on inventory
        if state.inventory > 0:
            # Long inventory - prefer to sell
            direction = SignalDirection.SHORT
            strength = -0.3
        elif state.inventory < 0:
            # Short inventory - prefer to buy
            direction = SignalDirection.LONG
            strength = 0.3
        else:
            # Neutral - market make both sides
            direction = SignalDirection.FLAT
            strength = 0.0

        return SignalEvent(
            source_agent=self.name,
            strategy_name="market_making",
            symbol=state.symbol,
            direction=direction,
            strength=strength,
            confidence=0.5,
            target_price=state.fair_value,
            rationale=(
                f"MM opportunity: spread={market_data.spread:.4f} "
                f"({(market_data.spread/market_data.mid)*10000:.1f}bps), "
                f"optimal={optimal_spread:.1f}bps, inventory={state.inventory}"
            ),
            data_sources=("ib_market_data", "order_book", "market_making_indicator"),
        )

    def _generate_inventory_signal(
        self,
        state: MarketMakingState,
        market_data: MarketDataEvent,
    ) -> SignalEvent:
        """Generate signal to reduce inventory."""
        if state.inventory > 0:
            direction = SignalDirection.SHORT
            strength = -0.7
            action = "reduce long"
        else:
            direction = SignalDirection.LONG
            strength = 0.7
            action = "reduce short"

        return SignalEvent(
            source_agent=self.name,
            strategy_name="market_making_inventory",
            symbol=state.symbol,
            direction=direction,
            strength=strength,
            confidence=0.7,
            target_price=market_data.mid,
            rationale=(
                f"Inventory management: {action} inventory "
                f"({state.inventory}/{self._max_inventory})"
            ),
            data_sources=("ib_market_data", "inventory", "market_making_indicator"),
        )

    def update_inventory(self, symbol: str, quantity_change: int) -> None:
        """
        Update inventory after a fill.

        Called by execution agent to track positions.
        """
        if symbol in self._symbols:
            self._symbols[symbol].inventory += quantity_change

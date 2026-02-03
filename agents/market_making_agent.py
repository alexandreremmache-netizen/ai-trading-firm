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

        # Handle delayed data: use mid/last price if bid/ask unavailable
        bid = event.bid if event.bid > 0 else (event.mid or event.last or 0)
        ask = event.ask if event.ask > 0 else (event.mid or event.last or 0)
        mid = event.mid if event.mid > 0 else ((bid + ask) / 2 if bid > 0 and ask > 0 else event.last or 0)

        # Skip if no valid price data at all
        if mid <= 0:
            return

        # For delayed data, estimate spread from typical values
        if event.bid <= 0 or event.ask <= 0:
            # Estimate spread as 0.05% for liquid instruments
            estimated_spread = mid * 0.0005
            bid = mid - estimated_spread / 2
            ask = mid + estimated_spread / 2

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

        # Update state with computed values
        spread = ask - bid if ask > bid else mid * 0.0005
        state.bid_prices.append(bid)
        state.ask_prices.append(ask)
        state.mid_prices.append(mid)
        state.spreads.append(spread)
        state.volumes.append(event.volume if event.volume > 0 else 1000)

        # Check quote refresh rate (not HFT)
        now = datetime.now(timezone.utc)
        if state.last_quote_time:
            elapsed_ms = (now - state.last_quote_time).total_seconds() * 1000
            if elapsed_ms < self._quote_refresh_ms:
                return  # Rate limiting

        state.last_quote_time = now

        # Generate signal if we have enough data (reduced from 20 to 10 for faster startup)
        if len(state.mid_prices) >= 10:
            signal = self._generate_mm_signal(state, mid, spread)
            if signal:
                await self._event_bus.publish_signal(signal)
                self._audit_logger.log_event(signal)

    def _generate_mm_signal(
        self,
        state: MarketMakingState,
        mid: float,
        spread: float,
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
        state.fair_value = mid

        # Calculate realized volatility
        mid_array = np.array(list(state.mid_prices))
        if len(mid_array) >= 2:
            # Avoid log of zero
            mid_array = mid_array[mid_array > 0]
            if len(mid_array) >= 2:
                returns = np.diff(np.log(mid_array))
                state.volatility = np.std(returns) * np.sqrt(252 * 390)  # Annualized

        # Calculate optimal spread based on volatility and inventory
        optimal_spread = self._calculate_optimal_spread(state, mid)

        # Check if current spread is attractive
        current_spread_bps = (spread / mid) * 10000 if mid > 0 else 0

        # Generate signal based on spread opportunity (relaxed threshold)
        if current_spread_bps > optimal_spread * 1.2:
            # Wide spread - opportunity to provide liquidity
            return self._generate_liquidity_signal(state, mid, spread, optimal_spread)

        # Inventory management signal
        if abs(state.inventory) > self._max_inventory * 0.8:
            return self._generate_inventory_signal(state, mid)

        # Also generate neutral MM signal periodically for active symbols
        if len(state.mid_prices) % 5 == 0:  # Every 5th update
            return self._generate_neutral_mm_signal(state, mid, spread, optimal_spread)

        return None

    def _calculate_optimal_spread(
        self,
        state: MarketMakingState,
        mid: float,
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
        mid: float,
        spread: float,
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

        current_spread_bps = (spread / mid) * 10000 if mid > 0 else 0

        return SignalEvent(
            source_agent=self.name,
            strategy_name="market_making",
            symbol=state.symbol,
            direction=direction,
            strength=strength,
            confidence=0.5,
            target_price=state.fair_value,
            rationale=(
                f"MM opportunity: spread={spread:.4f} "
                f"({current_spread_bps:.1f}bps), "
                f"optimal={optimal_spread:.1f}bps, inventory={state.inventory}"
            ),
            data_sources=("ib_market_data", "order_book", "market_making_indicator"),
        )

    def _generate_neutral_mm_signal(
        self,
        state: MarketMakingState,
        mid: float,
        spread: float,
        optimal_spread: float,
    ) -> SignalEvent:
        """Generate neutral market making signal for activity."""
        current_spread_bps = (spread / mid) * 10000 if mid > 0 else 0

        return SignalEvent(
            source_agent=self.name,
            strategy_name="market_making",
            symbol=state.symbol,
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.4,
            target_price=state.fair_value,
            rationale=(
                f"MM neutral: spread={current_spread_bps:.1f}bps, "
                f"optimal={optimal_spread:.1f}bps, vol={state.volatility:.1%}"
            ),
            data_sources=("ib_market_data", "order_book", "market_making_indicator"),
        )

    def _generate_inventory_signal(
        self,
        state: MarketMakingState,
        mid: float,
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
            target_price=mid,
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

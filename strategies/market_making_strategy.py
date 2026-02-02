"""
Market Making Strategy
======================

Implements market making logic (IB-compatible latency).

NOTE: Not HFT - operates at 100ms+ latencies.

TODO: This is a placeholder - implement actual market making models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """A bid/ask quote."""
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    mid_price: float
    spread: float


@dataclass
class MarketMakingSignal:
    """Market making signal output."""
    symbol: str
    action: str  # "quote", "cancel", "adjust"
    quote: Quote | None
    inventory_action: str | None  # "reduce_long", "reduce_short", None
    urgency: float  # 0 to 1


class MarketMakingStrategy:
    """
    Market Making Strategy Implementation.

    Implements:
    1. Fair value estimation
    2. Optimal spread calculation
    3. Inventory management
    4. Adverse selection avoidance

    NOT HFT - All quotes refreshed at IB-compatible intervals (100ms+).

    TODO: Implement proper models:
    - Avellaneda-Stoikov optimal market making
    - Kyle's lambda for adverse selection
    - Order flow toxicity (VPIN)
    """

    def __init__(self, config: dict[str, Any]):
        self._min_spread_bps = config.get("spread_bps", 10)
        self._max_inventory = config.get("max_inventory", 1000)
        self._risk_aversion = config.get("risk_aversion", 0.1)
        self._quote_refresh_ms = config.get("quote_refresh_ms", 1000)

        # Ensure not HFT
        if self._quote_refresh_ms < 100:
            logger.warning("Quote refresh < 100ms, adjusting to 100ms (IB compatible)")
            self._quote_refresh_ms = 100

    def estimate_fair_value(
        self,
        bid: float,
        ask: float,
        bid_size: int,
        ask_size: int,
    ) -> float:
        """
        Estimate fair value from order book.

        Uses size-weighted mid price.

        TODO: Implement more sophisticated fair value:
        - Micro-price
        - Order flow imbalance
        - Volume-weighted price
        """
        if bid_size + ask_size == 0:
            return (bid + ask) / 2

        # Size-weighted mid (micro-price approximation)
        fair_value = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

        return fair_value

    def calculate_optimal_spread(
        self,
        volatility: float,
        inventory: int,
        time_horizon: float = 1.0,
    ) -> float:
        """
        Calculate optimal spread using Avellaneda-Stoikov framework.

        s* = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)

        Simplified version - full implementation would need order arrival rate.

        TODO: Implement full A-S model with:
        - Order arrival rate estimation
        - Inventory penalty
        - Terminal inventory constraint
        """
        gamma = self._risk_aversion
        sigma = volatility

        # Simplified spread calculation
        vol_component = gamma * (sigma ** 2) * time_horizon

        # Inventory adjustment - widen spread as inventory grows
        inventory_pct = abs(inventory) / self._max_inventory
        inventory_component = inventory_pct * self._min_spread_bps / 10000

        optimal_spread = vol_component + inventory_component

        # Apply minimum spread
        min_spread = self._min_spread_bps / 10000
        optimal_spread = max(optimal_spread, min_spread)

        return optimal_spread

    def calculate_quote_skew(self, inventory: int) -> float:
        """
        Calculate quote skew based on inventory.

        Positive skew = higher ask (want to sell)
        Negative skew = lower bid (want to buy)
        """
        inventory_pct = inventory / self._max_inventory
        max_skew = 0.0005  # 5 bps max skew

        skew = inventory_pct * max_skew

        return skew

    def generate_quotes(
        self,
        symbol: str,
        fair_value: float,
        volatility: float,
        inventory: int,
        bid_size: int = 100,
        ask_size: int = 100,
    ) -> MarketMakingSignal:
        """
        Generate bid/ask quotes.
        """
        # Calculate optimal spread
        spread = self.calculate_optimal_spread(volatility, inventory)

        # Calculate skew based on inventory
        skew = self.calculate_quote_skew(inventory)

        # Calculate bid and ask
        half_spread = spread / 2
        bid_price = fair_value - half_spread - skew
        ask_price = fair_value + half_spread - skew

        # Round to tick size (assume 0.01)
        bid_price = round(bid_price, 2)
        ask_price = round(ask_price, 2)

        # Ensure positive spread
        if ask_price <= bid_price:
            ask_price = bid_price + 0.01

        quote = Quote(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            mid_price=fair_value,
            spread=ask_price - bid_price,
        )

        # Determine if inventory management needed
        inventory_action = None
        urgency = 0.0

        if inventory > self._max_inventory * 0.8:
            inventory_action = "reduce_long"
            urgency = 0.8
        elif inventory < -self._max_inventory * 0.8:
            inventory_action = "reduce_short"
            urgency = 0.8

        return MarketMakingSignal(
            symbol=symbol,
            action="quote",
            quote=quote,
            inventory_action=inventory_action,
            urgency=urgency,
        )

    def should_quote(
        self,
        current_spread: float,
        our_spread: float,
        inventory: int,
    ) -> bool:
        """
        Determine if we should be quoting.

        Don't quote if:
        - Spread is too tight (can't make money)
        - Inventory is at limit
        - Market is too fast (adverse selection)
        """
        # Check if spread is profitable
        if current_spread < our_spread * 0.8:
            return False  # Can't compete

        # Check inventory limits
        if abs(inventory) >= self._max_inventory:
            return False

        return True

"""
Smart Order Router (SOR)
========================

Implements smart order routing for best execution per MiFID II RTS 27/28.

Key features:
- Multi-venue price comparison
- Order splitting across venues
- Best execution logic
- Venue selection based on:
  - Price (best bid/ask)
  - Liquidity (order book depth)
  - Fees (maker/taker, rebates)
  - Latency (venue responsiveness)

#E11 - Smart Order Routing implementation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class VenueType(Enum):
    """Trading venue type."""
    PRIMARY = "primary"       # Primary exchange (NYSE, NASDAQ)
    REGIONAL = "regional"     # Regional exchange
    ECN = "ecn"              # Electronic Communication Network
    ATS = "ats"              # Alternative Trading System
    DARK_POOL = "dark_pool"  # Dark pool / non-displayed liquidity


class RoutingStrategy(Enum):
    """Order routing strategy."""
    BEST_PRICE = "best_price"          # Always route to best price
    LOWEST_COST = "lowest_cost"        # Consider fees/rebates
    FASTEST = "fastest"                 # Route to fastest venue
    LIQUIDITY = "liquidity"            # Route to deepest liquidity
    SPLIT_ORDER = "split_order"        # Split across multiple venues
    ADAPTIVE = "adaptive"              # Dynamically choose based on conditions


@dataclass
class VenueQuote:
    """Quote from a single venue."""
    venue_id: str
    venue_type: VenueType
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    latency_ms: float = 0.0  # Estimated latency to venue

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid > 0:
            return (self.spread / self.mid) * 10000
        return 0.0


@dataclass
class VenueConfig:
    """Configuration for a trading venue."""
    venue_id: str
    venue_type: VenueType
    name: str
    enabled: bool = True
    maker_fee_bps: float = 0.0      # Fee for providing liquidity (can be negative = rebate)
    taker_fee_bps: float = 0.0      # Fee for taking liquidity
    avg_latency_ms: float = 10.0    # Average latency
    min_order_size: int = 1
    max_order_size: int = 100000
    supports_market_orders: bool = True
    supports_limit_orders: bool = True
    supports_stop_orders: bool = True
    supports_iceberg: bool = False
    supports_hidden: bool = False
    connection_status: str = "disconnected"

    def get_fee_bps(self, is_maker: bool) -> float:
        """Get fee in basis points."""
        return self.maker_fee_bps if is_maker else self.taker_fee_bps


@dataclass
class RouteDecision:
    """Result of smart order routing decision."""
    routes: list[tuple[str, int, float]]  # List of (venue_id, quantity, limit_price)
    total_quantity: int
    strategy_used: RoutingStrategy
    expected_avg_price: float
    expected_total_fee_bps: float
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "routes": [
                {"venue": v, "quantity": q, "price": p}
                for v, q, p in self.routes
            ],
            "total_quantity": self.total_quantity,
            "strategy": self.strategy_used.value,
            "expected_avg_price": self.expected_avg_price,
            "expected_fee_bps": self.expected_total_fee_bps,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
        }


class SmartOrderRouter:
    """
    Smart Order Router for multi-venue execution (#E11).

    Implements best execution logic per MiFID II requirements:
    - Considers price, cost, speed, and likelihood of execution
    - Provides audit trail of routing decisions
    - Supports multiple routing strategies
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize smart order router.

        Args:
            config: Configuration with:
                - default_strategy: Default routing strategy
                - max_venue_split: Maximum number of venues to split order
                - min_split_size: Minimum size per venue when splitting
        """
        self._config = config or {}
        self._default_strategy = RoutingStrategy(
            self._config.get("default_strategy", "best_price")
        )
        self._max_venue_split = self._config.get("max_venue_split", 3)
        self._min_split_size = self._config.get("min_split_size", 100)

        # Venue configurations
        self._venues: dict[str, VenueConfig] = {}

        # Quote cache
        self._quotes: dict[str, dict[str, VenueQuote]] = {}  # symbol -> {venue_id -> quote}

        # Routing history for audit
        self._routing_history: list[RouteDecision] = []

        # Add default venues (would be configured from external source in production)
        self._initialize_default_venues()

        logger.info(
            f"SmartOrderRouter initialized: strategy={self._default_strategy.value}, "
            f"max_split={self._max_venue_split}"
        )

    def _initialize_default_venues(self) -> None:
        """Initialize default venue configurations."""
        # Primary exchanges
        self.add_venue(VenueConfig(
            venue_id="NYSE",
            venue_type=VenueType.PRIMARY,
            name="New York Stock Exchange",
            taker_fee_bps=3.0,
            maker_fee_bps=-2.0,  # Maker rebate
            avg_latency_ms=5.0,
        ))

        self.add_venue(VenueConfig(
            venue_id="NASDAQ",
            venue_type=VenueType.PRIMARY,
            name="NASDAQ",
            taker_fee_bps=3.0,
            maker_fee_bps=-2.5,  # Maker rebate
            avg_latency_ms=3.0,
        ))

        # ECNs
        self.add_venue(VenueConfig(
            venue_id="ARCA",
            venue_type=VenueType.ECN,
            name="NYSE Arca",
            taker_fee_bps=3.0,
            maker_fee_bps=-2.0,
            avg_latency_ms=4.0,
            supports_iceberg=True,
        ))

        self.add_venue(VenueConfig(
            venue_id="BATS",
            venue_type=VenueType.ECN,
            name="Cboe BZX Exchange",
            taker_fee_bps=2.5,
            maker_fee_bps=-2.5,
            avg_latency_ms=2.0,
        ))

        self.add_venue(VenueConfig(
            venue_id="IEX",
            venue_type=VenueType.ATS,
            name="Investors Exchange",
            taker_fee_bps=0.9,
            maker_fee_bps=0.0,  # No rebate
            avg_latency_ms=10.0,  # Speed bump
            supports_hidden=True,
        ))

        # SMART routing (IB's internal SOR)
        self.add_venue(VenueConfig(
            venue_id="SMART",
            venue_type=VenueType.PRIMARY,
            name="IB Smart Router",
            taker_fee_bps=2.0,
            maker_fee_bps=0.0,
            avg_latency_ms=5.0,
        ))

    def add_venue(self, venue_config: VenueConfig) -> None:
        """Add or update venue configuration."""
        self._venues[venue_config.venue_id] = venue_config
        logger.debug(f"Added venue: {venue_config.venue_id} ({venue_config.name})")

    def update_quote(self, quote: VenueQuote) -> None:
        """
        Update quote for a venue.

        Called by market data handlers when venue quotes update.
        """
        symbol = quote.symbol

        if symbol not in self._quotes:
            self._quotes[symbol] = {}

        self._quotes[symbol][quote.venue_id] = quote

    def get_quotes(self, symbol: str) -> list[VenueQuote]:
        """Get all quotes for a symbol."""
        return list(self._quotes.get(symbol, {}).values())

    def route_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: int,
        strategy: RoutingStrategy | None = None,
        max_price: float | None = None,  # For buys: max price willing to pay
        min_price: float | None = None,  # For sells: min price willing to accept
    ) -> RouteDecision:
        """
        Determine optimal routing for an order (#E11).

        Args:
            symbol: Instrument symbol
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            strategy: Routing strategy (uses default if None)
            max_price: Maximum price for buys
            min_price: Minimum price for sells

        Returns:
            RouteDecision with routing instructions
        """
        if strategy is None:
            strategy = self._default_strategy

        quotes = self.get_quotes(symbol)

        if not quotes:
            # No quotes available, fall back to SMART routing
            logger.warning(f"No venue quotes for {symbol}, using SMART routing")
            decision = RouteDecision(
                routes=[("SMART", quantity, 0.0)],
                total_quantity=quantity,
                strategy_used=strategy,
                expected_avg_price=0.0,
                expected_total_fee_bps=2.0,  # Default SMART fee
                rationale="No venue quotes available, using IB SMART routing",
            )
            self._routing_history.append(decision)
            return decision

        is_buy = side.lower() == "buy"

        if strategy == RoutingStrategy.BEST_PRICE:
            decision = self._route_best_price(symbol, is_buy, quantity, quotes, max_price, min_price)
        elif strategy == RoutingStrategy.LOWEST_COST:
            decision = self._route_lowest_cost(symbol, is_buy, quantity, quotes, max_price, min_price)
        elif strategy == RoutingStrategy.SPLIT_ORDER:
            decision = self._route_split_order(symbol, is_buy, quantity, quotes, max_price, min_price)
        elif strategy == RoutingStrategy.FASTEST:
            decision = self._route_fastest(symbol, is_buy, quantity, quotes, max_price, min_price)
        elif strategy == RoutingStrategy.LIQUIDITY:
            decision = self._route_liquidity(symbol, is_buy, quantity, quotes, max_price, min_price)
        else:  # ADAPTIVE
            decision = self._route_adaptive(symbol, is_buy, quantity, quotes, max_price, min_price)

        self._routing_history.append(decision)

        logger.info(
            f"SOR decision for {symbol}: {len(decision.routes)} route(s), "
            f"strategy={strategy.value}, expected_price={decision.expected_avg_price:.2f}"
        )

        return decision

    def _route_best_price(
        self,
        symbol: str,
        is_buy: bool,
        quantity: int,
        quotes: list[VenueQuote],
        max_price: float | None,
        min_price: float | None,
    ) -> RouteDecision:
        """Route to venue with best price."""
        # Filter valid quotes
        valid_quotes = []
        for q in quotes:
            venue = self._venues.get(q.venue_id)
            if not venue or not venue.enabled:
                continue

            if is_buy:
                if max_price is not None and q.ask > max_price:
                    continue
                if q.ask_size > 0:
                    valid_quotes.append((q, q.ask))
            else:
                if min_price is not None and q.bid < min_price:
                    continue
                if q.bid_size > 0:
                    valid_quotes.append((q, q.bid))

        if not valid_quotes:
            return RouteDecision(
                routes=[("SMART", quantity, 0.0)],
                total_quantity=quantity,
                strategy_used=RoutingStrategy.BEST_PRICE,
                expected_avg_price=0.0,
                expected_total_fee_bps=2.0,
                rationale="No valid quotes, using SMART routing",
            )

        # Sort by price (lowest ask for buys, highest bid for sells)
        valid_quotes.sort(key=lambda x: x[1], reverse=not is_buy)
        best_quote, best_price = valid_quotes[0]
        venue = self._venues[best_quote.venue_id]

        return RouteDecision(
            routes=[(best_quote.venue_id, quantity, best_price)],
            total_quantity=quantity,
            strategy_used=RoutingStrategy.BEST_PRICE,
            expected_avg_price=best_price,
            expected_total_fee_bps=venue.get_fee_bps(is_maker=False),
            rationale=f"Best price at {best_quote.venue_id}: {best_price}",
        )

    def _route_lowest_cost(
        self,
        symbol: str,
        is_buy: bool,
        quantity: int,
        quotes: list[VenueQuote],
        max_price: float | None,
        min_price: float | None,
    ) -> RouteDecision:
        """Route to venue with lowest total cost (price + fees)."""
        # Calculate effective cost for each venue
        venue_costs = []

        for q in quotes:
            venue = self._venues.get(q.venue_id)
            if not venue or not venue.enabled:
                continue

            if is_buy:
                if max_price is not None and q.ask > max_price:
                    continue
                if q.ask_size <= 0:
                    continue
                price = q.ask
            else:
                if min_price is not None and q.bid < min_price:
                    continue
                if q.bid_size <= 0:
                    continue
                price = q.bid

            # Total cost = price + fees
            fee_cost = price * venue.get_fee_bps(is_maker=False) / 10000
            if is_buy:
                total_cost = price + fee_cost
            else:
                total_cost = -price + fee_cost  # For sells, higher price is better

            venue_costs.append((q, price, total_cost, venue.get_fee_bps(is_maker=False)))

        if not venue_costs:
            return RouteDecision(
                routes=[("SMART", quantity, 0.0)],
                total_quantity=quantity,
                strategy_used=RoutingStrategy.LOWEST_COST,
                expected_avg_price=0.0,
                expected_total_fee_bps=2.0,
                rationale="No valid quotes, using SMART routing",
            )

        # Sort by total cost
        venue_costs.sort(key=lambda x: x[2])
        best = venue_costs[0]

        return RouteDecision(
            routes=[(best[0].venue_id, quantity, best[1])],
            total_quantity=quantity,
            strategy_used=RoutingStrategy.LOWEST_COST,
            expected_avg_price=best[1],
            expected_total_fee_bps=best[3],
            rationale=f"Lowest cost at {best[0].venue_id}: price={best[1]}, fee={best[3]}bps",
        )

    def _route_split_order(
        self,
        symbol: str,
        is_buy: bool,
        quantity: int,
        quotes: list[VenueQuote],
        max_price: float | None,
        min_price: float | None,
    ) -> RouteDecision:
        """Split order across multiple venues to access more liquidity."""
        # Get available liquidity at each venue
        venue_liquidity = []

        for q in quotes:
            venue = self._venues.get(q.venue_id)
            if not venue or not venue.enabled:
                continue

            if is_buy:
                if max_price is not None and q.ask > max_price:
                    continue
                available = min(q.ask_size, quantity)
                price = q.ask
            else:
                if min_price is not None and q.bid < min_price:
                    continue
                available = min(q.bid_size, quantity)
                price = q.bid

            if available >= self._min_split_size:
                venue_liquidity.append((q.venue_id, available, price, venue.get_fee_bps(is_maker=False)))

        if not venue_liquidity:
            return self._route_best_price(symbol, is_buy, quantity, quotes, max_price, min_price)

        # Sort by price (best first)
        venue_liquidity.sort(key=lambda x: x[2], reverse=not is_buy)

        # Allocate quantity across venues
        routes = []
        remaining = quantity
        total_value = 0.0
        total_fee_weighted = 0.0

        for venue_id, available, price, fee_bps in venue_liquidity[:self._max_venue_split]:
            if remaining <= 0:
                break

            alloc = min(available, remaining)
            routes.append((venue_id, alloc, price))
            remaining -= alloc
            total_value += alloc * price
            total_fee_weighted += alloc * fee_bps

        # Handle remaining with best available venue
        if remaining > 0 and routes:
            best_venue = routes[0][0]
            routes[0] = (best_venue, routes[0][1] + remaining, routes[0][2])
            total_value += remaining * routes[0][2]
            total_fee_weighted += remaining * venue_liquidity[0][3]

        total_qty = sum(r[1] for r in routes)
        avg_price = total_value / total_qty if total_qty > 0 else 0
        avg_fee = total_fee_weighted / total_qty if total_qty > 0 else 0

        return RouteDecision(
            routes=routes,
            total_quantity=total_qty,
            strategy_used=RoutingStrategy.SPLIT_ORDER,
            expected_avg_price=avg_price,
            expected_total_fee_bps=avg_fee,
            rationale=f"Split across {len(routes)} venues for liquidity access",
        )

    def _route_fastest(
        self,
        symbol: str,
        is_buy: bool,
        quantity: int,
        quotes: list[VenueQuote],
        max_price: float | None,
        min_price: float | None,
    ) -> RouteDecision:
        """Route to fastest venue."""
        # Filter valid and sort by latency
        valid_venues = []

        for q in quotes:
            venue = self._venues.get(q.venue_id)
            if not venue or not venue.enabled:
                continue

            if is_buy:
                if max_price is not None and q.ask > max_price:
                    continue
                if q.ask_size > 0:
                    valid_venues.append((q, venue.avg_latency_ms, q.ask))
            else:
                if min_price is not None and q.bid < min_price:
                    continue
                if q.bid_size > 0:
                    valid_venues.append((q, venue.avg_latency_ms, q.bid))

        if not valid_venues:
            return RouteDecision(
                routes=[("SMART", quantity, 0.0)],
                total_quantity=quantity,
                strategy_used=RoutingStrategy.FASTEST,
                expected_avg_price=0.0,
                expected_total_fee_bps=2.0,
                rationale="No valid quotes, using SMART routing",
            )

        # Sort by latency (fastest first)
        valid_venues.sort(key=lambda x: x[1])
        fastest = valid_venues[0]
        venue = self._venues[fastest[0].venue_id]

        return RouteDecision(
            routes=[(fastest[0].venue_id, quantity, fastest[2])],
            total_quantity=quantity,
            strategy_used=RoutingStrategy.FASTEST,
            expected_avg_price=fastest[2],
            expected_total_fee_bps=venue.get_fee_bps(is_maker=False),
            rationale=f"Fastest venue: {fastest[0].venue_id} ({venue.avg_latency_ms}ms)",
        )

    def _route_liquidity(
        self,
        symbol: str,
        is_buy: bool,
        quantity: int,
        quotes: list[VenueQuote],
        max_price: float | None,
        min_price: float | None,
    ) -> RouteDecision:
        """Route to venue with deepest liquidity."""
        # Sort by available size
        valid_venues = []

        for q in quotes:
            venue = self._venues.get(q.venue_id)
            if not venue or not venue.enabled:
                continue

            if is_buy:
                if max_price is not None and q.ask > max_price:
                    continue
                if q.ask_size > 0:
                    valid_venues.append((q, q.ask_size, q.ask))
            else:
                if min_price is not None and q.bid < min_price:
                    continue
                if q.bid_size > 0:
                    valid_venues.append((q, q.bid_size, q.bid))

        if not valid_venues:
            return RouteDecision(
                routes=[("SMART", quantity, 0.0)],
                total_quantity=quantity,
                strategy_used=RoutingStrategy.LIQUIDITY,
                expected_avg_price=0.0,
                expected_total_fee_bps=2.0,
                rationale="No valid quotes, using SMART routing",
            )

        # Sort by size (largest first)
        valid_venues.sort(key=lambda x: x[1], reverse=True)
        deepest = valid_venues[0]
        venue = self._venues[deepest[0].venue_id]

        return RouteDecision(
            routes=[(deepest[0].venue_id, quantity, deepest[2])],
            total_quantity=quantity,
            strategy_used=RoutingStrategy.LIQUIDITY,
            expected_avg_price=deepest[2],
            expected_total_fee_bps=venue.get_fee_bps(is_maker=False),
            rationale=f"Deepest liquidity: {deepest[0].venue_id} ({deepest[1]} available)",
        )

    def _route_adaptive(
        self,
        symbol: str,
        is_buy: bool,
        quantity: int,
        quotes: list[VenueQuote],
        max_price: float | None,
        min_price: float | None,
    ) -> RouteDecision:
        """
        Adaptive routing based on order characteristics.

        - Small orders: Route to best price
        - Large orders (> 25% of best venue): Split across venues
        - Urgent orders: Route to fastest venue
        """
        # Determine best available size
        best_size = 0
        for q in quotes:
            if is_buy:
                best_size = max(best_size, q.ask_size)
            else:
                best_size = max(best_size, q.bid_size)

        # Decide strategy based on order size vs liquidity
        if quantity > best_size * 0.5:
            # Large order relative to liquidity - split
            return self._route_split_order(symbol, is_buy, quantity, quotes, max_price, min_price)
        elif quantity > best_size * 0.25:
            # Medium order - lowest cost
            return self._route_lowest_cost(symbol, is_buy, quantity, quotes, max_price, min_price)
        else:
            # Small order - best price
            return self._route_best_price(symbol, is_buy, quantity, quotes, max_price, min_price)

    def get_routing_history(self, limit: int = 100) -> list[RouteDecision]:
        """Get recent routing decisions for audit."""
        return self._routing_history[-limit:]

    def get_venue_stats(self) -> dict[str, Any]:
        """Get venue statistics for monitoring."""
        return {
            venue_id: {
                "name": venue.name,
                "type": venue.venue_type.value,
                "enabled": venue.enabled,
                "maker_fee": venue.maker_fee_bps,
                "taker_fee": venue.taker_fee_bps,
                "latency_ms": venue.avg_latency_ms,
            }
            for venue_id, venue in self._venues.items()
        }

    def get_status(self) -> dict[str, Any]:
        """Get router status for monitoring."""
        return {
            "venues_configured": len(self._venues),
            "venues_enabled": len([v for v in self._venues.values() if v.enabled]),
            "symbols_with_quotes": len(self._quotes),
            "routing_decisions_today": len(self._routing_history),
            "default_strategy": self._default_strategy.value,
        }

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
class VenuePerformance:
    """Performance metrics for a trading venue (P2)."""
    venue_id: str
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    rejected_orders: int = 0
    total_latency_ms: float = 0.0
    total_slippage_bps: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def fill_rate(self) -> float:
        """Full fill rate (0-1)."""
        if self.total_orders == 0:
            return 0.0
        return self.filled_orders / self.total_orders

    @property
    def partial_fill_rate(self) -> float:
        """Partial fill rate (0-1)."""
        if self.total_orders == 0:
            return 0.0
        return self.partial_fills / self.total_orders

    @property
    def rejection_rate(self) -> float:
        """Rejection rate (0-1)."""
        if self.total_orders == 0:
            return 0.0
        return self.rejected_orders / self.total_orders

    @property
    def avg_latency_ms(self) -> float:
        """Average execution latency in milliseconds."""
        if self.total_orders == 0:
            return 0.0
        return self.total_latency_ms / self.total_orders

    @property
    def avg_slippage_bps(self) -> float:
        """Average slippage in basis points."""
        filled = self.filled_orders + self.partial_fills
        if filled == 0:
            return 0.0
        return self.total_slippage_bps / filled

    def record_execution(
        self,
        filled: bool,
        partial: bool,
        rejected: bool,
        latency_ms: float,
        slippage_bps: float = 0.0
    ) -> None:
        """Record an execution result."""
        self.total_orders += 1
        if filled:
            self.filled_orders += 1
        elif partial:
            self.partial_fills += 1
        elif rejected:
            self.rejected_orders += 1
        self.total_latency_ms += latency_ms
        if filled or partial:
            self.total_slippage_bps += slippage_bps
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "venue_id": self.venue_id,
            "total_orders": self.total_orders,
            "fill_rate": self.fill_rate,
            "partial_fill_rate": self.partial_fill_rate,
            "rejection_rate": self.rejection_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_slippage_bps": self.avg_slippage_bps,
            "last_updated": self.last_updated.isoformat(),
        }


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
    - Reg NMS trade-through protection (NBBO validation)
    - Tiered fee schedules based on monthly volume
    """

    # Fee tier thresholds (monthly share volume)
    FEE_TIER_HIGH_VOLUME = 10_000_000   # 10M+ shares: 30% discount
    FEE_TIER_MID_VOLUME = 1_000_000     # 1M+ shares: 15% discount

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

        # Monthly volume tracking for tiered fees (would be updated from external source)
        self._monthly_volume: int = self._config.get("monthly_volume", 0)

        # Venue configurations
        self._venues: dict[str, VenueConfig] = {}

        # Quote cache
        self._quotes: dict[str, dict[str, VenueQuote]] = {}  # symbol -> {venue_id -> quote}

        # Routing history for audit
        self._routing_history: list[RouteDecision] = []

        # Venue performance tracking (P2)
        self._venue_performance: dict[str, VenuePerformance] = {}

        # Fill rate optimization settings (P2)
        self._min_fill_rate_threshold = self._config.get("min_fill_rate_threshold", 0.7)
        self._latency_weight = self._config.get("latency_weight", 0.3)
        self._fill_rate_weight = self._config.get("fill_rate_weight", 0.4)
        self._price_weight = self._config.get("price_weight", 0.3)

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

        # IEX has a 350 microsecond (0.35ms) speed bump - intentional delay to protect
        # against latency arbitrage. This is NOT network latency but a regulatory feature.
        # The speed bump applies to all incoming orders, giving IEX time to adjust prices.
        self.add_venue(VenueConfig(
            venue_id="IEX",
            venue_type=VenueType.ATS,
            name="Investors Exchange",
            taker_fee_bps=0.9,
            maker_fee_bps=0.0,  # No rebate
            avg_latency_ms=0.35,  # 350 microsecond speed bump (regulatory feature, not network latency)
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

    def validate_no_trade_through(
        self, side: str, price: float, nbbo: tuple[float, float]
    ) -> bool:
        """
        Validate order price against NBBO per Reg NMS Rule 611 (Trade-Through Protection).

        Reg NMS prohibits executing trades at prices inferior to protected quotations
        displayed by other trading centers (National Best Bid/Offer).

        Args:
            side: Order side ("BUY" or "SELL")
            price: Proposed execution price
            nbbo: Tuple of (best_bid, best_ask) - National Best Bid and Offer

        Returns:
            True if order would NOT trade through (compliant)
            False if order would trade through protected quote (violation)
        """
        best_bid, best_ask = nbbo

        # Validate NBBO is sensible
        if best_bid <= 0 or best_ask <= 0:
            logger.warning("Invalid NBBO values, skipping trade-through validation")
            return True

        if best_bid >= best_ask:
            logger.warning(f"Crossed NBBO (bid={best_bid} >= ask={best_ask}), market may be locked")
            return True  # Crossed/locked market, allow order

        side_upper = side.upper()

        if side_upper == "BUY":
            # Buy orders must not execute at a price higher than the best offer
            # (would be paying more than necessary, trading through better prices)
            if price > best_ask:
                logger.warning(
                    f"Trade-through violation: BUY at {price} > NBBO ask {best_ask}"
                )
                return False
        elif side_upper == "SELL":
            # Sell orders must not execute at a price lower than the best bid
            # (would be receiving less than available, trading through better prices)
            if price < best_bid:
                logger.warning(
                    f"Trade-through violation: SELL at {price} < NBBO bid {best_bid}"
                )
                return False

        return True

    def get_tiered_fee(self, venue: str, base_fee_bps: float, monthly_volume: int | None = None) -> float:
        """
        Calculate tiered fee based on monthly trading volume.

        Many exchanges offer volume-based fee discounts to incentivize trading activity.
        This implements a standard tiered structure:
        - > 10M shares/month: 30% discount
        - > 1M shares/month: 15% discount
        - Otherwise: base fee

        Args:
            venue: Venue ID (for venue-specific tiers in future)
            base_fee_bps: Base fee in basis points
            monthly_volume: Monthly share volume (uses instance default if None)

        Returns:
            Adjusted fee in basis points
        """
        volume = monthly_volume if monthly_volume is not None else self._monthly_volume

        if volume > self.FEE_TIER_HIGH_VOLUME:
            # High volume tier: 30% discount
            return base_fee_bps * 0.7
        elif volume > self.FEE_TIER_MID_VOLUME:
            # Mid volume tier: 15% discount
            return base_fee_bps * 0.85
        else:
            # Standard tier: no discount
            return base_fee_bps

    def set_monthly_volume(self, volume: int) -> None:
        """
        Update monthly volume for tiered fee calculation.

        Should be called periodically (e.g., daily) with updated volume data.
        """
        self._monthly_volume = volume
        logger.info(f"Updated monthly volume to {volume:,} shares")

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

    def get_nbbo(self, symbol: str) -> tuple[float, float] | None:
        """
        Calculate National Best Bid and Offer (NBBO) from all venue quotes.

        The NBBO represents the best available bid and ask prices across all
        protected trading venues, as required by Reg NMS.

        Args:
            symbol: Instrument symbol

        Returns:
            Tuple of (best_bid, best_ask) or None if no quotes available
        """
        quotes = self.get_quotes(symbol)
        if not quotes:
            return None

        best_bid = 0.0
        best_ask = float('inf')

        for q in quotes:
            venue = self._venues.get(q.venue_id)
            if not venue or not venue.enabled:
                continue

            # Best bid is the highest bid
            if q.bid > 0 and q.bid_size > 0:
                best_bid = max(best_bid, q.bid)

            # Best ask is the lowest ask
            if q.ask > 0 and q.ask_size > 0:
                best_ask = min(best_ask, q.ask)

        if best_bid <= 0 or best_ask == float('inf'):
            return None

        return (best_bid, best_ask)

    def route_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: int,
        strategy: RoutingStrategy | None = None,
        max_price: float | None = None,  # For buys: max price willing to pay
        min_price: float | None = None,  # For sells: min price willing to accept
        validate_nbbo: bool = True,  # Validate against NBBO per Reg NMS
    ) -> RouteDecision:
        """
        Determine optimal routing for an order (#E11).

        Includes Reg NMS trade-through protection by validating proposed
        execution prices against NBBO before routing.

        Args:
            symbol: Instrument symbol
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            strategy: Routing strategy (uses default if None)
            max_price: Maximum price for buys
            min_price: Minimum price for sells
            validate_nbbo: Whether to validate against NBBO (default True)

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

        # Get NBBO for trade-through validation
        nbbo = self.get_nbbo(symbol) if validate_nbbo else None

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

        # Validate routes against NBBO per Reg NMS Rule 611
        if nbbo is not None and decision.routes:
            validated_routes = []
            for venue_id, qty, price in decision.routes:
                if price > 0:  # Only validate priced routes
                    if self.validate_no_trade_through(side, price, nbbo):
                        validated_routes.append((venue_id, qty, price))
                    else:
                        # Route would trade through - adjust to NBBO
                        best_bid, best_ask = nbbo
                        adjusted_price = best_ask if is_buy else best_bid
                        logger.info(
                            f"Adjusted {venue_id} route price from {price} to {adjusted_price} "
                            f"to comply with Reg NMS NBBO"
                        )
                        validated_routes.append((venue_id, qty, adjusted_price))
                else:
                    validated_routes.append((venue_id, qty, price))

            # Update decision with validated routes
            if validated_routes:
                total_value = sum(qty * price for _, qty, price in validated_routes if price > 0)
                total_qty = sum(qty for _, qty, _ in validated_routes)
                decision = RouteDecision(
                    routes=validated_routes,
                    total_quantity=decision.total_quantity,
                    strategy_used=decision.strategy_used,
                    expected_avg_price=total_value / total_qty if total_qty > 0 else 0,
                    expected_total_fee_bps=decision.expected_total_fee_bps,
                    rationale=decision.rationale + " (NBBO validated)",
                )

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
            "venues_with_performance_data": len(self._venue_performance),
        }

    # =========================================================================
    # VENUE PERFORMANCE TRACKING (P2)
    # =========================================================================

    def record_execution_result(
        self,
        venue_id: str,
        filled: bool,
        partial: bool = False,
        rejected: bool = False,
        latency_ms: float = 0.0,
        expected_price: float = 0.0,
        actual_price: float = 0.0
    ) -> None:
        """
        Record execution result for venue performance tracking (P2).

        Call this after each order execution to build performance statistics.

        Args:
            venue_id: Venue identifier
            filled: Whether order was fully filled
            partial: Whether order was partially filled
            rejected: Whether order was rejected
            latency_ms: Execution latency in milliseconds
            expected_price: Expected execution price
            actual_price: Actual execution price
        """
        if venue_id not in self._venue_performance:
            self._venue_performance[venue_id] = VenuePerformance(venue_id=venue_id)

        # Calculate slippage
        slippage_bps = 0.0
        if expected_price > 0 and actual_price > 0:
            slippage_bps = abs(actual_price - expected_price) / expected_price * 10000

        self._venue_performance[venue_id].record_execution(
            filled=filled,
            partial=partial,
            rejected=rejected,
            latency_ms=latency_ms,
            slippage_bps=slippage_bps
        )

        logger.debug(
            f"Recorded execution for {venue_id}: filled={filled}, "
            f"latency={latency_ms}ms, slippage={slippage_bps:.2f}bps"
        )

    def get_venue_performance(self, venue_id: str) -> VenuePerformance | None:
        """Get performance metrics for a specific venue."""
        return self._venue_performance.get(venue_id)

    def get_all_venue_performance(self) -> dict[str, dict[str, Any]]:
        """Get performance metrics for all venues."""
        return {
            venue_id: perf.to_dict()
            for venue_id, perf in self._venue_performance.items()
        }

    def get_best_performing_venues(
        self,
        metric: str = "fill_rate",
        min_orders: int = 10,
        limit: int = 5
    ) -> list[tuple[str, float]]:
        """
        Get best performing venues by specified metric (P2).

        Args:
            metric: Metric to rank by ("fill_rate", "latency", "slippage")
            min_orders: Minimum orders to be considered
            limit: Maximum number of venues to return

        Returns:
            List of (venue_id, metric_value) tuples, sorted best to worst
        """
        results = []

        for venue_id, perf in self._venue_performance.items():
            if perf.total_orders < min_orders:
                continue

            if metric == "fill_rate":
                value = perf.fill_rate
                reverse = True  # Higher is better
            elif metric == "latency":
                value = perf.avg_latency_ms
                reverse = False  # Lower is better
            elif metric == "slippage":
                value = perf.avg_slippage_bps
                reverse = False  # Lower is better
            else:
                continue

            results.append((venue_id, value))

        results.sort(key=lambda x: x[1], reverse=reverse)
        return results[:limit]

    # =========================================================================
    # FILL RATE OPTIMIZATION (P2)
    # =========================================================================

    def route_order_optimized(
        self,
        symbol: str,
        side: str,
        quantity: int,
        max_price: float | None = None,
        min_price: float | None = None,
        urgency: str = "normal"
    ) -> RouteDecision:
        """
        Route order with fill rate optimization (P2).

        Uses historical performance data to optimize routing for best
        expected fill rate while considering price and latency.

        Args:
            symbol: Instrument symbol
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            max_price: Maximum price for buys
            min_price: Minimum price for sells
            urgency: Order urgency ("low", "normal", "high")
                - low: Prioritize price
                - normal: Balance price/fill/latency
                - high: Prioritize fill rate and latency

        Returns:
            RouteDecision optimized for fill probability
        """
        quotes = self.get_quotes(symbol)
        if not quotes:
            return self.route_order(symbol, side, quantity, max_price=max_price, min_price=min_price)

        is_buy = side.lower() == "buy"

        # Score each venue based on multiple factors
        venue_scores = []

        for q in quotes:
            venue = self._venues.get(q.venue_id)
            if not venue or not venue.enabled:
                continue

            # Check price constraints
            if is_buy:
                if max_price is not None and q.ask > max_price:
                    continue
                if q.ask_size <= 0:
                    continue
                price = q.ask
                available = q.ask_size
            else:
                if min_price is not None and q.bid < min_price:
                    continue
                if q.bid_size <= 0:
                    continue
                price = q.bid
                available = q.bid_size

            # Get performance data
            perf = self._venue_performance.get(q.venue_id)

            # Calculate scores
            fill_score = 0.5  # Default if no history
            latency_score = 0.5
            if perf and perf.total_orders >= 5:
                fill_score = perf.fill_rate
                # Skip venues with very low fill rate
                if fill_score < self._min_fill_rate_threshold:
                    logger.debug(f"Skipping {q.venue_id}: fill_rate {fill_score:.2%} < threshold")
                    continue
                # Normalize latency (assuming 0-50ms range)
                latency_score = max(0, 1.0 - (perf.avg_latency_ms / 50.0))

            # Price score (best price = 1.0)
            best_price = min(q2.ask for q2 in quotes if q2.ask > 0) if is_buy else max(q2.bid for q2 in quotes if q2.bid > 0)
            if is_buy:
                price_score = best_price / price if price > 0 else 0
            else:
                price_score = price / best_price if best_price > 0 else 0

            # Liquidity score (can fill full order = 1.0)
            liquidity_score = min(1.0, available / quantity)

            # Adjust weights based on urgency
            if urgency == "high":
                weights = {"fill": 0.5, "latency": 0.3, "price": 0.1, "liquidity": 0.1}
            elif urgency == "low":
                weights = {"fill": 0.2, "latency": 0.1, "price": 0.5, "liquidity": 0.2}
            else:  # normal
                weights = {"fill": self._fill_rate_weight, "latency": self._latency_weight,
                          "price": self._price_weight, "liquidity": 0.1}

            # Calculate composite score
            total_score = (
                weights["fill"] * fill_score +
                weights["latency"] * latency_score +
                weights["price"] * price_score +
                weights["liquidity"] * liquidity_score
            )

            venue_scores.append({
                "venue_id": q.venue_id,
                "quote": q,
                "price": price,
                "available": available,
                "total_score": total_score,
                "fill_score": fill_score,
                "latency_score": latency_score,
                "price_score": price_score,
                "fee_bps": venue.get_fee_bps(is_maker=False),
            })

        if not venue_scores:
            # Fall back to standard routing
            return self.route_order(symbol, side, quantity, max_price=max_price, min_price=min_price)

        # Sort by total score (highest first)
        venue_scores.sort(key=lambda x: x["total_score"], reverse=True)

        # Route to best venue (or split if needed)
        best = venue_scores[0]

        if best["available"] >= quantity:
            # Single venue can fill entire order
            decision = RouteDecision(
                routes=[(best["venue_id"], quantity, best["price"])],
                total_quantity=quantity,
                strategy_used=RoutingStrategy.ADAPTIVE,
                expected_avg_price=best["price"],
                expected_total_fee_bps=best["fee_bps"],
                rationale=(
                    f"Optimized routing to {best['venue_id']}: "
                    f"score={best['total_score']:.2f}, fill_rate={best['fill_score']:.2%}"
                ),
            )
        else:
            # Split across multiple venues
            routes = []
            remaining = quantity
            total_value = 0.0
            total_fee_weighted = 0.0

            for vs in venue_scores[:self._max_venue_split]:
                if remaining <= 0:
                    break
                alloc = min(vs["available"], remaining)
                if alloc >= self._min_split_size:
                    routes.append((vs["venue_id"], alloc, vs["price"]))
                    remaining -= alloc
                    total_value += alloc * vs["price"]
                    total_fee_weighted += alloc * vs["fee_bps"]

            if routes:
                total_qty = sum(r[1] for r in routes)
                decision = RouteDecision(
                    routes=routes,
                    total_quantity=total_qty,
                    strategy_used=RoutingStrategy.SPLIT_ORDER,
                    expected_avg_price=total_value / total_qty if total_qty > 0 else 0,
                    expected_total_fee_bps=total_fee_weighted / total_qty if total_qty > 0 else 0,
                    rationale=f"Optimized split across {len(routes)} venues based on fill rates",
                )
            else:
                decision = self.route_order(symbol, side, quantity, max_price=max_price, min_price=min_price)

        self._routing_history.append(decision)
        return decision

    def update_venue_latency(self, venue_id: str, latency_ms: float) -> None:
        """
        Update measured latency for a venue (P2).

        Call this after measuring actual round-trip latency to update
        the venue's latency estimate.

        Args:
            venue_id: Venue identifier
            latency_ms: Measured latency in milliseconds
        """
        venue = self._venues.get(venue_id)
        if venue:
            # Exponential moving average update
            alpha = 0.2  # Weight for new observation
            venue.avg_latency_ms = alpha * latency_ms + (1 - alpha) * venue.avg_latency_ms
            logger.debug(f"Updated {venue_id} latency to {venue.avg_latency_ms:.2f}ms")

    def get_fill_rate_analysis(self) -> dict[str, Any]:
        """
        Get analysis of fill rates across venues (P2).

        Returns:
            Dictionary with fill rate analysis including:
            - best_fill_venues: Top venues by fill rate
            - worst_fill_venues: Bottom venues by fill rate
            - average_fill_rate: Overall average
            - recommendations: Suggested routing changes
        """
        if not self._venue_performance:
            return {
                "best_fill_venues": [],
                "worst_fill_venues": [],
                "average_fill_rate": 0.0,
                "recommendations": ["No execution data available yet"],
            }

        # Calculate metrics
        fill_rates = []
        for venue_id, perf in self._venue_performance.items():
            if perf.total_orders >= 5:  # Minimum sample
                fill_rates.append((venue_id, perf.fill_rate, perf.total_orders))

        if not fill_rates:
            return {
                "best_fill_venues": [],
                "worst_fill_venues": [],
                "average_fill_rate": 0.0,
                "recommendations": ["Insufficient execution data for analysis"],
            }

        fill_rates.sort(key=lambda x: x[1], reverse=True)
        avg_fill = np.mean([x[1] for x in fill_rates])

        recommendations = []
        # Check for underperforming venues
        for venue_id, rate, orders in fill_rates:
            if rate < self._min_fill_rate_threshold:
                recommendations.append(
                    f"Consider disabling {venue_id}: fill rate {rate:.1%} is below threshold"
                )

        if not recommendations:
            recommendations.append("All venues performing within acceptable parameters")

        return {
            "best_fill_venues": [(v, f"{r:.1%}") for v, r, _ in fill_rates[:3]],
            "worst_fill_venues": [(v, f"{r:.1%}") for v, r, _ in fill_rates[-3:]],
            "average_fill_rate": float(avg_fill),
            "total_venues_analyzed": len(fill_rates),
            "recommendations": recommendations,
        }

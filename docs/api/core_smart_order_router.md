# smart_order_router

**Path**: `C:\Users\Alexa\ai-trading-firm\core\smart_order_router.py`

## Overview

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

## Classes

### VenueType

**Inherits from**: Enum

Trading venue type.

### RoutingStrategy

**Inherits from**: Enum

Order routing strategy.

### VenueQuote

Quote from a single venue.

#### Methods

##### `def mid(self) -> float`

Mid price.

##### `def spread(self) -> float`

Bid-ask spread.

##### `def spread_bps(self) -> float`

Spread in basis points.

### VenueConfig

Configuration for a trading venue.

#### Methods

##### `def get_fee_bps(self, is_maker: bool) -> float`

Get fee in basis points.

### RouteDecision

Result of smart order routing decision.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### SmartOrderRouter

Smart Order Router for multi-venue execution (#E11).

Implements best execution logic per MiFID II requirements:
- Considers price, cost, speed, and likelihood of execution
- Provides audit trail of routing decisions
- Supports multiple routing strategies

#### Methods

##### `def __init__(self, config: )`

Initialize smart order router.

Args:
    config: Configuration with:
        - default_strategy: Default routing strategy
        - max_venue_split: Maximum number of venues to split order
        - min_split_size: Minimum size per venue when splitting

##### `def add_venue(self, venue_config: VenueConfig) -> None`

Add or update venue configuration.

##### `def update_quote(self, quote: VenueQuote) -> None`

Update quote for a venue.

Called by market data handlers when venue quotes update.

##### `def get_quotes(self, symbol: str) -> list[VenueQuote]`

Get all quotes for a symbol.

##### `def route_order(self, symbol: str, side: str, quantity: int, strategy: , max_price: , min_price: ) -> RouteDecision`

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

##### `def get_routing_history(self, limit: int) -> list[RouteDecision]`

Get recent routing decisions for audit.

##### `def get_venue_stats(self) -> dict[str, Any]`

Get venue statistics for monitoring.

##### `def get_status(self) -> dict[str, Any]`

Get router status for monitoring.

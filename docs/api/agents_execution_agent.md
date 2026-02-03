# execution_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\execution_agent.py`

## Overview

Execution Agent
===============

THE ONLY AGENT AUTHORIZED TO SEND ORDERS TO THE BROKER.

Receives validated decisions and executes them via Interactive Brokers.
Implements execution algorithms (TWAP, VWAP) to minimize market impact.

Responsibility: Order execution ONLY.
Does NOT make trading decisions.

## Classes

### OrderBookLevel

Single level in order book.

### OrderBookSnapshot

Order book snapshot for depth analysis (#E15).

Tracks bid/ask levels and derived metrics.

#### Methods

##### `def best_bid(self)`

Best bid price.

##### `def best_ask(self)`

Best ask price.

##### `def mid_price(self)`

Mid-point price.

##### `def spread_bps(self)`

Bid-ask spread in basis points.

##### `def total_bid_depth(self, n_levels: ) -> int`

Total bid size up to n_levels.

##### `def total_ask_depth(self, n_levels: ) -> int`

Total ask size up to n_levels.

##### `def depth_imbalance(self, n_levels: int) -> float`

Order book imbalance ratio.

Positive = more bid depth (buy pressure)
Negative = more ask depth (sell pressure)

##### `def vwap_to_size(self, side: str, target_size: int) -> tuple[float, int]`

Calculate VWAP and filled size to execute target_size.

Args:
    side: 'buy' (consume asks) or 'sell' (consume bids)
    target_size: Target quantity

Returns:
    (vwap, filled_size) tuple

### FillCategory

Categorization of a fill as passive or aggressive (#E20).

### MarketImpactEstimate

Estimated market impact for an order (#E21).

### SliceFill

Tracks fills for a single TWAP/VWAP slice (#E4, #E5).

#### Methods

##### `def __post_init__(self)`

##### `def add_fill(self, quantity: int, price: float) -> None`

Add a fill to this slice.

##### `def fill_rate(self) -> float`

Percentage of target quantity filled.

##### `def slippage_bps(self)`

Calculate slippage in basis points vs arrival price.

##### `def price_improvement_bps(self)`

Calculate price improvement in basis points (#E5).

Price improvement occurs when:
- For buys: fill_price < arrival_price (paid less)
- For sells: fill_price > arrival_price (received more)

Returns positive value if there was improvement, negative if there was slippage.

##### `def has_price_improvement(self) -> bool`

Check if fill received price improvement (#E5).

### PendingOrder

Tracks a pending order with state machine and slice-level fills (#E4).

#### Methods

##### `def __post_init__(self)`

##### `def register_slice(self, broker_id: int, target_quantity: int, arrival_price: , is_buy: bool) -> None`

Register a new slice for tracking (#E4, #E5).

##### `def get_slice_fill(self, broker_id: int)`

Get fill info for a specific slice.

##### `def add_slice_fill(self, broker_id: int, quantity: int, price: float) -> bool`

Add a fill to a slice and update totals (#E4).

Returns True if this was a known slice, False otherwise.

##### `def transition_state(self, new_state: OrderState, reason: str) -> bool`

Attempt to transition to a new state.

Returns True if transition is valid and was applied.

##### `def is_terminal(self) -> bool`

Check if order is in a terminal state.

### ExecutionAgentImpl

**Inherits from**: ExecutionAgentBase

Execution Agent Implementation.

THE ONLY AGENT THAT CAN SEND ORDERS TO INTERACTIVE BROKERS.

Execution algorithms:
- MARKET: Immediate execution at market price
- TWAP: Time-weighted average price
- VWAP: Volume-weighted average price (simplified)

All orders are logged for audit compliance.

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger, broker: IBBroker)`

##### `async def initialize(self) -> None`

Initialize execution agent.

##### `def get_subscribed_events(self) -> list[EventType]`

Execution agent subscribes to validated decisions and kill switch.

##### `async def process_event(self, event: Event) -> None`

Process validated decisions and execute orders.

##### `def update_market_volume(self, symbol: str, volume: int) -> None`

Update market volume observation for participation tracking (#E10).

Called by orchestrator when market data volume updates are received.

Args:
    symbol: Instrument symbol
    volume: Current market volume (typically cumulative or tick volume)

##### `def get_participation_stats(self, symbol: str) -> dict[str, Any]`

Get participation rate statistics for monitoring (#E10).

Args:
    symbol: Instrument symbol

Returns:
    Dictionary with participation statistics

##### `def set_volume_profile(self, symbol: str, profile: list[float]) -> None`

Set historical volume profile for a symbol.

Args:
    symbol: The symbol
    profile: List of relative volume weights for each time bucket

##### `async def cancel_order(self, order_id: str) -> bool`

Cancel a pending order.

##### `def get_pending_orders(self) -> list[dict]`

Get all pending orders for monitoring.

##### `def set_best_execution_analyzer(self, analyzer) -> None`

Set the best execution analyzer for RTS 27/28 compliance tracking.

##### `async def start_stop_order_monitor(self) -> None`

Start the background task that monitors and triggers stop orders.

This should be called when the execution agent starts.

##### `async def stop_stop_order_monitor(self) -> None`

Stop the stop order monitor task.

##### `def start_order_timeout_monitor(self) -> None`

Start background task to monitor order timeouts (#E24).

Cancels orders that exceed configured timeout thresholds.

##### `async def stop_order_timeout_monitor(self) -> None`

Stop the order timeout monitor task.

##### `def get_partial_fill_timeouts(self) -> list[dict]`

P1-14: Get list of orders that timed out with partial fills.

These represent execution failures where the full order was not completed
and remaining quantity was not executed.

##### `def get_timed_out_orders(self) -> list[str]`

Get list of order IDs that have been cancelled due to timeout.

##### `def get_order_age(self, order_id: str)`

Get age of an order in seconds (#E24).

Returns None if order not found.

##### `def register_stop_order(self, pending: PendingOrder) -> None`

Register a stop order for monitoring.

Called when a stop order is created but not yet triggered.

##### `def update_price(self, symbol: str, price: float) -> None`

Update the last known price for a symbol.

Called by market data feed to enable stop order triggering.

##### `def get_stop_orders(self) -> list[dict]`

Get all pending stop orders for monitoring.

##### `def get_fill_quality_report(self, order_id: str)`

Get detailed fill quality report for an order (#E4, #E5).

Args:
    order_id: Order ID

Returns:
    Fill quality metrics dict, or None if order not found

##### `def get_status(self) -> dict`

Get execution agent status for monitoring.

##### `def persist_orders_to_file(self, filepath: str) -> int`

Persist pending orders to file for recovery after restart (#E12).

Saves all non-terminal pending orders to JSON for later recovery.

Args:
    filepath: Path to save orders

Returns:
    Number of orders persisted

##### `async def recover_orders_from_file(self, filepath: str) -> int`

Recover pending orders from file after restart (#E12).

Loads persisted orders and attempts to reconcile with broker state.

Args:
    filepath: Path to load orders from

Returns:
    Number of orders recovered

##### `def get_aggregate_fill_metrics(self) -> dict`

Get aggregate fill quality metrics across all orders (#E13).

Returns summary statistics for execution quality monitoring.

##### `def calculate_implementation_shortfall(self, order_id: str)`

Calculate implementation shortfall for an order (#E14).

Implementation shortfall = (Actual Execution Cost) - (Paper Portfolio Cost)
Components:
1. Delay cost: Price movement from decision to execution start
2. Trading impact: Price movement during execution
3. Opportunity cost: Unfilled portion

Args:
    order_id: Order ID to analyze

Returns:
    Implementation shortfall breakdown, or None if order not found

##### `def get_implementation_shortfall_summary(self) -> dict`

Get aggregate implementation shortfall metrics (#E14).

Returns summary across all completed orders.

##### `def update_order_book(self, symbol: str, bids: list[tuple[float, int, int]], asks: list[tuple[float, int, int]]) -> None`

Update order book snapshot for a symbol (#E15).

Called by market data handler when order book updates received.

Args:
    symbol: Instrument symbol
    bids: List of (price, size, num_orders) tuples, best first
    asks: List of (price, size, num_orders) tuples, best first

##### `def analyze_order_book(self, symbol: str)`

Analyze current order book depth (#E15).

Returns metrics useful for execution decisions.

Args:
    symbol: Instrument symbol

Returns:
    Order book analysis or None if no data

##### `def estimate_execution_cost(self, symbol: str, side: str, quantity: int)`

Estimate execution cost using order book (#E15).

Calculates expected VWAP and slippage for given order size.

Args:
    symbol: Instrument symbol
    side: 'buy' or 'sell'
    quantity: Order quantity

Returns:
    Execution cost estimate or None if no book data

##### `def categorize_fill(self, order_id: str, fill_price: float, fill_quantity: int, fill_side: str) -> FillCategory`

Categorize a fill as passive or aggressive (#E20).

Aggressive: Takes liquidity (crosses spread)
Passive: Provides liquidity (rests in book)
Midpoint: Filled at or near mid price

Args:
    order_id: Order identifier
    fill_price: Price at which filled
    fill_quantity: Quantity filled
    fill_side: 'buy' or 'sell'

Returns:
    FillCategory with classification

##### `def get_fill_categorization_summary(self) -> dict`

Get summary of fill categorizations (#E20).

Returns breakdown of aggressive vs passive fills.

##### `def estimate_market_impact(self, symbol: str, side: str, quantity: int, price: , adv: , volatility: ) -> MarketImpactEstimate`

Estimate market impact for an order (#E21).

Uses a square-root market impact model (Almgren-Chriss simplified):

Impact = η * σ * √(Q/V)

Where:
- η = market impact coefficient
- σ = volatility
- Q = order quantity
- V = average daily volume

Args:
    symbol: Instrument symbol
    side: 'buy' or 'sell'
    quantity: Order quantity
    price: Current price (uses last price if None)
    adv: Average daily volume (uses default if None)
    volatility: Daily volatility (uses default if None)

Returns:
    MarketImpactEstimate with impact breakdown

##### `def configure_market_impact_model(self, eta: , gamma: , alpha: ) -> None`

Configure market impact model parameters (#E21).

Args:
    eta: Temporary impact coefficient (default 0.1)
    gamma: Permanent impact coefficient (default 0.1)
    alpha: Square-root power (default 0.5)

##### `def should_cross_spread(self, symbol: str, side: str, urgency: float, opportunity_cost_bps: ) -> dict`

Determine whether to cross the spread for immediate execution (#E16).

Compares the cost of crossing (paying the spread) against:
- Opportunity cost of waiting (missed alpha)
- Queue position probability
- Market conditions

Args:
    symbol: Instrument symbol
    side: 'buy' or 'sell'
    urgency: Urgency factor 0-1 (1 = max urgency)
    opportunity_cost_bps: Estimated alpha decay per unit time (bps/min)

Returns:
    Decision dict with recommendation and breakdown

##### `def execute_with_spread_awareness(self, order: OrderEvent, urgency: float) -> dict`

Execute order with spread-aware logic (#E16).

Automatically chooses between aggressive (crossing) and passive
(limit at bid/ask) execution based on market conditions.

Args:
    order: Order to execute
    urgency: Urgency factor 0-1

Returns:
    Execution decision details

##### `def estimate_queue_position(self, symbol: str, side: str, size: int, price: ) -> dict`

Estimate queue position and expected fill time (#E17).

Args:
    symbol: Instrument symbol
    side: 'buy' or 'sell'
    size: Order size
    price: Limit price (uses NBBO if None)

Returns:
    Queue position analysis

##### `async def execute_midpoint_peg(self, order: OrderEvent, max_deviation_bps: float, timeout_seconds: float) -> dict`

Execute order with midpoint pegging (#E18).

Places limit order at midpoint and re-pegs on price changes.
Common for reducing spread costs in less urgent executions.

Args:
    order: Order to execute
    max_deviation_bps: Max price deviation before re-peg (bps from mid)
    timeout_seconds: Maximum time to attempt execution

Returns:
    Execution result

##### `async def execute_iceberg(self, order: OrderEvent, display_size: int, variance_pct: float, min_replenish_seconds: float, price_offset_ticks: int) -> dict`

Execute iceberg order with hidden quantity (#E19).

Shows only display_size at a time, replenishing as fills occur.
Useful for large orders to avoid market impact from signaling.

Args:
    order: Full order (total quantity)
    display_size: Visible quantity per slice
    variance_pct: Random variance in display size (0-1)
    min_replenish_seconds: Minimum time between slice submissions
    price_offset_ticks: Ticks from NBBO (0 = at NBBO)

Returns:
    Execution result

##### `def calculate_post_trade_tca(self, order_id: str, benchmark: str)`

Calculate comprehensive post-trade Transaction Cost Analysis (#E22).

Analyzes execution quality against multiple benchmarks.

Args:
    order_id: Order ID to analyze
    benchmark: Primary benchmark ('arrival', 'vwap', 'twap', 'close')

Returns:
    TCA report or None if order not found

##### `def generate_tca_report(self, start_date: , end_date: ) -> dict`

Generate aggregate TCA report for a period (#E22).

Args:
    start_date: Report start (default: all data)
    end_date: Report end (default: now)

Returns:
    Aggregate TCA statistics

##### `def __init_venue_latency(self)`

Initialize venue latency tracking structures.

##### `def record_venue_latency(self, venue: str, latency_ms: float, event_type: str) -> None`

Record latency measurement for a venue (#E23).

Args:
    venue: Venue/exchange identifier
    latency_ms: Round-trip latency in milliseconds
    event_type: Type of event ('order', 'fill', 'cancel', 'quote')

##### `def get_venue_latency_stats(self, venue: , lookback_minutes: int) -> dict`

Get venue latency statistics (#E23).

Args:
    venue: Specific venue or None for all
    lookback_minutes: Analysis window

Returns:
    Latency statistics by venue and event type

##### `def set_latency_threshold(self, threshold_ms: float) -> None`

Set latency alert threshold in milliseconds (#E23).

##### `def check_venue_health(self) -> dict`

Check overall venue connectivity health (#E23).

Returns:
    Health status for each venue

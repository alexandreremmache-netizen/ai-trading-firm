"""
Execution Agent
===============

THE ONLY AGENT AUTHORIZED TO SEND ORDERS TO THE BROKER.

Receives validated decisions and executes them via Interactive Brokers.
Implements execution algorithms (TWAP, VWAP) to minimize market impact.

Responsibility: Order execution ONLY.
Does NOT make trading decisions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from core.agent_base import ExecutionAgent as ExecutionAgentBase, AgentConfig
from core.events import (
    Event,
    EventType,
    ValidatedDecisionEvent,
    DecisionEvent,
    OrderEvent,
    FillEvent,
    OrderSide,
    OrderType,
    OrderState,
    OrderStateChangeEvent,
    KillSwitchEvent,
    is_valid_state_transition,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.broker import IBBroker

from core.contract_specs import ContractSpecsManager


logger = logging.getLogger(__name__)


# =========================================================================
# ORDER BOOK DEPTH ANALYSIS (#E15)
# =========================================================================

@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: int
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """
    Order book snapshot for depth analysis (#E15).

    Tracks bid/ask levels and derived metrics.
    """
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel]  # Best bid first
    asks: list[OrderBookLevel]  # Best ask first

    @property
    def best_bid(self) -> float | None:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        """Mid-point price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread_bps(self) -> float | None:
        """Bid-ask spread in basis points."""
        if self.best_bid and self.best_ask and self.mid_price:
            return (self.best_ask - self.best_bid) / self.mid_price * 10000
        return None

    def total_bid_depth(self, n_levels: int | None = None) -> int:
        """Total bid size up to n_levels."""
        levels = self.bids[:n_levels] if n_levels else self.bids
        return sum(level.size for level in levels)

    def total_ask_depth(self, n_levels: int | None = None) -> int:
        """Total ask size up to n_levels."""
        levels = self.asks[:n_levels] if n_levels else self.asks
        return sum(level.size for level in levels)

    def depth_imbalance(self, n_levels: int = 5) -> float:
        """
        Order book imbalance ratio.

        Positive = more bid depth (buy pressure)
        Negative = more ask depth (sell pressure)
        """
        bid_depth = self.total_bid_depth(n_levels)
        ask_depth = self.total_ask_depth(n_levels)
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        return (bid_depth - ask_depth) / total

    def vwap_to_size(self, side: str, target_size: int) -> tuple[float, int]:
        """
        Calculate VWAP and filled size to execute target_size.

        Args:
            side: 'buy' (consume asks) or 'sell' (consume bids)
            target_size: Target quantity

        Returns:
            (vwap, filled_size) tuple
        """
        levels = self.asks if side == 'buy' else self.bids
        remaining = target_size
        total_value = 0.0
        total_filled = 0

        for level in levels:
            fill_at_level = min(remaining, level.size)
            total_value += fill_at_level * level.price
            total_filled += fill_at_level
            remaining -= fill_at_level
            if remaining <= 0:
                break

        vwap = total_value / total_filled if total_filled > 0 else 0.0
        return vwap, total_filled


@dataclass
class FillCategory:
    """Categorization of a fill as passive or aggressive (#E20)."""
    is_aggressive: bool
    category: str  # "aggressive", "passive", "midpoint", "unknown"
    price_vs_arrival_bps: float  # Positive = paid more (buy) / received less (sell)
    price_vs_spread_position: float  # 0 = at bid (buy), 1 = at ask (buy)


@dataclass
class MarketImpactEstimate:
    """Estimated market impact for an order (#E21)."""
    symbol: str
    side: str
    quantity: int
    temporary_impact_bps: float  # Price impact during execution
    permanent_impact_bps: float  # Long-term price impact
    total_impact_bps: float
    estimated_cost: float  # In currency
    model_used: str  # "square_root", "linear", "almgren_chriss"


@dataclass
class SliceFill:
    """Tracks fills for a single TWAP/VWAP slice (#E4, #E5)."""
    slice_index: int
    broker_order_id: int
    target_quantity: int
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fills: list[tuple[datetime, int, float]] = None  # (timestamp, quantity, price)
    arrival_price: float | None = None  # Price when slice was sent
    is_complete: bool = False
    is_buy: bool = True  # Direction for price improvement calculation (#E5)

    def __post_init__(self):
        self.fills = self.fills or []

    def add_fill(self, quantity: int, price: float) -> None:
        """Add a fill to this slice."""
        now = datetime.now(timezone.utc)
        self.fills.append((now, quantity, price))
        prev_filled = self.filled_quantity
        self.filled_quantity += quantity

        # Update weighted average price
        if self.filled_quantity > 0:
            total_value = (self.avg_fill_price * prev_filled) + (price * quantity)
            self.avg_fill_price = total_value / self.filled_quantity

        # Check if complete
        if self.filled_quantity >= self.target_quantity:
            self.is_complete = True

    @property
    def fill_rate(self) -> float:
        """Percentage of target quantity filled."""
        if self.target_quantity <= 0:
            return 0.0
        return self.filled_quantity / self.target_quantity

    @property
    def slippage_bps(self) -> float | None:
        """Calculate slippage in basis points vs arrival price."""
        if self.arrival_price is None or self.arrival_price <= 0 or self.filled_quantity == 0:
            return None
        return (self.avg_fill_price - self.arrival_price) / self.arrival_price * 10000

    @property
    def price_improvement_bps(self) -> float | None:
        """
        Calculate price improvement in basis points (#E5).

        Price improvement occurs when:
        - For buys: fill_price < arrival_price (paid less)
        - For sells: fill_price > arrival_price (received more)

        Returns positive value if there was improvement, negative if there was slippage.
        """
        if self.arrival_price is None or self.arrival_price <= 0 or self.filled_quantity == 0:
            return None

        price_diff = self.arrival_price - self.avg_fill_price

        if self.is_buy:
            # For buys, paying less is improvement (positive diff = good)
            improvement = price_diff
        else:
            # For sells, receiving more is improvement (negative diff = good)
            improvement = -price_diff

        return (improvement / self.arrival_price) * 10000

    @property
    def has_price_improvement(self) -> bool:
        """Check if fill received price improvement (#E5)."""
        improvement = self.price_improvement_bps
        return improvement is not None and improvement > 0


@dataclass
class PendingOrder:
    """Tracks a pending order with state machine and slice-level fills (#E4)."""
    order_event: OrderEvent
    decision_event: DecisionEvent
    broker_order_id: int | None = None
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    status: str = "pending"
    state: OrderState = OrderState.CREATED
    created_at: datetime = None
    slices: list[int] = None  # Broker IDs for TWAP/VWAP slices
    slice_fills: dict[int, SliceFill] = None  # broker_id -> SliceFill (#E4)
    state_history: list[tuple[datetime, OrderState, str]] = None  # (timestamp, state, reason)
    arrival_price: float | None = None  # Price at order submission

    def __post_init__(self):
        self.created_at = self.created_at or datetime.now(timezone.utc)
        self.slices = self.slices or []
        self.slice_fills = self.slice_fills if self.slice_fills is not None else {}
        self.state_history = self.state_history or [(self.created_at, OrderState.CREATED, "Order created")]
        self.remaining_quantity = self.order_event.quantity

    def register_slice(
        self,
        broker_id: int,
        target_quantity: int,
        arrival_price: float | None = None,
        is_buy: bool = True
    ) -> None:
        """Register a new slice for tracking (#E4, #E5)."""
        slice_index = len(self.slice_fills)
        self.slices.append(broker_id)
        self.slice_fills[broker_id] = SliceFill(
            slice_index=slice_index,
            broker_order_id=broker_id,
            target_quantity=target_quantity,
            arrival_price=arrival_price,
            is_buy=is_buy,
        )

    def get_slice_fill(self, broker_id: int) -> SliceFill | None:
        """Get fill info for a specific slice."""
        return self.slice_fills.get(broker_id)

    def add_slice_fill(self, broker_id: int, quantity: int, price: float) -> bool:
        """
        Add a fill to a slice and update totals (#E4).

        Returns True if this was a known slice, False otherwise.
        """
        slice_fill = self.slice_fills.get(broker_id)
        if slice_fill is not None:
            slice_fill.add_fill(quantity, price)
            return True
        return False

    def transition_state(self, new_state: OrderState, reason: str = "") -> bool:
        """
        Attempt to transition to a new state.

        Returns True if transition is valid and was applied.
        """
        if not is_valid_state_transition(self.state, new_state):
            logger.warning(
                f"Invalid state transition for order {self.order_event.event_id}: "
                f"{self.state.value} -> {new_state.value}"
            )
            return False

        old_state = self.state
        self.state = new_state
        self.state_history.append((datetime.now(timezone.utc), new_state, reason))

        # Update legacy status field for compatibility
        self.status = new_state.value

        logger.debug(f"Order state: {old_state.value} -> {new_state.value} ({reason})")
        return True

    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        )


class ExecutionAgentImpl(ExecutionAgentBase):
    """
    Execution Agent Implementation.

    THE ONLY AGENT THAT CAN SEND ORDERS TO INTERACTIVE BROKERS.

    Execution algorithms:
    - MARKET: Immediate execution at market price
    - TWAP: Time-weighted average price
    - VWAP: Volume-weighted average price (simplified)

    All orders are logged for audit compliance.
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
        broker: IBBroker,
    ):
        super().__init__(config, event_bus, audit_logger)
        self._broker = broker

        # Configuration
        self._default_algo = config.parameters.get("default_algo", "TWAP")
        self._slice_interval = config.parameters.get("slice_interval_seconds", 60)
        self._max_slippage_bps = config.parameters.get("max_slippage_bps", 50)

        # Rate limiting configuration (anti-HFT per CLAUDE.md)
        self._max_orders_per_minute = config.parameters.get("max_orders_per_minute", 10)
        self._min_order_interval_ms = config.parameters.get("min_order_interval_ms", 100)

        # State
        self._pending_orders: dict[str, PendingOrder] = {}  # order_id -> PendingOrder
        self._decision_cache: dict[str, DecisionEvent] = {}  # decision_id -> DecisionEvent
        self._order_timestamps: list[datetime] = []  # For rate limiting
        self._last_order_time: datetime | None = None  # For interval checking

        # Stop order tracking (#E8)
        self._stop_orders: dict[str, PendingOrder] = {}  # order_id -> stop orders waiting to trigger
        self._last_prices: dict[str, float] = {}  # symbol -> last known price
        self._stop_check_interval_seconds = config.parameters.get("stop_check_interval_seconds", 0.1)
        self._stop_order_monitor_task: asyncio.Task | None = None

        # Kill switch state
        self._kill_switch_active = False
        self._kill_switch_reason = ""

        # VWAP execution state
        # Historical volume profiles by symbol (hourly buckets, 0-23 representing market hours)
        self._volume_profiles: dict[str, list[float]] = {}  # symbol -> normalized hourly volumes
        self._vwap_participation_rate = config.parameters.get("vwap_participation_rate", 0.1)  # 10% of volume

        # Dynamic VWAP participation rate tracking (#E10)
        self._vwap_target_participation = config.parameters.get("vwap_target_participation", 0.10)  # Target 10%
        self._vwap_min_participation = config.parameters.get("vwap_min_participation", 0.05)  # Min 5%
        self._vwap_max_participation = config.parameters.get("vwap_max_participation", 0.25)  # Max 25%
        self._market_volume_history: dict[str, list[tuple[datetime, int]]] = {}  # symbol -> [(time, volume)]
        self._our_volume_history: dict[str, list[tuple[datetime, int]]] = {}  # symbol -> [(time, filled)]
        self._volume_window_minutes = config.parameters.get("volume_window_minutes", 5)  # Rolling window
        self._participation_adjustment_factor = config.parameters.get("participation_adjustment_factor", 0.5)  # Smoothing

        # Contract specs for tick size validation
        self._contract_specs = ContractSpecsManager()

        # Optional components
        self._best_execution_analyzer = None

        # Order book tracking (#E15)
        self._order_books: dict[str, OrderBookSnapshot] = {}

        # Fill categorization tracking (#E20)
        self._fill_categories: dict[str, list[FillCategory]] = {}  # order_id -> categories

        # Market impact model parameters (#E21)
        self._market_impact_params = {
            "eta": 0.1,  # Temporary impact coefficient
            "gamma": 0.1,  # Permanent impact coefficient
            "alpha": 0.5,  # Power for square-root model
        }

        # Thread safety for fill handling
        import threading
        self._fill_lock = threading.Lock()

        # Register fill callback
        self._broker.on_fill(self._handle_fill)

    async def initialize(self) -> None:
        """Initialize execution agent."""
        logger.info(
            f"ExecutionAgent initializing with algo={self._default_algo}, "
            f"slice_interval={self._slice_interval}s"
        )

    def get_subscribed_events(self) -> list[EventType]:
        """Execution agent subscribes to validated decisions and kill switch."""
        return [EventType.VALIDATED_DECISION, EventType.DECISION, EventType.KILL_SWITCH]

    async def process_event(self, event: Event) -> None:
        """
        Process validated decisions and execute orders.
        """
        # Handle kill switch events
        if isinstance(event, KillSwitchEvent):
            await self._handle_kill_switch(event)
            return

        # Cache decisions for reference
        if isinstance(event, DecisionEvent):
            self._decision_cache[event.event_id] = event
            return

        if not isinstance(event, ValidatedDecisionEvent):
            return

        # Check kill switch before executing
        if self._kill_switch_active:
            logger.warning(
                f"Kill switch active, rejecting order for decision {event.original_decision_id}: "
                f"{self._kill_switch_reason}"
            )
            return

        # Only execute approved decisions
        if not event.approved:
            logger.info(f"Decision {event.original_decision_id} rejected, not executing")
            return

        # Get original decision
        decision = self._decision_cache.get(event.original_decision_id)
        if not decision:
            logger.error(f"Decision {event.original_decision_id} not found in cache")
            return

        # Use adjusted quantity if provided
        quantity = event.adjusted_quantity or decision.quantity

        # Execute the decision
        await self._execute_decision(decision, quantity, event)

    async def _handle_kill_switch(self, event: KillSwitchEvent) -> None:
        """Handle kill switch activation/deactivation."""
        if event.activated:
            self._kill_switch_active = True
            self._kill_switch_reason = event.reason
            logger.critical(
                f"KILL SWITCH ACTIVATED: {event.reason} (trigger: {event.trigger_type})"
            )

            # Cancel all pending orders if requested
            if event.cancel_pending_orders:
                await self._cancel_all_pending_orders(event.reason)

            # Audit log
            self._audit_logger.log_event(event)

        else:
            self._kill_switch_active = False
            self._kill_switch_reason = ""
            logger.info(f"Kill switch deactivated: {event.reason}")

    async def _cancel_all_pending_orders(self, reason: str) -> None:
        """Cancel all pending orders (kill switch response)."""
        pending_ids = list(self._pending_orders.keys())
        cancelled_count = 0

        for order_id in pending_ids:
            pending = self._pending_orders.get(order_id)
            if pending and not pending.is_terminal():
                success = await self.cancel_order(order_id)
                if success:
                    cancelled_count += 1
                    pending.transition_state(OrderState.CANCELLED, f"Kill switch: {reason}")

        logger.info(f"Kill switch: cancelled {cancelled_count}/{len(pending_ids)} pending orders")

    def _enforce_tick_size(self, symbol: str, price: float | None) -> float | None:
        """
        Round price to nearest valid tick for the symbol.

        Args:
            symbol: Contract symbol
            price: Price to round

        Returns:
            Tick-aligned price, or original price if symbol not found
        """
        if price is None:
            return None

        rounded = self._contract_specs.round_to_tick(symbol, price)
        if rounded is not None and rounded != price:
            logger.info(f"Price adjusted to tick size for {symbol}: {price} -> {rounded}")
            return rounded

        return price

    async def _execute_decision(
        self,
        decision: DecisionEvent,
        quantity: int,
        validation: ValidatedDecisionEvent,
    ) -> None:
        """
        Execute a validated trading decision.

        This is THE execution point - all orders go through here.
        """
        if not self._broker.is_connected:
            logger.error("Cannot execute: broker not connected")
            return

        # Enforce tick size on limit/stop prices
        limit_price = self._enforce_tick_size(decision.symbol, decision.limit_price)
        stop_price = self._enforce_tick_size(decision.symbol, decision.stop_price)

        # Create order event
        order = OrderEvent(
            source_agent=self.name,
            decision_id=decision.event_id,
            validation_id=validation.event_id,
            symbol=decision.symbol,
            side=decision.action or OrderSide.BUY,
            quantity=quantity,
            order_type=decision.order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            algo=self._default_algo,
        )

        # Track pending order
        pending = PendingOrder(
            order_event=order,
            decision_event=decision,
            remaining_quantity=quantity,
        )
        self._pending_orders[order.event_id] = pending

        # Check rate limits (anti-HFT per CLAUDE.md)
        if not self._check_rate_limits():
            logger.warning(f"Rate limit exceeded, rejecting order for {decision.symbol}")
            pending.status = "rejected_rate_limit"
            return

        # Handle stop orders separately - they wait for trigger
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            self.register_stop_order(pending)
            logger.info(
                f"Stop order registered (waiting for trigger): {order.side.value} "
                f"{order.quantity} {order.symbol} @ stop={order.stop_price}"
            )
            self._audit_logger.log_event(order)
            return

        # Execute based on algorithm
        if self._default_algo == "MARKET":
            await self._execute_market(pending)
        elif self._default_algo == "TWAP":
            await self._execute_twap(pending)
        elif self._default_algo == "VWAP":
            await self._execute_vwap(pending)
        else:
            logger.warning(f"Unknown algo {self._default_algo}, using MARKET")
            await self._execute_market(pending)

        # Log order for audit
        self._audit_logger.log_event(order)

    async def _execute_market(self, pending: PendingOrder) -> None:
        """
        Execute order immediately at market price with slippage protection (#E6).

        Instead of a pure market order, uses an aggressive limit order with
        a price cap to prevent excessive slippage. The limit price is set at
        arrival_price +/- max_slippage_bps based on order direction.
        """
        order = pending.order_event

        # Transition to pending state
        pending.transition_state(OrderState.PENDING, "Starting market execution")

        # Get arrival price for slippage cap (#E6)
        arrival_price = self._last_prices.get(order.symbol)

        if arrival_price and arrival_price > 0 and self._max_slippage_bps > 0:
            # Calculate slippage cap limit price
            slippage_factor = self._max_slippage_bps / 10000.0  # Convert bps to decimal

            if order.side == OrderSide.BUY:
                # For buys, cap at arrival + max slippage
                cap_price = arrival_price * (1 + slippage_factor)
            else:
                # For sells, cap at arrival - max slippage
                cap_price = arrival_price * (1 - slippage_factor)

            # Round to 2 decimal places
            cap_price = round(cap_price, 2)

            # Create capped limit order instead of market order
            capped_order = OrderEvent(
                source_agent=self.name,
                decision_id=order.decision_id,
                validation_id=order.validation_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=OrderType.LIMIT,
                limit_price=cap_price,
                algo="MARKET_WITH_CAP",
            )

            logger.info(
                f"Market order converted to limit with slippage cap: "
                f"{order.side.value} {order.quantity} {order.symbol} "
                f"arrival={arrival_price:.2f}, cap={cap_price:.2f} "
                f"(max_slippage={self._max_slippage_bps}bps)"
            )

            broker_id = await self._broker.place_order(capped_order)
        else:
            # No price data or slippage cap disabled - use regular market order
            if not arrival_price:
                logger.warning(
                    f"No arrival price for {order.symbol}, using uncapped market order"
                )
            broker_id = await self._broker.place_order(order)

        if broker_id:
            pending.broker_order_id = broker_id
            pending.arrival_price = arrival_price
            pending.transition_state(OrderState.SUBMITTED, f"Submitted to broker (id={broker_id})")
            self._record_order_timestamp()  # Record for rate limiting
            logger.info(
                f"Market order submitted: {order.side.value} {order.quantity} "
                f"{order.symbol} (broker_id={broker_id})"
            )
        else:
            pending.transition_state(OrderState.FAILED, "Broker submission failed")
            logger.error(f"Failed to submit market order for {order.symbol}")

    def _get_lot_size(self, symbol: str) -> int:
        """
        Get the lot size for a symbol (#E9).

        Different instruments have different minimum trading quantities:
        - Stocks: Usually 1 share (but some prefer round lots of 100)
        - Futures: Contract-specific (e.g., ES=1, crude oil=1)
        - Options: Usually 1 contract (100 shares underlying)

        Returns:
            Minimum lot size for the symbol
        """
        # Try to get from contract specs if available
        if self._broker and hasattr(self._broker, 'get_contract_specs'):
            specs = self._broker.get_contract_specs(symbol)
            if specs and hasattr(specs, 'lot_size'):
                return specs.lot_size

        # Default lot sizes by symbol pattern
        symbol_upper = symbol.upper()

        # Futures contracts typically trade in single contracts
        futures_symbols = ['ES', 'NQ', 'YM', 'RTY', 'CL', 'GC', 'SI', 'ZB', 'ZN', 'ZC', 'ZW', 'ZS']
        for fut in futures_symbols:
            if symbol_upper.startswith(fut):
                return 1

        # FX pairs - micro lots (1000 units) or standard lots
        fx_pairs = ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF']
        if any(ccy in symbol_upper for ccy in fx_pairs) and len(symbol_upper) == 6:
            return 1000  # Micro lot

        # Stocks - prefer round lots of 100 for larger orders, 1 for small
        # Round lot preference only kicks in for orders > 100
        return 1  # Default to 1 share

    def _round_to_lot_size(self, quantity: int, lot_size: int, round_up: bool = False) -> int:
        """
        Round quantity to the nearest lot size (#E9).

        Args:
            quantity: Desired quantity
            lot_size: Lot size to round to
            round_up: If True, always round up; if False, round to nearest

        Returns:
            Rounded quantity
        """
        if lot_size <= 1:
            return quantity

        if round_up:
            return ((quantity + lot_size - 1) // lot_size) * lot_size
        else:
            return max(lot_size, (quantity // lot_size) * lot_size)

    async def _execute_twap(self, pending: PendingOrder) -> None:
        """
        Execute order using Time-Weighted Average Price algorithm.

        Splits order into equal slices over time to minimize market impact.
        Properly handles lot sizes for different instrument types (#E9).
        """
        order = pending.order_event
        total_quantity = pending.remaining_quantity

        # Transition to pending state
        pending.transition_state(OrderState.PENDING, "Starting TWAP execution")

        # Get lot size for the symbol (#E9)
        lot_size = self._get_lot_size(order.symbol)
        min_slice_size = max(lot_size, 10)  # Minimum slice of 10 or lot size

        # Calculate slices with lot size consideration (#E9)
        # Target slice size: at least min_slice_size, aim for 100 shares for stocks
        target_slice_size = max(min_slice_size, 100) if lot_size == 1 else lot_size
        num_slices = max(1, total_quantity // target_slice_size)
        num_slices = min(num_slices, 10)  # Max 10 slices

        # Calculate base slice size rounded to lot size
        base_slice_size = self._round_to_lot_size(total_quantity // num_slices, lot_size)

        # Ensure slice is at least minimum size
        if base_slice_size < min_slice_size:
            base_slice_size = min_slice_size
            num_slices = max(1, total_quantity // base_slice_size)

        # Build slice sizes array (#E9)
        slices = []
        remaining = total_quantity
        for i in range(num_slices):
            if i == num_slices - 1:
                # Last slice gets remainder (rounded to lot size)
                slice_qty = self._round_to_lot_size(remaining, lot_size)
            else:
                slice_qty = base_slice_size

            if slice_qty >= min_slice_size and slice_qty <= remaining:
                slices.append(slice_qty)
                remaining -= slice_qty
            elif remaining > 0:
                # Handle case where remaining is less than min slice
                if remaining >= lot_size:
                    slices.append(self._round_to_lot_size(remaining, lot_size))
                    remaining = 0
                break

        # Handle any remaining quantity (add to last slice)
        if remaining > 0 and slices:
            slices[-1] += remaining

        num_slices = len(slices)

        logger.info(
            f"TWAP execution: {total_quantity} {order.symbol} in {num_slices} slices, "
            f"lot_size={lot_size}, interval={self._slice_interval}s, "
            f"sizes={slices}"
        )

        # Execute slices (#E9 - lot size aware)
        for i, current_size in enumerate(slices):
            if pending.status == "cancelled":
                break

            # Create slice order
            slice_order = OrderEvent(
                source_agent=self.name,
                decision_id=order.decision_id,
                validation_id=order.validation_id,
                symbol=order.symbol,
                side=order.side,
                quantity=current_size,
                order_type=OrderType.MARKET,
                algo="TWAP_SLICE",
            )

            # Check rate limits for each slice (anti-HFT)
            if not self._check_rate_limits():
                logger.warning(f"Rate limit hit during TWAP, pausing slice {i+1}")
                await asyncio.sleep(1.0)  # Wait 1 second before retrying
                if not self._check_rate_limits():
                    logger.error(f"TWAP slice {i+1} cancelled due to rate limits")
                    continue

            # Get arrival price for slippage tracking (#E4)
            arrival_price = self._last_prices.get(order.symbol)

            broker_id = await self._broker.place_order(slice_order)

            if broker_id:
                # Register slice for fill tracking (#E4, #E5)
                is_buy = order.side == OrderSide.BUY
                pending.register_slice(broker_id, current_size, arrival_price, is_buy=is_buy)
                self._record_order_timestamp()  # Record for rate limiting
                logger.info(f"TWAP slice {i+1}/{num_slices}: {current_size} shares (arrival={arrival_price})")
            else:
                logger.error(f"TWAP slice {i+1} failed")

            # Wait before next slice (except for last one)
            if i < num_slices - 1:
                await asyncio.sleep(self._slice_interval)

        pending.status = "completed" if len(pending.slices) == num_slices else "partial"

        # Log fill quality summary (#E4)
        self._log_fill_quality(pending)

    # =========================================================================
    # DYNAMIC VWAP PARTICIPATION (#E10)
    # =========================================================================

    def update_market_volume(self, symbol: str, volume: int) -> None:
        """
        Update market volume observation for participation tracking (#E10).

        Called by orchestrator when market data volume updates are received.

        Args:
            symbol: Instrument symbol
            volume: Current market volume (typically cumulative or tick volume)
        """
        now = datetime.now(timezone.utc)

        if symbol not in self._market_volume_history:
            self._market_volume_history[symbol] = []

        self._market_volume_history[symbol].append((now, volume))

        # Trim old data beyond window
        cutoff = now - timedelta(minutes=self._volume_window_minutes * 2)
        self._market_volume_history[symbol] = [
            (t, v) for t, v in self._market_volume_history[symbol]
            if t > cutoff
        ]

    def _record_our_volume(self, symbol: str, filled_quantity: int) -> None:
        """
        Record our filled volume for participation tracking (#E10).

        Called when orders are filled to track our market participation.

        Args:
            symbol: Instrument symbol
            filled_quantity: Quantity we filled
        """
        now = datetime.now(timezone.utc)

        if symbol not in self._our_volume_history:
            self._our_volume_history[symbol] = []

        self._our_volume_history[symbol].append((now, filled_quantity))

        # Trim old data
        cutoff = now - timedelta(minutes=self._volume_window_minutes * 2)
        self._our_volume_history[symbol] = [
            (t, v) for t, v in self._our_volume_history[symbol]
            if t > cutoff
        ]

    def _get_current_participation_rate(self, symbol: str) -> float | None:
        """
        Calculate current participation rate (#E10).

        Returns:
            Current participation rate (our volume / market volume), or None if insufficient data
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=self._volume_window_minutes)

        # Get market volume in window
        market_volumes = self._market_volume_history.get(symbol, [])
        recent_market = [(t, v) for t, v in market_volumes if t > window_start]

        if len(recent_market) < 2:
            return None

        # Calculate market volume change (assuming cumulative volume)
        market_vol_change = recent_market[-1][1] - recent_market[0][1]

        if market_vol_change <= 0:
            return None

        # Get our volume in window
        our_volumes = self._our_volume_history.get(symbol, [])
        recent_ours = [(t, v) for t, v in our_volumes if t > window_start]
        our_vol = sum(v for _, v in recent_ours)

        if market_vol_change > 0:
            return our_vol / market_vol_change

        return None

    def _get_adjusted_participation_rate(self, symbol: str) -> float:
        """
        Get dynamically adjusted participation rate (#E10).

        Adjusts participation based on:
        - Current market volume vs expected
        - Our recent participation rate vs target
        - Market conditions (volatility, spread)

        Returns:
            Adjusted participation rate (bounded by min/max)
        """
        current_rate = self._get_current_participation_rate(symbol)
        target_rate = self._vwap_target_participation

        if current_rate is None:
            # No data yet, use target rate
            return target_rate

        # Calculate adjustment needed
        rate_diff = target_rate - current_rate

        # Apply smoothed adjustment
        adjustment = rate_diff * self._participation_adjustment_factor

        # Calculate new rate
        new_rate = current_rate + adjustment

        # Bound by min/max
        bounded_rate = max(self._vwap_min_participation,
                         min(self._vwap_max_participation, new_rate))

        if abs(bounded_rate - target_rate) > 0.02:
            logger.info(
                f"VWAP participation rate adjusted for {symbol}: "
                f"current={current_rate:.1%} -> target={target_rate:.1%} -> "
                f"adjusted={bounded_rate:.1%}"
            )

        return bounded_rate

    def _calculate_vwap_slice_size(
        self,
        symbol: str,
        total_remaining: int,
        expected_market_volume: float,
    ) -> int:
        """
        Calculate dynamic VWAP slice size based on current participation (#E10).

        Args:
            symbol: Instrument symbol
            total_remaining: Remaining quantity to fill
            expected_market_volume: Expected market volume for this period

        Returns:
            Slice size adjusted for participation rate
        """
        # Get dynamically adjusted participation rate
        participation_rate = self._get_adjusted_participation_rate(symbol)

        # Calculate slice based on expected volume and participation
        target_slice = int(expected_market_volume * participation_rate)

        # Don't exceed remaining quantity
        slice_size = min(target_slice, total_remaining)

        # Ensure minimum slice size
        lot_size = self._get_lot_size(symbol)
        min_slice = max(lot_size, 10)

        if slice_size < min_slice:
            slice_size = min_slice

        # Round to lot size
        slice_size = self._round_to_lot_size(slice_size, lot_size)

        return slice_size

    def get_participation_stats(self, symbol: str) -> dict[str, Any]:
        """
        Get participation rate statistics for monitoring (#E10).

        Args:
            symbol: Instrument symbol

        Returns:
            Dictionary with participation statistics
        """
        current_rate = self._get_current_participation_rate(symbol)
        adjusted_rate = self._get_adjusted_participation_rate(symbol)

        return {
            "symbol": symbol,
            "target_participation": self._vwap_target_participation,
            "current_participation": current_rate,
            "adjusted_participation": adjusted_rate,
            "min_participation": self._vwap_min_participation,
            "max_participation": self._vwap_max_participation,
            "volume_window_minutes": self._volume_window_minutes,
            "market_observations": len(self._market_volume_history.get(symbol, [])),
            "our_fills": len(self._our_volume_history.get(symbol, [])),
        }

    def _get_et_time(self) -> tuple[float, float, float]:
        """
        Get current time in Eastern Time for VWAP calculations.

        Returns:
            Tuple of (current_hour_decimal, hours_into_session, hours_remaining)
        """
        now = datetime.now(timezone.utc)

        # Convert UTC to Eastern Time
        # ET is UTC-5 (EST) or UTC-4 (EDT during daylight saving)
        # Approximate: use -5 hours (EST) as baseline
        # For production, use pytz or zoneinfo for proper DST handling
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            now_et = now.astimezone(et_tz)
        except ImportError:
            # Fallback: approximate EST (UTC-5)
            now_et = now.replace(tzinfo=None) - timedelta(hours=5)

        current_hour = now_et.hour + now_et.minute / 60.0

        # US equity market hours: 9:30 AM - 4:00 PM ET
        market_open_hour = 9.5  # 9:30 AM
        market_close_hour = 16.0  # 4:00 PM

        hours_into_session = max(0, current_hour - market_open_hour)
        hours_remaining = max(0.5, market_close_hour - current_hour)

        return current_hour, hours_into_session, hours_remaining

    async def _execute_vwap(self, pending: PendingOrder) -> None:
        """
        Execute order using Volume-Weighted Average Price algorithm.

        VWAP execution distributes order slices proportionally to expected
        intraday volume. This helps minimize market impact by trading
        heavier during high-volume periods.

        Uses historical volume profile to determine slice sizes.
        """
        order = pending.order_event
        symbol = order.symbol
        total_quantity = pending.remaining_quantity

        # Transition state
        pending.transition_state(OrderState.PENDING, "Starting VWAP execution")

        # Get volume profile for this symbol
        volume_profile = self._get_volume_profile(symbol)

        if not volume_profile or all(v == 0 for v in volume_profile):
            logger.warning(f"No volume profile for {symbol}, falling back to TWAP")
            await self._execute_twap(pending)
            return

        # Get current time in Eastern Time (proper timezone handling)
        current_hour, hours_into_session, hours_remaining = self._get_et_time()

        # Get remaining volume profile
        start_bucket = int(hours_into_session)
        remaining_profile = volume_profile[start_bucket:]

        if not remaining_profile or sum(remaining_profile) == 0:
            logger.warning(f"No remaining volume profile for {symbol}, using TWAP")
            await self._execute_twap(pending)
            return

        # Normalize remaining profile
        total_remaining_volume = sum(remaining_profile)
        normalized_profile = [v / total_remaining_volume for v in remaining_profile]

        # Calculate slice sizes based on volume profile
        slices = []
        for i, pct in enumerate(normalized_profile):
            slice_qty = int(total_quantity * pct)
            if slice_qty >= 10:  # Minimum slice size
                slices.append(slice_qty)

        # Handle remainder
        allocated = sum(slices)
        if allocated < total_quantity and slices:
            slices[-1] += (total_quantity - allocated)

        if not slices:
            slices = [total_quantity]

        num_slices = len(slices)
        interval_minutes = (hours_remaining * 60) / num_slices if num_slices > 0 else 5

        logger.info(
            f"VWAP execution: {total_quantity} {symbol} in {num_slices} slices, "
            f"interval={interval_minutes:.1f}min"
        )

        # Execute slices
        for i, slice_qty in enumerate(slices):
            if pending.status == "cancelled":
                break

            # Create slice order
            slice_order = OrderEvent(
                source_agent=self.name,
                decision_id=order.decision_id,
                validation_id=order.validation_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=OrderType.MARKET,
                algo="VWAP_SLICE",
            )

            # Check rate limits for each slice (anti-HFT)
            if not self._check_rate_limits():
                logger.warning(f"Rate limit hit during VWAP, pausing slice {i+1}")
                await asyncio.sleep(1.0)
                if not self._check_rate_limits():
                    logger.error(f"VWAP slice {i+1} cancelled due to rate limits")
                    continue

            broker_id = await self._broker.place_order(slice_order)

            if broker_id:
                pending.slices.append(broker_id)
                self._record_order_timestamp()
                logger.info(
                    f"VWAP slice {i+1}/{num_slices}: {slice_qty} shares "
                    f"({slice_qty/total_quantity*100:.1f}% of order)"
                )
            else:
                logger.error(f"VWAP slice {i+1} failed")

            # Wait before next slice (except for last one)
            if i < num_slices - 1:
                await asyncio.sleep(interval_minutes * 60)

        pending.status = "completed" if len(pending.slices) == num_slices else "partial"

    def _get_volume_profile(self, symbol: str) -> list[float]:
        """
        Get historical volume profile for a symbol.

        Returns list of 7 values representing normalized hourly volume
        from 9:30 AM to 4:00 PM (0.5 hour buckets would be more precise).

        In production, this would fetch from historical data.
        """
        if symbol in self._volume_profiles:
            return self._volume_profiles[symbol]

        # Default U-shaped volume profile (typical equity pattern)
        # Higher volume at open and close, lower mid-day
        # Represents hours: 9:30-10:30, 10:30-11:30, 11:30-12:30, 12:30-1:30, 1:30-2:30, 2:30-3:30, 3:30-4:00
        default_profile = [
            0.20,  # 9:30-10:30 - High opening volume
            0.12,  # 10:30-11:30
            0.10,  # 11:30-12:30 - Lunch lull
            0.08,  # 12:30-1:30 - Lowest volume
            0.10,  # 1:30-2:30
            0.15,  # 2:30-3:30 - Increasing toward close
            0.25,  # 3:30-4:00 - High closing volume
        ]

        return default_profile

    def set_volume_profile(self, symbol: str, profile: list[float]) -> None:
        """
        Set historical volume profile for a symbol.

        Args:
            symbol: The symbol
            profile: List of relative volume weights for each time bucket
        """
        # Normalize the profile
        total = sum(profile)
        if total > 0:
            normalized = [v / total for v in profile]
            self._volume_profiles[symbol] = normalized
            logger.info(f"Volume profile set for {symbol}: {len(profile)} buckets")

    def _handle_fill(self, fill: FillEvent) -> None:
        """
        Handle fill notification from broker.

        Called by broker when an order is filled.
        Handles both direct orders (MARKET) and TWAP/VWAP slice orders.

        Thread-safe: uses lock to prevent race conditions when
        multiple fills arrive simultaneously.
        """
        with self._fill_lock:
            self._process_fill_internal(fill)

    def _process_fill_internal(self, fill: FillEvent) -> None:
        """
        Internal fill processing (called within lock).

        Separated for testability and clarity.
        """
        # Find matching pending order
        matched_order_id = None
        matched_pending = None
        match_type = None

        for order_id, pending in self._pending_orders.items():
            # Check direct broker_order_id match (MARKET orders)
            is_direct_match = pending.broker_order_id == fill.broker_order_id

            # Check TWAP/VWAP slice match (fill.broker_order_id in pending.slices)
            is_slice_match = (
                pending.slices is not None and
                fill.broker_order_id in pending.slices
            )

            if is_direct_match or is_slice_match:
                matched_order_id = order_id
                matched_pending = pending
                match_type = "direct" if is_direct_match else "slice"
                break

        if not matched_pending:
            logger.warning(f"Received fill for unknown order: {fill.broker_order_id}")
            return

        # Track slice-level fill if applicable (#E4)
        if match_type == "slice":
            matched_pending.add_slice_fill(fill.broker_order_id, fill.filled_quantity, fill.fill_price)

        # Update fill tracking
        prev_filled = matched_pending.filled_quantity
        matched_pending.filled_quantity += fill.filled_quantity
        matched_pending.remaining_quantity -= fill.filled_quantity

        # Record our volume for VWAP participation tracking (#E10)
        self._record_our_volume(fill.symbol, fill.filled_quantity)

        # Sanity check
        if matched_pending.remaining_quantity < 0:
            logger.warning(
                f"Fill overflow for {fill.symbol}: remaining={matched_pending.remaining_quantity}"
            )
            matched_pending.remaining_quantity = 0

        # Update average fill price (volume-weighted)
        if matched_pending.filled_quantity > 0:
            total_value = (matched_pending.avg_fill_price * prev_filled
                           + fill.fill_price * fill.filled_quantity)
            matched_pending.avg_fill_price = total_value / matched_pending.filled_quantity

        # Update status and state
        if matched_pending.remaining_quantity <= 0:
            matched_pending.status = "filled"
            matched_pending.transition_state(OrderState.FILLED, "All quantity filled")
        elif matched_pending.filled_quantity > 0 and matched_pending.state != OrderState.PARTIAL:
            matched_pending.transition_state(OrderState.PARTIAL, f"Partial fill: {matched_pending.filled_quantity}/{matched_pending.order_event.quantity}")

        # Record execution for best execution analysis
        if self._best_execution_analyzer is not None:
            try:
                self._best_execution_analyzer.record_execution(
                    symbol=fill.symbol,
                    side=fill.side.value,
                    quantity=fill.filled_quantity,
                    fill_price=fill.fill_price,
                    commission=fill.commission,
                )
            except Exception as e:
                logger.warning(f"Best execution recording failed: {e}")

        # Log trade for audit
        self._audit_logger.log_trade(
            agent_name=self.name,
            order_id=matched_order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.filled_quantity,
            price=fill.fill_price,
            commission=fill.commission,
            decision_id=matched_pending.decision_event.event_id,
        )

        algo_info = f" ({matched_pending.order_event.algo} {match_type})" if match_type == "slice" else ""
        logger.info(
            f"Fill{algo_info}: {fill.side.value} {fill.filled_quantity} {fill.symbol} "
            f"@ {fill.fill_price} (filled={matched_pending.filled_quantity}, "
            f"remaining={matched_pending.remaining_quantity}, "
            f"avg_price={matched_pending.avg_fill_price:.4f})"
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self._pending_orders:
            return False

        pending = self._pending_orders[order_id]

        if pending.broker_order_id:
            success = await self._broker.cancel_order(pending.broker_order_id)
            if success:
                pending.status = "cancelled"
                return True

        return False

    def get_pending_orders(self) -> list[dict]:
        """Get all pending orders for monitoring."""
        return [
            {
                "order_id": order_id,
                "symbol": pending.order_event.symbol,
                "side": pending.order_event.side.value,
                "quantity": pending.order_event.quantity,
                "filled": pending.filled_quantity,
                "remaining": pending.remaining_quantity,
                "status": pending.status,
                "algo": pending.order_event.algo,
            }
            for order_id, pending in self._pending_orders.items()
        ]

    def set_best_execution_analyzer(self, analyzer) -> None:
        """Set the best execution analyzer for RTS 27/28 compliance tracking."""
        self._best_execution_analyzer = analyzer

    def _check_rate_limits(self) -> bool:
        """
        Check rate limits before placing order (anti-HFT per CLAUDE.md).

        Returns True if order can proceed, False if rate limited.
        """
        now = datetime.now(timezone.utc)

        # Clean old timestamps (older than 1 minute)
        one_minute_ago = now - timedelta(minutes=1)
        self._order_timestamps = [t for t in self._order_timestamps if t > one_minute_ago]

        # Check orders per minute limit
        if len(self._order_timestamps) >= self._max_orders_per_minute:
            logger.warning(
                f"Rate limit: {len(self._order_timestamps)}/{self._max_orders_per_minute} "
                f"orders in last minute"
            )
            return False

        # Check minimum interval between orders
        if self._last_order_time is not None:
            elapsed_ms = (now - self._last_order_time).total_seconds() * 1000
            if elapsed_ms < self._min_order_interval_ms:
                logger.warning(
                    f"Order interval too short: {elapsed_ms:.0f}ms < {self._min_order_interval_ms}ms minimum"
                )
                return False

        return True

    def _record_order_timestamp(self) -> None:
        """Record timestamp after successful order placement."""
        now = datetime.now(timezone.utc)
        self._order_timestamps.append(now)
        self._last_order_time = now

    # =========================================================================
    # STOP ORDER MANAGEMENT (#E8)
    # =========================================================================

    async def start_stop_order_monitor(self) -> None:
        """
        Start the background task that monitors and triggers stop orders.

        This should be called when the execution agent starts.
        """
        if self._stop_order_monitor_task is not None:
            logger.warning("Stop order monitor already running")
            return

        self._stop_order_monitor_task = asyncio.create_task(
            self._stop_order_monitor_loop()
        )
        logger.info("Stop order monitor started")

    async def stop_stop_order_monitor(self) -> None:
        """Stop the stop order monitor task."""
        if self._stop_order_monitor_task is not None:
            self._stop_order_monitor_task.cancel()
            try:
                await self._stop_order_monitor_task
            except asyncio.CancelledError:
                pass
            self._stop_order_monitor_task = None
            logger.info("Stop order monitor stopped")

    async def _stop_order_monitor_loop(self) -> None:
        """
        Background loop that checks if any stop orders should be triggered.

        Runs continuously with configurable interval (default 100ms).
        """
        while True:
            try:
                await self._check_and_trigger_stop_orders()
                await asyncio.sleep(self._stop_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stop order monitor: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _check_and_trigger_stop_orders(self) -> None:
        """
        Check all pending stop orders and trigger those that meet conditions.

        Stop order trigger logic:
        - BUY STOP: triggers when last_price >= stop_price
        - SELL STOP: triggers when last_price <= stop_price
        """
        if self._kill_switch_active:
            return

        # Get list of stop orders to check (copy to avoid modification during iteration)
        stop_order_ids = list(self._stop_orders.keys())

        for order_id in stop_order_ids:
            pending = self._stop_orders.get(order_id)
            if pending is None:
                continue

            order = pending.order_event
            symbol = order.symbol
            stop_price = order.stop_price

            if stop_price is None:
                logger.warning(f"Stop order {order_id} has no stop price, removing")
                del self._stop_orders[order_id]
                continue

            # Get last known price
            last_price = self._last_prices.get(symbol)
            if last_price is None:
                continue  # No price data yet

            # Check trigger condition
            triggered = False
            if order.side == OrderSide.BUY:
                # Buy stop triggers when price rises to/above stop price
                triggered = last_price >= stop_price
            elif order.side == OrderSide.SELL:
                # Sell stop triggers when price falls to/below stop price
                triggered = last_price <= stop_price

            if triggered:
                await self._trigger_stop_order(pending, last_price)

    async def _trigger_stop_order(self, pending: PendingOrder, trigger_price: float) -> None:
        """
        Trigger a stop order by converting it to a market or limit order.

        Args:
            pending: The stop order to trigger
            trigger_price: The price that triggered the stop
        """
        order = pending.order_event
        order_id = order.event_id

        logger.info(
            f"STOP ORDER TRIGGERED: {order.side.value} {order.quantity} {order.symbol} "
            f"@ stop={order.stop_price}, trigger_price={trigger_price}"
        )

        # Remove from stop orders
        if order_id in self._stop_orders:
            del self._stop_orders[order_id]

        # Check kill switch
        if self._kill_switch_active:
            logger.warning(f"Kill switch active, not executing triggered stop order {order_id}")
            pending.transition_state(OrderState.CANCELLED, "Kill switch active during trigger")
            return

        # Determine order type after trigger
        # STOP -> MARKET, STOP_LIMIT -> LIMIT
        if order.order_type == OrderType.STOP:
            # Convert to market order
            pending.transition_state(OrderState.PENDING, f"Stop triggered at {trigger_price}")
            await self._execute_market(pending)

        elif order.order_type == OrderType.STOP_LIMIT:
            # Convert to limit order at the limit price
            pending.transition_state(OrderState.PENDING, f"Stop-limit triggered at {trigger_price}")
            await self._execute_limit_order(pending)

        else:
            logger.warning(f"Unexpected order type for stop order: {order.order_type}")
            await self._execute_market(pending)

        # Audit log the trigger
        self._audit_logger.log_event(order)

    async def _execute_limit_order(self, pending: PendingOrder) -> None:
        """
        Execute a limit order (used for triggered stop-limit orders).
        """
        order = pending.order_event

        if order.limit_price is None:
            logger.warning(f"Stop-limit order {order.event_id} has no limit price, using market")
            await self._execute_market(pending)
            return

        # Place limit order with broker
        broker_id = await self._broker.place_order(order)

        if broker_id:
            pending.broker_order_id = broker_id
            pending.transition_state(OrderState.SUBMITTED, f"Limit order submitted (id={broker_id})")
            self._record_order_timestamp()
            logger.info(
                f"Limit order submitted: {order.side.value} {order.quantity} "
                f"{order.symbol} @ {order.limit_price} (broker_id={broker_id})"
            )
        else:
            pending.transition_state(OrderState.FAILED, "Broker submission failed")
            logger.error(f"Failed to submit limit order for {order.symbol}")

    def register_stop_order(self, pending: PendingOrder) -> None:
        """
        Register a stop order for monitoring.

        Called when a stop order is created but not yet triggered.
        """
        order = pending.order_event
        if order.stop_price is None:
            logger.warning(f"Cannot register stop order without stop price: {order.event_id}")
            return

        self._stop_orders[order.event_id] = pending
        logger.info(
            f"Stop order registered: {order.side.value} {order.quantity} {order.symbol} "
            f"@ stop={order.stop_price}"
        )

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update the last known price for a symbol.

        Called by market data feed to enable stop order triggering.
        """
        if price > 0:
            self._last_prices[symbol] = price

    def get_stop_orders(self) -> list[dict]:
        """Get all pending stop orders for monitoring."""
        return [
            {
                "order_id": order_id,
                "symbol": pending.order_event.symbol,
                "side": pending.order_event.side.value,
                "quantity": pending.order_event.quantity,
                "stop_price": pending.order_event.stop_price,
                "limit_price": pending.order_event.limit_price,
                "order_type": pending.order_event.order_type.value,
                "last_price": self._last_prices.get(pending.order_event.symbol),
            }
            for order_id, pending in self._stop_orders.items()
        ]

    def _log_fill_quality(self, pending: PendingOrder) -> None:
        """
        Log fill quality metrics for TWAP/VWAP orders (#E4, #E5).

        Calculates:
        - Overall fill rate
        - Average slippage per slice
        - Fill time distribution
        - Price improvement detection (#E5)
        """
        if not pending.slice_fills:
            return

        total_slices = len(pending.slice_fills)
        completed_slices = sum(1 for sf in pending.slice_fills.values() if sf.is_complete)
        partial_slices = sum(1 for sf in pending.slice_fills.values() if 0 < sf.fill_rate < 1)

        # Calculate average slippage
        slippages = [sf.slippage_bps for sf in pending.slice_fills.values() if sf.slippage_bps is not None]
        avg_slippage = sum(slippages) / len(slippages) if slippages else None

        # Calculate price improvement stats (#E5)
        improvements = [sf.price_improvement_bps for sf in pending.slice_fills.values() if sf.price_improvement_bps is not None]
        avg_improvement = sum(improvements) / len(improvements) if improvements else None
        slices_with_improvement = sum(1 for sf in pending.slice_fills.values() if sf.has_price_improvement)
        total_improvement_value = sum(
            sf.filled_quantity * (sf.price_improvement_bps / 10000.0 * (sf.arrival_price or 0))
            for sf in pending.slice_fills.values()
            if sf.has_price_improvement and sf.arrival_price
        )

        # Fill rate by slice
        fill_rates = [sf.fill_rate for sf in pending.slice_fills.values()]
        avg_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0

        # Build log message
        slippage_str = f"avg_slippage={avg_slippage:.2f}bps" if avg_slippage is not None else "avg_slippage=N/A"
        improvement_str = f"avg_improvement={avg_improvement:.2f}bps" if avg_improvement is not None else ""

        logger.info(
            f"TWAP/VWAP Fill Quality for {pending.order_event.symbol}: "
            f"slices={completed_slices}/{total_slices} complete, {partial_slices} partial, "
            f"avg_fill_rate={avg_fill_rate*100:.1f}%, {slippage_str}"
        )

        # Log price improvement summary (#E5)
        if slices_with_improvement > 0:
            logger.info(
                f"  Price Improvement (#E5): {slices_with_improvement}/{total_slices} slices improved, "
                f"avg={avg_improvement:.2f}bps, total_value=${total_improvement_value:.2f}"
            )

        # Log individual slice performance if significant slippage or improvement
        for broker_id, sf in pending.slice_fills.items():
            if sf.slippage_bps is not None and abs(sf.slippage_bps) > 10:  # > 10 bps
                improvement_note = " (IMPROVED)" if sf.has_price_improvement else ""
                logger.warning(
                    f"  Slice {sf.slice_index+1}: slippage={sf.slippage_bps:.2f}bps{improvement_note} "
                    f"(arrival={sf.arrival_price}, fill={sf.avg_fill_price})"
                )
            elif sf.has_price_improvement and sf.price_improvement_bps and sf.price_improvement_bps > 5:
                logger.info(
                    f"  Slice {sf.slice_index+1}: PRICE IMPROVEMENT +{sf.price_improvement_bps:.2f}bps "
                    f"(arrival={sf.arrival_price}, fill={sf.avg_fill_price})"
                )

    def get_fill_quality_report(self, order_id: str) -> dict | None:
        """
        Get detailed fill quality report for an order (#E4, #E5).

        Args:
            order_id: Order ID

        Returns:
            Fill quality metrics dict, or None if order not found
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return None

        slices_data = []
        for broker_id, sf in pending.slice_fills.items():
            slices_data.append({
                "slice_index": sf.slice_index,
                "broker_order_id": broker_id,
                "target_quantity": sf.target_quantity,
                "filled_quantity": sf.filled_quantity,
                "fill_rate": sf.fill_rate,
                "avg_fill_price": sf.avg_fill_price,
                "arrival_price": sf.arrival_price,
                "slippage_bps": sf.slippage_bps,
                "price_improvement_bps": sf.price_improvement_bps,  # #E5
                "has_price_improvement": sf.has_price_improvement,  # #E5
                "is_complete": sf.is_complete,
                "fill_count": len(sf.fills),
            })

        # Calculate aggregate price improvement metrics (#E5)
        improvements = [sf.price_improvement_bps for sf in pending.slice_fills.values() if sf.price_improvement_bps is not None]
        avg_improvement = sum(improvements) / len(improvements) if improvements else None
        slices_with_improvement = sum(1 for sf in pending.slice_fills.values() if sf.has_price_improvement)

        return {
            "order_id": order_id,
            "symbol": pending.order_event.symbol,
            "algo": pending.order_event.algo,
            "total_quantity": pending.order_event.quantity,
            "filled_quantity": pending.filled_quantity,
            "remaining_quantity": pending.remaining_quantity,
            "avg_fill_price": pending.avg_fill_price,
            "overall_fill_rate": pending.filled_quantity / pending.order_event.quantity if pending.order_event.quantity > 0 else 0,
            "slice_count": len(pending.slice_fills),
            "slices": slices_data,
            # Price improvement summary (#E5)
            "price_improvement": {
                "avg_improvement_bps": avg_improvement,
                "slices_with_improvement": slices_with_improvement,
                "total_slices": len(pending.slice_fills),
                "improvement_rate": slices_with_improvement / len(pending.slice_fills) if pending.slice_fills else 0,
            },
        }

    def get_status(self) -> dict:
        """Get execution agent status for monitoring."""
        return {
            "pending_orders": len(self._pending_orders),
            "stop_orders": len(self._stop_orders),
            "kill_switch_active": self._kill_switch_active,
            "orders_this_minute": len(self._order_timestamps),
            "max_orders_per_minute": self._max_orders_per_minute,
            "stop_monitor_running": self._stop_order_monitor_task is not None and not self._stop_order_monitor_task.done(),
            "symbols_with_prices": len(self._last_prices),
            # Fill quality summary (#E13)
            "fill_quality": self.get_aggregate_fill_metrics(),
            # Implementation shortfall summary (#E14)
            "implementation_shortfall": self.get_implementation_shortfall_summary(),
        }

    # =========================================================================
    # ORDER PERSISTENCE ACROSS RESTARTS (#E12)
    # =========================================================================

    def persist_orders_to_file(self, filepath: str = "data/pending_orders.json") -> int:
        """
        Persist pending orders to file for recovery after restart (#E12).

        Saves all non-terminal pending orders to JSON for later recovery.

        Args:
            filepath: Path to save orders

        Returns:
            Number of orders persisted
        """
        import json
        import os

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        orders_to_persist = []
        for order_id, pending in self._pending_orders.items():
            if pending.is_terminal():
                continue

            order = pending.order_event
            orders_to_persist.append({
                "order_id": order_id,
                "broker_order_id": pending.broker_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "algo": order.algo,
                "filled_quantity": pending.filled_quantity,
                "remaining_quantity": pending.remaining_quantity,
                "avg_fill_price": pending.avg_fill_price,
                "status": pending.status,
                "state": pending.state.value,
                "arrival_price": pending.arrival_price,
                "created_at": pending.created_at.isoformat() if pending.created_at else None,
                "decision_id": order.decision_id,
                "slices": pending.slices,
            })

        # Also persist stop orders
        for order_id, pending in self._stop_orders.items():
            order = pending.order_event
            orders_to_persist.append({
                "order_id": order_id,
                "broker_order_id": pending.broker_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "algo": order.algo,
                "filled_quantity": 0,
                "remaining_quantity": order.quantity,
                "avg_fill_price": 0.0,
                "status": "stop_pending",
                "state": OrderState.CREATED.value,
                "arrival_price": None,
                "created_at": pending.created_at.isoformat() if pending.created_at else None,
                "decision_id": order.decision_id,
                "slices": [],
                "is_stop_order": True,
            })

        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "orders": orders_to_persist,
            }, f, indent=2)

        logger.info(f"Persisted {len(orders_to_persist)} orders to {filepath}")
        return len(orders_to_persist)

    async def recover_orders_from_file(self, filepath: str = "data/pending_orders.json") -> int:
        """
        Recover pending orders from file after restart (#E12).

        Loads persisted orders and attempts to reconcile with broker state.

        Args:
            filepath: Path to load orders from

        Returns:
            Number of orders recovered
        """
        import json
        import os

        if not os.path.exists(filepath):
            logger.info(f"No persisted orders file found at {filepath}")
            return 0

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            orders_data = data.get("orders", [])
            recovered_count = 0

            for order_data in orders_data:
                try:
                    # Recreate order event
                    order = OrderEvent(
                        source_agent=self.name,
                        decision_id=order_data.get("decision_id", ""),
                        validation_id="",
                        symbol=order_data["symbol"],
                        side=OrderSide(order_data["side"]),
                        quantity=order_data["quantity"],
                        order_type=OrderType(order_data["order_type"]),
                        limit_price=order_data.get("limit_price"),
                        stop_price=order_data.get("stop_price"),
                        algo=order_data.get("algo", "MARKET"),
                    )

                    # Try to get state from broker if we have broker_order_id
                    broker_order_id = order_data.get("broker_order_id")
                    broker_state = None

                    if broker_order_id and self._broker:
                        try:
                            broker_state = await self._broker.get_order_status(broker_order_id)
                        except Exception as e:
                            logger.warning(f"Could not get broker state for {broker_order_id}: {e}")

                    # If broker says order is still active, recreate tracking
                    if broker_state and broker_state.get("status") not in ["Filled", "Cancelled", "Error"]:
                        # Create a minimal DecisionEvent for tracking
                        from core.events import DecisionEvent
                        decision = DecisionEvent(
                            source_agent="recovery",
                            symbol=order_data["symbol"],
                            action=OrderSide(order_data["side"]),
                            quantity=order_data["quantity"],
                            order_type=OrderType(order_data["order_type"]),
                            limit_price=order_data.get("limit_price"),
                            stop_price=order_data.get("stop_price"),
                        )

                        pending = PendingOrder(
                            order_event=order,
                            decision_event=decision,
                            broker_order_id=broker_order_id,
                            filled_quantity=broker_state.get("filled", order_data.get("filled_quantity", 0)),
                            remaining_quantity=broker_state.get("remaining", order_data.get("remaining_quantity", order_data["quantity"])),
                            avg_fill_price=order_data.get("avg_fill_price", 0.0),
                            status=broker_state.get("status", order_data.get("status", "pending")),
                            arrival_price=order_data.get("arrival_price"),
                        )

                        if order_data.get("is_stop_order"):
                            self._stop_orders[order_data["order_id"]] = pending
                        else:
                            self._pending_orders[order_data["order_id"]] = pending

                        recovered_count += 1
                        logger.info(f"Recovered order {order_data['order_id']}: {order_data['symbol']} {order_data['side']} {order_data['quantity']}")

                    else:
                        logger.info(f"Skipped order {order_data['order_id']} - broker status: {broker_state}")

                except Exception as e:
                    logger.error(f"Failed to recover order: {e}")

            # Backup and remove the persistence file after recovery
            if recovered_count > 0:
                backup_path = filepath + f".recovered.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                os.rename(filepath, backup_path)
                logger.info(f"Moved persistence file to {backup_path}")

            logger.info(f"Recovered {recovered_count}/{len(orders_data)} orders from persistence file")
            return recovered_count

        except Exception as e:
            logger.error(f"Failed to recover orders from {filepath}: {e}")
            return 0

    # =========================================================================
    # FILL QUALITY METRICS (#E13)
    # =========================================================================

    def get_aggregate_fill_metrics(self) -> dict:
        """
        Get aggregate fill quality metrics across all orders (#E13).

        Returns summary statistics for execution quality monitoring.
        """
        total_orders = 0
        total_filled = 0
        total_slippage_bps = []
        total_improvement_bps = []
        orders_with_improvement = 0
        orders_with_slippage = 0

        for pending in self._pending_orders.values():
            if pending.filled_quantity == 0:
                continue

            total_orders += 1
            total_filled += pending.filled_quantity

            # Aggregate slice metrics
            for sf in pending.slice_fills.values():
                if sf.slippage_bps is not None:
                    total_slippage_bps.append(sf.slippage_bps)
                    if sf.slippage_bps > 0:
                        orders_with_slippage += 1
                if sf.price_improvement_bps is not None and sf.price_improvement_bps > 0:
                    total_improvement_bps.append(sf.price_improvement_bps)
                    orders_with_improvement += 1

        avg_slippage = sum(total_slippage_bps) / len(total_slippage_bps) if total_slippage_bps else 0
        avg_improvement = sum(total_improvement_bps) / len(total_improvement_bps) if total_improvement_bps else 0

        return {
            "total_orders": total_orders,
            "total_shares_filled": total_filled,
            "slippage": {
                "avg_bps": avg_slippage,
                "max_bps": max(total_slippage_bps) if total_slippage_bps else 0,
                "min_bps": min(total_slippage_bps) if total_slippage_bps else 0,
                "orders_with_slippage": orders_with_slippage,
            },
            "price_improvement": {
                "avg_bps": avg_improvement,
                "max_bps": max(total_improvement_bps) if total_improvement_bps else 0,
                "orders_with_improvement": orders_with_improvement,
                "improvement_rate": orders_with_improvement / max(len(total_slippage_bps), 1),
            },
        }

    # =========================================================================
    # IMPLEMENTATION SHORTFALL TRACKING (#E14)
    # =========================================================================

    def calculate_implementation_shortfall(self, order_id: str) -> dict | None:
        """
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
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return None

        order = pending.order_event
        decision = pending.decision_event

        # Get decision price (benchmark)
        decision_price = decision.limit_price or pending.arrival_price
        if decision_price is None or decision_price <= 0:
            return {"error": "No decision price available"}

        # Calculate total shares and value
        total_shares = order.quantity
        filled_shares = pending.filled_quantity
        unfilled_shares = pending.remaining_quantity
        avg_fill_price = pending.avg_fill_price if pending.avg_fill_price > 0 else decision_price

        is_buy = order.side == OrderSide.BUY
        sign = 1 if is_buy else -1

        # Paper portfolio cost (what we would have paid at decision price)
        paper_cost = decision_price * total_shares

        # Actual execution cost
        actual_cost = avg_fill_price * filled_shares

        # Current price for opportunity cost (unfilled portion)
        current_price = self._last_prices.get(order.symbol, decision_price)
        opportunity_cost_value = unfilled_shares * current_price

        # Component breakdown

        # 1. Delay cost: Price movement from decision to first fill
        # (Approximated by arrival_price vs decision_price)
        arrival_price = pending.arrival_price or decision_price
        delay_cost_per_share = sign * (arrival_price - decision_price)
        total_delay_cost = delay_cost_per_share * filled_shares

        # 2. Trading impact: Movement during execution
        # (Difference between avg_fill and arrival)
        impact_per_share = sign * (avg_fill_price - arrival_price)
        total_impact_cost = impact_per_share * filled_shares

        # 3. Opportunity cost: Price movement on unfilled portion
        # (Current price vs decision price on unfilled shares)
        if unfilled_shares > 0:
            opp_cost_per_share = sign * (current_price - decision_price)
            total_opportunity_cost = opp_cost_per_share * unfilled_shares
        else:
            total_opportunity_cost = 0

        # Total implementation shortfall
        total_shortfall = total_delay_cost + total_impact_cost + total_opportunity_cost

        # As percentage of paper cost
        shortfall_bps = (total_shortfall / paper_cost * 10000) if paper_cost > 0 else 0

        return {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "total_shares": total_shares,
            "filled_shares": filled_shares,
            "unfilled_shares": unfilled_shares,
            "benchmark": {
                "decision_price": decision_price,
                "arrival_price": arrival_price,
                "avg_fill_price": avg_fill_price,
                "current_price": current_price,
            },
            "costs": {
                "paper_cost": paper_cost,
                "actual_cost": actual_cost,
                "delay_cost": total_delay_cost,
                "impact_cost": total_impact_cost,
                "opportunity_cost": total_opportunity_cost,
                "total_shortfall": total_shortfall,
            },
            "metrics": {
                "shortfall_bps": shortfall_bps,
                "delay_bps": (delay_cost_per_share / decision_price * 10000) if decision_price > 0 else 0,
                "impact_bps": (impact_per_share / arrival_price * 10000) if arrival_price > 0 else 0,
                "fill_rate": filled_shares / total_shares if total_shares > 0 else 0,
            },
        }

    def get_implementation_shortfall_summary(self) -> dict:
        """
        Get aggregate implementation shortfall metrics (#E14).

        Returns summary across all completed orders.
        """
        shortfalls = []
        delay_costs = []
        impact_costs = []
        opp_costs = []

        for order_id, pending in self._pending_orders.items():
            if pending.filled_quantity == 0:
                continue

            is_data = self.calculate_implementation_shortfall(order_id)
            if is_data and "error" not in is_data:
                shortfalls.append(is_data["metrics"]["shortfall_bps"])
                delay_costs.append(is_data["metrics"]["delay_bps"])
                impact_costs.append(is_data["metrics"]["impact_bps"])
                if is_data["unfilled_shares"] > 0:
                    opp_costs.append(is_data["costs"]["opportunity_cost"])

        if not shortfalls:
            return {"orders_analyzed": 0}

        return {
            "orders_analyzed": len(shortfalls),
            "avg_shortfall_bps": sum(shortfalls) / len(shortfalls),
            "total_shortfall_bps": sum(shortfalls),
            "max_shortfall_bps": max(shortfalls),
            "min_shortfall_bps": min(shortfalls),
            "components": {
                "avg_delay_bps": sum(delay_costs) / len(delay_costs) if delay_costs else 0,
                "avg_impact_bps": sum(impact_costs) / len(impact_costs) if impact_costs else 0,
                "total_opportunity_cost": sum(opp_costs),
            },
        }

    # =========================================================================
    # ORDER BOOK DEPTH ANALYSIS (#E15)
    # =========================================================================

    def update_order_book(
        self,
        symbol: str,
        bids: list[tuple[float, int, int]],  # (price, size, num_orders)
        asks: list[tuple[float, int, int]]
    ) -> None:
        """
        Update order book snapshot for a symbol (#E15).

        Called by market data handler when order book updates received.

        Args:
            symbol: Instrument symbol
            bids: List of (price, size, num_orders) tuples, best first
            asks: List of (price, size, num_orders) tuples, best first
        """
        bid_levels = [OrderBookLevel(p, s, n) for p, s, n in bids]
        ask_levels = [OrderBookLevel(p, s, n) for p, s, n in asks]

        self._order_books[symbol] = OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bids=bid_levels,
            asks=ask_levels,
        )

    def analyze_order_book(self, symbol: str) -> dict | None:
        """
        Analyze current order book depth (#E15).

        Returns metrics useful for execution decisions.

        Args:
            symbol: Instrument symbol

        Returns:
            Order book analysis or None if no data
        """
        book = self._order_books.get(symbol)
        if book is None:
            return None

        return {
            "symbol": symbol,
            "timestamp": book.timestamp.isoformat(),
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "mid_price": book.mid_price,
            "spread_bps": book.spread_bps,
            "depth": {
                "bid_depth_5": book.total_bid_depth(5),
                "ask_depth_5": book.total_ask_depth(5),
                "bid_depth_10": book.total_bid_depth(10),
                "ask_depth_10": book.total_ask_depth(10),
                "total_bid_depth": book.total_bid_depth(),
                "total_ask_depth": book.total_ask_depth(),
            },
            "imbalance": {
                "imbalance_5": book.depth_imbalance(5),
                "imbalance_10": book.depth_imbalance(10),
            },
            "bid_levels": len(book.bids),
            "ask_levels": len(book.asks),
        }

    def estimate_execution_cost(
        self,
        symbol: str,
        side: str,
        quantity: int
    ) -> dict | None:
        """
        Estimate execution cost using order book (#E15).

        Calculates expected VWAP and slippage for given order size.

        Args:
            symbol: Instrument symbol
            side: 'buy' or 'sell'
            quantity: Order quantity

        Returns:
            Execution cost estimate or None if no book data
        """
        book = self._order_books.get(symbol)
        if book is None:
            return None

        vwap, filled = book.vwap_to_size(side, quantity)

        if filled == 0 or book.mid_price is None:
            return None

        # Calculate slippage vs mid price
        if side == 'buy':
            slippage_bps = (vwap - book.mid_price) / book.mid_price * 10000
        else:
            slippage_bps = (book.mid_price - vwap) / book.mid_price * 10000

        # Check if order is marketable (can fill at current prices)
        is_marketable = filled >= quantity

        return {
            "symbol": symbol,
            "side": side,
            "target_quantity": quantity,
            "expected_fill": filled,
            "expected_vwap": vwap,
            "mid_price": book.mid_price,
            "slippage_bps": slippage_bps,
            "slippage_cost": abs(vwap - book.mid_price) * filled,
            "is_fully_marketable": is_marketable,
            "fill_rate": filled / quantity if quantity > 0 else 0,
            "spread_bps": book.spread_bps,
        }

    # =========================================================================
    # PASSIVE/AGGRESSIVE FILL CATEGORIZATION (#E20)
    # =========================================================================

    def categorize_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: int,
        fill_side: str
    ) -> FillCategory:
        """
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
        """
        pending = self._pending_orders.get(order_id)
        symbol = pending.order_event.symbol if pending else None

        # Get order book for context
        book = self._order_books.get(symbol) if symbol else None

        if book is None or book.best_bid is None or book.best_ask is None:
            # Cannot categorize without order book
            category = FillCategory(
                is_aggressive=False,
                category="unknown",
                price_vs_arrival_bps=0.0,
                price_vs_spread_position=0.5,
            )
        else:
            mid = book.mid_price
            spread = book.best_ask - book.best_bid

            # Determine where in the spread the fill occurred
            if fill_side == 'buy':
                # For buys: at ask = aggressive, at bid = passive
                if fill_price >= book.best_ask:
                    is_aggressive = True
                    category_name = "aggressive"
                    spread_position = 1.0  # At or above ask
                elif fill_price <= book.best_bid:
                    is_aggressive = False
                    category_name = "passive"
                    spread_position = 0.0  # At or below bid
                else:
                    # Between bid and ask
                    spread_position = (fill_price - book.best_bid) / spread if spread > 0 else 0.5
                    is_aggressive = spread_position > 0.6  # More than 60% toward ask
                    category_name = "midpoint" if 0.4 <= spread_position <= 0.6 else ("aggressive" if is_aggressive else "passive")

                # Price vs arrival (positive = paid more)
                arrival = pending.arrival_price if pending and pending.arrival_price else mid
                price_vs_arrival = (fill_price - arrival) / arrival * 10000 if arrival else 0

            else:  # sell
                # For sells: at bid = aggressive, at ask = passive
                if fill_price <= book.best_bid:
                    is_aggressive = True
                    category_name = "aggressive"
                    spread_position = 0.0
                elif fill_price >= book.best_ask:
                    is_aggressive = False
                    category_name = "passive"
                    spread_position = 1.0
                else:
                    spread_position = (fill_price - book.best_bid) / spread if spread > 0 else 0.5
                    is_aggressive = spread_position < 0.4
                    category_name = "midpoint" if 0.4 <= spread_position <= 0.6 else ("aggressive" if is_aggressive else "passive")

                # Price vs arrival (positive = received less)
                arrival = pending.arrival_price if pending and pending.arrival_price else mid
                price_vs_arrival = (arrival - fill_price) / arrival * 10000 if arrival else 0

            category = FillCategory(
                is_aggressive=is_aggressive,
                category=category_name,
                price_vs_arrival_bps=price_vs_arrival,
                price_vs_spread_position=spread_position,
            )

        # Track for reporting
        if order_id not in self._fill_categories:
            self._fill_categories[order_id] = []
        self._fill_categories[order_id].append(category)

        return category

    def get_fill_categorization_summary(self) -> dict:
        """
        Get summary of fill categorizations (#E20).

        Returns breakdown of aggressive vs passive fills.
        """
        total_fills = 0
        aggressive_count = 0
        passive_count = 0
        midpoint_count = 0
        unknown_count = 0
        total_arrival_slippage = []

        for categories in self._fill_categories.values():
            for cat in categories:
                total_fills += 1
                if cat.category == "aggressive":
                    aggressive_count += 1
                elif cat.category == "passive":
                    passive_count += 1
                elif cat.category == "midpoint":
                    midpoint_count += 1
                else:
                    unknown_count += 1

                if cat.price_vs_arrival_bps != 0:
                    total_arrival_slippage.append(cat.price_vs_arrival_bps)

        return {
            "total_fills": total_fills,
            "aggressive": {
                "count": aggressive_count,
                "pct": aggressive_count / total_fills * 100 if total_fills > 0 else 0,
            },
            "passive": {
                "count": passive_count,
                "pct": passive_count / total_fills * 100 if total_fills > 0 else 0,
            },
            "midpoint": {
                "count": midpoint_count,
                "pct": midpoint_count / total_fills * 100 if total_fills > 0 else 0,
            },
            "unknown": unknown_count,
            "avg_arrival_slippage_bps": sum(total_arrival_slippage) / len(total_arrival_slippage) if total_arrival_slippage else 0,
        }

    # =========================================================================
    # MARKET IMPACT MODEL (#E21)
    # =========================================================================

    def estimate_market_impact(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None = None,
        adv: float | None = None,
        volatility: float | None = None
    ) -> MarketImpactEstimate:
        """
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
        """
        # Get price
        if price is None:
            price = self._last_prices.get(symbol, 100.0)

        # Default parameters if not provided
        if adv is None:
            adv = 1_000_000  # Default 1M shares/day
        if volatility is None:
            volatility = 0.02  # Default 2% daily vol

        # Model parameters
        eta = self._market_impact_params["eta"]
        gamma = self._market_impact_params["gamma"]
        alpha = self._market_impact_params["alpha"]

        # Participation rate
        participation = quantity / adv if adv > 0 else 0.01

        # Square-root model for temporary impact
        # Higher participation = more impact
        temporary_impact = eta * volatility * (participation ** alpha)

        # Permanent impact (information leakage)
        # Permanent impact is typically lower than temporary
        permanent_impact = gamma * volatility * (participation ** alpha)

        # Convert to basis points
        temporary_impact_bps = temporary_impact * 10000
        permanent_impact_bps = permanent_impact * 10000
        total_impact_bps = temporary_impact_bps + permanent_impact_bps

        # Estimate cost in currency
        estimated_cost = quantity * price * (total_impact_bps / 10000)

        return MarketImpactEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            temporary_impact_bps=temporary_impact_bps,
            permanent_impact_bps=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            estimated_cost=estimated_cost,
            model_used="square_root_almgren_chriss",
        )

    def configure_market_impact_model(
        self,
        eta: float | None = None,
        gamma: float | None = None,
        alpha: float | None = None
    ) -> None:
        """
        Configure market impact model parameters (#E21).

        Args:
            eta: Temporary impact coefficient (default 0.1)
            gamma: Permanent impact coefficient (default 0.1)
            alpha: Square-root power (default 0.5)
        """
        if eta is not None:
            self._market_impact_params["eta"] = eta
        if gamma is not None:
            self._market_impact_params["gamma"] = gamma
        if alpha is not None:
            self._market_impact_params["alpha"] = alpha

        logger.info(
            f"Market impact model configured: eta={self._market_impact_params['eta']}, "
            f"gamma={self._market_impact_params['gamma']}, alpha={self._market_impact_params['alpha']}"
        )

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
        }

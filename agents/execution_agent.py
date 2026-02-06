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
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any

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
        # Max slippage: 10 bps for liquid futures (was 50, too permissive)
        # Research: typical futures slippage is 2-5 bps for MES/MNQ
        self._max_slippage_bps = config.parameters.get("max_slippage_bps", 10)

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

        # Order timeout handling (#E24)
        self._order_timeout_seconds = config.parameters.get("order_timeout_seconds", 300)  # 5 min default
        self._algo_timeout_seconds = config.parameters.get("algo_timeout_seconds", 3600)  # 1 hour for TWAP/VWAP
        self._timeout_check_interval = config.parameters.get("timeout_check_interval_seconds", 10)
        self._order_timeout_monitor_task: asyncio.Task | None = None
        self._timed_out_orders: deque[str] = deque(maxlen=500)  # Track timed out order IDs

        # Order acknowledgment timeout handling (P2)
        self._ack_timeout_seconds = config.parameters.get("ack_timeout_seconds", 5.0)  # 5 second default for broker ack
        self._pending_ack_orders: dict[str, datetime] = {}  # order_id -> submission_time
        self._ack_timeout_alerts: deque[dict] = deque(maxlen=500)  # Track ack timeout events

        # Partial fill percentage tracking (P2, bounded)
        self._partial_fill_history: deque[dict] = deque(maxlen=1000)  # Historical partial fill data

        # QUICK WIN #9: ATR cache for spread validation
        self._atr_cache: dict[str, float] = {}  # symbol -> current ATR
        self._partial_fill_threshold_pct = config.parameters.get("partial_fill_alert_threshold_pct", 50.0)  # Alert if <50% filled
        self._partial_fill_time_window_seconds = config.parameters.get("partial_fill_time_window_seconds", 120)  # 2 min window

        # Execution quality scoring (P2)
        self._execution_quality_scores: dict[str, dict] = {}  # order_id -> quality metrics
        self._quality_score_weights = config.parameters.get("quality_score_weights", {
            "slippage": 0.30,  # Weight for slippage component
            "fill_rate": 0.25,  # Weight for fill rate component
            "speed": 0.20,  # Weight for execution speed
            "price_improvement": 0.15,  # Weight for price improvement
            "market_impact": 0.10,  # Weight for market impact
        })

        # Kill switch state
        self._kill_switch_active = False
        self._kill_switch_reason = ""

        # Zombie order management (Phase C improvement)
        self._zombie_order_timeout_seconds = config.parameters.get("zombie_order_timeout_seconds", 30.0)
        self._zombie_check_interval_seconds = config.parameters.get("zombie_check_interval_seconds", 5.0)
        self._zombie_order_monitor_task: asyncio.Task | None = None
        self._zombie_orders_cancelled: int = 0
        self._zombie_order_history: deque[dict] = deque(maxlen=100)  # Track cancelled zombie orders

        # VWAP execution state
        # Historical volume profiles by symbol (hourly buckets, 0-23 representing market hours)
        self._volume_profiles: dict[str, list[float]] = {}  # symbol -> normalized hourly volumes
        self._vwap_participation_rate = config.parameters.get("vwap_participation_rate", 0.1)  # 10% of volume

        # Dynamic VWAP participation rate tracking (#E10)
        # Industry standard: max participation 12.3% to avoid market impact detection
        # Research: SEC market microstructure studies show >15% participation triggers algorithmic detection
        self._vwap_target_participation = config.parameters.get("vwap_target_participation", 0.10)  # Target 10%
        self._vwap_min_participation = config.parameters.get("vwap_min_participation", 0.05)  # Min 5%
        self._vwap_max_participation = config.parameters.get("vwap_max_participation", 0.12)  # Max 12% (was 25%)
        self._market_volume_history: dict[str, list[tuple[datetime, int]]] = {}  # symbol -> [(time, volume)]
        self._our_volume_history: dict[str, list[tuple[datetime, int]]] = {}  # symbol -> [(time, filled)]
        self._volume_window_minutes = config.parameters.get("volume_window_minutes", 5)  # Rolling window
        self._participation_adjustment_factor = config.parameters.get("participation_adjustment_factor", 0.5)  # Smoothing

        # Contract specs for tick size validation
        self._contract_specs = ContractSpecsManager()

        # Optional components
        self._best_execution_analyzer = None
        self._stop_loss_manager = None  # StopLossManager for automatic stop placement

        # Automatic stop-loss configuration
        self._auto_stop_loss_enabled = config.parameters.get("auto_stop_loss_enabled", True)
        self._stop_loss_atr_multiplier = config.parameters.get("stop_loss_atr_multiplier", 2.5)
        self._default_stop_loss_pct = config.parameters.get("default_stop_loss_pct", 5.0)

        # Order book tracking (#E15)
        self._order_books: dict[str, OrderBookSnapshot] = {}

        # Fill categorization tracking (#E20)
        self._fill_categories: dict[str, list[FillCategory]] = {}  # order_id -> categories

        # Market impact model parameters (#E21)
        # Almgren-Chriss model: eta should be 5-10x gamma
        # eta = temporary impact (price reverts after trade)
        # gamma = permanent impact (price shift persists)
        # Research: Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"
        self._market_impact_params = {
            "eta": 0.5,  # Temporary impact coefficient (was 0.1, now 5x gamma)
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

        # Only process events from ComplianceAgent (final validation step)
        # Skip events from RiskAgent (those go to ComplianceAgent first)
        if event.source_agent != "ComplianceAgent":
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

        # P1-11: Pre-trade checks to prevent slippage and execution issues
        pre_trade_result = await self._pre_trade_checks(decision, quantity)
        if not pre_trade_result["passed"]:
            logger.warning(
                f"Pre-trade checks failed for {decision.symbol}: {pre_trade_result['reason']}"
            )
            # Still allow execution but log warning - don't block legitimate trades
            # Critical issues (broker disconnected) are already handled above

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
        futures_symbols = ['ES', 'NQ', 'YM', 'RTY', 'CL', 'GC', 'SI', 'ZB', 'ZN', 'ZC', 'ZW', 'ZS',
                           'MES', 'MNQ', 'MYM', 'M2K', 'MCL', 'MGC']
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
        if quantity == 0:
            return 0

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

        # EXE-002: Market hours awareness - adjust slice interval if needed
        # Get time remaining in trading session
        current_hour, hours_into_session, hours_remaining = self._get_et_time()
        total_execution_time_seconds = self._slice_interval * (num_slices - 1)  # Time for all waits
        total_execution_time_hours = total_execution_time_seconds / 3600.0

        adjusted_slice_interval = self._slice_interval
        if total_execution_time_hours > hours_remaining * 0.95:  # Would exceed 95% of remaining time
            # Adjust interval to complete within remaining market hours
            available_seconds = hours_remaining * 3600 * 0.9  # Use 90% of remaining time
            if num_slices > 1:
                adjusted_slice_interval = max(30, available_seconds / (num_slices - 1))  # Min 30s interval
                logger.warning(
                    f"TWAP adjusting interval for market hours: {self._slice_interval}s -> "
                    f"{adjusted_slice_interval:.0f}s (hours_remaining={hours_remaining:.2f})"
                )
            else:
                adjusted_slice_interval = 0  # Single slice, no wait needed

        logger.info(
            f"TWAP execution: {total_quantity} {order.symbol} in {num_slices} slices, "
            f"lot_size={lot_size}, interval={adjusted_slice_interval:.0f}s, "
            f"sizes={slices}"
        )

        # Execute slices (#E9 - lot size aware)
        for i, current_size in enumerate(slices):
            # CRITICAL FIX: Check both cancelled status AND kill-switch at start of each iteration
            # This ensures TWAP halts immediately when kill-switch triggers
            if pending.status == "cancelled" or self._kill_switch_active:
                logger.warning(f"TWAP halted at slice {i+1}/{num_slices}: kill_switch={self._kill_switch_active}")
                pending.status = "cancelled"
                pending.transition_state(OrderState.CANCELLED, "TWAP halted by kill-switch")
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
            # EXE-002: Use adjusted interval for market hours awareness
            if i < num_slices - 1:
                # Split sleep into chunks to check kill-switch more frequently
                sleep_chunks = max(1, int(adjusted_slice_interval / 5))  # Check every ~5 seconds
                chunk_duration = adjusted_slice_interval / sleep_chunks
                for _ in range(sleep_chunks):
                    await asyncio.sleep(chunk_duration)
                    # CRITICAL: Check kill-switch during sleep
                    if self._kill_switch_active or pending.status == "cancelled":
                        logger.warning(f"TWAP interrupted during sleep: kill_switch={self._kill_switch_active}")
                        break
                # Check again after full sleep completes
                if self._kill_switch_active or pending.status == "cancelled":
                    break

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

        # Normalize remaining profile - guard against division by zero
        total_remaining_volume = sum(remaining_profile)
        if total_remaining_volume == 0:
            logger.warning(f"No remaining volume profile for {symbol}, using TWAP")
            await self._execute_twap(pending)
            return
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

        # EXE-001: EOD acceleration logic to prevent stranded quantity at close
        # If we're in the last 30 minutes and intervals are too short, accelerate
        if interval_minutes < 1.0 and hours_remaining < 0.5:  # Last 30 min
            interval_minutes = max(0.5, (hours_remaining * 60 * 0.9) / num_slices)
            logger.warning(
                f"VWAP accelerating for EOD: interval reduced to {interval_minutes:.1f}min, "
                f"hours_remaining={hours_remaining:.2f}, slices={num_slices}"
            )
        elif hours_remaining < 0.25 and num_slices > 1:  # Last 15 min - more aggressive
            # Reduce number of slices to ensure completion
            new_num_slices = max(1, min(num_slices, int(hours_remaining * 60 / 0.5)))  # Min 30s intervals
            if new_num_slices < num_slices:
                # Redistribute slices
                new_slices = []
                total_allocated = 0
                slice_size = total_quantity // new_num_slices
                for i in range(new_num_slices):
                    if i == new_num_slices - 1:
                        new_slices.append(total_quantity - total_allocated)
                    else:
                        new_slices.append(slice_size)
                        total_allocated += slice_size
                slices = new_slices
                num_slices = new_num_slices
                interval_minutes = (hours_remaining * 60 * 0.9) / num_slices
                logger.warning(
                    f"VWAP EOD critical: reduced to {num_slices} slices, "
                    f"interval={interval_minutes:.1f}min to complete before close"
                )

        logger.info(
            f"VWAP execution: {total_quantity} {symbol} in {num_slices} slices, "
            f"interval={interval_minutes:.1f}min"
        )

        # Execute slices
        for i, slice_qty in enumerate(slices):
            # CRITICAL FIX: Check both cancelled status AND kill-switch at start of each iteration
            # This ensures VWAP halts immediately when kill-switch triggers
            if pending.status == "cancelled" or self._kill_switch_active:
                logger.warning(f"VWAP halted at slice {i+1}/{num_slices}: kill_switch={self._kill_switch_active}")
                pending.status = "cancelled"
                pending.transition_state(OrderState.CANCELLED, "VWAP halted by kill-switch")
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

            # Get arrival price for slippage tracking (#E4)
            arrival_price = self._last_prices.get(order.symbol)

            broker_id = await self._broker.place_order(slice_order)

            if broker_id:
                # Register slice for fill tracking (#E4, #E5) - same as TWAP
                is_buy = order.side == OrderSide.BUY
                pending.register_slice(broker_id, slice_qty, arrival_price, is_buy=is_buy)
                self._record_order_timestamp()
                logger.info(
                    f"VWAP slice {i+1}/{num_slices}: {slice_qty} shares "
                    f"({slice_qty/total_quantity*100:.1f}% of order, arrival={arrival_price})"
                )
            else:
                logger.error(f"VWAP slice {i+1} failed")

            # Wait before next slice (except for last one)
            if i < num_slices - 1:
                sleep_seconds = interval_minutes * 60
                # Split sleep into chunks to check kill-switch more frequently (~5 second chunks)
                sleep_chunks = max(1, int(sleep_seconds / 5))
                chunk_duration = sleep_seconds / sleep_chunks
                for _ in range(sleep_chunks):
                    await asyncio.sleep(chunk_duration)
                    # CRITICAL: Check kill-switch during sleep
                    if self._kill_switch_active or pending.status == "cancelled":
                        logger.warning(f"VWAP interrupted during sleep: kill_switch={self._kill_switch_active}")
                        break
                # Check again after full sleep completes
                if self._kill_switch_active or pending.status == "cancelled":
                    break

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

        # CRITICAL FIX: Check for overflow BEFORE any tracking to prevent inconsistent state
        # Calculate what the new totals would be
        projected_filled = matched_pending.filled_quantity + fill.filled_quantity
        projected_remaining = matched_pending.remaining_quantity - fill.filled_quantity

        if projected_remaining < 0:
            # Fill overflow detected - reject this fill entirely
            logger.error(
                f"FILL OVERFLOW REJECTED for {fill.symbol}: "
                f"would overfill by {abs(projected_remaining)}. "
                f"Order qty={matched_pending.order_event.quantity}, "
                f"already filled={matched_pending.filled_quantity}, "
                f"this fill={fill.filled_quantity}. "
                "Possible duplicate fill or race condition."
            )
            # Don't process this fill - it would cause inconsistent state
            return

        # Track slice-level fill if applicable (#E4) - ONLY after overflow check passed
        if match_type == "slice":
            matched_pending.add_slice_fill(fill.broker_order_id, fill.filled_quantity, fill.fill_price)

        # Update fill tracking - guaranteed to be valid after overflow check
        prev_filled = matched_pending.filled_quantity
        matched_pending.filled_quantity = projected_filled
        matched_pending.remaining_quantity = projected_remaining

        # Record our volume for VWAP participation tracking (#E10)
        self._record_our_volume(fill.symbol, fill.filled_quantity)

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
                # Non-critical operation - execution continues even if analytics fail
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

        # Register position with stop-loss manager for automatic stop placement
        if matched_pending.remaining_quantity <= 0:
            self._register_position_with_stop_manager(fill, matched_pending)

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

    def set_stop_loss_manager(self, manager) -> None:
        """
        Set the stop-loss manager for automatic stop placement.

        When set, the execution agent will automatically register positions
        with the stop-loss manager after fills, enabling:
        - ATR-based stop-loss placement
        - Trailing stop updates
        - Automatic position closing on stop hit
        """
        self._stop_loss_manager = manager
        logger.info("StopLossManager attached to ExecutionAgent")

    def _register_position_with_stop_manager(
        self,
        fill: "FillEvent",
        pending: PendingOrder,
    ) -> None:
        """
        Register a filled position with the stop-loss manager.

        Called after a fill is processed to set up automatic stop-loss.

        Args:
            fill: The fill event
            pending: The pending order that was filled
        """
        if not self._stop_loss_manager or not self._auto_stop_loss_enabled:
            return

        # Only register on complete fills for new positions
        if pending.remaining_quantity > 0:
            return  # Wait for complete fill

        # Determine if this is a long or short position
        is_long = fill.side == OrderSide.BUY

        # Get ATR if available from the decision context
        atr = None
        if pending.decision_event:
            # Check if stop_loss was provided in the decision
            stop_price = pending.decision_event.stop_price
            if stop_price:
                # Calculate implied ATR from provided stop
                if is_long:
                    implied_stop_distance = pending.avg_fill_price - stop_price
                else:
                    implied_stop_distance = stop_price - pending.avg_fill_price
                # Reverse engineer ATR assuming default multiplier
                atr = implied_stop_distance / self._stop_loss_atr_multiplier

        # Register with stop-loss manager
        self._stop_loss_manager.register_position(
            symbol=fill.symbol,
            entry_price=pending.avg_fill_price,
            quantity=pending.filled_quantity,
            is_long=is_long,
            atr=atr,
            stop_loss_price=pending.decision_event.stop_price if pending.decision_event else None,
        )

        logger.info(
            f"Position registered with StopLossManager: "
            f"{'LONG' if is_long else 'SHORT'} {pending.filled_quantity} {fill.symbol} "
            f"@ {pending.avg_fill_price:.2f}"
        )

    async def _pre_trade_checks(
        self, decision: "DecisionEvent", quantity: int
    ) -> dict[str, Any]:
        """
        P1-11: Pre-trade checks to prevent excessive slippage and execution issues.

        Checks:
        1. Market data freshness - data must be recent
        2. Spread sanity - spread should not be excessive
        3. Price sanity - price should be reasonable vs recent history
        4. Liquidity check - order size vs available volume

        Returns:
            dict with 'passed' bool and 'reason' if failed, plus 'warnings' list
        """
        result = {"passed": True, "reason": "", "warnings": []}
        symbol = decision.symbol

        # 1. Check market data freshness
        if hasattr(self._broker, 'check_data_staleness'):
            staleness = self._broker.check_data_staleness(symbol)
            if staleness and staleness.is_critical:
                result["warnings"].append(
                    f"Market data is stale ({staleness.age_seconds:.1f}s old)"
                )
                # Don't fail - just warn. Data staleness is sometimes normal.

        # 2. Get current market data for checks
        try:
            bid, ask = await self._get_current_bid_ask(symbol)
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                result["warnings"].append("Could not get valid bid/ask prices")
                return result

            mid_price = (bid + ask) / 2
            spread = ask - bid
            spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0

            # 3. Spread sanity check
            MAX_SPREAD_PCT = 2.0  # 2% max spread - very wide for most liquid instruments
            if spread_pct > MAX_SPREAD_PCT:
                result["warnings"].append(
                    f"Wide spread: {spread_pct:.2f}% (>{MAX_SPREAD_PCT}%) - "
                    f"consider using limit order"
                )

            # QUICK WIN #9: Dynamic spread check vs ATR
            # Spread should not exceed configured % of ATR for the asset
            if hasattr(self, '_config') and self._config:
                exec_config = self._config.parameters if hasattr(self._config, 'parameters') else self._config
                spread_limits = exec_config.get("spread_limits", {})
                max_spread_atr_pct = exec_config.get("max_spread_atr_pct", 0.10)

                # Check asset-specific spread limit first
                asset_spread_limit = spread_limits.get(symbol)
                if asset_spread_limit is not None:
                    if spread_pct > asset_spread_limit * 100:  # Convert to percentage
                        result["passed"] = False
                        result["reason"] = (
                            f"Spread {spread_pct:.3f}% exceeds limit "
                            f"{asset_spread_limit*100:.2f}% for {symbol}"
                        )
                        return result

                # Check spread vs ATR if we have ATR data
                if hasattr(self, '_atr_cache') and symbol in self._atr_cache:
                    atr = self._atr_cache[symbol]
                    if atr > 0 and mid_price > 0:
                        spread_as_atr_pct = spread / atr
                        if spread_as_atr_pct > max_spread_atr_pct:
                            result["warnings"].append(
                                f"Spread is {spread_as_atr_pct:.1%} of ATR "
                                f"(limit: {max_spread_atr_pct:.0%})"
                            )

            # 4. Order size vs typical volume (if available)
            # Large orders in illiquid markets can cause significant slippage
            if hasattr(self, '_avg_daily_volume') and symbol in self._avg_daily_volume:
                adv = self._avg_daily_volume[symbol]
                order_pct_of_adv = (quantity / adv) * 100 if adv > 0 else 0

                if order_pct_of_adv > 5.0:  # Order is >5% of daily volume
                    result["warnings"].append(
                        f"Large order: {order_pct_of_adv:.1f}% of ADV - "
                        f"consider TWAP/VWAP algo"
                    )

            # 5. Price sanity vs limit (if limit order)
            if decision.limit_price and decision.limit_price > 0:
                limit_vs_mid = abs(decision.limit_price - mid_price) / mid_price * 100
                if limit_vs_mid > 5.0:  # Limit price >5% from mid
                    result["warnings"].append(
                        f"Limit price {decision.limit_price:.2f} is {limit_vs_mid:.1f}% "
                        f"from mid price {mid_price:.2f}"
                    )

        except Exception as e:
            result["warnings"].append(f"Pre-trade check error: {e}")

        # Log all warnings
        for warning in result["warnings"]:
            logger.warning(f"Pre-trade [{symbol}]: {warning}")

        return result

    async def _get_current_bid_ask(self, symbol: str) -> tuple[float | None, float | None]:
        """Get current bid/ask from broker or cache."""
        if hasattr(self._broker, 'get_bid_ask'):
            return await self._broker.get_bid_ask(symbol)

        # Fallback to last known prices from market data
        if hasattr(self, '_last_bid') and hasattr(self, '_last_ask'):
            return self._last_bid.get(symbol), self._last_ask.get(symbol)

        return None, None

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

    # =========================================================================
    # ZOMBIE ORDER MANAGEMENT (Phase C)
    # =========================================================================

    async def start_zombie_order_monitor(self) -> None:
        """
        Start the background task that cancels zombie (unfilled) orders.

        A zombie order is an order that has been pending for longer than
        the zombie_order_timeout_seconds threshold without being filled.
        """
        if self._zombie_order_monitor_task is not None:
            logger.warning("Zombie order monitor already running")
            return

        self._zombie_order_monitor_task = asyncio.create_task(
            self._zombie_order_monitor_loop()
        )
        logger.info(
            f"Zombie order monitor started (timeout: {self._zombie_order_timeout_seconds}s, "
            f"check interval: {self._zombie_check_interval_seconds}s)"
        )

    async def stop_zombie_order_monitor(self) -> None:
        """Stop the zombie order monitor task."""
        if self._zombie_order_monitor_task is not None:
            self._zombie_order_monitor_task.cancel()
            try:
                await self._zombie_order_monitor_task
            except asyncio.CancelledError:
                pass
            self._zombie_order_monitor_task = None
            logger.info("Zombie order monitor stopped")

    async def _zombie_order_monitor_loop(self) -> None:
        """
        Background loop that checks for and cancels zombie orders.

        Runs continuously with configurable interval (default 5s).
        """
        while True:
            try:
                await self._check_and_cancel_zombie_orders()
                await asyncio.sleep(self._zombie_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in zombie order monitor: {e}")
                await asyncio.sleep(5.0)  # Back off on error

    async def _check_and_cancel_zombie_orders(self) -> int:
        """
        Check all pending orders and cancel those that have exceeded timeout.

        Returns:
            Number of zombie orders cancelled
        """
        if self._kill_switch_active:
            return 0

        now = datetime.now(timezone.utc)
        cancelled_count = 0

        # Get list of pending orders to check (copy to avoid modification during iteration)
        pending_order_ids = list(self._pending_orders.keys())

        for order_id in pending_order_ids:
            pending = self._pending_orders.get(order_id)
            if pending is None:
                continue

            # Skip orders in terminal states
            if pending.is_terminal():
                continue

            # Check if order is in a zombieable state (not yet filled/acknowledged)
            if pending.state not in (
                OrderState.PENDING,
                OrderState.SUBMITTED,
                OrderState.ACKNOWLEDGED,
                OrderState.PARTIAL,
            ):
                continue

            # Calculate order age
            order_age = (now - pending.created_at).total_seconds()

            # Use algo timeout for TWAP/VWAP orders (they take longer by design)
            if pending.algo in ("TWAP", "VWAP"):
                timeout = self._algo_timeout_seconds
            else:
                timeout = self._zombie_order_timeout_seconds

            # Check if order has exceeded timeout
            if order_age > timeout:
                logger.warning(
                    f"Cancelling zombie order {order_id} after {order_age:.1f}s "
                    f"(timeout: {timeout}s, state: {pending.state.value}, algo: {pending.algo})"
                )

                try:
                    # Attempt to cancel via broker
                    cancelled = await self._cancel_order_via_broker(order_id)

                    if cancelled:
                        self._zombie_orders_cancelled += 1
                        cancelled_count += 1

                        # Record zombie order for analysis
                        self._zombie_order_history.append({
                            "order_id": order_id,
                            "symbol": pending.order_event.symbol,
                            "side": pending.order_event.side.value if pending.order_event.side else "unknown",
                            "quantity": pending.order_event.quantity,
                            "algo": pending.algo,
                            "age_seconds": order_age,
                            "cancelled_at": now.isoformat(),
                            "state_at_cancel": pending.state.value,
                        })

                        # Transition order state
                        pending.transition_state(OrderState.CANCELLED, "Zombie timeout")

                except Exception as e:
                    logger.error(f"Failed to cancel zombie order {order_id}: {e}")

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} zombie orders")

        return cancelled_count

    async def _cancel_order_via_broker(self, order_id: str) -> bool:
        """
        Cancel an order via the broker.

        Args:
            order_id: The order ID to cancel

        Returns:
            True if cancel was submitted successfully
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return False

        # Find broker order ID (may be stored differently)
        broker_order_id = pending.broker_order_id

        if broker_order_id is None:
            logger.warning(f"No broker order ID for {order_id}, cannot cancel")
            return False

        try:
            await self._broker.cancel_order(broker_order_id)
            return True
        except Exception as e:
            logger.error(f"Broker cancel failed for {broker_order_id}: {e}")
            return False

    def get_zombie_order_stats(self) -> dict:
        """Get statistics about zombie order management."""
        return {
            "zombie_orders_cancelled": self._zombie_orders_cancelled,
            "recent_zombies": list(self._zombie_order_history),
            "zombie_timeout_seconds": self._zombie_order_timeout_seconds,
            "algo_timeout_seconds": self._algo_timeout_seconds,
            "monitor_running": self._zombie_order_monitor_task is not None,
        }

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
                # Broad catch intentional - monitor must stay alive for system safety
                logger.exception(f"Error in stop order monitor: {e}")  # P0: preserve stack trace
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

    # =========================================================================
    # ORDER TIMEOUT HANDLING (#E24)
    # =========================================================================

    def start_order_timeout_monitor(self) -> None:
        """
        Start background task to monitor order timeouts (#E24).

        Cancels orders that exceed configured timeout thresholds.
        """
        if self._order_timeout_monitor_task is not None:
            logger.warning("Order timeout monitor already running")
            return

        self._order_timeout_monitor_task = asyncio.create_task(
            self._order_timeout_monitor_loop()
        )
        logger.info(
            f"Order timeout monitor started (simple={self._order_timeout_seconds}s, "
            f"algo={self._algo_timeout_seconds}s)"
        )

    async def stop_order_timeout_monitor(self) -> None:
        """Stop the order timeout monitor task."""
        if self._order_timeout_monitor_task is not None:
            self._order_timeout_monitor_task.cancel()
            try:
                await self._order_timeout_monitor_task
            except asyncio.CancelledError:
                pass
            self._order_timeout_monitor_task = None
            logger.info("Order timeout monitor stopped")

    async def _order_timeout_monitor_loop(self) -> None:
        """
        Background loop that checks for timed-out orders (#E24).

        Orders are cancelled if they exceed their timeout threshold:
        - Simple orders (MARKET/LIMIT): order_timeout_seconds (default 5 min)
        - Algorithmic orders (TWAP/VWAP): algo_timeout_seconds (default 1 hour)
        """
        while True:
            try:
                await self._check_and_cancel_timed_out_orders()
                await asyncio.sleep(self._timeout_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Broad catch intentional - monitor must stay alive for order management
                logger.exception(f"Error in order timeout monitor: {e}")  # P0: preserve stack trace
                await asyncio.sleep(5.0)  # Back off on error

    async def _check_and_cancel_timed_out_orders(self) -> None:
        """
        Check all pending orders for timeout and cancel expired ones (#E24).

        Timeout thresholds:
        - MARKET/LIMIT: _order_timeout_seconds
        - TWAP/VWAP: _algo_timeout_seconds
        """
        if self._kill_switch_active:
            return

        now = datetime.now(timezone.utc)
        orders_to_cancel = []

        for order_id, pending in self._pending_orders.items():
            order = pending.order_event

            # Skip already completed/cancelled orders
            if pending.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
                continue

            # Determine timeout based on order type
            if order.algo in ("TWAP", "VWAP"):
                timeout_seconds = self._algo_timeout_seconds
            else:
                timeout_seconds = self._order_timeout_seconds

            # Calculate age
            order_age = (now - pending.submitted_at).total_seconds()

            # Check for timeout
            if order_age > timeout_seconds:
                orders_to_cancel.append((order_id, order_age, timeout_seconds))

        # Cancel timed out orders
        for order_id, age, timeout in orders_to_cancel:
            await self._cancel_timed_out_order(order_id, age, timeout)

    async def _cancel_timed_out_order(
        self,
        order_id: str,
        age_seconds: float,
        timeout_seconds: float,
    ) -> None:
        """
        Cancel a single timed-out order (#E24).

        Args:
            order_id: Order to cancel
            age_seconds: How old the order is
            timeout_seconds: The timeout threshold it exceeded
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return

        order = pending.order_event

        # P1-14: Log partial fill info on timeout
        filled_qty = pending.filled_quantity
        unfilled_qty = pending.remaining_quantity
        fill_pct = (filled_qty / order.quantity * 100) if order.quantity > 0 else 0

        if filled_qty > 0 and unfilled_qty > 0:
            # Partial fill on timeout - this needs attention!
            logger.error(
                f"PARTIAL FILL TIMEOUT: {order.side.value} {order.symbol} - "
                f"filled {filled_qty}/{order.quantity} ({fill_pct:.1f}%), "
                f"UNFILLED {unfilled_qty} shares will NOT be executed! "
                f"(order_id={order_id}, age={age_seconds:.0f}s)"
            )
        else:
            logger.warning(
                f"ORDER TIMEOUT: {order.side.value} {order.quantity} {order.symbol} "
                f"(order_id={order_id}, age={age_seconds:.0f}s, timeout={timeout_seconds}s)"
            )

        # Track timed out orders
        self._timed_out_orders.append(order_id)

        # P1-14: Track partial fills separately for monitoring
        if filled_qty > 0 and unfilled_qty > 0:
            if not hasattr(self, '_partial_fill_timeouts'):
                self._partial_fill_timeouts = []
            self._partial_fill_timeouts.append({
                "order_id": order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "total_quantity": order.quantity,
                "filled_quantity": filled_qty,
                "unfilled_quantity": unfilled_qty,
                "avg_fill_price": pending.avg_fill_price,
                "timeout_at": datetime.now(timezone.utc).isoformat(),
            })

        # ERR-003: Attempt to cancel any open broker orders with proper error tracking
        cancellation_failures = []
        for broker_order_id in pending.active_broker_orders:
            try:
                self._broker.cancel_order(broker_order_id)
                logger.info(f"Cancelled broker order {broker_order_id} due to timeout")
            except Exception as e:
                # ERR-003: Track cancellation failures - these create "ghost orders"
                cancellation_failures.append({
                    "broker_order_id": broker_order_id,
                    "error": str(e),
                    "symbol": order.symbol,
                    "order_id": order_id,
                })
                logger.critical(
                    f"CRITICAL: Failed to cancel broker order {broker_order_id} for "
                    f"{order.symbol} (order_id={order_id}): {e}. "
                    f"GHOST ORDER may exist - manual intervention required!"
                )

        # ERR-003: Track failed cancellations for monitoring
        if cancellation_failures:
            if not hasattr(self, '_cancellation_failures'):
                self._cancellation_failures = []
            self._cancellation_failures.extend(cancellation_failures)

            # Mark order as UNCERTAIN since we don't know broker state
            pending.status = "uncertain"
            logger.critical(
                f"Order {order_id} marked as UNCERTAIN - {len(cancellation_failures)} "
                f"cancellation(s) failed. Broker orders may still be active!"
            )

        # Update state - use UNCERTAIN if cancellation failed
        final_state = OrderState.CANCELLED
        if cancellation_failures:
            # Note: OrderState.UNCERTAIN may not exist, keep CANCELLED but status field tracks it
            pass

        pending.transition_state(
            final_state,
            f"Timed out after {age_seconds:.0f} seconds (limit={timeout_seconds}s)"
        )

        # Publish state change event
        state_change = OrderStateChangeEvent(
            source_agent=self.agent_id,
            order_id=order_id,
            symbol=order.symbol,
            old_state=OrderState.WORKING,
            new_state=OrderState.CANCELLED,
            reason=f"Order timeout ({age_seconds:.0f}s)",
        )
        await self._event_bus.publish(state_change)

        # Audit log
        self._audit_logger.log_agent_event(
            agent_name=self.agent_id,
            event_type="order_timeout",
            details={
                "order_id": order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "filled_quantity": pending.filled_quantity,
                "age_seconds": age_seconds,
                "timeout_seconds": timeout_seconds,
                "algo": order.algo,
            },
        )

    def get_partial_fill_timeouts(self) -> list[dict]:
        """
        P1-14: Get list of orders that timed out with partial fills.

        These represent execution failures where the full order was not completed
        and remaining quantity was not executed.
        """
        if not hasattr(self, '_partial_fill_timeouts'):
            return []
        return list(self._partial_fill_timeouts)

    def get_timed_out_orders(self) -> list[str]:
        """Get list of order IDs that have been cancelled due to timeout."""
        return self._timed_out_orders.copy()

    def get_cancellation_failures(self) -> list[dict]:
        """
        ERR-003: Get list of broker orders that failed to cancel.

        These represent potential "ghost orders" that may still be active
        at the broker but are no longer tracked by the system.

        Returns:
            List of dicts with broker_order_id, error, symbol, order_id
        """
        if not hasattr(self, '_cancellation_failures'):
            return []
        return list(self._cancellation_failures)

    def get_uncertain_orders(self) -> list[str]:
        """
        ERR-003: Get list of orders in UNCERTAIN state.

        Orders become uncertain when cancellation fails and we don't know
        if broker orders are still active.

        Returns:
            List of order_ids in uncertain state
        """
        uncertain = []
        for order_id, pending in self._pending_orders.items():
            if pending.status == "uncertain":
                uncertain.append(order_id)
        return uncertain

    def get_order_age(self, order_id: str) -> float | None:
        """
        Get age of an order in seconds (#E24).

        Returns None if order not found.
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return None

        now = datetime.now(timezone.utc)
        return (now - pending.submitted_at).total_seconds()

    # =========================================================================
    # ORDER ACKNOWLEDGMENT TIMEOUT HANDLING (P2)
    # =========================================================================

    def register_pending_ack(self, order_id: str) -> None:
        """
        Register an order as waiting for broker acknowledgment (P2).

        Called when an order is submitted to track ack timeout.

        Args:
            order_id: Order ID waiting for acknowledgment
        """
        self._pending_ack_orders[order_id] = datetime.now(timezone.utc)
        logger.debug(f"Order {order_id} pending broker acknowledgment")

    def confirm_order_ack(self, order_id: str) -> float | None:
        """
        Confirm broker acknowledgment received for an order (P2).

        Args:
            order_id: Order ID that was acknowledged

        Returns:
            Ack latency in seconds, or None if not tracked
        """
        if order_id not in self._pending_ack_orders:
            return None

        submission_time = self._pending_ack_orders.pop(order_id)
        ack_latency = (datetime.now(timezone.utc) - submission_time).total_seconds()

        logger.debug(f"Order {order_id} acknowledged in {ack_latency:.3f}s")
        return ack_latency

    async def check_ack_timeouts(self) -> list[dict]:
        """
        Check for orders that have not received acknowledgment (P2).

        Returns:
            List of timed-out orders with details
        """
        now = datetime.now(timezone.utc)
        timed_out = []

        for order_id, submission_time in list(self._pending_ack_orders.items()):
            elapsed = (now - submission_time).total_seconds()

            if elapsed > self._ack_timeout_seconds:
                pending = self._pending_orders.get(order_id)
                symbol = pending.order_event.symbol if pending else "unknown"

                timeout_info = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "submission_time": submission_time.isoformat(),
                    "elapsed_seconds": elapsed,
                    "timeout_threshold": self._ack_timeout_seconds,
                }
                timed_out.append(timeout_info)
                self._ack_timeout_alerts.append(timeout_info)

                logger.warning(
                    f"ORDER ACK TIMEOUT: {order_id} ({symbol}) - "
                    f"no acknowledgment after {elapsed:.2f}s (threshold: {self._ack_timeout_seconds}s)"
                )

                # Remove from pending ack tracking
                del self._pending_ack_orders[order_id]

                # Optionally transition order state
                if pending and pending.state == OrderState.SUBMITTED:
                    pending.transition_state(
                        OrderState.PENDING,
                        f"Ack timeout after {elapsed:.2f}s - broker may not have received order"
                    )

        return timed_out

    def get_ack_timeout_alerts(self) -> list[dict]:
        """
        Get all acknowledgment timeout alerts (P2).

        Returns:
            List of ack timeout events
        """
        return list(self._ack_timeout_alerts)

    def get_pending_ack_count(self) -> int:
        """Get count of orders pending acknowledgment (P2)."""
        return len(self._pending_ack_orders)

    # =========================================================================
    # PARTIAL FILL PERCENTAGE TRACKING (P2)
    # =========================================================================

    def track_partial_fill(self, order_id: str) -> dict | None:
        """
        Track partial fill percentage for an order (P2).

        Calculates fill percentage and logs alerts if below threshold.

        Args:
            order_id: Order ID to track

        Returns:
            Partial fill metrics dict, or None if order not found
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return None

        order = pending.order_event
        total_qty = order.quantity
        filled_qty = pending.filled_quantity
        remaining_qty = pending.remaining_quantity

        fill_pct = (filled_qty / total_qty * 100) if total_qty > 0 else 0

        # Calculate time since order was created
        order_age = (datetime.now(timezone.utc) - pending.created_at).total_seconds()

        metrics = {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "total_quantity": total_qty,
            "filled_quantity": filled_qty,
            "remaining_quantity": remaining_qty,
            "fill_percentage": fill_pct,
            "order_age_seconds": order_age,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Check if fill rate is concerning
        if (order_age > self._partial_fill_time_window_seconds and
            fill_pct < self._partial_fill_threshold_pct and
            remaining_qty > 0):

            metrics["alert"] = True
            metrics["alert_reason"] = (
                f"Only {fill_pct:.1f}% filled after {order_age:.0f}s "
                f"(threshold: {self._partial_fill_threshold_pct}%)"
            )

            logger.warning(
                f"PARTIAL FILL ALERT: {order.side.value} {order.symbol} - "
                f"{fill_pct:.1f}% filled ({filled_qty}/{total_qty}) "
                f"after {order_age:.0f}s"
            )

            self._partial_fill_history.append(metrics)

        return metrics

    def get_partial_fill_summary(self) -> dict:
        """
        Get summary of partial fill tracking (P2).

        Returns:
            Summary statistics of partial fills
        """
        if not self._partial_fill_history:
            return {
                "total_alerts": 0,
                "avg_fill_pct_at_alert": None,
                "symbols_affected": [],
            }

        fill_pcts = [h["fill_percentage"] for h in self._partial_fill_history]
        symbols = list(set(h["symbol"] for h in self._partial_fill_history))

        return {
            "total_alerts": len(self._partial_fill_history),
            "avg_fill_pct_at_alert": sum(fill_pcts) / len(fill_pcts) if fill_pcts else None,
            "min_fill_pct": min(fill_pcts) if fill_pcts else None,
            "max_fill_pct": max(fill_pcts) if fill_pcts else None,
            "symbols_affected": symbols,
            "recent_alerts": self._partial_fill_history[-10:],  # Last 10 alerts
        }

    def get_current_fill_rates(self) -> list[dict]:
        """
        Get current fill rates for all active orders (P2).

        Returns:
            List of fill rate info for all pending orders
        """
        fill_rates = []

        for order_id, pending in self._pending_orders.items():
            if pending.is_terminal():
                continue

            order = pending.order_event
            total_qty = order.quantity
            filled_qty = pending.filled_quantity
            fill_pct = (filled_qty / total_qty * 100) if total_qty > 0 else 0
            order_age = (datetime.now(timezone.utc) - pending.created_at).total_seconds()

            fill_rates.append({
                "order_id": order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "fill_percentage": fill_pct,
                "filled_qty": filled_qty,
                "total_qty": total_qty,
                "order_age_seconds": order_age,
                "state": pending.state.value,
            })

        # Sort by fill percentage ascending (worst first)
        fill_rates.sort(key=lambda x: x["fill_percentage"])
        return fill_rates

    # =========================================================================
    # EXECUTION QUALITY SCORING (P2)
    # =========================================================================

    def calculate_execution_quality_score(self, order_id: str) -> dict | None:
        """
        Calculate comprehensive execution quality score for an order (P2).

        Components:
        - Slippage score: Based on actual vs expected price
        - Fill rate score: Percentage of order filled
        - Speed score: How quickly the order was filled
        - Price improvement score: Did we get better than expected?
        - Market impact score: Did our order move the market?

        Args:
            order_id: Order ID to score

        Returns:
            Quality score metrics dict with overall score 0-100, or None if not found
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return None

        order = pending.order_event
        scores = {}

        # 1. Slippage Score (0-100, 100 = no slippage)
        slippage_score = 100.0
        if pending.arrival_price and pending.arrival_price > 0 and pending.avg_fill_price > 0:
            slippage_bps = abs(pending.avg_fill_price - pending.arrival_price) / pending.arrival_price * 10000
            # Score decreases as slippage increases: 100 at 0 bps, 0 at 100 bps
            slippage_score = max(0, 100 - slippage_bps)
        scores["slippage"] = slippage_score

        # 2. Fill Rate Score (0-100)
        fill_rate = pending.filled_quantity / order.quantity if order.quantity > 0 else 0
        fill_rate_score = fill_rate * 100
        scores["fill_rate"] = fill_rate_score

        # 3. Speed Score (0-100, based on how quickly filled relative to expectation)
        speed_score = 100.0
        if pending.created_at:
            execution_time = (datetime.now(timezone.utc) - pending.created_at).total_seconds()

            # Expected time based on algo type
            if order.algo in ("TWAP", "VWAP"):
                expected_time = self._algo_timeout_seconds * 0.5  # Expect 50% of max time
            else:
                expected_time = self._order_timeout_seconds * 0.25  # Expect 25% of max time

            if execution_time <= expected_time:
                speed_score = 100.0
            else:
                # Decrease linearly: 50% at 2x expected, 0% at 4x expected
                excess_ratio = execution_time / expected_time
                speed_score = max(0, 100 - (excess_ratio - 1) * 33.3)
        scores["speed"] = speed_score

        # 4. Price Improvement Score (0-100)
        price_improvement_score = 50.0  # Neutral baseline
        total_improvement_bps = 0.0
        improvement_count = 0

        for sf in pending.slice_fills.values():
            if sf.price_improvement_bps is not None:
                total_improvement_bps += sf.price_improvement_bps
                improvement_count += 1

        if improvement_count > 0:
            avg_improvement = total_improvement_bps / improvement_count
            # Score: 50 at 0 bps, 100 at +20 bps improvement, 0 at -20 bps
            price_improvement_score = min(100, max(0, 50 + avg_improvement * 2.5))
        scores["price_improvement"] = price_improvement_score

        # 5. Market Impact Score (0-100, estimate based on order size and fill pattern)
        market_impact_score = 80.0  # Default to good if we can't measure
        if order.quantity > 0 and pending.filled_quantity > 0:
            # Estimate impact based on fill dispersion
            if len(pending.slice_fills) > 1:
                fill_prices = [sf.avg_fill_price for sf in pending.slice_fills.values() if sf.avg_fill_price > 0]
                if len(fill_prices) > 1:
                    price_range = max(fill_prices) - min(fill_prices)
                    avg_price = sum(fill_prices) / len(fill_prices)
                    range_bps = (price_range / avg_price * 10000) if avg_price > 0 else 0
                    # Score decreases as price range increases: 100 at 0 bps, 0 at 50 bps range
                    market_impact_score = max(0, 100 - range_bps * 2)
        scores["market_impact"] = market_impact_score

        # Calculate weighted overall score
        weights = self._quality_score_weights
        overall_score = (
            scores["slippage"] * weights.get("slippage", 0.30) +
            scores["fill_rate"] * weights.get("fill_rate", 0.25) +
            scores["speed"] * weights.get("speed", 0.20) +
            scores["price_improvement"] * weights.get("price_improvement", 0.15) +
            scores["market_impact"] * weights.get("market_impact", 0.10)
        )

        quality_metrics = {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "algo": order.algo,
            "overall_score": round(overall_score, 2),
            "grade": self._score_to_grade(overall_score),
            "component_scores": scores,
            "weights_used": weights,
            "filled_quantity": pending.filled_quantity,
            "total_quantity": order.quantity,
            "arrival_price": pending.arrival_price,
            "avg_fill_price": pending.avg_fill_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Store for later retrieval
        self._execution_quality_scores[order_id] = quality_metrics

        return quality_metrics

    def _score_to_grade(self, score: float) -> str:
        """
        Convert numeric score to letter grade (P2).

        Args:
            score: Score 0-100

        Returns:
            Letter grade (A, B, C, D, F)
        """
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def get_execution_quality_score(self, order_id: str) -> dict | None:
        """
        Get previously calculated execution quality score (P2).

        Args:
            order_id: Order ID

        Returns:
            Quality metrics dict or None
        """
        return self._execution_quality_scores.get(order_id)

    def get_execution_quality_summary(self) -> dict:
        """
        Get summary of execution quality across all scored orders (P2).

        Returns:
            Summary statistics of execution quality
        """
        if not self._execution_quality_scores:
            return {
                "total_scored": 0,
                "avg_overall_score": None,
                "grade_distribution": {},
            }

        scores = list(self._execution_quality_scores.values())
        overall_scores = [s["overall_score"] for s in scores]

        # Grade distribution
        grades = [s["grade"] for s in scores]
        grade_dist = {g: grades.count(g) for g in ["A", "B", "C", "D", "F"]}

        # Component averages
        component_avgs = {}
        for component in ["slippage", "fill_rate", "speed", "price_improvement", "market_impact"]:
            vals = [s["component_scores"].get(component, 0) for s in scores]
            component_avgs[component] = sum(vals) / len(vals) if vals else 0

        return {
            "total_scored": len(scores),
            "avg_overall_score": sum(overall_scores) / len(overall_scores),
            "min_score": min(overall_scores),
            "max_score": max(overall_scores),
            "grade_distribution": grade_dist,
            "component_averages": component_avgs,
            "recent_scores": scores[-10:],  # Last 10 scored orders
        }

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

    def update_atr(self, symbol: str, atr: float) -> None:
        """
        QUICK WIN #9: Update the ATR value for a symbol.

        Used for spread validation in pre-trade checks.
        Called by market data processor or signal agents.

        Args:
            symbol: Trading symbol
            atr: Current ATR value
        """
        if atr > 0:
            self._atr_cache[symbol] = atr

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
            "timeout_monitor_running": self._order_timeout_monitor_task is not None and not self._order_timeout_monitor_task.done(),  # #E24
            "timed_out_orders": len(self._timed_out_orders),  # #E24
            "symbols_with_prices": len(self._last_prices),
            # Fill quality summary (#E13)
            "fill_quality": self.get_aggregate_fill_metrics(),
            # Implementation shortfall summary (#E14)
            "implementation_shortfall": self.get_implementation_shortfall_summary(),
            # P2: Order acknowledgment timeout tracking
            "ack_timeout": {
                "pending_ack_count": self.get_pending_ack_count(),
                "timeout_threshold_seconds": self._ack_timeout_seconds,
                "total_timeout_alerts": len(self._ack_timeout_alerts),
            },
            # P2: Partial fill tracking
            "partial_fill_tracking": self.get_partial_fill_summary(),
            # P2: Execution quality scoring
            "execution_quality_scoring": self.get_execution_quality_summary(),
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
                    # Best-effort recovery - log full trace for debugging
                    logger.exception(f"Failed to recover order: {e}")

            # Backup and remove the persistence file after recovery
            if recovered_count > 0:
                backup_path = filepath + f".recovered.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                os.rename(filepath, backup_path)
                logger.info(f"Moved persistence file to {backup_path}")

            logger.info(f"Recovered {recovered_count}/{len(orders_data)} orders from persistence file")
            return recovered_count

        except Exception as e:
            # Broad catch for resilience - order recovery is best-effort and
            # system should start even if recovery fails (JSON parse errors,
            # corrupted files, schema changes, etc.)
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

    # =========================================================================
    # SPREAD CROSSING LOGIC (#E16)
    # =========================================================================

    def should_cross_spread(
        self,
        symbol: str,
        side: str,
        urgency: float = 0.5,
        opportunity_cost_bps: float | None = None
    ) -> dict:
        """
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
        """
        book = self._order_books.get(symbol)
        if not book or not book.best_bid or not book.best_ask:
            return {
                "should_cross": True,  # Default to crossing when no book data
                "reason": "no_book_data",
                "spread_cost_bps": None,
                "expected_queue_cost_bps": None,
                "net_benefit_bps": None,
            }

        spread = book.best_ask - book.best_bid
        spread_bps = book.spread_bps or 0

        # Cost of crossing = half spread (we're at mid, pay to best)
        crossing_cost_bps = spread_bps / 2

        # Estimate queue cost (opportunity cost while waiting)
        queue_time_estimate = self._estimate_queue_time(symbol, side)

        if opportunity_cost_bps is None:
            # Default opportunity cost based on volatility
            volatility = self._market_impact_params.get("default_volatility", 0.02)
            opportunity_cost_bps = volatility * 100 / 60  # Rough per-minute alpha decay

        queue_cost_bps = opportunity_cost_bps * queue_time_estimate

        # Adjust for urgency
        urgency_factor = 1 + urgency * 2  # 1-3x multiplier on opportunity cost
        adjusted_queue_cost = queue_cost_bps * urgency_factor

        # Consider market conditions
        imbalance = book.depth_imbalance(5)
        # If buying and bid-heavy (imbalance > 0), more likely to get filled passively
        # If buying and ask-heavy (imbalance < 0), queue might grow
        imbalance_factor = 1.0
        if side == 'buy':
            imbalance_factor = 1 - imbalance * 0.5  # Less likely to queue if ask-heavy
        else:  # sell
            imbalance_factor = 1 + imbalance * 0.5  # Less likely to queue if bid-heavy

        adjusted_queue_cost *= imbalance_factor

        # Decision: cross if queue cost exceeds crossing cost
        net_benefit = adjusted_queue_cost - crossing_cost_bps
        should_cross = net_benefit > 0 or urgency > 0.8

        return {
            "should_cross": should_cross,
            "reason": "queue_cost_exceeds_spread" if should_cross else "passive_more_efficient",
            "spread_cost_bps": crossing_cost_bps,
            "expected_queue_time_min": queue_time_estimate,
            "expected_queue_cost_bps": adjusted_queue_cost,
            "net_benefit_bps": net_benefit,
            "book_imbalance": imbalance,
            "urgency": urgency,
        }

    def execute_with_spread_awareness(
        self,
        order: OrderEvent,
        urgency: float = 0.5
    ) -> dict:
        """
        Execute order with spread-aware logic (#E16).

        Automatically chooses between aggressive (crossing) and passive
        (limit at bid/ask) execution based on market conditions.

        Args:
            order: Order to execute
            urgency: Urgency factor 0-1

        Returns:
            Execution decision details
        """
        symbol = order.symbol
        side = 'buy' if order.side == OrderSide.BUY else 'sell'

        decision = self.should_cross_spread(symbol, side, urgency)

        if decision["should_cross"]:
            # Execute aggressively - market or aggressive limit
            execution_type = "aggressive"
            logger.info(
                f"Spread crossing for {symbol}: crossing cost={decision['spread_cost_bps']:.1f}bps, "
                f"queue cost={decision['expected_queue_cost_bps']:.1f}bps, "
                f"net benefit={decision['net_benefit_bps']:.1f}bps"
            )
        else:
            # Execute passively - post at NBBO
            execution_type = "passive"
            logger.info(
                f"Passive execution for {symbol}: spread cost={decision['spread_cost_bps']:.1f}bps "
                f"exceeds queue cost={decision['expected_queue_cost_bps']:.1f}bps"
            )

        return {
            "execution_type": execution_type,
            "order": order,
            **decision,
        }

    # =========================================================================
    # QUEUE POSITION ESTIMATION (#E17)
    # =========================================================================

    def _estimate_queue_time(self, symbol: str, side: str, size: int = 100) -> float:
        """
        Estimate time to get filled when posting at NBBO (#E17).

        Uses simplified queue model based on:
        - Current queue depth at NBBO
        - Historical fill rate
        - Order size relative to queue

        Args:
            symbol: Instrument symbol
            side: 'buy' (posting at bid) or 'sell' (posting at ask)
            size: Order size

        Returns:
            Estimated minutes to fill
        """
        book = self._order_books.get(symbol)
        if not book:
            return 5.0  # Default 5 min when no data

        # Get queue depth at NBBO
        if side == 'buy':
            # Posting at bid - queue behind existing bids
            queue_depth = book.bids[0].size if book.bids else 0
        else:
            # Posting at ask - queue behind existing asks
            queue_depth = book.asks[0].size if book.asks else 0

        # Estimate fill rate (shares/minute) from historical data
        fill_rate = self._estimate_fill_rate(symbol, side)

        if fill_rate <= 0:
            fill_rate = 100  # Default 100 shares/min

        # Position in queue (simplified - assumes we're at back)
        queue_position = queue_depth + size

        # Time = position / fill_rate
        estimated_time = queue_position / fill_rate

        return estimated_time

    def _estimate_fill_rate(self, symbol: str, side: str) -> float:
        """
        Estimate passive fill rate in shares/minute (#E17).

        Uses historical fill data if available, otherwise market volume.

        Args:
            symbol: Instrument symbol
            side: 'buy' or 'sell'

        Returns:
            Estimated fill rate in shares/minute
        """
        # Try to use historical market volume
        volume_history = self._market_volume_history.get(symbol, [])

        if len(volume_history) >= 2:
            # Calculate average volume per minute
            time_span_minutes = (
                volume_history[-1][0] - volume_history[0][0]
            ).total_seconds() / 60

            if time_span_minutes > 0:
                vol_change = volume_history[-1][1] - volume_history[0][1]
                vol_per_minute = vol_change / time_span_minutes

                # Assume ~50% of volume crosses the spread (marketable)
                # The other 50% is passive fills that clear queue
                fill_rate = vol_per_minute * 0.5

                # Adjust by side based on imbalance
                book = self._order_books.get(symbol)
                if book:
                    imbalance = book.depth_imbalance(3)
                    if side == 'buy':
                        # Bid-heavy = more sellers hitting bids = faster bid fills
                        fill_rate *= (1 - imbalance * 0.3)
                    else:
                        # Ask-heavy = more buyers lifting offers = faster ask fills
                        fill_rate *= (1 + imbalance * 0.3)

                return max(fill_rate, 10)  # Minimum 10 shares/min

        # Default based on typical liquid stock
        return 200  # 200 shares/minute default

    def estimate_queue_position(
        self,
        symbol: str,
        side: str,
        size: int,
        price: float | None = None
    ) -> dict:
        """
        Estimate queue position and expected fill time (#E17).

        Args:
            symbol: Instrument symbol
            side: 'buy' or 'sell'
            size: Order size
            price: Limit price (uses NBBO if None)

        Returns:
            Queue position analysis
        """
        book = self._order_books.get(symbol)
        if not book:
            return {
                "queue_position": None,
                "queue_depth_ahead": None,
                "estimated_fill_time_min": None,
                "fill_probability_5min": None,
                "reason": "no_book_data",
            }

        # Determine queue level
        at_nbbo = True
        if price is not None:
            if side == 'buy' and book.best_bid:
                at_nbbo = abs(price - book.best_bid) < 0.0001
            elif side == 'sell' and book.best_ask:
                at_nbbo = abs(price - book.best_ask) < 0.0001

        if side == 'buy':
            # Posting at bid
            nbbo_depth = book.bids[0].size if book.bids else 0
            our_price = price or (book.best_bid or 0)
            levels_ahead = [l for l in book.bids if l.price > our_price] if not at_nbbo else []
            depth_ahead = sum(l.size for l in levels_ahead) + (nbbo_depth if at_nbbo else 0)
        else:
            # Posting at ask
            nbbo_depth = book.asks[0].size if book.asks else 0
            our_price = price or (book.best_ask or float('inf'))
            levels_ahead = [l for l in book.asks if l.price < our_price] if not at_nbbo else []
            depth_ahead = sum(l.size for l in levels_ahead) + (nbbo_depth if at_nbbo else 0)

        # Estimate fill time
        fill_rate = self._estimate_fill_rate(symbol, side)
        queue_position = depth_ahead + size
        estimated_fill_time = queue_position / fill_rate if fill_rate > 0 else float('inf')

        # Calculate fill probability in next 5 minutes
        # Using exponential arrival model
        expected_fills_5min = fill_rate * 5
        # P(fill) = 1 - P(queue_position not reached in 5 min)
        # Simplified: if expected fills > queue_position, high probability
        fill_prob_5min = min(1.0, expected_fills_5min / queue_position) if queue_position > 0 else 1.0

        return {
            "symbol": symbol,
            "side": side,
            "size": size,
            "at_nbbo": at_nbbo,
            "queue_position": queue_position,
            "queue_depth_ahead": depth_ahead,
            "nbbo_depth": nbbo_depth,
            "estimated_fill_rate": fill_rate,
            "estimated_fill_time_min": estimated_fill_time,
            "fill_probability_5min": fill_prob_5min,
        }

    # =========================================================================
    # MIDPOINT PEG ORDER SUPPORT (#E18)
    # =========================================================================

    async def execute_midpoint_peg(
        self,
        order: OrderEvent,
        max_deviation_bps: float = 5.0,
        timeout_seconds: float = 300.0
    ) -> dict:
        """
        Execute order with midpoint pegging (#E18).

        Places limit order at midpoint and re-pegs on price changes.
        Common for reducing spread costs in less urgent executions.

        Args:
            order: Order to execute
            max_deviation_bps: Max price deviation before re-peg (bps from mid)
            timeout_seconds: Maximum time to attempt execution

        Returns:
            Execution result
        """
        symbol = order.symbol
        side = 'buy' if order.side == OrderSide.BUY else 'sell'
        remaining = order.quantity

        start_time = datetime.now(timezone.utc)
        fills = []
        repeg_count = 0
        current_order_id = None

        while remaining > 0:
            # Check timeout
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > timeout_seconds:
                logger.warning(f"Midpoint peg timeout for {symbol} after {elapsed:.0f}s")
                break

            # Get current midpoint
            book = self._order_books.get(symbol)
            if not book or not book.mid_price:
                await asyncio.sleep(0.5)
                continue

            mid = book.mid_price
            tick = self._specs_manager.get_spec(symbol).tick_size if self._specs_manager else 0.01

            # Round to tick
            if side == 'buy':
                # Round down for buys (don't overpay)
                limit_price = (mid // tick) * tick
            else:
                # Round up for sells (don't undersell)
                limit_price = ((mid + tick - 0.0001) // tick) * tick

            # Check if we need to re-peg
            need_repeg = True
            if current_order_id is not None:
                # Check if current order's price is still near mid
                pending = self._pending_orders.get(current_order_id)
                if pending and pending.order_event.limit_price:
                    current_price = pending.order_event.limit_price
                    deviation_bps = abs(current_price - mid) / mid * 10000
                    if deviation_bps <= max_deviation_bps:
                        need_repeg = False

            if need_repeg:
                # Cancel existing order if any
                if current_order_id is not None:
                    await self._cancel_order(current_order_id)
                    repeg_count += 1

                # Create new midpoint order
                mid_order = OrderEvent(
                    source_agent=self.name,
                    decision_id=order.decision_id,
                    validation_id=order.validation_id,
                    symbol=symbol,
                    side=order.side,
                    quantity=remaining,
                    order_type=OrderType.LIMIT,
                    limit_price=limit_price,
                    algo="MIDPOINT_PEG",
                )

                broker_id = await self._broker.place_order(mid_order)
                if broker_id:
                    current_order_id = broker_id
                    logger.debug(f"Midpoint peg order placed: {symbol} @ {limit_price:.4f}")

            # Wait for fill or price change
            await asyncio.sleep(0.5)

            # Check for fills
            if current_order_id:
                pending = self._pending_orders.get(current_order_id)
                if pending:
                    filled = pending.filled_quantity
                    if filled > 0:
                        fills.append({
                            "quantity": filled,
                            "avg_price": pending.avg_fill_price,
                        })
                        remaining -= filled

        # Final status
        total_filled = sum(f["quantity"] for f in fills)
        if total_filled > 0:
            avg_price = sum(f["quantity"] * f["avg_price"] for f in fills) / total_filled
        else:
            avg_price = None

        return {
            "symbol": symbol,
            "side": side,
            "requested_quantity": order.quantity,
            "filled_quantity": total_filled,
            "remaining_quantity": remaining,
            "avg_fill_price": avg_price,
            "repeg_count": repeg_count,
            "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
            "status": "filled" if remaining == 0 else ("partial" if total_filled > 0 else "unfilled"),
        }

    async def _cancel_order(self, broker_order_id: int) -> bool:
        """Cancel an order by broker ID."""
        try:
            if self._broker:
                await self._broker.cancel_order(broker_order_id)
                return True
        except Exception as e:
            # Broad catch - cancellation is best-effort, system must continue
            logger.exception(f"Failed to cancel order {broker_order_id}: {e}")
        return False

    # =========================================================================
    # ICEBERG ORDER SUPPORT (#E19)
    # =========================================================================

    async def execute_iceberg(
        self,
        order: OrderEvent,
        display_size: int,
        variance_pct: float = 0.2,
        min_replenish_seconds: float = 1.0,
        price_offset_ticks: int = 0
    ) -> dict:
        """
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
        """
        import random

        symbol = order.symbol
        side = 'buy' if order.side == OrderSide.BUY else 'sell'
        total_quantity = order.quantity
        remaining = total_quantity

        start_time = datetime.now(timezone.utc)
        fills = []
        slice_count = 0
        current_slice_id = None
        last_replenish_time = None

        logger.info(
            f"Starting iceberg execution: {symbol} {side} {total_quantity} "
            f"(display={display_size}, variance={variance_pct:.0%})"
        )

        while remaining > 0:
            # Check rate limits
            if not self._check_rate_limits():
                await asyncio.sleep(0.5)
                continue

            # Check minimum replenish interval
            if last_replenish_time:
                elapsed = (datetime.now(timezone.utc) - last_replenish_time).total_seconds()
                if elapsed < min_replenish_seconds:
                    await asyncio.sleep(min_replenish_seconds - elapsed)

            # Check if current slice is filled
            need_new_slice = current_slice_id is None
            if current_slice_id:
                pending = self._pending_orders.get(current_slice_id)
                if pending and pending.status in ("filled", "completed"):
                    filled = pending.filled_quantity
                    if filled > 0:
                        fills.append({
                            "slice": slice_count,
                            "quantity": filled,
                            "avg_price": pending.avg_fill_price,
                        })
                        remaining -= filled
                    need_new_slice = True
                elif pending and pending.status == "cancelled":
                    need_new_slice = True

            if need_new_slice and remaining > 0:
                # Calculate slice size with variance
                variance = random.uniform(-variance_pct, variance_pct)
                slice_size = int(display_size * (1 + variance))
                slice_size = min(slice_size, remaining)
                slice_size = max(1, slice_size)

                # Determine limit price
                book = self._order_books.get(symbol)
                if book:
                    tick = self._specs_manager.get_spec(symbol).tick_size if self._specs_manager else 0.01
                    if side == 'buy':
                        base_price = book.best_bid or self._last_prices.get(symbol, 100)
                        limit_price = base_price + (price_offset_ticks * tick)
                    else:
                        base_price = book.best_ask or self._last_prices.get(symbol, 100)
                        limit_price = base_price - (price_offset_ticks * tick)
                else:
                    limit_price = self._last_prices.get(symbol, 100)

                # Create slice order
                slice_order = OrderEvent(
                    source_agent=self.name,
                    decision_id=order.decision_id,
                    validation_id=order.validation_id,
                    symbol=symbol,
                    side=order.side,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    limit_price=limit_price,
                    algo="ICEBERG_SLICE",
                )

                broker_id = await self._broker.place_order(slice_order)
                if broker_id:
                    current_slice_id = broker_id
                    slice_count += 1
                    last_replenish_time = datetime.now(timezone.utc)
                    logger.debug(
                        f"Iceberg slice {slice_count}: {slice_size} @ {limit_price:.4f} "
                        f"(remaining={remaining})"
                    )

            await asyncio.sleep(0.1)  # Small delay between checks

        # Final tally
        total_filled = sum(f["quantity"] for f in fills)
        if total_filled > 0:
            avg_price = sum(f["quantity"] * f["avg_price"] for f in fills) / total_filled
        else:
            avg_price = None

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = {
            "symbol": symbol,
            "side": side,
            "total_quantity": total_quantity,
            "filled_quantity": total_filled,
            "remaining_quantity": remaining,
            "avg_fill_price": avg_price,
            "slice_count": slice_count,
            "display_size": display_size,
            "duration_seconds": duration,
            "fill_rate_per_minute": total_filled / (duration / 60) if duration > 0 else 0,
            "status": "filled" if remaining == 0 else ("partial" if total_filled > 0 else "unfilled"),
        }

        logger.info(
            f"Iceberg complete: {symbol} filled {total_filled}/{total_quantity} "
            f"in {slice_count} slices over {duration:.1f}s"
        )

        return result

    # =========================================================================
    # POST-TRADE TCA (#E22)
    # =========================================================================

    def calculate_post_trade_tca(
        self,
        order_id: str,
        benchmark: str = "arrival"
    ) -> dict | None:
        """
        Calculate comprehensive post-trade Transaction Cost Analysis (#E22).

        Analyzes execution quality against multiple benchmarks.

        Args:
            order_id: Order ID to analyze
            benchmark: Primary benchmark ('arrival', 'vwap', 'twap', 'close')

        Returns:
            TCA report or None if order not found
        """
        pending = self._pending_orders.get(order_id)
        if not pending:
            return None

        symbol = pending.order_event.symbol
        side = 'buy' if pending.order_event.side == OrderSide.BUY else 'sell'
        quantity = pending.order_event.quantity
        filled = pending.filled_quantity
        avg_price = pending.avg_fill_price

        if filled == 0 or avg_price == 0:
            return {
                "order_id": order_id,
                "status": "unfilled",
                "message": "No fills to analyze",
            }

        # Get benchmark prices
        arrival_price = pending.arrival_price
        vwap_price = self._calculate_interval_vwap(symbol, pending.created_at)
        close_price = self._last_prices.get(symbol)

        # Calculate costs vs each benchmark
        def calc_cost(benchmark_price: float | None) -> float | None:
            if benchmark_price is None or benchmark_price == 0:
                return None
            if side == 'buy':
                return (avg_price - benchmark_price) / benchmark_price * 10000  # bps
            else:
                return (benchmark_price - avg_price) / benchmark_price * 10000  # bps

        arrival_cost = calc_cost(arrival_price)
        vwap_cost = calc_cost(vwap_price)
        close_cost = calc_cost(close_price)

        # Implementation shortfall breakdown
        impl_shortfall = self.calculate_implementation_shortfall(order_id)

        # Market impact estimate
        impact = self.estimate_market_impact(
            symbol, side, filled, arrival_price,
            adv=self._get_adv(symbol)
        )

        # Fill quality metrics
        fill_metrics = self.get_aggregate_fill_metrics()

        # Categorization summary
        categorization = self.get_fill_categorization_summary()

        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "filled_quantity": filled,
            "fill_rate_pct": (filled / quantity) * 100 if quantity > 0 else 0,
            "avg_fill_price": avg_price,

            # Benchmark prices
            "benchmarks": {
                "arrival": arrival_price,
                "vwap": vwap_price,
                "close": close_price,
            },

            # Costs vs benchmarks (positive = underperformed)
            "costs_bps": {
                "vs_arrival": arrival_cost,
                "vs_vwap": vwap_cost,
                "vs_close": close_cost,
            },

            # Primary benchmark
            "primary_benchmark": benchmark,
            "primary_cost_bps": {
                "arrival": arrival_cost,
                "vwap": vwap_cost,
                "close": close_cost,
            }.get(benchmark),

            # Implementation shortfall
            "implementation_shortfall": impl_shortfall,

            # Market impact
            "estimated_market_impact_bps": impact.total_impact_bps if impact else None,

            # Execution quality
            "execution_quality": {
                "price_improvement_rate": fill_metrics.get("price_improvement_rate", 0),
                "avg_slippage_bps": fill_metrics.get("avg_slippage_bps", 0),
                "aggressive_fill_pct": categorization.get("aggressive", {}).get("pct", 0),
                "passive_fill_pct": categorization.get("passive", {}).get("pct", 0),
            },

            # Timing
            "created_at": pending.created_at.isoformat() if pending.created_at else None,
            "duration_seconds": (
                datetime.now(timezone.utc) - pending.created_at
            ).total_seconds() if pending.created_at else None,
        }

    def _calculate_interval_vwap(
        self,
        symbol: str,
        start_time: datetime
    ) -> float | None:
        """Calculate VWAP from start_time to now."""
        # This would ideally use tick-level data
        # Simplified: use stored prices if available
        # In production, would query historical tick data
        return self._last_prices.get(symbol)

    def _get_adv(self, symbol: str) -> float:
        """Get average daily volume for symbol."""
        # Simplified - would query historical data in production
        return 1_000_000  # Default

    def generate_tca_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> dict:
        """
        Generate aggregate TCA report for a period (#E22).

        Args:
            start_date: Report start (default: all data)
            end_date: Report end (default: now)

        Returns:
            Aggregate TCA statistics
        """
        reports = []

        for order_id, pending in self._pending_orders.items():
            # Filter by date if specified
            if start_date and pending.created_at and pending.created_at < start_date:
                continue
            if end_date and pending.created_at and pending.created_at > end_date:
                continue

            tca = self.calculate_post_trade_tca(order_id)
            if tca and tca.get("status") != "unfilled":
                reports.append(tca)

        if not reports:
            return {
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
                "order_count": 0,
                "message": "No completed orders in period",
            }

        # Aggregate statistics
        arrival_costs = [r["costs_bps"]["vs_arrival"] for r in reports if r["costs_bps"]["vs_arrival"] is not None]
        vwap_costs = [r["costs_bps"]["vs_vwap"] for r in reports if r["costs_bps"]["vs_vwap"] is not None]

        return {
            "period": {
                "start": start_date.isoformat() if start_date else "all",
                "end": end_date.isoformat() if end_date else "now",
            },
            "order_count": len(reports),
            "total_filled_quantity": sum(r["filled_quantity"] for r in reports),
            "avg_fill_rate_pct": sum(r["fill_rate_pct"] for r in reports) / len(reports),

            "arrival_cost_bps": {
                "mean": sum(arrival_costs) / len(arrival_costs) if arrival_costs else None,
                "median": sorted(arrival_costs)[len(arrival_costs)//2] if arrival_costs else None,
                "min": min(arrival_costs) if arrival_costs else None,
                "max": max(arrival_costs) if arrival_costs else None,
            },

            "vwap_cost_bps": {
                "mean": sum(vwap_costs) / len(vwap_costs) if vwap_costs else None,
                "median": sorted(vwap_costs)[len(vwap_costs)//2] if vwap_costs else None,
            },

            "execution_quality": {
                "avg_price_improvement_rate": sum(
                    r["execution_quality"]["price_improvement_rate"] for r in reports
                ) / len(reports),
                "avg_aggressive_fill_pct": sum(
                    r["execution_quality"]["aggressive_fill_pct"] for r in reports
                ) / len(reports),
            },
        }

    # =========================================================================
    # VENUE LATENCY MONITORING (#E23)
    # =========================================================================

    def __init_venue_latency(self):
        """Initialize venue latency tracking structures."""
        self._venue_latencies: dict[str, deque[tuple[datetime, float]]] = {}
        self._venue_latency_alerts: deque[dict] = deque(maxlen=500)
        self._latency_threshold_ms: float = 100.0  # Alert threshold

    def record_venue_latency(
        self,
        venue: str,
        latency_ms: float,
        event_type: str = "order"
    ) -> None:
        """
        Record latency measurement for a venue (#E23).

        Args:
            venue: Venue/exchange identifier
            latency_ms: Round-trip latency in milliseconds
            event_type: Type of event ('order', 'fill', 'cancel', 'quote')
        """
        if not hasattr(self, '_venue_latencies'):
            self.__init_venue_latency()

        now = datetime.now(timezone.utc)
        key = f"{venue}:{event_type}"

        if key not in self._venue_latencies:
            self._venue_latencies[key] = []

        self._venue_latencies[key].append((now, latency_ms))

        # Trim old data (keep 1 hour)
        cutoff = now - timedelta(hours=1)
        self._venue_latencies[key] = [
            (t, l) for t, l in self._venue_latencies[key]
            if t > cutoff
        ]

        # Check for latency spike
        if latency_ms > self._latency_threshold_ms:
            self._venue_latency_alerts.append({
                "timestamp": now.isoformat(),
                "venue": venue,
                "event_type": event_type,
                "latency_ms": latency_ms,
                "threshold_ms": self._latency_threshold_ms,
            })
            logger.warning(
                f"Venue latency spike: {venue} {event_type} = {latency_ms:.1f}ms "
                f"(threshold={self._latency_threshold_ms}ms)"
            )

    def get_venue_latency_stats(
        self,
        venue: str | None = None,
        lookback_minutes: int = 60
    ) -> dict:
        """
        Get venue latency statistics (#E23).

        Args:
            venue: Specific venue or None for all
            lookback_minutes: Analysis window

        Returns:
            Latency statistics by venue and event type
        """
        if not hasattr(self, '_venue_latencies'):
            self.__init_venue_latency()

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=lookback_minutes)

        stats = {}

        for key, measurements in self._venue_latencies.items():
            # Filter venue if specified
            venue_name, event_type = key.split(":", 1)
            if venue and venue_name != venue:
                continue

            recent = [l for t, l in measurements if t > cutoff]

            if not recent:
                continue

            if venue_name not in stats:
                stats[venue_name] = {}

            stats[venue_name][event_type] = {
                "count": len(recent),
                "mean_ms": sum(recent) / len(recent),
                "min_ms": min(recent),
                "max_ms": max(recent),
                "p50_ms": sorted(recent)[len(recent)//2],
                "p95_ms": sorted(recent)[int(len(recent)*0.95)] if len(recent) >= 20 else max(recent),
                "p99_ms": sorted(recent)[int(len(recent)*0.99)] if len(recent) >= 100 else max(recent),
            }

        return {
            "lookback_minutes": lookback_minutes,
            "venues": stats,
            "recent_alerts": self._venue_latency_alerts[-10:] if hasattr(self, '_venue_latency_alerts') else [],
        }

    def set_latency_threshold(self, threshold_ms: float) -> None:
        """Set latency alert threshold in milliseconds (#E23)."""
        if not hasattr(self, '_latency_threshold_ms'):
            self.__init_venue_latency()
        self._latency_threshold_ms = threshold_ms
        logger.info(f"Venue latency alert threshold set to {threshold_ms}ms")

    def check_venue_health(self) -> dict:
        """
        Check overall venue connectivity health (#E23).

        Returns:
            Health status for each venue
        """
        stats = self.get_venue_latency_stats(lookback_minutes=5)
        health = {}

        for venue, event_stats in stats.get("venues", {}).items():
            # Check order latency specifically
            order_stats = event_stats.get("order", {})
            if not order_stats:
                health[venue] = {
                    "status": "unknown",
                    "reason": "no_recent_data",
                }
                continue

            mean_latency = order_stats.get("mean_ms", 0)
            p95_latency = order_stats.get("p95_ms", 0)

            if p95_latency > self._latency_threshold_ms * 2:
                status = "degraded"
                reason = f"high_p95_latency ({p95_latency:.0f}ms)"
            elif mean_latency > self._latency_threshold_ms:
                status = "warning"
                reason = f"elevated_mean_latency ({mean_latency:.0f}ms)"
            else:
                status = "healthy"
                reason = None

            health[venue] = {
                "status": status,
                "reason": reason,
                "mean_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "sample_count": order_stats.get("count", 0),
            }

        return health

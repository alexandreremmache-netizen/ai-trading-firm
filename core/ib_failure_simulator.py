"""
IB Failure Simulator - Test broker failure scenarios.

This module simulates various Interactive Brokers failure scenarios
for testing system resilience and recovery mechanisms.

Scenarios covered:
1. Order accepted without fill (zombie orders)
2. Fill without callback (silent fills)
3. Disconnect during cancel
4. Partial fill then disconnect
5. Connection timeout during order submission
6. Duplicate fill callbacks
7. Price rejection / price moved
8. Market data gaps
"""

import asyncio
import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


class FailureScenario(Enum):
    """Types of broker failure scenarios to simulate."""

    # Order lifecycle failures
    ORDER_ACCEPTED_NO_FILL = "order_accepted_no_fill"  # Order sits in "working" state forever
    FILL_WITHOUT_CALLBACK = "fill_without_callback"   # Fill happens but no callback received
    DISCONNECT_DURING_CANCEL = "disconnect_during_cancel"  # Connection lost during cancel
    PARTIAL_FILL_DISCONNECT = "partial_fill_disconnect"    # Partial fill, then connection lost

    # Connection failures
    TIMEOUT_ON_SUBMIT = "timeout_on_submit"           # Connection timeout during order submission
    RANDOM_DISCONNECT = "random_disconnect"           # Random disconnection
    RECONNECT_STALE_DATA = "reconnect_stale_data"     # After reconnect, stale position data

    # Callback failures
    DUPLICATE_FILL_CALLBACK = "duplicate_fill_callback"  # Same fill callback sent twice
    OUT_OF_ORDER_CALLBACKS = "out_of_order_callbacks"    # Callbacks arrive out of sequence

    # Price/execution failures
    PRICE_REJECTION = "price_rejection"               # Order rejected due to price moved
    PARTIAL_FILL_PRICE_MOVE = "partial_fill_price_move"  # Partial fill, then price moves

    # Market data failures
    MARKET_DATA_GAP = "market_data_gap"               # Gap in market data stream
    STALE_QUOTE = "stale_quote"                       # Quote timestamps are stale


@dataclass
class SimulatedOrder:
    """Track simulated order state."""
    order_id: str
    symbol: str
    action: str  # BUY or SELL
    quantity: float
    price: float
    status: str = "pending"
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    callbacks_received: list = field(default_factory=list)


@dataclass
class FailureEvent:
    """Record of a simulated failure event."""
    scenario: FailureScenario
    order_id: Optional[str]
    timestamp: datetime
    details: dict
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class IBFailureSimulator:
    """
    Simulates Interactive Brokers failure scenarios for testing.

    This can be injected into the broker module during testing to
    verify that the system handles various edge cases correctly.

    Usage:
        simulator = IBFailureSimulator()
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, probability=0.1)

        # In broker code:
        if simulator.should_trigger(order_id):
            scenario = simulator.get_triggered_scenario(order_id)
            # Handle based on scenario
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_failure_history: int = 1000,
    ):
        """
        Initialize the failure simulator.

        Args:
            seed: Random seed for reproducibility
            max_failure_history: Max number of failure events to track
        """
        self._random = random.Random(seed)
        self._enabled_scenarios: dict[FailureScenario, float] = {}  # scenario -> probability
        self._active_failures: dict[str, FailureEvent] = {}  # order_id -> active failure
        self._failure_history: deque[FailureEvent] = deque(maxlen=max_failure_history)
        self._simulated_orders: dict[str, SimulatedOrder] = {}
        self._connection_state = "connected"
        self._callbacks_pending: list[tuple[float, Callable]] = []  # (delay, callback)
        self._running = False
        self._callback_processor_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "total_failures_triggered": 0,
            "failures_by_scenario": {s.value: 0 for s in FailureScenario},
            "failures_resolved": 0,
            "average_resolution_time_seconds": 0.0,
        }

        # Thread safety lock
        self._lock = threading.RLock()

        logger.info("IBFailureSimulator initialized")

    def enable_scenario(
        self,
        scenario: FailureScenario,
        probability: float = 0.1,
    ) -> None:
        """
        Enable a failure scenario with given probability.

        Args:
            scenario: The failure scenario to enable
            probability: Probability of triggering (0.0 to 1.0)
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")

        self._enabled_scenarios[scenario] = probability
        logger.info(f"Enabled failure scenario: {scenario.value} with probability {probability}")

    def disable_scenario(self, scenario: FailureScenario) -> None:
        """Disable a failure scenario."""
        if scenario in self._enabled_scenarios:
            del self._enabled_scenarios[scenario]
            logger.info(f"Disabled failure scenario: {scenario.value}")

    def disable_all(self) -> None:
        """Disable all failure scenarios."""
        self._enabled_scenarios.clear()
        logger.info("All failure scenarios disabled")

    def should_trigger(self, order_id: Optional[str] = None) -> bool:
        """
        Check if any failure scenario should trigger.

        Args:
            order_id: Optional order ID to check for order-specific failures

        Returns:
            True if a failure scenario was triggered
        """
        if not self._enabled_scenarios:
            return False

        # Check each enabled scenario
        for scenario, probability in self._enabled_scenarios.items():
            if self._random.random() < probability:
                self._trigger_failure(scenario, order_id)
                return True

        return False

    def _trigger_failure(
        self,
        scenario: FailureScenario,
        order_id: Optional[str],
        details: Optional[dict] = None,
    ) -> FailureEvent:
        """
        Trigger a specific failure scenario.

        Args:
            scenario: The scenario to trigger
            order_id: Related order ID if applicable
            details: Additional details about the failure

        Returns:
            The created FailureEvent
        """
        event = FailureEvent(
            scenario=scenario,
            order_id=order_id,
            timestamp=datetime.now(timezone.utc),
            details=details or {},
        )

        with self._lock:
            if order_id:
                self._active_failures[order_id] = event

            self._failure_history.append(event)
            self._stats["total_failures_triggered"] += 1
            self._stats["failures_by_scenario"][scenario.value] += 1

        logger.warning(
            f"SIMULATED FAILURE TRIGGERED: {scenario.value} | "
            f"Order: {order_id or 'N/A'} | Details: {details}"
        )

        return event

    def get_triggered_scenario(self, order_id: str) -> Optional[FailureScenario]:
        """Get the active failure scenario for an order."""
        if order_id in self._active_failures:
            return self._active_failures[order_id].scenario
        return None

    def resolve_failure(self, order_id: str) -> bool:
        """
        Mark a failure as resolved.

        Args:
            order_id: The order ID whose failure to resolve

        Returns:
            True if a failure was resolved
        """
        with self._lock:
            if order_id not in self._active_failures:
                return False

            event = self._active_failures.pop(order_id)
            event.resolved = True
            event.resolved_at = datetime.now(timezone.utc)

            # Update statistics
            self._stats["failures_resolved"] += 1
            resolution_time = (event.resolved_at - event.timestamp).total_seconds()
            n = self._stats["failures_resolved"]
            avg = self._stats["average_resolution_time_seconds"]
            # Division by zero protection
            if n > 0:
                self._stats["average_resolution_time_seconds"] = avg + (resolution_time - avg) / n

        logger.info(f"Simulated failure resolved: {event.scenario.value} | Order: {order_id}")
        return True

    # --- Scenario-specific simulation methods ---

    async def simulate_order_accepted_no_fill(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        on_status: Callable[[str, str], None],
    ) -> None:
        """
        Simulate order accepted but never filled (zombie order).

        The order will receive "accepted" status but will never get
        fill callbacks.
        """
        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            status="working",
        )
        self._simulated_orders[order_id] = order

        # Send accepted callback
        await asyncio.sleep(0.05)  # Simulate network latency
        on_status(order_id, "accepted")
        order.callbacks_received.append(("status", "accepted"))

        # Never send fill - order stays in working state
        self._trigger_failure(
            FailureScenario.ORDER_ACCEPTED_NO_FILL,
            order_id,
            {"symbol": symbol, "quantity": quantity},
        )

    async def simulate_fill_without_callback(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        on_status: Callable[[str, str], None],
        on_fill: Callable[[str, float, float], None],
        actual_fill_price: Optional[float] = None,
    ) -> None:
        """
        Simulate fill that happens but callback is lost.

        The broker actually fills the order but the callback never
        reaches our system. The reconciliation agent should catch this.
        """
        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            status="filled",
            filled_qty=quantity,
            avg_fill_price=actual_fill_price or price,
        )
        self._simulated_orders[order_id] = order

        # Send accepted callback
        await asyncio.sleep(0.05)
        on_status(order_id, "accepted")
        order.callbacks_received.append(("status", "accepted"))

        # Mark order as filled internally but DON'T send callback
        self._trigger_failure(
            FailureScenario.FILL_WITHOUT_CALLBACK,
            order_id,
            {"symbol": symbol, "quantity": quantity, "fill_price": order.avg_fill_price},
        )

        # The fill actually happened - return the fill details for reconciliation testing
        return {"filled_qty": quantity, "fill_price": order.avg_fill_price}

    async def simulate_disconnect_during_cancel(
        self,
        order_id: str,
        on_disconnect: Callable[[], None],
        reconnect_after_seconds: float = 5.0,
        on_reconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Simulate connection loss during order cancel.

        The cancel request is sent but connection drops before
        confirmation. Order state becomes unknown.
        """
        self._trigger_failure(
            FailureScenario.DISCONNECT_DURING_CANCEL,
            order_id,
            {"reconnect_after": reconnect_after_seconds},
        )

        # Simulate disconnect
        self._connection_state = "disconnected"
        on_disconnect()

        if reconnect_after_seconds > 0:
            await asyncio.sleep(reconnect_after_seconds)
            self._connection_state = "connected"
            if on_reconnect:
                on_reconnect()
            self.resolve_failure(order_id)

    async def simulate_partial_fill_disconnect(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        partial_fill_pct: float,
        on_status: Callable[[str, str], None],
        on_fill: Callable[[str, float, float], None],
        on_disconnect: Callable[[], None],
        reconnect_after_seconds: float = 5.0,
    ) -> None:
        """
        Simulate partial fill followed by disconnect.

        Order gets partial fill, then connection is lost.
        System must handle reconciling the partial fill.
        """
        partial_qty = quantity * partial_fill_pct

        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            status="partial",
            filled_qty=partial_qty,
            avg_fill_price=price,
        )
        self._simulated_orders[order_id] = order

        # Send accepted callback
        await asyncio.sleep(0.05)
        on_status(order_id, "accepted")

        # Send partial fill
        await asyncio.sleep(0.1)
        on_fill(order_id, partial_qty, price)
        order.callbacks_received.append(("fill", partial_qty, price))

        # Disconnect
        self._trigger_failure(
            FailureScenario.PARTIAL_FILL_DISCONNECT,
            order_id,
            {"partial_qty": partial_qty, "remaining": quantity - partial_qty},
        )

        self._connection_state = "disconnected"
        on_disconnect()

        if reconnect_after_seconds > 0:
            await asyncio.sleep(reconnect_after_seconds)
            self._connection_state = "connected"

    async def simulate_duplicate_fill_callback(
        self,
        order_id: str,
        symbol: str,
        quantity: float,
        price: float,
        on_fill: Callable[[str, float, float], None],
    ) -> None:
        """
        Simulate duplicate fill callbacks.

        The same fill callback is sent twice. System must be
        idempotent and not double-count the fill.
        """
        self._trigger_failure(
            FailureScenario.DUPLICATE_FILL_CALLBACK,
            order_id,
            {"symbol": symbol, "quantity": quantity},
        )

        # Send fill callback twice
        await asyncio.sleep(0.05)
        on_fill(order_id, quantity, price)

        await asyncio.sleep(0.01)  # Very short delay
        on_fill(order_id, quantity, price)  # Duplicate!

    async def simulate_market_data_gap(
        self,
        symbol: str,
        gap_duration_seconds: float,
        on_gap_start: Callable[[str], None],
        on_gap_end: Callable[[str, float], None],
        price_change_during_gap: float = 0.0,
    ) -> None:
        """
        Simulate gap in market data stream.

        No quotes received for a period, then data resumes
        possibly at a different price.
        """
        self._trigger_failure(
            FailureScenario.MARKET_DATA_GAP,
            None,
            {"symbol": symbol, "gap_seconds": gap_duration_seconds},
        )

        on_gap_start(symbol)

        await asyncio.sleep(gap_duration_seconds)

        on_gap_end(symbol, price_change_during_gap)

    # --- Helper methods ---

    def get_simulated_order(self, order_id: str) -> Optional[SimulatedOrder]:
        """Get a simulated order by ID."""
        return self._simulated_orders.get(order_id)

    def get_actual_position(self, symbol: str) -> float:
        """
        Get the actual simulated position (what IB would report).

        This may differ from what our system thinks due to
        fill_without_callback scenarios.
        """
        total = 0.0
        for order in self._simulated_orders.values():
            if order.symbol == symbol and order.filled_qty > 0:
                if order.action == "BUY":
                    total += order.filled_qty
                else:
                    total -= order.filled_qty
        return total

    def is_connected(self) -> bool:
        """Check simulated connection state."""
        return self._connection_state == "connected"

    def get_statistics(self) -> dict:
        """Get failure simulation statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_failures": len(self._active_failures),
                "enabled_scenarios": list(self._enabled_scenarios.keys()),
                "connection_state": self._connection_state,
            }

    def get_active_failures(self) -> list[FailureEvent]:
        """Get list of currently active (unresolved) failures."""
        with self._lock:
            return list(self._active_failures.values())

    def get_failure_history(self) -> list[FailureEvent]:
        """Get history of all triggered failures."""
        return list(self._failure_history)

    def reset(self) -> None:
        """Reset simulator state."""
        self._active_failures.clear()
        self._simulated_orders.clear()
        self._connection_state = "connected"
        logger.info("IBFailureSimulator reset")


# --- Convenience functions for testing ---

def create_failure_simulator(
    scenarios: Optional[dict[FailureScenario, float]] = None,
    seed: Optional[int] = None,
) -> IBFailureSimulator:
    """
    Create a failure simulator with preconfigured scenarios.

    Args:
        scenarios: Dict of scenario -> probability
        seed: Random seed for reproducibility

    Returns:
        Configured IBFailureSimulator
    """
    simulator = IBFailureSimulator(seed=seed)

    if scenarios:
        for scenario, probability in scenarios.items():
            simulator.enable_scenario(scenario, probability)

    return simulator


def create_test_simulator_all_scenarios(probability: float = 0.1) -> IBFailureSimulator:
    """
    Create a simulator with all scenarios enabled at given probability.

    Useful for stress testing.
    """
    simulator = IBFailureSimulator()

    for scenario in FailureScenario:
        simulator.enable_scenario(scenario, probability)

    return simulator

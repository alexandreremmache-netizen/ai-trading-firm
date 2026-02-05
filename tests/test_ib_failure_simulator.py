"""
Tests for IB Failure Simulator
==============================

Tests cover:
- Failure scenario triggering
- Order state simulation
- Statistics tracking
- Thread safety
- Recovery and resolution
"""

import pytest
import asyncio
from datetime import datetime, timezone

from core.ib_failure_simulator import (
    IBFailureSimulator,
    FailureScenario,
    SimulatedOrder,
    FailureEvent,
    create_failure_simulator,
    create_test_simulator_all_scenarios,
)


class TestFailureScenario:
    """Tests for failure scenario enum."""

    def test_all_scenarios_have_values(self):
        """All scenarios should have string values."""
        for scenario in FailureScenario:
            assert isinstance(scenario.value, str)
            assert len(scenario.value) > 0

    def test_scenario_count(self):
        """Should have expected number of scenarios."""
        assert len(FailureScenario) >= 10


class TestSimulatedOrder:
    """Tests for SimulatedOrder dataclass."""

    def test_order_creation(self):
        """Should create order with default values."""
        order = SimulatedOrder(
            order_id="TEST-001",
            symbol="MES",
            action="BUY",
            quantity=5.0,
            price=5000.0,
        )
        assert order.status == "pending"
        assert order.filled_qty == 0.0
        assert order.avg_fill_price == 0.0

    def test_order_with_values(self):
        """Should accept all values."""
        order = SimulatedOrder(
            order_id="TEST-002",
            symbol="MNQ",
            action="SELL",
            quantity=3.0,
            price=20000.0,
            status="filled",
            filled_qty=3.0,
            avg_fill_price=20001.0,
        )
        assert order.status == "filled"
        assert order.filled_qty == 3.0


class TestIBFailureSimulator:
    """Tests for IBFailureSimulator."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator for testing."""
        return IBFailureSimulator(seed=42)

    def test_initialization(self, simulator):
        """Should initialize with empty state."""
        assert len(simulator._enabled_scenarios) == 0
        assert len(simulator._active_failures) == 0
        assert simulator.is_connected() is True

    def test_enable_scenario(self, simulator):
        """Should enable scenario with probability."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 0.5)

        assert FailureScenario.ORDER_ACCEPTED_NO_FILL in simulator._enabled_scenarios
        assert simulator._enabled_scenarios[FailureScenario.ORDER_ACCEPTED_NO_FILL] == 0.5

    def test_enable_scenario_invalid_probability(self, simulator):
        """Should reject invalid probability."""
        with pytest.raises(ValueError):
            simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.5)

        with pytest.raises(ValueError):
            simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, -0.1)

    def test_disable_scenario(self, simulator):
        """Should disable scenario."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 0.5)
        simulator.disable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL)

        assert FailureScenario.ORDER_ACCEPTED_NO_FILL not in simulator._enabled_scenarios

    def test_disable_all(self, simulator):
        """Should disable all scenarios."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 0.5)
        simulator.enable_scenario(FailureScenario.FILL_WITHOUT_CALLBACK, 0.3)
        simulator.disable_all()

        assert len(simulator._enabled_scenarios) == 0

    def test_should_trigger_no_scenarios(self, simulator):
        """Should not trigger when no scenarios enabled."""
        result = simulator.should_trigger("ORDER-001")
        assert result is False

    def test_should_trigger_with_scenario(self, simulator):
        """Should trigger based on probability."""
        # Set probability to 100% to ensure trigger
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)

        result = simulator.should_trigger("ORDER-001")
        assert result is True

    def test_should_trigger_zero_probability(self, simulator):
        """Should not trigger with 0% probability."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 0.0)

        # Try many times - should never trigger
        for _ in range(100):
            result = simulator.should_trigger("ORDER-001")
            assert result is False

    def test_get_triggered_scenario(self, simulator):
        """Should return active scenario for order."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)
        simulator.should_trigger("ORDER-001")

        scenario = simulator.get_triggered_scenario("ORDER-001")
        assert scenario == FailureScenario.ORDER_ACCEPTED_NO_FILL

    def test_get_triggered_scenario_none(self, simulator):
        """Should return None for non-failing order."""
        scenario = simulator.get_triggered_scenario("UNKNOWN")
        assert scenario is None

    def test_resolve_failure(self, simulator):
        """Should resolve active failure."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)
        simulator.should_trigger("ORDER-001")

        result = simulator.resolve_failure("ORDER-001")

        assert result is True
        assert "ORDER-001" not in simulator._active_failures

    def test_resolve_failure_not_found(self, simulator):
        """Should return False for non-existent failure."""
        result = simulator.resolve_failure("UNKNOWN")
        assert result is False

    def test_statistics_tracking(self, simulator):
        """Should track failure statistics."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)

        for i in range(5):
            simulator.should_trigger(f"ORDER-{i}")

        stats = simulator.get_statistics()

        assert stats["total_failures_triggered"] == 5
        assert stats["failures_by_scenario"]["order_accepted_no_fill"] == 5

    def test_statistics_thread_safe(self, simulator):
        """Should get statistics safely with lock."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)
        simulator.should_trigger("ORDER-001")

        # This should not raise
        stats = simulator.get_statistics()
        assert "active_failures" in stats

    def test_get_active_failures(self, simulator):
        """Should return list of active failures."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)
        simulator.should_trigger("ORDER-001")
        simulator.should_trigger("ORDER-002")

        failures = simulator.get_active_failures()

        assert len(failures) == 2
        assert all(isinstance(f, FailureEvent) for f in failures)

    def test_get_failure_history(self, simulator):
        """Should return failure history."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)
        simulator.should_trigger("ORDER-001")
        simulator.resolve_failure("ORDER-001")

        history = simulator.get_failure_history()

        assert len(history) == 1
        assert history[0].resolved is True

    def test_reset(self, simulator):
        """Should reset all state."""
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)
        simulator.should_trigger("ORDER-001")
        simulator._connection_state = "disconnected"

        simulator.reset()

        assert len(simulator._active_failures) == 0
        assert len(simulator._simulated_orders) == 0
        assert simulator.is_connected() is True

    def test_is_connected(self, simulator):
        """Should track connection state."""
        assert simulator.is_connected() is True

        simulator._connection_state = "disconnected"
        assert simulator.is_connected() is False


class TestAsyncScenarios:
    """Tests for async scenario simulations."""

    @pytest.fixture
    def simulator(self):
        return IBFailureSimulator(seed=42)

    @pytest.mark.asyncio
    async def test_simulate_order_accepted_no_fill(self, simulator):
        """Should simulate zombie order."""
        statuses = []

        def on_status(order_id, status):
            statuses.append((order_id, status))

        await simulator.simulate_order_accepted_no_fill(
            order_id="ORDER-001",
            symbol="MES",
            action="BUY",
            quantity=5.0,
            price=5000.0,
            on_status=on_status,
        )

        # Should receive accepted status but no fill
        assert len(statuses) == 1
        assert statuses[0] == ("ORDER-001", "accepted")

        # Order should be in simulated state
        order = simulator.get_simulated_order("ORDER-001")
        assert order is not None
        assert order.status == "working"

    @pytest.mark.asyncio
    async def test_simulate_fill_without_callback(self, simulator):
        """Should simulate silent fill."""
        statuses = []
        fills = []

        def on_status(order_id, status):
            statuses.append((order_id, status))

        def on_fill(order_id, qty, price):
            fills.append((order_id, qty, price))

        result = await simulator.simulate_fill_without_callback(
            order_id="ORDER-002",
            symbol="MNQ",
            action="BUY",
            quantity=3.0,
            price=20000.0,
            on_status=on_status,
            on_fill=on_fill,
        )

        # Should receive accepted but NOT fill callback
        assert len(statuses) == 1
        assert len(fills) == 0  # No fill callback!

        # But order should actually be filled
        order = simulator.get_simulated_order("ORDER-002")
        assert order.status == "filled"
        assert order.filled_qty == 3.0

    @pytest.mark.asyncio
    async def test_simulate_duplicate_fill_callback(self, simulator):
        """Should simulate duplicate fill callbacks."""
        fills = []

        def on_fill(order_id, qty, price):
            fills.append((order_id, qty, price))

        await simulator.simulate_duplicate_fill_callback(
            order_id="ORDER-003",
            symbol="MES",
            quantity=5.0,
            price=5000.0,
            on_fill=on_fill,
        )

        # Should receive TWO fill callbacks
        assert len(fills) == 2
        assert fills[0] == fills[1]  # Identical

    @pytest.mark.asyncio
    async def test_simulate_disconnect_during_cancel(self, simulator):
        """Should simulate disconnect during cancel."""
        disconnected = []
        reconnected = []

        def on_disconnect():
            disconnected.append(datetime.now(timezone.utc))

        def on_reconnect():
            reconnected.append(datetime.now(timezone.utc))

        await simulator.simulate_disconnect_during_cancel(
            order_id="ORDER-004",
            on_disconnect=on_disconnect,
            reconnect_after_seconds=0.1,
            on_reconnect=on_reconnect,
        )

        assert len(disconnected) == 1
        assert len(reconnected) == 1
        assert simulator.is_connected() is True

    @pytest.mark.asyncio
    async def test_simulate_partial_fill_disconnect(self, simulator):
        """Should simulate partial fill then disconnect."""
        statuses = []
        fills = []
        disconnected = []

        await simulator.simulate_partial_fill_disconnect(
            order_id="ORDER-005",
            symbol="MES",
            action="BUY",
            quantity=10.0,
            price=5000.0,
            partial_fill_pct=0.5,
            on_status=lambda oid, s: statuses.append((oid, s)),
            on_fill=lambda oid, q, p: fills.append((oid, q, p)),
            on_disconnect=lambda: disconnected.append(True),
            reconnect_after_seconds=0.1,
        )

        # Should have partial fill
        assert len(fills) == 1
        assert fills[0][1] == 5.0  # 50% of 10

        # Should have disconnected
        assert len(disconnected) == 1


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_failure_simulator_empty(self):
        """Should create simulator with no scenarios."""
        sim = create_failure_simulator()
        assert len(sim._enabled_scenarios) == 0

    def test_create_failure_simulator_with_scenarios(self):
        """Should create simulator with specified scenarios."""
        scenarios = {
            FailureScenario.ORDER_ACCEPTED_NO_FILL: 0.1,
            FailureScenario.FILL_WITHOUT_CALLBACK: 0.05,
        }
        sim = create_failure_simulator(scenarios=scenarios, seed=42)

        assert len(sim._enabled_scenarios) == 2
        assert sim._enabled_scenarios[FailureScenario.ORDER_ACCEPTED_NO_FILL] == 0.1

    def test_create_test_simulator_all_scenarios(self):
        """Should create simulator with all scenarios enabled."""
        sim = create_test_simulator_all_scenarios(probability=0.1)

        assert len(sim._enabled_scenarios) == len(FailureScenario)
        for scenario in FailureScenario:
            assert sim._enabled_scenarios[scenario] == 0.1


class TestPositionTracking:
    """Tests for position tracking in simulator."""

    @pytest.fixture
    def simulator(self):
        return IBFailureSimulator(seed=42)

    @pytest.mark.asyncio
    async def test_get_actual_position(self, simulator):
        """Should calculate actual position from fills."""
        # Simulate some fills
        simulator._simulated_orders["O1"] = SimulatedOrder(
            order_id="O1", symbol="MES", action="BUY",
            quantity=5.0, price=5000.0,
            status="filled", filled_qty=5.0,
        )
        simulator._simulated_orders["O2"] = SimulatedOrder(
            order_id="O2", symbol="MES", action="BUY",
            quantity=3.0, price=5010.0,
            status="filled", filled_qty=3.0,
        )
        simulator._simulated_orders["O3"] = SimulatedOrder(
            order_id="O3", symbol="MES", action="SELL",
            quantity=2.0, price=5020.0,
            status="filled", filled_qty=2.0,
        )

        position = simulator.get_actual_position("MES")

        assert position == 6.0  # 5 + 3 - 2

    @pytest.mark.asyncio
    async def test_get_actual_position_empty(self, simulator):
        """Should return 0 for no orders."""
        position = simulator.get_actual_position("MES")
        assert position == 0.0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_triggers(self):
        """Should handle concurrent triggers safely."""
        import threading

        simulator = IBFailureSimulator()
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)

        results = []
        errors = []

        def trigger_many(n):
            try:
                for i in range(n):
                    simulator.should_trigger(f"ORDER-{threading.current_thread().name}-{i}")
                    results.append(1)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=trigger_many, args=(20,), name=f"T{i}")
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_statistics(self):
        """Should get statistics safely under concurrent access."""
        import threading

        simulator = IBFailureSimulator()
        simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, 1.0)

        errors = []

        def trigger_and_stats():
            try:
                for i in range(50):
                    simulator.should_trigger(f"ORDER-{i}")
                    simulator.get_statistics()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=trigger_and_stats) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

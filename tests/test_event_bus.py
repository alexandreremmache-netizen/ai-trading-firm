"""
Tests for Event Bus
===================
"""

import pytest
import asyncio

from core.event_bus import EventBus, SignalBarrier
from core.events import (
    Event,
    MarketDataEvent,
    SignalEvent,
    SignalDirection,
    EventType,
)


class TestEventBus:
    """Test EventBus functionality."""

    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing."""
        return EventBus(max_queue_size=100, signal_timeout=1.0, barrier_timeout=2.0)

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self, event_bus):
        """Test basic publish/subscribe."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.subscribe(EventType.MARKET_DATA, handler)

        # Start event bus in background
        bus_task = asyncio.create_task(event_bus.start())

        # Publish event
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=100.05,
        )
        await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Stop event bus
        await event_bus.stop()

        assert len(received_events) == 1
        assert received_events[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_signal_agent_registration(self, event_bus):
        """Test signal agent registration."""
        event_bus.register_signal_agent("MacroAgent")
        event_bus.register_signal_agent("MomentumAgent")

        assert "MacroAgent" in event_bus._signal_agents
        assert "MomentumAgent" in event_bus._signal_agents

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribe functionality."""
        received = []

        async def handler(event: Event):
            received.append(event)

        event_bus.subscribe(EventType.MARKET_DATA, handler)
        event_bus.unsubscribe(EventType.MARKET_DATA, handler)

        bus_task = asyncio.create_task(event_bus.start())

        event = MarketDataEvent(source_agent="test", symbol="AAPL", bid=100.0, ask=100.05)
        await event_bus.publish(event)

        await asyncio.sleep(0.1)
        await event_bus.stop()

        assert len(received) == 0

    def test_queue_size_property(self, event_bus):
        """Test queue size property."""
        assert event_bus.queue_size == 0

    def test_is_running_property(self, event_bus):
        """Test is_running property."""
        assert event_bus.is_running is False


class TestSignalBarrier:
    """Test SignalBarrier synchronization."""

    def test_barrier_creation(self):
        """Test barrier creation."""
        barrier = SignalBarrier(
            expected_agents={"MacroAgent", "MomentumAgent"},
            timeout_seconds=5.0,
        )

        assert len(barrier.expected_agents) == 2
        assert barrier.is_complete() is False

    def test_barrier_add_signal(self):
        """Test adding signals to barrier."""
        barrier = SignalBarrier(
            expected_agents={"MacroAgent", "MomentumAgent"},
            timeout_seconds=5.0,
        )

        signal1 = SignalEvent(
            source_agent="MacroAgent",
            strategy_name="macro",
            symbol="SPY",
            direction=SignalDirection.LONG,
            strength=0.5,
            confidence=0.6,
            rationale="test",
            data_sources=(),
        )

        complete = barrier.add_signal("MacroAgent", signal1)
        assert complete is False
        assert barrier.is_complete() is False

        signal2 = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="SPY",
            direction=SignalDirection.LONG,
            strength=0.7,
            confidence=0.8,
            rationale="test",
            data_sources=(),
        )

        complete = barrier.add_signal("MomentumAgent", signal2)
        assert complete is True
        assert barrier.is_complete() is True

    @pytest.mark.asyncio
    async def test_barrier_wait_complete(self):
        """Test barrier wait when complete."""
        barrier = SignalBarrier(
            expected_agents={"Agent1"},
            timeout_seconds=1.0,
        )

        signal = SignalEvent(
            source_agent="Agent1",
            strategy_name="test",
            symbol="SPY",
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.5,
            rationale="test",
            data_sources=(),
        )

        barrier.add_signal("Agent1", signal)

        signals = await barrier.wait()
        assert "Agent1" in signals

    @pytest.mark.asyncio
    async def test_barrier_wait_timeout(self):
        """Test barrier wait timeout."""
        barrier = SignalBarrier(
            expected_agents={"Agent1", "Agent2"},
            timeout_seconds=0.1,
        )

        signal = SignalEvent(
            source_agent="Agent1",
            strategy_name="test",
            symbol="SPY",
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.5,
            rationale="test",
            data_sources=(),
        )

        barrier.add_signal("Agent1", signal)

        # Should timeout since Agent2 never reports
        signals = await barrier.wait()

        # Should have partial signals
        assert "Agent1" in signals
        assert "Agent2" not in signals

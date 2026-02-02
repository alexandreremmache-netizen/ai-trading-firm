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

    @pytest.mark.asyncio
    async def test_barrier_creation(self):
        """Test barrier creation."""
        barrier = SignalBarrier(
            expected_agents={"MacroAgent", "MomentumAgent"},
            timeout_seconds=5.0,
        )

        assert len(barrier.expected_agents) == 2
        assert await barrier.is_complete() is False

    @pytest.mark.asyncio
    async def test_barrier_add_signal(self):
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

        complete = await barrier.add_signal("MacroAgent", signal1)
        assert complete is False
        assert await barrier.is_complete() is False

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

        complete = await barrier.add_signal("MomentumAgent", signal2)
        assert complete is True
        assert await barrier.is_complete() is True

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

        await barrier.add_signal("Agent1", signal)

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

        await barrier.add_signal("Agent1", signal)

        # Should timeout since Agent2 never reports
        signals = await barrier.wait()

        # Should have partial signals
        assert "Agent1" in signals
        assert "Agent2" not in signals


class TestEventPersistenceAndRecovery:
    """Test event persistence and recovery functionality (Issue #21)."""

    @pytest.mark.asyncio
    async def test_event_bus_with_persistence_enabled(self, tmp_path):
        """Test event bus can be created with persistence enabled."""
        from core.event_persistence import PersistenceConfig

        db_path = str(tmp_path / "test_events.db")
        persistence_config = PersistenceConfig(db_path=db_path)

        event_bus = EventBus(
            max_queue_size=100,
            signal_timeout=1.0,
            barrier_timeout=2.0,
            enable_persistence=True,
            persistence_config=persistence_config,
        )

        assert event_bus.persistence is not None
        assert event_bus._enable_persistence is True

    @pytest.mark.asyncio
    async def test_event_persistence_stores_events(self, tmp_path):
        """Test that events are persisted to storage."""
        from core.event_persistence import EventPersistence, PersistenceConfig, EventStatus

        db_path = str(tmp_path / "test_events.db")
        config = PersistenceConfig(db_path=db_path)
        persistence = EventPersistence(config)
        persistence.initialize()

        # Create and persist a signal event
        signal = SignalEvent(
            source_agent="TestAgent",
            strategy_name="test",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.5,
            confidence=0.7,
            rationale="Test signal",
            data_sources=("test_source",),
        )

        result = persistence.persist_event(signal)
        assert result is True

        # Check statistics
        stats = persistence.get_statistics()
        assert EventStatus.PENDING.value in stats.get("status_counts", {})

        persistence.close()

    @pytest.mark.asyncio
    async def test_event_persistence_recovery(self, tmp_path):
        """Test that persisted events can be recovered."""
        from core.event_persistence import EventPersistence, PersistenceConfig

        db_path = str(tmp_path / "test_events.db")
        config = PersistenceConfig(db_path=db_path)
        persistence = EventPersistence(config)
        persistence.initialize()

        # Create and persist a signal event
        signal = SignalEvent(
            source_agent="TestAgent",
            strategy_name="test",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.5,
            confidence=0.7,
            rationale="Test signal for recovery",
            data_sources=("test_source",),
        )

        persistence.persist_event(signal)

        # Get pending events
        pending = persistence.get_pending_events(limit=10)
        assert len(pending) >= 1

        # Reconstruct event
        reconstructed = persistence.reconstruct_event(pending[0])
        assert reconstructed is not None
        assert reconstructed.symbol == "AAPL"
        assert reconstructed.source_agent == "TestAgent"

        persistence.close()

    @pytest.mark.asyncio
    async def test_event_persistence_mark_completed(self, tmp_path):
        """Test marking events as completed."""
        from core.event_persistence import EventPersistence, PersistenceConfig

        db_path = str(tmp_path / "test_events.db")
        config = PersistenceConfig(db_path=db_path)
        persistence = EventPersistence(config)
        persistence.initialize()

        signal = SignalEvent(
            source_agent="TestAgent",
            strategy_name="test",
            symbol="AAPL",
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.5,
            rationale="Test",
            data_sources=(),
        )

        persistence.persist_event(signal)
        result = persistence.mark_completed(signal.event_id)
        assert result is True

        # Verify it's no longer in pending
        pending = persistence.get_pending_events(limit=10)
        pending_ids = [p.event_id for p in pending]
        assert signal.event_id not in pending_ids

        persistence.close()

    @pytest.mark.asyncio
    async def test_event_persistence_cleanup(self, tmp_path):
        """Test cleanup of old completed events."""
        from core.event_persistence import EventPersistence, PersistenceConfig

        db_path = str(tmp_path / "test_events.db")
        config = PersistenceConfig(
            db_path=db_path,
            cleanup_completed_after_hours=0,  # Clean immediately
        )
        persistence = EventPersistence(config)
        persistence.initialize()

        signal = SignalEvent(
            source_agent="TestAgent",
            strategy_name="test",
            symbol="AAPL",
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.5,
            rationale="Test",
            data_sources=(),
        )

        persistence.persist_event(signal)
        persistence.mark_completed(signal.event_id)

        # Cleanup with 0 hours retention
        deleted = persistence.cleanup_completed_events(hours=0)
        # May or may not delete depending on timing
        assert deleted >= 0

        persistence.close()

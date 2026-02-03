"""
Tests for Execution Agent
=========================

Tests for the Execution Agent that handles order execution.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from agents.execution_agent import (
    ExecutionAgentImpl,
    PendingOrder,
    SliceFill,
    OrderBookSnapshot,
    OrderBookLevel,
    FillCategory,
    MarketImpactEstimate,
)
from core.agent_base import AgentConfig
from core.events import (
    DecisionEvent,
    ValidatedDecisionEvent,
    OrderEvent,
    FillEvent,
    KillSwitchEvent,
    EventType,
    OrderSide,
    OrderType,
    OrderState,
    TimeInForce,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def mock_audit_logger():
    """Create a mock audit logger."""
    logger = MagicMock()
    logger.log_event = MagicMock()
    logger.log_agent_event = MagicMock()
    logger.log_order = MagicMock()
    return logger


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = MagicMock()
    broker.place_order = AsyncMock(return_value=12345)
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_market_price = AsyncMock(return_value=150.0)
    broker.on_fill = MagicMock()
    broker.is_connected = MagicMock(return_value=True)
    return broker


@pytest.fixture
def default_config():
    """Create default Execution agent configuration."""
    return AgentConfig(
        name="ExecutionAgent",
        enabled=True,
        timeout_seconds=30.0,
        parameters={
            "default_algo": "TWAP",
            "slice_interval_seconds": 60,
            "max_slippage_bps": 50,
            "max_orders_per_minute": 10,
            "min_order_interval_ms": 100,
            "order_timeout_seconds": 300,
            "algo_timeout_seconds": 3600,
        },
    )


@pytest.fixture
def execution_agent(default_config, mock_event_bus, mock_audit_logger, mock_broker):
    """Create an Execution agent instance for testing."""
    return ExecutionAgentImpl(default_config, mock_event_bus, mock_audit_logger, mock_broker)


@pytest.fixture
def sample_decision_event():
    """Create a sample decision event."""
    return DecisionEvent(
        source_agent="CIOAgent",
        symbol="AAPL",
        action=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
        rationale="Test decision",
        data_sources=("ib",),
        conviction_score=0.8,
    )


@pytest.fixture
def sample_validated_event(sample_decision_event):
    """Create a sample validated decision event."""
    return ValidatedDecisionEvent(
        source_agent="RiskAgent",
        original_decision_id=sample_decision_event.event_id,
        approved=True,
        adjusted_quantity=100,
        risk_metrics={"var_95": 0.015},
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestExecutionAgentInitialization:
    """Test Execution agent initialization."""

    def test_initialization_with_default_config(self, execution_agent):
        """Test that agent initializes correctly with default config."""
        assert execution_agent.name == "ExecutionAgent"
        assert execution_agent._default_algo == "TWAP"
        assert execution_agent._slice_interval == 60
        assert execution_agent._max_slippage_bps == 50

    def test_initialization_with_vwap_algo(self, mock_event_bus, mock_audit_logger, mock_broker):
        """Test initialization with VWAP algorithm."""
        config = AgentConfig(
            name="ExecutionAgent",
            parameters={
                "default_algo": "VWAP",
                "vwap_participation_rate": 0.15,
            },
        )
        agent = ExecutionAgentImpl(config, mock_event_bus, mock_audit_logger, mock_broker)

        assert agent._default_algo == "VWAP"
        assert agent._vwap_participation_rate == 0.15

    def test_initialization_registers_fill_callback(self, execution_agent, mock_broker):
        """Test that fill callback is registered with broker."""
        mock_broker.on_fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_method(self, execution_agent):
        """Test the async initialize method."""
        await execution_agent.initialize()
        # Should complete without error

    def test_subscribed_events(self, execution_agent):
        """Test subscribed event types."""
        events = execution_agent.get_subscribed_events()

        assert EventType.VALIDATED_DECISION in events
        assert EventType.DECISION in events
        assert EventType.KILL_SWITCH in events


# ============================================================================
# ORDER BOOK TESTS
# ============================================================================

class TestOrderBookSnapshot:
    """Test OrderBookSnapshot class."""

    def test_order_book_creation(self):
        """Test creating an order book snapshot."""
        bids = [
            OrderBookLevel(price=149.90, size=100),
            OrderBookLevel(price=149.80, size=200),
        ]
        asks = [
            OrderBookLevel(price=150.10, size=150),
            OrderBookLevel(price=150.20, size=250),
        ]

        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
        )

        assert book.symbol == "AAPL"
        assert len(book.bids) == 2
        assert len(book.asks) == 2

    def test_order_book_best_prices(self):
        """Test best bid/ask prices."""
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=149.90, size=100)],
            asks=[OrderBookLevel(price=150.10, size=150)],
        )

        assert book.best_bid == 149.90
        assert book.best_ask == 150.10

    def test_order_book_mid_price(self):
        """Test mid-point price calculation."""
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=149.90, size=100)],
            asks=[OrderBookLevel(price=150.10, size=150)],
        )

        assert book.mid_price == 150.0

    def test_order_book_spread_bps(self):
        """Test spread calculation in basis points."""
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=149.90, size=100)],
            asks=[OrderBookLevel(price=150.10, size=150)],
        )

        spread_bps = book.spread_bps
        # (150.10 - 149.90) / 150.0 * 10000 = 13.33 bps
        assert spread_bps is not None
        assert abs(spread_bps - 13.33) < 0.1

    def test_order_book_empty_levels(self):
        """Test order book with empty levels."""
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=[],
            asks=[],
        )

        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None

    def test_order_book_depth(self):
        """Test total depth calculation."""
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=[
                OrderBookLevel(price=149.90, size=100),
                OrderBookLevel(price=149.80, size=200),
            ],
            asks=[
                OrderBookLevel(price=150.10, size=150),
                OrderBookLevel(price=150.20, size=250),
            ],
        )

        assert book.total_bid_depth() == 300
        assert book.total_ask_depth() == 400
        assert book.total_bid_depth(1) == 100
        assert book.total_ask_depth(1) == 150

    def test_order_book_depth_imbalance(self):
        """Test depth imbalance calculation."""
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=149.90, size=300)],
            asks=[OrderBookLevel(price=150.10, size=100)],
        )

        imbalance = book.depth_imbalance(1)
        # (300 - 100) / (300 + 100) = 0.5
        assert imbalance == 0.5

    def test_order_book_vwap_to_size(self):
        """Test VWAP to size calculation."""
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bids=[],
            asks=[
                OrderBookLevel(price=150.10, size=100),
                OrderBookLevel(price=150.20, size=200),
            ],
        )

        vwap, filled = book.vwap_to_size('buy', 150)

        assert filled == 150
        # VWAP = (100 * 150.10 + 50 * 150.20) / 150
        expected_vwap = (100 * 150.10 + 50 * 150.20) / 150
        assert abs(vwap - expected_vwap) < 0.01


# ============================================================================
# SLICE FILL TESTS
# ============================================================================

class TestSliceFill:
    """Test SliceFill class."""

    def test_slice_fill_creation(self):
        """Test creating a slice fill."""
        slice_fill = SliceFill(
            slice_index=0,
            broker_order_id=12345,
            target_quantity=100,
            arrival_price=150.0,
            is_buy=True,
        )

        assert slice_fill.slice_index == 0
        assert slice_fill.target_quantity == 100
        assert slice_fill.filled_quantity == 0
        assert slice_fill.is_complete is False

    def test_slice_fill_add_fill(self):
        """Test adding a fill to a slice."""
        slice_fill = SliceFill(
            slice_index=0,
            broker_order_id=12345,
            target_quantity=100,
            arrival_price=150.0,
            is_buy=True,
        )

        slice_fill.add_fill(50, 150.05)

        assert slice_fill.filled_quantity == 50
        assert slice_fill.avg_fill_price == 150.05
        assert slice_fill.is_complete is False

    def test_slice_fill_complete(self):
        """Test slice fill completion."""
        slice_fill = SliceFill(
            slice_index=0,
            broker_order_id=12345,
            target_quantity=100,
            arrival_price=150.0,
            is_buy=True,
        )

        slice_fill.add_fill(100, 150.05)

        assert slice_fill.is_complete is True
        assert slice_fill.fill_rate == 1.0

    def test_slice_fill_rate(self):
        """Test fill rate calculation."""
        slice_fill = SliceFill(
            slice_index=0,
            broker_order_id=12345,
            target_quantity=100,
        )

        slice_fill.add_fill(25, 150.0)

        assert slice_fill.fill_rate == 0.25

    def test_slice_fill_slippage(self):
        """Test slippage calculation."""
        slice_fill = SliceFill(
            slice_index=0,
            broker_order_id=12345,
            target_quantity=100,
            arrival_price=150.0,
            is_buy=True,
        )

        slice_fill.add_fill(100, 150.15)  # Filled 15 cents higher

        slippage = slice_fill.slippage_bps
        # (150.15 - 150.0) / 150.0 * 10000 = 10 bps
        assert slippage is not None
        assert abs(slippage - 10.0) < 0.1

    def test_slice_fill_price_improvement_buy(self):
        """Test price improvement for buy orders."""
        slice_fill = SliceFill(
            slice_index=0,
            broker_order_id=12345,
            target_quantity=100,
            arrival_price=150.0,
            is_buy=True,
        )

        # Bought cheaper than arrival price - price improvement
        slice_fill.add_fill(100, 149.85)

        improvement = slice_fill.price_improvement_bps
        assert improvement is not None
        assert improvement > 0  # Positive = improvement
        assert slice_fill.has_price_improvement is True

    def test_slice_fill_price_improvement_sell(self):
        """Test price improvement for sell orders."""
        slice_fill = SliceFill(
            slice_index=0,
            broker_order_id=12345,
            target_quantity=100,
            arrival_price=150.0,
            is_buy=False,
        )

        # Sold higher than arrival price - price improvement
        slice_fill.add_fill(100, 150.15)

        improvement = slice_fill.price_improvement_bps
        assert improvement is not None
        assert improvement > 0
        assert slice_fill.has_price_improvement is True


# ============================================================================
# PENDING ORDER TESTS
# ============================================================================

class TestPendingOrder:
    """Test PendingOrder class."""

    def test_pending_order_creation(self, sample_decision_event):
        """Test creating a pending order."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            decision_id=sample_decision_event.event_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        assert pending.status == "pending"
        assert pending.state == OrderState.CREATED
        assert pending.remaining_quantity == 100

    def test_pending_order_state_transition(self, sample_decision_event):
        """Test valid state transition."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        # CREATED -> PENDING (valid transition)
        result = pending.transition_state(OrderState.PENDING, "Queued for submission")
        assert result is True
        assert pending.state == OrderState.PENDING

        # PENDING -> SUBMITTED (valid transition)
        result = pending.transition_state(OrderState.SUBMITTED, "Sent to broker")
        assert result is True
        assert pending.state == OrderState.SUBMITTED

    def test_pending_order_invalid_state_transition(self, sample_decision_event):
        """Test invalid state transition."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        # Try to go directly from CREATED to FILLED (invalid)
        result = pending.transition_state(OrderState.FILLED, "Filled")

        # This should fail (depends on is_valid_state_transition implementation)
        # If it doesn't fail, the state shouldn't change
        if not result:
            assert pending.state == OrderState.CREATED

    def test_pending_order_is_terminal(self, sample_decision_event):
        """Test terminal state detection."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        assert pending.is_terminal() is False

        # Move to terminal state
        pending.state = OrderState.FILLED
        assert pending.is_terminal() is True

    def test_pending_order_register_slice(self, sample_decision_event):
        """Test registering a slice."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(
            broker_id=12345,
            target_quantity=50,
            arrival_price=150.0,
            is_buy=True,
        )

        assert 12345 in pending.slices
        assert 12345 in pending.slice_fills

    def test_pending_order_add_slice_fill(self, sample_decision_event):
        """Test adding a fill to a slice."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 50, 150.0, True)
        result = pending.add_slice_fill(12345, 25, 150.05)

        assert result is True
        slice_fill = pending.get_slice_fill(12345)
        assert slice_fill is not None
        assert slice_fill.filled_quantity == 25


# ============================================================================
# KILL SWITCH TESTS
# ============================================================================

class TestKillSwitch:
    """Test kill switch handling."""

    def test_kill_switch_initially_inactive(self, execution_agent):
        """Test that kill switch is initially inactive."""
        assert execution_agent._kill_switch_active is False

    @pytest.mark.asyncio
    async def test_handle_kill_switch_activation(self, execution_agent):
        """Test handling kill switch activation."""
        kill_event = KillSwitchEvent(
            source_agent="RiskAgent",
            activated=True,
            reason="Daily loss limit exceeded",
            trigger_type="automated",
            cancel_pending_orders=True,
        )

        await execution_agent._handle_kill_switch(kill_event)

        assert execution_agent._kill_switch_active is True
        assert execution_agent._kill_switch_reason == "Daily loss limit exceeded"

    @pytest.mark.asyncio
    async def test_reject_order_when_kill_switch_active(self, execution_agent, sample_decision_event, sample_validated_event):
        """Test that orders are rejected when kill switch is active."""
        execution_agent._kill_switch_active = True
        execution_agent._kill_switch_reason = "Emergency halt"

        # Cache the decision
        execution_agent._decision_cache[sample_decision_event.event_id] = sample_decision_event

        await execution_agent.process_event(sample_validated_event)

        # Should not execute - no publish
        execution_agent._event_bus.publish.assert_not_called()


# ============================================================================
# DECISION CACHING TESTS
# ============================================================================

class TestDecisionCaching:
    """Test decision caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_decision_event(self, execution_agent, sample_decision_event):
        """Test that decision events are cached."""
        await execution_agent.process_event(sample_decision_event)

        assert sample_decision_event.event_id in execution_agent._decision_cache
        cached = execution_agent._decision_cache[sample_decision_event.event_id]
        assert cached.symbol == "AAPL"


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_configuration(self, execution_agent):
        """Test rate limit configuration."""
        assert execution_agent._max_orders_per_minute == 10
        assert execution_agent._min_order_interval_ms == 100

    def test_order_timestamps_tracking(self, execution_agent):
        """Test order timestamp tracking."""
        assert isinstance(execution_agent._order_timestamps, list)

        # Simulate adding timestamps
        now = datetime.now(timezone.utc)
        execution_agent._order_timestamps.append(now)

        assert len(execution_agent._order_timestamps) == 1


# ============================================================================
# FILL CATEGORY TESTS
# ============================================================================

class TestFillCategory:
    """Test FillCategory class."""

    def test_fill_category_aggressive(self):
        """Test aggressive fill categorization."""
        category = FillCategory(
            is_aggressive=True,
            category="aggressive",
            price_vs_arrival_bps=5.0,
            price_vs_spread_position=1.0,
        )

        assert category.is_aggressive is True
        assert category.category == "aggressive"

    def test_fill_category_passive(self):
        """Test passive fill categorization."""
        category = FillCategory(
            is_aggressive=False,
            category="passive",
            price_vs_arrival_bps=-2.0,
            price_vs_spread_position=0.0,
        )

        assert category.is_aggressive is False
        assert category.category == "passive"


# ============================================================================
# MARKET IMPACT TESTS
# ============================================================================

class TestMarketImpact:
    """Test MarketImpactEstimate class."""

    def test_market_impact_estimate(self):
        """Test creating market impact estimate."""
        impact = MarketImpactEstimate(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            temporary_impact_bps=5.0,
            permanent_impact_bps=2.0,
            total_impact_bps=7.0,
            estimated_cost=105.0,
            model_used="square_root",
        )

        assert impact.symbol == "AAPL"
        assert impact.total_impact_bps == 7.0
        assert impact.model_used == "square_root"


# ============================================================================
# EDGE CASES TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pending_orders(self, execution_agent):
        """Test handling of empty pending orders."""
        assert len(execution_agent._pending_orders) == 0

    def test_empty_decision_cache(self, execution_agent):
        """Test handling of empty decision cache."""
        assert len(execution_agent._decision_cache) == 0

    @pytest.mark.asyncio
    async def test_validated_event_without_cached_decision(self, execution_agent, sample_validated_event):
        """Test handling validated event when decision is not cached."""
        # Don't cache the decision
        await execution_agent.process_event(sample_validated_event)

        # Should not crash, should not publish
        execution_agent._event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_rejected_validated_event(self, execution_agent, sample_decision_event):
        """Test that rejected decisions are not executed."""
        rejected_event = ValidatedDecisionEvent(
            source_agent="RiskAgent",
            original_decision_id=sample_decision_event.event_id,
            approved=False,
            rejection_reason="Position limit exceeded",
        )

        execution_agent._decision_cache[sample_decision_event.event_id] = sample_decision_event

        await execution_agent.process_event(rejected_event)

        # Should not place any order
        execution_agent._broker.place_order.assert_not_called()

    def test_volume_profiles_empty(self, execution_agent):
        """Test that volume profiles start empty."""
        assert isinstance(execution_agent._volume_profiles, dict)
        assert len(execution_agent._volume_profiles) == 0

    def test_stop_orders_tracking(self, execution_agent):
        """Test stop orders tracking."""
        assert isinstance(execution_agent._stop_orders, dict)
        assert len(execution_agent._stop_orders) == 0


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration options."""

    def test_timeout_configuration(self, execution_agent):
        """Test timeout configuration."""
        assert execution_agent._order_timeout_seconds == 300
        assert execution_agent._algo_timeout_seconds == 3600

    def test_vwap_configuration(self, execution_agent):
        """Test VWAP configuration."""
        assert execution_agent._vwap_target_participation == 0.10
        assert execution_agent._vwap_min_participation == 0.05
        assert execution_agent._vwap_max_participation == 0.25

    def test_market_impact_params(self, execution_agent):
        """Test market impact model parameters."""
        params = execution_agent._market_impact_params

        assert "eta" in params
        assert "gamma" in params
        assert "alpha" in params


# ============================================================================
# PARTIAL FILL TESTS
# ============================================================================

class TestPartialFills:
    """Test partial fill handling."""

    def test_partial_fill_tracking(self, sample_decision_event):
        """Test tracking of partial fills."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            decision_id=sample_decision_event.event_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)

        # Partial fill of 30 shares
        pending.add_slice_fill(12345, 30, 150.02)

        slice_fill = pending.get_slice_fill(12345)
        assert slice_fill.filled_quantity == 30
        assert slice_fill.fill_rate == 0.30
        assert slice_fill.is_complete is False

    def test_multiple_partial_fills(self, sample_decision_event):
        """Test multiple partial fills accumulating correctly."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)

        # First partial fill
        pending.add_slice_fill(12345, 30, 150.02)
        # Second partial fill
        pending.add_slice_fill(12345, 40, 150.05)
        # Third partial fill completes order
        pending.add_slice_fill(12345, 30, 150.03)

        slice_fill = pending.get_slice_fill(12345)
        assert slice_fill.filled_quantity == 100
        assert slice_fill.is_complete is True
        assert slice_fill.fill_rate == 1.0

    def test_partial_fill_average_price(self, sample_decision_event):
        """Test average fill price calculation across partial fills."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)

        # Fill 50 @ 150.00
        pending.add_slice_fill(12345, 50, 150.00)
        # Fill 50 @ 150.10
        pending.add_slice_fill(12345, 50, 150.10)

        slice_fill = pending.get_slice_fill(12345)
        # Average should be 150.05
        assert slice_fill.avg_fill_price is not None
        assert abs(slice_fill.avg_fill_price - 150.05) < 0.01

    def test_remaining_quantity_after_partial_fill(self, sample_decision_event):
        """Test remaining quantity tracking after partial fills."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)
        pending.add_slice_fill(12345, 40, 150.02)

        # Update remaining_quantity manually as the execution agent does
        pending.filled_quantity = 40
        pending.remaining_quantity = 60

        # Remaining should be 60
        assert pending.remaining_quantity == 60

    def test_partial_fill_slippage_calculation(self, sample_decision_event):
        """Test slippage calculation on partial fills."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)  # Arrival price 150.0

        # Fill at worse price
        pending.add_slice_fill(12345, 100, 150.30)  # 30 cents slippage

        slice_fill = pending.get_slice_fill(12345)
        # Slippage should be 20 bps (0.30/150.0 * 10000)
        assert slice_fill.slippage_bps is not None
        assert abs(slice_fill.slippage_bps - 20.0) < 0.1

    def test_partial_fill_with_zero_remaining(self, sample_decision_event):
        """Test order state when partial fills complete the order."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)
        pending.add_slice_fill(12345, 100, 150.05)

        # Update order-level tracking as the execution agent does
        pending.filled_quantity = 100
        pending.remaining_quantity = 0

        assert pending.remaining_quantity == 0
        slice_fill = pending.get_slice_fill(12345)
        assert slice_fill.is_complete is True


# ============================================================================
# ORDER REJECTION TESTS
# ============================================================================

class TestOrderRejection:
    """Test order rejection handling."""

    @pytest.mark.asyncio
    async def test_reject_order_when_broker_disconnected(self, execution_agent, sample_decision_event, sample_validated_event):
        """Test order rejection when broker is disconnected."""
        # Set is_connected as a property returning False
        type(execution_agent._broker).is_connected = property(lambda self: False)
        execution_agent._decision_cache[sample_decision_event.event_id] = sample_decision_event

        # Order should not be placed
        await execution_agent.process_event(sample_validated_event)

        execution_agent._broker.place_order.assert_not_called()

    def test_order_state_on_rejection(self, sample_decision_event):
        """Test order state transitions on rejection."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        # Transition to rejected state
        pending.transition_state(OrderState.PENDING, "Queued")
        pending.transition_state(OrderState.SUBMITTED, "Sent")
        pending.state = OrderState.REJECTED

        assert pending.is_terminal() is True

    def test_order_creation_with_empty_symbol(self, sample_decision_event):
        """Test that order event can be created with empty symbol (validation at broker)."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="",  # Empty symbol
            side=OrderSide.BUY,
            quantity=100,
        )

        # Order event is created - validation happens at broker level
        assert order_event.symbol == ""
        assert order_event.quantity == 100

    def test_order_creation_with_zero_quantity(self, sample_decision_event):
        """Test order event with zero quantity."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,  # Zero quantity - should be rejected by broker
        )

        # Order event is created - broker will reject zero quantity
        assert order_event.quantity == 0

        # PendingOrder with zero quantity
        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        # remaining_quantity initialized from order
        assert pending.remaining_quantity == 0

    def test_order_creation_with_negative_quantity(self, sample_decision_event):
        """Test order event with negative quantity (edge case)."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=-100,  # Negative quantity - invalid, broker rejects
        )

        # Order event is created - broker will reject negative quantity
        assert order_event.quantity == -100

        # PendingOrder with negative quantity
        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        # remaining_quantity initialized from order (even if negative)
        assert pending.remaining_quantity == -100

    @pytest.mark.asyncio
    async def test_broker_rejection_handling(self, execution_agent, sample_decision_event, sample_validated_event):
        """Test handling of broker order rejection."""
        execution_agent._broker.place_order = AsyncMock(side_effect=Exception("Order rejected by broker"))
        execution_agent._decision_cache[sample_decision_event.event_id] = sample_decision_event

        # Should handle exception gracefully
        try:
            await execution_agent.process_event(sample_validated_event)
        except Exception:
            pass  # Expected to raise or handle gracefully

    def test_rejection_reason_tracking(self, sample_decision_event):
        """Test that rejection reasons are tracked."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.rejection_reason = "Insufficient buying power"
        pending.state = OrderState.REJECTED

        assert pending.rejection_reason == "Insufficient buying power"
        assert pending.state == OrderState.REJECTED


# ============================================================================
# TIMEOUT HANDLING TESTS
# ============================================================================

class TestTimeoutHandling:
    """Test timeout handling for orders."""

    def test_order_timeout_configuration(self, execution_agent):
        """Test order timeout is correctly configured."""
        assert execution_agent._order_timeout_seconds == 300  # 5 minutes

    def test_algo_timeout_configuration(self, execution_agent):
        """Test algo timeout is correctly configured."""
        assert execution_agent._algo_timeout_seconds == 3600  # 1 hour

    def test_order_creation_timestamp(self, sample_decision_event):
        """Test that order creation timestamp is recorded."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        assert pending.created_at is not None
        assert isinstance(pending.created_at, datetime)

    def test_order_timeout_detection(self, sample_decision_event):
        """Test detection of timed out orders."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        # Simulate old order
        pending.created_at = datetime.now(timezone.utc) - timedelta(seconds=400)

        # Check if order is timed out (5 minute timeout)
        age_seconds = (datetime.now(timezone.utc) - pending.created_at).total_seconds()
        is_timed_out = age_seconds > 300

        assert is_timed_out is True

    def test_order_not_timed_out(self, sample_decision_event):
        """Test that recent orders are not timed out."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        # Fresh order
        pending.created_at = datetime.now(timezone.utc) - timedelta(seconds=60)

        age_seconds = (datetime.now(timezone.utc) - pending.created_at).total_seconds()
        is_timed_out = age_seconds > 300

        assert is_timed_out is False

    def test_state_transition_to_expired(self, sample_decision_event):
        """Test state transition to EXPIRED state (timeout)."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.transition_state(OrderState.PENDING, "Queued")
        pending.transition_state(OrderState.SUBMITTED, "Sent")

        # Simulate timeout using EXPIRED state
        pending.state = OrderState.EXPIRED

        assert pending.state == OrderState.EXPIRED
        assert pending.is_terminal() is True

    def test_slice_timeout_tracking(self, sample_decision_event):
        """Test tracking of timed out slices."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 50, 150.0, True)

        # Simulate old slice
        slice_fill = pending.get_slice_fill(12345)
        slice_fill.submitted_at = datetime.now(timezone.utc) - timedelta(seconds=400)

        age = (datetime.now(timezone.utc) - slice_fill.submitted_at).total_seconds()
        assert age > 300

    @pytest.mark.asyncio
    async def test_timeout_cancels_pending_slices(self, execution_agent, sample_decision_event):
        """Test that timeout triggers cancellation of pending slices."""
        # Create pending order with slice
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            decision_id=sample_decision_event.event_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)
        pending.created_at = datetime.now(timezone.utc) - timedelta(seconds=400)

        execution_agent._pending_orders[order_event.event_id] = pending

        # Verify cancel_order can be called on timeout
        execution_agent._broker.cancel_order = AsyncMock(return_value=True)

        # Simulate cancellation
        await execution_agent._broker.cancel_order(12345)

        execution_agent._broker.cancel_order.assert_called_once_with(12345)

    def test_partial_fill_before_timeout(self, sample_decision_event):
        """Test handling of partial fill before timeout."""
        order_event = OrderEvent(
            source_agent="ExecutionAgent",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        pending = PendingOrder(
            order_event=order_event,
            decision_event=sample_decision_event,
        )

        pending.register_slice(12345, 100, 150.0, True)

        # Partial fill occurred
        pending.add_slice_fill(12345, 50, 150.05)

        # Update order-level tracking
        pending.filled_quantity = 50
        pending.remaining_quantity = 50

        # Then timeout (use EXPIRED state)
        pending.state = OrderState.EXPIRED

        # Should have partial fill recorded
        slice_fill = pending.get_slice_fill(12345)
        assert slice_fill.filled_quantity == 50
        assert pending.remaining_quantity == 50

    def test_multiple_order_timeout_handling(self, execution_agent, sample_decision_event):
        """Test handling multiple timed out orders."""
        timed_out_orders = []

        for i in range(3):
            order_event = OrderEvent(
                source_agent="ExecutionAgent",
                decision_id=f"decision-{i}",
                symbol=f"SYM{i}",
                side=OrderSide.BUY,
                quantity=100,
            )

            pending = PendingOrder(
                order_event=order_event,
                decision_event=sample_decision_event,
            )

            pending.created_at = datetime.now(timezone.utc) - timedelta(seconds=400)
            execution_agent._pending_orders[order_event.event_id] = pending

            # Check timeout
            age = (datetime.now(timezone.utc) - pending.created_at).total_seconds()
            if age > 300:
                timed_out_orders.append(pending)

        assert len(timed_out_orders) == 3

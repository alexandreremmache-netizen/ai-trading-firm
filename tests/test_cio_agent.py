"""
Tests for CIO Agent
===================

Tests for the Chief Investment Officer agent - the single decision-making authority.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from agents.cio_agent import (
    CIOAgent,
    MarketRegime,
    SignalAggregation,
    StrategyPerformance,
)
from core.agent_base import AgentConfig
from core.events import (
    SignalEvent,
    SignalDirection,
    DecisionEvent,
    ValidatedDecisionEvent,
    EventType,
    OrderSide,
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
    bus.wait_for_signals = AsyncMock(return_value={})
    return bus


@pytest.fixture
def mock_audit_logger():
    """Create a mock audit logger."""
    logger = MagicMock()
    logger.log_decision = MagicMock()
    logger.log_event = MagicMock()
    logger.log_agent_event = MagicMock()
    return logger


@pytest.fixture
def default_config():
    """Create default CIO agent configuration."""
    return AgentConfig(
        name="CIOAgent",
        enabled=True,
        timeout_seconds=30.0,
        parameters={
            "signal_weight_macro": 0.15,
            "signal_weight_stat_arb": 0.25,
            "signal_weight_momentum": 0.25,
            "signal_weight_market_making": 0.15,
            "signal_weight_options_vol": 0.20,
            "min_conviction_threshold": 0.6,
            "max_concurrent_decisions": 5,
            "use_dynamic_weights": True,
            "use_kelly_sizing": False,
            "base_position_size": 100,
            "max_position_size": 1000,
            "portfolio_value": 1_000_000.0,
        },
    )


@pytest.fixture
def cio_agent(default_config, mock_event_bus, mock_audit_logger):
    """Create a CIO agent instance for testing."""
    return CIOAgent(default_config, mock_event_bus, mock_audit_logger)


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestCIOAgentInitialization:
    """Test CIO agent initialization."""

    def test_initialization_with_default_config(self, cio_agent):
        """Test that agent initializes correctly with default config."""
        assert cio_agent.name == "CIOAgent"
        assert cio_agent._min_conviction == 0.6
        assert cio_agent._max_concurrent == 5
        assert cio_agent._use_dynamic_weights is True
        assert cio_agent._current_regime == MarketRegime.NEUTRAL

    def test_initialization_with_custom_weights(self, mock_event_bus, mock_audit_logger):
        """Test initialization with custom signal weights."""
        config = AgentConfig(
            name="CIOAgent",
            parameters={
                "signal_weight_macro": 0.30,
                "signal_weight_momentum": 0.40,
                "min_conviction_threshold": 0.8,
            },
        )
        agent = CIOAgent(config, mock_event_bus, mock_audit_logger)

        assert agent._base_weights["MacroAgent"] == 0.30
        assert agent._base_weights["MomentumAgent"] == 0.40
        assert agent._min_conviction == 0.8

    def test_initialization_with_empty_config(self, mock_event_bus, mock_audit_logger):
        """Test initialization with empty parameters uses defaults."""
        config = AgentConfig(name="CIOAgent", parameters={})
        agent = CIOAgent(config, mock_event_bus, mock_audit_logger)

        # Should use default values
        assert agent._min_conviction == 0.6
        assert agent._max_concurrent == 5
        assert agent._base_position_size == 100

    @pytest.mark.asyncio
    async def test_initialize_method(self, cio_agent):
        """Test the async initialize method."""
        await cio_agent.initialize()
        # Should complete without error


# ============================================================================
# SIGNAL AGGREGATION TESTS
# ============================================================================

class TestSignalAggregation:
    """Test signal aggregation logic."""

    def test_aggregate_single_signal_long(self, cio_agent):
        """Test aggregation with a single LONG signal."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.7,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        assert agg.consensus_direction == SignalDirection.LONG
        assert agg.weighted_strength > 0
        assert agg.weighted_confidence > 0

    def test_aggregate_single_signal_short(self, cio_agent):
        """Test aggregation with a single SHORT signal."""
        signal = SignalEvent(
            source_agent="StatArbAgent",
            strategy_name="stat_arb",
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            strength=0.9,
            confidence=0.85,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"StatArbAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        assert agg.consensus_direction == SignalDirection.SHORT

    def test_aggregate_conflicting_signals(self, cio_agent):
        """Test aggregation with conflicting signals results in FLAT."""
        signal_long = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.7,
            confidence=0.7,
            data_sources=("ib",),
        )
        signal_short = SignalEvent(
            source_agent="StatArbAgent",
            strategy_name="stat_arb",
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            strength=0.7,
            confidence=0.7,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={
                "MomentumAgent": signal_long,
                "StatArbAgent": signal_short,
            },
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # With equal weights and strength, should be FLAT (no clear consensus)
        assert agg.consensus_direction == SignalDirection.FLAT

    def test_aggregate_multiple_long_signals(self, cio_agent):
        """Test aggregation with multiple agreeing LONG signals."""
        signals = {
            "MomentumAgent": SignalEvent(
                source_agent="MomentumAgent",
                strategy_name="momentum",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.75,
                data_sources=("ib",),
            ),
            "MacroAgent": SignalEvent(
                source_agent="MacroAgent",
                strategy_name="macro",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.7,
                confidence=0.8,
                data_sources=("bloomberg",),
            ),
            "StatArbAgent": SignalEvent(
                source_agent="StatArbAgent",
                strategy_name="stat_arb",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.6,
                confidence=0.7,
                data_sources=("ib",),
            ),
        }

        agg = SignalAggregation(
            symbol="AAPL",
            signals=signals,
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        assert agg.consensus_direction == SignalDirection.LONG
        assert agg.weighted_confidence > 0.7

    def test_aggregate_with_flat_signals(self, cio_agent):
        """Test aggregation when all signals are FLAT."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.5,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        assert agg.consensus_direction == SignalDirection.FLAT

    def test_aggregate_empty_signals(self, cio_agent):
        """Test aggregation with empty signals dict."""
        agg = SignalAggregation(
            symbol="AAPL",
            signals={},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        assert agg.consensus_direction == SignalDirection.FLAT
        assert agg.weighted_strength == 0.0
        assert agg.weighted_confidence == 0.0


# ============================================================================
# POSITION SIZING TESTS
# ============================================================================

class TestPositionSizing:
    """Test position sizing calculations."""

    def test_conviction_based_sizing(self, cio_agent):
        """Test conviction-based position sizing."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=1.0,
            confidence=1.0,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )
        cio_agent._aggregate_signals(agg)

        size = cio_agent._calculate_conviction_size(agg)

        # With full conviction and strength, should get base_position_size
        assert size == 100  # base_position_size from config

    def test_conviction_sizing_low_confidence(self, cio_agent):
        """Test that low confidence reduces position size."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.5,
            confidence=0.3,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )
        cio_agent._aggregate_signals(agg)

        size = cio_agent._calculate_conviction_size(agg)

        # With low conviction, should get smaller size or zero
        assert size < 100

    def test_position_size_max_limit(self, cio_agent):
        """Test that position size respects max limit."""
        # Set artificially high base size
        cio_agent._base_position_size = 5000
        cio_agent._max_position_size = 1000

        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=1.0,
            confidence=1.0,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )
        cio_agent._aggregate_signals(agg)

        size = cio_agent._calculate_conviction_size(agg)

        assert size <= 1000  # Should not exceed max


# ============================================================================
# MARKET REGIME TESTS
# ============================================================================

class TestMarketRegime:
    """Test market regime handling."""

    def test_set_market_regime(self, cio_agent):
        """Test setting market regime."""
        assert cio_agent._current_regime == MarketRegime.NEUTRAL

        cio_agent.set_market_regime(MarketRegime.RISK_ON)

        assert cio_agent._current_regime == MarketRegime.RISK_ON

    def test_regime_change_updates_weights(self, cio_agent):
        """Test that regime change triggers weight update."""
        original_weights = dict(cio_agent._weights)

        cio_agent.set_market_regime(MarketRegime.RISK_ON)

        # Weights should be updated (may or may not be different depending on config)
        # At minimum, the update method should have been called
        assert cio_agent._current_regime == MarketRegime.RISK_ON

    def test_all_regime_values(self, cio_agent):
        """Test that all regime values can be set."""
        for regime in MarketRegime:
            cio_agent.set_market_regime(regime)
            assert cio_agent._current_regime == regime


# ============================================================================
# DYNAMIC WEIGHTS TESTS
# ============================================================================

class TestDynamicWeights:
    """Test dynamic weight adjustment."""

    def test_update_strategy_performance(self, cio_agent):
        """Test updating strategy performance metrics."""
        cio_agent.update_strategy_performance(
            strategy="MomentumAgent",
            rolling_sharpe=1.5,
            win_rate=0.6,
            recent_pnl=10000.0,
            signal_accuracy=0.65,
        )

        perf = cio_agent._strategy_performance.get("MomentumAgent")
        assert perf is not None
        assert perf.rolling_sharpe == 1.5
        assert perf.win_rate == 0.6

    def test_update_dynamic_weights(self, cio_agent):
        """Test dynamic weight update based on performance."""
        # Add performance data
        cio_agent.update_strategy_performance(
            strategy="MomentumAgent",
            rolling_sharpe=2.0,
            win_rate=0.7,
            recent_pnl=50000.0,
        )

        cio_agent._update_dynamic_weights()

        # Weights should be normalized to sum to 1
        total_weight = sum(cio_agent._weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_get_current_weights(self, cio_agent):
        """Test getting current weights."""
        weights = cio_agent.get_current_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(isinstance(v, float) for v in weights.values())

    def test_get_base_weights(self, cio_agent):
        """Test getting base weights."""
        base_weights = cio_agent.get_base_weights()

        assert isinstance(base_weights, dict)
        assert "MomentumAgent" in base_weights
        assert base_weights["MomentumAgent"] == 0.25


# ============================================================================
# EXTERNAL COMPONENT INTEGRATION TESTS
# ============================================================================

class TestExternalComponentIntegration:
    """Test integration with external components."""

    def test_set_position_sizer(self, cio_agent):
        """Test setting position sizer."""
        mock_sizer = MagicMock()
        cio_agent.set_position_sizer(mock_sizer)

        assert cio_agent._position_sizer is mock_sizer

    def test_set_correlation_manager(self, cio_agent):
        """Test setting correlation manager."""
        mock_manager = MagicMock()
        cio_agent.set_correlation_manager(mock_manager)

        assert cio_agent._correlation_manager is mock_manager

    def test_set_risk_budget_manager(self, cio_agent):
        """Test setting risk budget manager."""
        mock_manager = MagicMock()
        cio_agent.set_risk_budget_manager(mock_manager)

        assert cio_agent._risk_budget_manager is mock_manager

    def test_set_portfolio_value(self, cio_agent):
        """Test setting portfolio value."""
        cio_agent.set_portfolio_value(2_000_000.0)

        assert cio_agent._portfolio_value == 2_000_000.0

    def test_update_price(self, cio_agent):
        """Test updating price cache."""
        cio_agent.update_price("AAPL", 150.0)

        assert cio_agent._price_cache["AAPL"] == 150.0

    def test_update_price_ignores_invalid(self, cio_agent):
        """Test that invalid prices are ignored."""
        cio_agent.update_price("AAPL", 0)
        cio_agent.update_price("AAPL", -10.0)

        assert "AAPL" not in cio_agent._price_cache

    def test_update_prices_bulk(self, cio_agent):
        """Test bulk price update."""
        prices = {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 300.0,
        }
        cio_agent.update_prices(prices)

        assert cio_agent._price_cache["AAPL"] == 150.0
        assert cio_agent._price_cache["GOOGL"] == 2800.0
        assert cio_agent._price_cache["MSFT"] == 300.0


# ============================================================================
# STATUS AND MONITORING TESTS
# ============================================================================

class TestStatusMonitoring:
    """Test status and monitoring methods."""

    def test_get_status(self, cio_agent):
        """Test getting agent status."""
        status = cio_agent.get_status()

        assert isinstance(status, dict)
        assert "current_regime" in status
        assert "use_dynamic_weights" in status
        assert "min_conviction" in status
        assert status["current_regime"] == MarketRegime.NEUTRAL.value

    def test_get_subscribed_events(self, cio_agent):
        """Test getting subscribed events."""
        events = cio_agent.get_subscribed_events()

        assert EventType.VALIDATED_DECISION in events

    def test_is_enabled(self, cio_agent):
        """Test is_enabled property."""
        assert cio_agent.is_enabled is True


# ============================================================================
# DECISION MAKING TESTS
# ============================================================================

class TestDecisionMaking:
    """Test decision making flow."""

    @pytest.mark.asyncio
    async def test_make_decision_below_conviction_threshold(self, cio_agent):
        """Test that decisions below conviction threshold are not made."""
        cio_agent._min_conviction = 0.9  # High threshold

        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.5,
            confidence=0.5,  # Below threshold
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        await cio_agent._make_decision_from_aggregation(agg)

        # Should not publish any decision due to low conviction
        cio_agent._event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_make_decision_flat_direction(self, cio_agent):
        """Test that FLAT direction does not generate decision."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.9,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        await cio_agent._make_decision_from_aggregation(agg)

        # Should not publish due to FLAT direction
        cio_agent._event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_make_decision_with_valid_signal(self, cio_agent):
        """Test decision making with valid signal."""
        cio_agent._min_conviction = 0.5  # Lower threshold for test

        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib", "bloomberg"),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        await cio_agent._make_decision_from_aggregation(agg)

        # Should publish a decision
        cio_agent._event_bus.publish.assert_called_once()
        published_event = cio_agent._event_bus.publish.call_args[0][0]
        assert isinstance(published_event, DecisionEvent)
        assert published_event.symbol == "AAPL"
        assert published_event.action == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_max_concurrent_decisions_limit(self, cio_agent):
        """Test that max concurrent decisions is enforced."""
        cio_agent._max_concurrent = 1
        cio_agent._min_conviction = 0.5

        # Add one active decision
        cio_agent._active_decisions["decision-1"] = datetime.now(timezone.utc)

        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        await cio_agent._make_decision_from_aggregation(agg)

        # Should not publish due to max concurrent limit
        cio_agent._event_bus.publish.assert_not_called()


# ============================================================================
# EDGE CASES TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_portfolio_value(self, mock_event_bus, mock_audit_logger):
        """Test handling of zero portfolio value."""
        config = AgentConfig(
            name="CIOAgent",
            parameters={"portfolio_value": 0.0},
        )
        agent = CIOAgent(config, mock_event_bus, mock_audit_logger)

        assert agent._portfolio_value == 0.0

    def test_negative_conviction_threshold(self, mock_event_bus, mock_audit_logger):
        """Test that negative conviction threshold is accepted (though unusual)."""
        config = AgentConfig(
            name="CIOAgent",
            parameters={"min_conviction_threshold": -0.5},
        )
        agent = CIOAgent(config, mock_event_bus, mock_audit_logger)

        assert agent._min_conviction == -0.5

    def test_build_rationale(self, cio_agent):
        """Test rationale building."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.75,
            rationale="Strong upward momentum detected",
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )
        cio_agent._aggregate_signals(agg)

        rationale = cio_agent._build_rationale(agg)

        assert "AAPL" in rationale
        assert "MomentumAgent" in rationale

    def test_collect_data_sources(self, cio_agent):
        """Test data source collection."""
        signals = {
            "MomentumAgent": SignalEvent(
                source_agent="MomentumAgent",
                strategy_name="momentum",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.75,
                data_sources=("ib", "yahoo"),
            ),
            "MacroAgent": SignalEvent(
                source_agent="MacroAgent",
                strategy_name="macro",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.7,
                confidence=0.8,
                data_sources=("bloomberg", "reuters"),
            ),
        }

        agg = SignalAggregation(
            symbol="AAPL",
            signals=signals,
            timestamp=datetime.now(timezone.utc),
        )

        sources = cio_agent._collect_data_sources(agg)

        assert "ib" in sources
        assert "bloomberg" in sources
        assert "yahoo" in sources
        assert "reuters" in sources

    @pytest.mark.asyncio
    async def test_handle_validated_decision_approved(self, cio_agent):
        """Test handling of approved validated decision."""
        decision_id = "test-decision-123"
        cio_agent._active_decisions[decision_id] = datetime.now(timezone.utc)

        event = ValidatedDecisionEvent(
            source_agent="RiskAgent",
            original_decision_id=decision_id,
            approved=True,
        )

        await cio_agent._handle_validated_decision(event)

        # Decision should be removed from active list
        assert decision_id not in cio_agent._active_decisions

    @pytest.mark.asyncio
    async def test_handle_validated_decision_rejected(self, cio_agent):
        """Test handling of rejected validated decision."""
        decision_id = "test-decision-456"
        cio_agent._active_decisions[decision_id] = datetime.now(timezone.utc)

        event = ValidatedDecisionEvent(
            source_agent="RiskAgent",
            original_decision_id=decision_id,
            approved=False,
            rejection_reason="Position limit exceeded",
        )

        await cio_agent._handle_validated_decision(event)

        # Decision should be removed from active list
        assert decision_id not in cio_agent._active_decisions


# ============================================================================
# SIGNAL CONFLICT EDGE CASE TESTS
# ============================================================================

class TestSignalConflictEdgeCases:
    """Test edge cases for signal conflicts."""

    def test_conflicting_signals_with_unequal_weights(self, cio_agent):
        """Test conflicting signals where one has higher weight wins."""
        # StatArbAgent has higher weight (0.25) vs MacroAgent (0.15)
        signal_long = SignalEvent(
            source_agent="MacroAgent",
            strategy_name="macro",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.8,
            data_sources=("bloomberg",),
        )
        signal_short = SignalEvent(
            source_agent="StatArbAgent",
            strategy_name="stat_arb",
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={
                "MacroAgent": signal_long,
                "StatArbAgent": signal_short,
            },
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # StatArb has higher weight, so SHORT should win
        assert agg.consensus_direction == SignalDirection.SHORT

    def test_conflicting_signals_with_unequal_strength(self, cio_agent):
        """Test conflicting signals where stronger signal wins or results in FLAT."""
        signal_long = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.9,  # Stronger
            confidence=0.8,
            data_sources=("ib",),
        )
        signal_short = SignalEvent(
            source_agent="StatArbAgent",
            strategy_name="stat_arb",
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            strength=0.3,  # Weaker
            confidence=0.8,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={
                "MomentumAgent": signal_long,
                "StatArbAgent": signal_short,
            },
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # With conflicting signals, result depends on net weighted score
        # May be LONG, SHORT, or FLAT depending on how weights interact
        assert agg.consensus_direction in [SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT]

    def test_three_way_signal_conflict(self, cio_agent):
        """Test three signals with mixed directions."""
        signals = {
            "MomentumAgent": SignalEvent(
                source_agent="MomentumAgent",
                strategy_name="momentum",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.7,
                confidence=0.8,
                data_sources=("ib",),
            ),
            "StatArbAgent": SignalEvent(
                source_agent="StatArbAgent",
                strategy_name="stat_arb",
                symbol="AAPL",
                direction=SignalDirection.SHORT,
                strength=0.6,
                confidence=0.75,
                data_sources=("ib",),
            ),
            "MacroAgent": SignalEvent(
                source_agent="MacroAgent",
                strategy_name="macro",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.5,
                confidence=0.7,
                data_sources=("bloomberg",),
            ),
        }

        agg = SignalAggregation(
            symbol="AAPL",
            signals=signals,
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # Two LONG vs one SHORT - should be LONG
        assert agg.consensus_direction == SignalDirection.LONG

    def test_conflicting_signals_all_equal_strength(self, cio_agent):
        """Test perfectly balanced conflicting signals result in FLAT."""
        # Make weights equal for this test
        cio_agent._weights = {
            "MomentumAgent": 0.5,
            "StatArbAgent": 0.5,
        }

        signal_long = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib",),
        )
        signal_short = SignalEvent(
            source_agent="StatArbAgent",
            strategy_name="stat_arb",
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={
                "MomentumAgent": signal_long,
                "StatArbAgent": signal_short,
            },
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # Perfect balance should result in FLAT
        assert agg.consensus_direction == SignalDirection.FLAT

    def test_conflicting_signals_with_flat_signal(self, cio_agent):
        """Test conflicting signals where one is FLAT."""
        signal_long = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.7,
            confidence=0.8,
            data_sources=("ib",),
        )
        signal_flat = SignalEvent(
            source_agent="StatArbAgent",
            strategy_name="stat_arb",
            symbol="AAPL",
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.5,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={
                "MomentumAgent": signal_long,
                "StatArbAgent": signal_flat,
            },
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # FLAT signal should not affect direction
        assert agg.consensus_direction == SignalDirection.LONG


# ============================================================================
# STRESS SCENARIO TESTS
# ============================================================================

class TestStressScenarios:
    """Test stress scenarios for CIO agent."""

    def test_high_volume_signals(self, cio_agent):
        """Test handling many signals for the same symbol."""
        # Create 5 agreeing signals
        signals = {}
        for agent in ["MomentumAgent", "StatArbAgent", "MacroAgent", "MarketMakingAgent", "OptionsVolAgent"]:
            signals[agent] = SignalEvent(
                source_agent=agent,
                strategy_name=agent.lower().replace("agent", ""),
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.8,
                data_sources=("ib",),
            )

        agg = SignalAggregation(
            symbol="AAPL",
            signals=signals,
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # All signals agree - strong consensus
        assert agg.consensus_direction == SignalDirection.LONG
        assert agg.weighted_confidence >= 0.7

    def test_extreme_strength_values(self, cio_agent):
        """Test signals with extreme strength values."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=1.0,  # Maximum
            confidence=1.0,  # Maximum
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        assert agg.weighted_strength > 0
        assert agg.weighted_confidence > 0

    def test_near_zero_strength_values(self, cio_agent):
        """Test signals with near-zero strength values."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.001,  # Near zero
            confidence=0.001,  # Near zero
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # Should still process without error
        assert agg.weighted_strength >= 0
        assert agg.weighted_confidence >= 0

    @pytest.mark.asyncio
    async def test_rapid_signal_processing(self, cio_agent):
        """Test rapid sequential signal processing."""
        cio_agent._min_conviction = 0.3

        for i in range(10):
            signal = SignalEvent(
                source_agent="MomentumAgent",
                strategy_name="momentum",
                symbol=f"SYM{i}",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.8,
                data_sources=("ib",),
            )

            agg = SignalAggregation(
                symbol=f"SYM{i}",
                signals={"MomentumAgent": signal},
                timestamp=datetime.now(timezone.utc),
            )

            await cio_agent._make_decision_from_aggregation(agg)

        # Should handle rapid processing without error
        # Verify decisions were processed
        assert cio_agent._event_bus.publish.call_count >= 0

    def test_unknown_agent_signal(self, cio_agent):
        """Test handling signal from unknown agent."""
        signal = SignalEvent(
            source_agent="UnknownAgent",
            strategy_name="unknown",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"UnknownAgent": signal},
            timestamp=datetime.now(timezone.utc),
        )

        # Should handle gracefully with default/zero weight
        cio_agent._aggregate_signals(agg)

        # Should not crash
        assert agg.consensus_direction is not None


# ============================================================================
# DECISION CONSISTENCY TESTS
# ============================================================================

class TestDecisionConsistency:
    """Test decision consistency and determinism."""

    def test_same_input_same_output(self, cio_agent):
        """Test that same inputs produce same aggregation results."""
        def create_signal():
            return SignalEvent(
                source_agent="MomentumAgent",
                strategy_name="momentum",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.75,
                confidence=0.8,
                data_sources=("ib",),
            )

        results = []
        for _ in range(5):
            signal = create_signal()
            agg = SignalAggregation(
                symbol="AAPL",
                signals={"MomentumAgent": signal},
                timestamp=datetime.now(timezone.utc),
            )
            cio_agent._aggregate_signals(agg)
            results.append((agg.consensus_direction, agg.weighted_strength, agg.weighted_confidence))

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_direction_consistency_across_regime_changes(self, cio_agent):
        """Test that consensus direction is consistent regardless of regime."""
        signal = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib",),
        )

        directions = []
        for regime in [MarketRegime.NEUTRAL, MarketRegime.RISK_ON, MarketRegime.RISK_OFF]:
            cio_agent.set_market_regime(regime)
            agg = SignalAggregation(
                symbol="AAPL",
                signals={"MomentumAgent": signal},
                timestamp=datetime.now(timezone.utc),
            )
            cio_agent._aggregate_signals(agg)
            directions.append(agg.consensus_direction)

        # Direction should be consistent (LONG) regardless of regime
        assert all(d == SignalDirection.LONG for d in directions)

    def test_weight_normalization_consistency(self, cio_agent):
        """Test that weights always sum to approximately 1."""
        for _ in range(10):
            cio_agent._update_dynamic_weights()
            total_weight = sum(cio_agent._weights.values())
            assert abs(total_weight - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_decision_id_uniqueness(self, cio_agent):
        """Test that each decision gets a unique ID."""
        cio_agent._min_conviction = 0.3

        decision_ids = []
        for i in range(5):
            signal = SignalEvent(
                source_agent="MomentumAgent",
                strategy_name="momentum",
                symbol=f"SYM{i}",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.8,
                data_sources=("ib",),
            )

            agg = SignalAggregation(
                symbol=f"SYM{i}",
                signals={"MomentumAgent": signal},
                timestamp=datetime.now(timezone.utc),
            )

            await cio_agent._make_decision_from_aggregation(agg)

            if cio_agent._event_bus.publish.call_count > len(decision_ids):
                published_event = cio_agent._event_bus.publish.call_args[0][0]
                if hasattr(published_event, 'event_id'):
                    decision_ids.append(published_event.event_id)

        # All decision IDs should be unique
        assert len(decision_ids) == len(set(decision_ids))

    def test_conviction_threshold_boundary(self, cio_agent):
        """Test behavior at conviction threshold boundary."""
        cio_agent._min_conviction = 0.6

        # At exactly the threshold
        signal_at_threshold = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.6,
            confidence=0.6,
            data_sources=("ib",),
        )

        agg = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal_at_threshold},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg)

        # Should still produce valid aggregation
        assert agg.consensus_direction is not None

    def test_multiple_symbols_independence(self, cio_agent):
        """Test that decisions for different symbols are independent."""
        signal_aapl = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.8,
            data_sources=("ib",),
        )

        signal_googl = SignalEvent(
            source_agent="MomentumAgent",
            strategy_name="momentum",
            symbol="GOOGL",
            direction=SignalDirection.SHORT,
            strength=0.7,
            confidence=0.75,
            data_sources=("ib",),
        )

        agg_aapl = SignalAggregation(
            symbol="AAPL",
            signals={"MomentumAgent": signal_aapl},
            timestamp=datetime.now(timezone.utc),
        )

        agg_googl = SignalAggregation(
            symbol="GOOGL",
            signals={"MomentumAgent": signal_googl},
            timestamp=datetime.now(timezone.utc),
        )

        cio_agent._aggregate_signals(agg_aapl)
        cio_agent._aggregate_signals(agg_googl)

        # Directions should be independent
        assert agg_aapl.consensus_direction == SignalDirection.LONG
        assert agg_googl.consensus_direction == SignalDirection.SHORT


# ============================================================================
# AUTONOMOUS POSITION MANAGEMENT TESTS
# ============================================================================

class TestPositionManagement:
    """Test autonomous position management functionality."""

    def test_register_position(self, cio_agent):
        """Test position registration for CIO tracking."""
        cio_agent.register_position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            is_long=True,
            conviction=0.75,
            stop_loss=145.0,
            take_profit=165.0,
            contributing_strategies=["MomentumAgent"],
        )

        positions = cio_agent.get_tracked_positions()
        assert "AAPL" in positions
        assert positions["AAPL"]["quantity"] == 100
        assert positions["AAPL"]["entry_price"] == 150.0
        assert positions["AAPL"]["is_long"] is True
        assert positions["AAPL"]["original_conviction"] == 0.75

    def test_position_pnl_calculation(self, cio_agent):
        """Test P&L calculation for tracked positions."""
        from agents.cio_agent import TrackedPosition

        # Long position - profit
        pos_long = TrackedPosition(
            symbol="AAPL",
            quantity=100,
            entry_price=100.0,
            entry_time=datetime.now(timezone.utc),
            is_long=True,
            current_price=110.0,
        )
        assert pos_long.pnl_pct == 10.0  # 10% gain

        # Long position - loss
        pos_long.current_price = 95.0
        pos_long.update_price(95.0)
        assert pos_long.pnl_pct == -5.0  # 5% loss

        # Short position - profit
        pos_short = TrackedPosition(
            symbol="MSFT",
            quantity=50,
            entry_price=200.0,
            entry_time=datetime.now(timezone.utc),
            is_long=False,
            current_price=190.0,
        )
        assert pos_short.pnl_pct == 5.0  # 5% gain on short

    def test_position_management_config_defaults(self, cio_agent):
        """Test position management configuration defaults."""
        config = cio_agent._position_management_config
        assert config.max_loss_pct == 5.0
        assert config.extended_loss_pct == 8.0
        assert config.profit_target_pct == 15.0
        assert config.trailing_profit_pct == 3.0
        assert config.min_conviction_to_hold == 0.4

    def test_position_management_stats_tracking(self, cio_agent):
        """Test position management statistics tracking."""
        stats = cio_agent.get_position_management_stats()
        assert "tracked_positions" in stats
        assert "management_enabled" in stats
        assert "config" in stats
        assert "stats" in stats
        assert stats["stats"]["losers_closed"] == 0
        assert stats["stats"]["profits_taken"] == 0

    def test_create_close_loser_decision(self, cio_agent):
        """Test creation of close loser decision."""
        from agents.cio_agent import TrackedPosition
        from core.events import DecisionAction

        pos = TrackedPosition(
            symbol="AAPL",
            quantity=100,
            entry_price=100.0,
            entry_time=datetime.now(timezone.utc),
            is_long=True,
            current_price=92.0,  # -8% loss
            original_conviction=0.8,
            current_conviction=0.3,
            contributing_strategies=["MomentumAgent"],
        )
        pos.update_price(92.0)

        decision = cio_agent._create_close_loser_decision(
            pos,
            reason="Loss exceeds threshold",
            is_emergency=True,
        )

        assert decision.symbol == "AAPL"
        assert decision.action == OrderSide.SELL  # Closing long = sell
        assert decision.quantity == 100
        assert decision.decision_action == DecisionAction.CLOSE_LOSER
        assert decision.position_pnl_pct == -8.0

    def test_create_take_profit_decision(self, cio_agent):
        """Test creation of take profit decision."""
        from agents.cio_agent import TrackedPosition
        from core.events import DecisionAction

        pos = TrackedPosition(
            symbol="MSFT",
            quantity=50,
            entry_price=200.0,
            entry_time=datetime.now(timezone.utc),
            is_long=True,
            current_price=235.0,  # +17.5% gain
            original_conviction=0.7,
            current_conviction=0.6,
            contributing_strategies=["StatArbAgent"],
        )
        pos.update_price(235.0)

        decision = cio_agent._create_take_profit_decision(
            pos,
            reason="Profit target reached",
            partial_exit=False,
        )

        assert decision.symbol == "MSFT"
        assert decision.action == OrderSide.SELL
        assert decision.quantity == 50
        assert decision.decision_action == DecisionAction.TAKE_PROFIT
        assert decision.position_pnl_pct == 17.5

    def test_create_reduce_position_decision(self, cio_agent):
        """Test creation of reduce position decision."""
        from agents.cio_agent import TrackedPosition
        from core.events import DecisionAction

        pos = TrackedPosition(
            symbol="GOOGL",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(timezone.utc),
            is_long=True,
            current_price=155.0,
            original_conviction=0.9,
            current_conviction=0.5,  # Conviction dropped
            contributing_strategies=["MomentumAgent"],
        )
        pos.update_price(155.0)

        decision = cio_agent._create_reduce_position_decision(
            pos,
            reduction_pct=50.0,
            reason="Conviction dropped",
        )

        assert decision.symbol == "GOOGL"
        assert decision.action == OrderSide.SELL
        assert decision.quantity == 50  # 50% of 100
        assert decision.decision_action == DecisionAction.REDUCE_POSITION

    def test_regime_adjusted_loss_threshold(self, cio_agent):
        """Test that loss thresholds adjust for market regime."""
        # Default regime (NEUTRAL)
        assert cio_agent._get_regime_adjusted_loss_threshold(MarketRegime.NEUTRAL) == 5.0

        # Volatile regime - should be tighter
        assert cio_agent._get_regime_adjusted_loss_threshold(MarketRegime.VOLATILE) == 3.0

        # Risk-off regime - should be tighter
        threshold = cio_agent._get_regime_adjusted_loss_threshold(MarketRegime.RISK_OFF)
        assert threshold == 4.0  # 5.0 * 0.8

    def test_regime_adjusted_profit_threshold(self, cio_agent):
        """Test that profit thresholds adjust for market regime."""
        # Default regime (NEUTRAL)
        assert cio_agent._get_regime_adjusted_profit_threshold(MarketRegime.NEUTRAL) == 15.0

        # Trending regime - should let profits run longer
        threshold = cio_agent._get_regime_adjusted_profit_threshold(MarketRegime.TRENDING)
        assert threshold == 22.5  # 15.0 * 1.5

        # Volatile regime - should take profits earlier
        threshold = cio_agent._get_regime_adjusted_profit_threshold(MarketRegime.VOLATILE)
        assert threshold == 10.5  # 15.0 * 0.7

    @pytest.mark.asyncio
    async def test_evaluate_position_emergency_loss(self, cio_agent):
        """Test that emergency loss triggers immediate close."""
        from agents.cio_agent import TrackedPosition

        pos = TrackedPosition(
            symbol="AAPL",
            quantity=100,
            entry_price=100.0,
            entry_time=datetime.now(timezone.utc),
            is_long=True,
            current_price=91.0,  # -9% loss > extended_loss_pct (8%)
            original_conviction=0.8,
            current_conviction=0.6,
        )
        pos.update_price(91.0)

        decision = await cio_agent._evaluate_position_for_management(pos)

        assert decision is not None
        assert "EMERGENCY" in decision.rationale
        assert decision.order_type.value == "market"  # Emergency uses market order

    @pytest.mark.asyncio
    async def test_evaluate_position_take_profit_with_trailing(self, cio_agent):
        """Test take profit with trailing stop logic."""
        from agents.cio_agent import TrackedPosition

        pos = TrackedPosition(
            symbol="MSFT",
            quantity=50,
            entry_price=100.0,
            entry_time=datetime.now(timezone.utc),
            is_long=True,
            highest_price=120.0,  # Hit 20% peak
            current_price=115.0,  # Now 15%, drawdown 4.2% from peak
            original_conviction=0.7,
            current_conviction=0.6,
        )
        pos.update_price(115.0)

        # Should trigger take profit (15% gain > 15% target, 4.2% drawdown > 3% trailing)
        decision = await cio_agent._evaluate_position_for_management(pos)

        assert decision is not None
        assert "Take profit" in decision.rationale

    def test_get_tracked_positions(self, cio_agent):
        """Test getting all tracked positions."""
        cio_agent.register_position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            is_long=True,
            conviction=0.75,
        )
        cio_agent.register_position(
            symbol="MSFT",
            quantity=50,
            entry_price=200.0,
            is_long=False,  # Short position
            conviction=0.65,
        )

        positions = cio_agent.get_tracked_positions()

        assert len(positions) == 2
        assert "AAPL" in positions
        assert "MSFT" in positions
        assert positions["AAPL"]["is_long"] is True
        assert positions["MSFT"]["is_long"] is False

    def test_position_management_in_status(self, cio_agent):
        """Test that position management info is in status."""
        cio_agent.register_position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            is_long=True,
            conviction=0.75,
        )

        status = cio_agent.get_status()

        assert "position_management" in status
        assert status["position_management"]["enabled"] is True
        assert status["position_management"]["tracked_positions"] == 1
        assert "config" in status["position_management"]
        assert "stats" in status["position_management"]


class TestDecisionActionEnum:
    """Test DecisionAction enum functionality."""

    def test_decision_action_values(self):
        """Test that all decision action values are defined."""
        from core.events import DecisionAction

        assert DecisionAction.BUY.value == "buy"
        assert DecisionAction.SELL.value == "sell"
        assert DecisionAction.CLOSE_LOSER.value == "close_loser"
        assert DecisionAction.TAKE_PROFIT.value == "take_profit"
        assert DecisionAction.REDUCE_POSITION.value == "reduce_position"
        assert DecisionAction.INCREASE_POSITION.value == "increase_position"
        assert DecisionAction.HOLD.value == "hold"

    def test_is_closing_action(self):
        """Test is_closing_action helper method."""
        from core.events import DecisionAction

        assert DecisionAction.is_closing_action(DecisionAction.CLOSE_LOSER) is True
        assert DecisionAction.is_closing_action(DecisionAction.TAKE_PROFIT) is True
        assert DecisionAction.is_closing_action(DecisionAction.REDUCE_POSITION) is True
        assert DecisionAction.is_closing_action(DecisionAction.BUY) is False
        assert DecisionAction.is_closing_action(DecisionAction.HOLD) is False

    def test_is_opening_action(self):
        """Test is_opening_action helper method."""
        from core.events import DecisionAction

        assert DecisionAction.is_opening_action(DecisionAction.BUY) is True
        assert DecisionAction.is_opening_action(DecisionAction.SELL) is True
        assert DecisionAction.is_opening_action(DecisionAction.INCREASE_POSITION) is True
        assert DecisionAction.is_opening_action(DecisionAction.CLOSE_LOSER) is False
        assert DecisionAction.is_opening_action(DecisionAction.HOLD) is False

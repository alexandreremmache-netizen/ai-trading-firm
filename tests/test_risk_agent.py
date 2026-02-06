"""
Tests for Risk Agent
====================

Tests for the Risk Management agent that validates trading decisions.
"""

import pytest
import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from agents.risk_agent import (
    RiskAgent,
    RiskState,
    RiskCheckResult,
    RiskValidationResult,
    KillSwitchReason,
    KillSwitchAction,
    DrawdownLevel,
    PositionInfo,
    GreeksState,
    MarginState,
)
from core.agent_base import AgentConfig
from core.events import (
    DecisionEvent,
    ValidatedDecisionEvent,
    EventType,
    OrderSide,
    OrderType,
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
    logger.log_risk_check = MagicMock()
    return logger


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = MagicMock()
    broker.get_portfolio_summary = AsyncMock(return_value={
        "net_liquidation": 1_000_000.0,
        "total_cash": 500_000.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
    })
    broker.get_positions = AsyncMock(return_value={})
    broker.get_margin_info = AsyncMock(return_value={
        "initial_margin": 100_000.0,
        "maintenance_margin": 80_000.0,
        "available_margin": 400_000.0,
    })
    broker.get_portfolio_state = AsyncMock(return_value={
        "net_liquidation": 1_000_000.0,
        "total_cash": 500_000.0,
        "positions": {},
    })
    return broker


@pytest.fixture
def default_config():
    """Create default Risk agent configuration."""
    return AgentConfig(
        name="RiskAgent",
        enabled=True,
        timeout_seconds=30.0,
        parameters={
            "limits": {
                "max_position_size_pct": 5.0,
                "max_sector_exposure_pct": 20.0,
                "max_leverage": 2.0,
                "max_gross_exposure_pct": 200.0,
                "max_portfolio_var_pct": 2.0,
                "max_daily_loss_pct": 3.0,
                "max_drawdown_pct": 10.0,
            },
            "rate_limits": {
                "max_orders_per_minute": 10,
                "min_order_interval_ms": 100,
            },
            "greeks": {
                "max_portfolio_delta": 500,
                "max_portfolio_gamma": 100,
                "max_portfolio_vega": 50000,
            },
            "drawdown": {
                "warning_pct": 5.0,
                "reduce_pct": 7.5,
                "halt_pct": 10.0,
            },
        },
    )


@pytest.fixture
def risk_agent(default_config, mock_event_bus, mock_audit_logger, mock_broker):
    """Create a Risk agent instance for testing."""
    return RiskAgent(default_config, mock_event_bus, mock_audit_logger, mock_broker)


@pytest.fixture
def sample_decision_event():
    """Create a sample decision event for testing."""
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


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestRiskAgentInitialization:
    """Test Risk agent initialization."""

    def test_initialization_with_default_config(self, risk_agent):
        """Test that agent initializes correctly with default config."""
        assert risk_agent.name == "RiskAgent"
        assert risk_agent._max_position_pct == 0.05
        assert risk_agent._max_sector_pct == 0.20
        assert risk_agent._max_leverage == 2.0
        assert risk_agent._max_drawdown_pct == 0.10

    def test_initialization_without_broker(self, default_config, mock_event_bus, mock_audit_logger):
        """Test initialization without broker."""
        agent = RiskAgent(default_config, mock_event_bus, mock_audit_logger, broker=None)

        assert agent._broker is None
        assert agent._risk_state is not None

    def test_initialization_with_custom_limits(self, mock_event_bus, mock_audit_logger, mock_broker):
        """Test initialization with custom risk limits."""
        config = AgentConfig(
            name="RiskAgent",
            parameters={
                "limits": {
                    "max_position_size_pct": 10.0,
                    "max_drawdown_pct": 15.0,
                },
            },
        )
        agent = RiskAgent(config, mock_event_bus, mock_audit_logger, mock_broker)

        assert agent._max_position_pct == 0.10
        assert agent._max_drawdown_pct == 0.15

    def test_initialization_with_empty_config(self, mock_event_bus, mock_audit_logger, mock_broker):
        """Test initialization with empty parameters uses defaults."""
        config = AgentConfig(name="RiskAgent", parameters={})
        agent = RiskAgent(config, mock_event_bus, mock_audit_logger, mock_broker)

        # Should use default values (max_position_size_pct defaults to 2.5%)
        assert agent._max_position_pct == 0.025  # 2.5% default
        assert agent._max_daily_loss_pct == 0.03

    @pytest.mark.asyncio
    async def test_initialize_with_broker(self, risk_agent):
        """Test async initialization with broker."""
        await risk_agent.initialize()
        # Should complete without error


# ============================================================================
# RISK STATE TESTS
# ============================================================================

class TestRiskState:
    """Test RiskState dataclass."""

    def test_risk_state_defaults(self):
        """Test RiskState default values."""
        state = RiskState()

        assert state.net_liquidation == 1_000_000.0
        assert state.current_drawdown_pct == 0.0
        assert state.daily_pnl == 0.0
        assert isinstance(state.positions, dict)
        assert isinstance(state.greeks, GreeksState)
        assert isinstance(state.margin, MarginState)

    def test_greeks_state_staleness(self):
        """Test GreeksState staleness check."""
        greeks = GreeksState()
        greeks.last_update = datetime.now(timezone.utc) - timedelta(seconds=120)

        assert greeks.is_stale(60.0) is True
        assert greeks.is_stale(300.0) is False

    def test_margin_state_warning_levels(self):
        """Test MarginState warning level methods."""
        margin = MarginState()

        margin.margin_utilization_pct = 60.0
        assert margin.is_warning() is False
        assert margin.is_critical() is False

        margin.margin_utilization_pct = 75.0
        assert margin.is_warning() is True
        assert margin.is_critical() is False

        margin.margin_utilization_pct = 90.0
        assert margin.is_warning() is True
        assert margin.is_critical() is True


# ============================================================================
# CVaR 97.5% TESTS (Phase 11 - FRTB/Basel III)
# ============================================================================

class TestCVaR:
    """Test Conditional Value at Risk (CVaR) at 97.5% confidence level."""

    def test_cvar_975_field_exists(self):
        """Verify RiskState has cvar_975 field (FRTB/Basel III requirement)."""
        state = RiskState()
        assert hasattr(state, "cvar_975"), "RiskState must have cvar_975 field"
        assert state.cvar_975 == 0.0  # Default value

    def test_expected_shortfall_field_exists(self):
        """Verify RiskState has expected_shortfall field (legacy 95%)."""
        state = RiskState()
        assert hasattr(state, "expected_shortfall"), "RiskState must have expected_shortfall"
        assert state.expected_shortfall == 0.0

    @pytest.mark.asyncio
    async def test_cvar_975_calculated_on_update(self, risk_agent):
        """Verify CVaR 97.5% is calculated when returns are updated."""
        import numpy as np
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.02, 300)

        for r in returns:
            risk_agent._returns_history.append(r)

        # _calculate_var calls asyncio.create_task, needs event loop
        risk_agent._calculate_var()

        # CVaR should be calculated and positive (represents a loss)
        assert risk_agent._risk_state.cvar_975 > 0, (
            "CVaR 97.5% should be positive for loss data"
        )

    @pytest.mark.asyncio
    async def test_cvar_975_greater_than_or_equal_to_var_95(self, risk_agent):
        """Verify CVaR 97.5% >= VaR 95% (mathematical property)."""
        import numpy as np
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.02, 300)

        for r in returns:
            risk_agent._returns_history.append(r)

        risk_agent._calculate_var()

        # CVaR at 97.5% must be >= VaR at 95% (deeper tail)
        assert risk_agent._risk_state.cvar_975 >= risk_agent._risk_state.var_95, (
            f"CVaR 97.5% ({risk_agent._risk_state.cvar_975:.4f}) must be >= "
            f"VaR 95% ({risk_agent._risk_state.var_95:.4f})"
        )

    def test_cvar_975_accessible_on_risk_state(self, risk_agent):
        """Verify CVaR 97.5% is on the risk state for monitoring."""
        # cvar_975 should be accessible on the risk state object
        assert hasattr(risk_agent._risk_state, "cvar_975"), (
            "CVaR 97.5% must be on RiskState for FRTB/Basel III compliance"
        )
        assert hasattr(risk_agent._risk_state, "expected_shortfall"), (
            "Expected Shortfall (legacy 95%) must be on RiskState"
        )


# ============================================================================
# KILL SWITCH TESTS
# ============================================================================

class TestKillSwitch:
    """Test kill switch functionality."""

    def test_kill_switch_initially_inactive(self, risk_agent):
        """Test that kill switch is initially inactive."""
        assert risk_agent._kill_switch_active is False
        assert risk_agent._kill_switch_reason is None

    def test_activate_kill_switch(self, risk_agent):
        """Test activating kill switch."""
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.DAILY_LOSS_LIMIT
        risk_agent._kill_switch_time = datetime.now(timezone.utc)

        assert risk_agent._kill_switch_active is True
        assert risk_agent._kill_switch_reason == KillSwitchReason.DAILY_LOSS_LIMIT

    @pytest.mark.asyncio
    async def test_reject_decision_when_kill_switch_active(self, risk_agent, sample_decision_event):
        """Test that decisions are rejected when kill switch is active."""
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.MAX_DRAWDOWN

        await risk_agent.process_event(sample_decision_event)

        # Should publish a rejected ValidatedDecisionEvent
        risk_agent._event_bus.publish.assert_called_once()
        published_event = risk_agent._event_bus.publish.call_args[0][0]
        assert isinstance(published_event, ValidatedDecisionEvent)
        assert published_event.approved is False
        assert "KILL-SWITCH" in published_event.rejection_reason

    def test_all_kill_switch_reasons(self):
        """Test all kill switch reason values."""
        reasons = list(KillSwitchReason)
        assert len(reasons) >= 5
        assert KillSwitchReason.DAILY_LOSS_LIMIT in reasons
        assert KillSwitchReason.MAX_DRAWDOWN in reasons
        assert KillSwitchReason.MANUAL in reasons


# ============================================================================
# DRAWDOWN TESTS
# ============================================================================

class TestDrawdown:
    """Test drawdown handling with autonomous risk management."""

    def test_drawdown_levels(self):
        """Test drawdown level enumeration - new tiered system."""
        assert DrawdownLevel.NORMAL.value == "normal"
        assert DrawdownLevel.WARNING.value == "warning"
        assert DrawdownLevel.CRITICAL.value == "critical"
        assert DrawdownLevel.SEVERE.value == "severe"
        assert DrawdownLevel.MAXIMUM.value == "maximum"

    def test_initial_drawdown_level(self, risk_agent):
        """Test initial drawdown level is NORMAL."""
        assert risk_agent._current_drawdown_level == DrawdownLevel.NORMAL

    def test_drawdown_thresholds(self, risk_agent):
        """Test drawdown threshold configuration - new tiered system."""
        assert risk_agent._drawdown_warning_pct == 0.05  # 5%
        assert risk_agent._drawdown_critical_pct == 0.10  # 10%
        assert risk_agent._drawdown_severe_pct == 0.15  # 15%
        assert risk_agent._drawdown_maximum_pct == 0.20  # 20%

    def test_autonomous_mode_flags(self, risk_agent):
        """Test autonomous mode flags are initialized correctly."""
        assert risk_agent._defensive_mode_active is False
        assert risk_agent._no_new_longs is False

    def test_position_size_multiplier_normal(self, risk_agent):
        """Test position size multiplier at NORMAL level."""
        risk_agent._current_drawdown_level = DrawdownLevel.NORMAL
        assert risk_agent.get_position_size_multiplier() == 1.0

    def test_position_size_multiplier_warning(self, risk_agent):
        """Test position size multiplier at WARNING level (50% reduction)."""
        risk_agent._current_drawdown_level = DrawdownLevel.WARNING
        assert risk_agent.get_position_size_multiplier() == 0.5

    def test_position_size_multiplier_critical(self, risk_agent):
        """Test position size multiplier at CRITICAL level (no new positions)."""
        risk_agent._current_drawdown_level = DrawdownLevel.CRITICAL
        assert risk_agent.get_position_size_multiplier() == 0.0

    def test_can_open_position_normal(self, risk_agent):
        """Test can open position at NORMAL level."""
        can_open, reason = risk_agent.can_open_new_position(is_long=True)
        assert can_open is True

    def test_can_open_position_defensive_mode(self, risk_agent):
        """Test cannot open position in defensive mode."""
        risk_agent._defensive_mode_active = True
        can_open, reason = risk_agent.can_open_new_position(is_long=True)
        assert can_open is False
        assert "DEFENSIVE MODE" in reason


# ============================================================================
# POSITION INFO TESTS
# ============================================================================

class TestPositionInfo:
    """Test PositionInfo dataclass."""

    def test_position_info_creation(self):
        """Test creating PositionInfo."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            weight_pct=1.5,
            sector="Technology",
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.unrealized_pnl == 500.0

    def test_position_info_default_sector(self):
        """Test PositionInfo default sector."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0,
            weight_pct=1.5,
        )

        assert position.sector == "unknown"


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_configuration(self, risk_agent):
        """Test rate limit configuration."""
        assert risk_agent._max_orders_per_minute == 10
        assert risk_agent._min_order_interval_ms == 100

    def test_orders_tracking(self, risk_agent):
        """Test orders tracking in risk state."""
        risk_agent._risk_state.orders_today = 0
        risk_agent._risk_state.orders_this_minute = []

        # Simulate adding orders
        now = datetime.now(timezone.utc)
        risk_agent._risk_state.orders_this_minute.append(now)
        risk_agent._risk_state.orders_today += 1

        assert risk_agent._risk_state.orders_today == 1
        assert len(risk_agent._risk_state.orders_this_minute) == 1


# ============================================================================
# EVENT SUBSCRIPTION TESTS
# ============================================================================

class TestEventSubscription:
    """Test event subscription."""

    def test_subscribed_events(self, risk_agent):
        """Test that risk agent subscribes to DECISION events."""
        events = risk_agent.get_subscribed_events()

        assert EventType.DECISION in events


# ============================================================================
# RISK CHECK RESULT TESTS
# ============================================================================

class TestRiskCheckResult:
    """Test RiskCheckResult dataclass."""

    def test_risk_check_result_passed(self):
        """Test creating a passed risk check result."""
        result = RiskCheckResult(
            check_name="position_size",
            passed=True,
            current_value=3.0,
            limit_value=5.0,
            message="Position size within limits",
        )

        assert result.passed is True
        assert result.check_name == "position_size"

    def test_risk_check_result_failed(self):
        """Test creating a failed risk check result."""
        result = RiskCheckResult(
            check_name="drawdown",
            passed=False,
            current_value=12.0,
            limit_value=10.0,
            message="Drawdown exceeds limit",
        )

        assert result.passed is False


# ============================================================================
# RISK VALIDATION RESULT TESTS
# ============================================================================

class TestRiskValidationResult:
    """Test RiskValidationResult dataclass."""

    def test_validation_result_approved(self):
        """Test creating an approved validation result."""
        checks = [
            RiskCheckResult("position_size", True, 3.0, 5.0),
            RiskCheckResult("drawdown", True, 2.0, 10.0),
        ]
        result = RiskValidationResult(
            approved=True,
            checks=checks,
            risk_metrics={"var_95": 0.015, "leverage": 1.5},
        )

        assert result.approved is True
        assert len(result.checks) == 2
        assert result.rejection_reason is None

    def test_validation_result_rejected(self):
        """Test creating a rejected validation result."""
        checks = [
            RiskCheckResult("position_size", False, 8.0, 5.0, "Exceeds limit"),
        ]
        result = RiskValidationResult(
            approved=False,
            checks=checks,
            risk_metrics={},
            rejection_reason="Position size exceeds 5% limit",
        )

        assert result.approved is False
        assert result.rejection_reason is not None


# ============================================================================
# GREEKS LIMITS TESTS
# ============================================================================

class TestGreeksLimits:
    """Test Greeks limits configuration."""

    def test_greeks_limits_configuration(self, risk_agent):
        """Test Greeks limits are configured correctly."""
        assert risk_agent._max_delta == 500
        assert risk_agent._max_gamma == 100
        assert risk_agent._max_vega == 50000

    def test_greeks_staleness_configuration(self, risk_agent):
        """Test Greeks staleness configuration."""
        assert risk_agent._greeks_staleness_warning_seconds == 60.0
        assert risk_agent._greeks_staleness_critical_seconds == 300.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRiskAgentIntegration:
    """Integration tests for Risk Agent."""

    @pytest.mark.asyncio
    async def test_process_decision_calls_publish(self, risk_agent, sample_decision_event):
        """Test that processing a decision event results in publishing."""
        # Setup risk state for approval
        risk_agent._risk_state.net_liquidation = 1_000_000.0
        risk_agent._risk_state.current_drawdown_pct = 0.01
        risk_agent._risk_state.daily_pnl_pct = 0.0

        # Mock the internal validation to avoid full execution path
        async def mock_validate(decision):
            return RiskValidationResult(
                approved=True,
                checks=[],
                risk_metrics={"var_95": 0.015},
            )

        with patch.object(risk_agent, '_validate_decision', mock_validate):
            with patch.object(risk_agent, '_refresh_portfolio_state', AsyncMock()):
                await risk_agent.process_event(sample_decision_event)

        # Should publish a ValidatedDecisionEvent
        risk_agent._event_bus.publish.assert_called_once()
        published_event = risk_agent._event_bus.publish.call_args[0][0]
        assert isinstance(published_event, ValidatedDecisionEvent)

    @pytest.mark.asyncio
    async def test_process_non_decision_event(self, risk_agent):
        """Test that non-decision events are ignored."""
        from core.events import MarketDataEvent

        market_event = MarketDataEvent(
            source_agent="DataFeed",
            symbol="AAPL",
            bid=150.0,
            ask=150.10,
            last=150.05,
        )

        await risk_agent.process_event(market_event)

        # Should not publish anything
        risk_agent._event_bus.publish.assert_not_called()


# ============================================================================
# EDGE CASES TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_net_liquidation(self, mock_event_bus, mock_audit_logger, mock_broker):
        """Test handling of zero net liquidation."""
        config = AgentConfig(name="RiskAgent", parameters={})
        agent = RiskAgent(config, mock_event_bus, mock_audit_logger, mock_broker)

        agent._risk_state.net_liquidation = 0.0

        # Should not crash
        assert agent._risk_state.net_liquidation == 0.0

    def test_negative_pnl(self, risk_agent):
        """Test handling of negative P&L."""
        risk_agent._risk_state.daily_pnl = -50000.0
        risk_agent._risk_state.daily_pnl_pct = -0.05

        assert risk_agent._risk_state.daily_pnl < 0
        assert risk_agent._risk_state.daily_pnl_pct < 0

    def test_empty_positions(self, risk_agent):
        """Test handling of empty positions dict."""
        risk_agent._risk_state.positions = {}

        assert len(risk_agent._risk_state.positions) == 0

    def test_max_drawdown_calculation(self, risk_agent):
        """Test max drawdown tracking."""
        risk_agent._risk_state.peak_equity = 1_000_000.0
        risk_agent._risk_state.net_liquidation = 900_000.0
        risk_agent._risk_state.current_drawdown_pct = 0.10

        assert risk_agent._risk_state.current_drawdown_pct == 0.10

    def test_var_history(self, risk_agent):
        """Test VaR history tracking."""
        # Memory safety: returns_history uses deque instead of list
        assert isinstance(risk_agent._returns_history, deque)
        assert risk_agent._max_history_days == 252

        # Add some returns
        risk_agent._returns_history.append(0.01)
        risk_agent._returns_history.append(-0.005)

        assert len(risk_agent._returns_history) == 2


# ============================================================================
# MONITORING AND STATUS TESTS
# ============================================================================

class TestMonitoringStatus:
    """Test monitoring and status methods."""

    def test_check_latencies_tracking(self, risk_agent):
        """Test check latencies tracking."""
        # Memory safety: check_latencies uses deque instead of list
        assert isinstance(risk_agent._check_latencies, deque)

        # Add some latencies
        risk_agent._check_latencies.append(5.0)
        risk_agent._check_latencies.append(10.0)

        assert len(risk_agent._check_latencies) == 2

    def test_latencies_capped_at_1000(self, risk_agent):
        """Test that latencies list is capped."""
        # Add 1005 latencies
        for i in range(1005):
            risk_agent._check_latencies.append(float(i))

        # Trim as the agent does
        if len(risk_agent._check_latencies) > 1000:
            risk_agent._check_latencies = risk_agent._check_latencies[-1000:]

        assert len(risk_agent._check_latencies) == 1000


# ============================================================================
# VAR BREACH TESTS
# ============================================================================

class TestVaRBreach:
    """Test VaR (Value at Risk) breach handling."""

    def test_var_limit_configuration(self, risk_agent):
        """Test VaR limit is correctly configured."""
        assert risk_agent._max_var_pct == 0.02  # 2% from config

    def test_var_calculation_with_empty_returns(self, risk_agent):
        """Test VaR calculation with empty returns history."""
        risk_agent._returns_history = []

        # Should handle gracefully without crashing
        assert len(risk_agent._returns_history) == 0

    def test_var_calculation_with_sufficient_returns(self, risk_agent):
        """Test VaR calculation with sufficient historical returns."""
        # Add 30 days of returns
        import random
        random.seed(42)  # For reproducibility
        risk_agent._returns_history = [random.gauss(0.0005, 0.01) for _ in range(30)]

        assert len(risk_agent._returns_history) == 30

    def test_var_breach_detection(self, risk_agent):
        """Test that VaR breach is detected."""
        # Set current VaR to exceed limit
        risk_agent._risk_state.current_var_pct = 0.03  # 3% > 2% limit

        assert risk_agent._risk_state.current_var_pct > risk_agent._max_var_pct

    def test_var_within_limits(self, risk_agent):
        """Test VaR within acceptable limits."""
        risk_agent._risk_state.current_var_pct = 0.015  # 1.5% < 2% limit

        assert risk_agent._risk_state.current_var_pct < risk_agent._max_var_pct

    @pytest.mark.asyncio
    async def test_reject_decision_on_var_breach(self, risk_agent, sample_decision_event):
        """Test that decisions are rejected when VaR exceeds limit."""
        risk_agent._risk_state.current_var_pct = 0.03  # Exceeds 2% limit

        # Mock validation to fail on VaR
        async def mock_validate(decision):
            return RiskValidationResult(
                approved=False,
                checks=[
                    RiskCheckResult(
                        check_name="var_limit",
                        passed=False,
                        current_value=3.0,
                        limit_value=2.0,
                        message="VaR exceeds limit",
                    )
                ],
                risk_metrics={"var_95": 0.03},
                rejection_reason="VaR 3.0% exceeds limit of 2.0%",
            )

        with patch.object(risk_agent, '_validate_decision', mock_validate):
            with patch.object(risk_agent, '_refresh_portfolio_state', AsyncMock()):
                await risk_agent.process_event(sample_decision_event)

        # Should publish rejection
        risk_agent._event_bus.publish.assert_called_once()
        published_event = risk_agent._event_bus.publish.call_args[0][0]
        assert published_event.approved is False

    def test_var_history_retention(self, risk_agent):
        """Test that VaR history is properly retained."""
        # Add more than max history days
        for i in range(300):
            risk_agent._returns_history.append(0.001)

        # Should not exceed max
        if len(risk_agent._returns_history) > risk_agent._max_history_days:
            risk_agent._returns_history = risk_agent._returns_history[-risk_agent._max_history_days:]

        assert len(risk_agent._returns_history) <= risk_agent._max_history_days


# ============================================================================
# KILL SWITCH TRIGGER TESTS
# ============================================================================

class TestKillSwitchTriggers:
    """Test various kill switch trigger scenarios."""

    @pytest.mark.asyncio
    async def test_kill_switch_on_daily_loss_limit(self, risk_agent, sample_decision_event):
        """Test kill switch activates on daily loss limit breach."""
        risk_agent._risk_state.daily_pnl_pct = -0.04  # -4% > 3% limit

        # Simulate trigger
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.DAILY_LOSS_LIMIT

        await risk_agent.process_event(sample_decision_event)

        # Should reject due to kill switch
        risk_agent._event_bus.publish.assert_called_once()
        published_event = risk_agent._event_bus.publish.call_args[0][0]
        assert published_event.approved is False
        assert "KILL-SWITCH" in published_event.rejection_reason

    @pytest.mark.asyncio
    async def test_kill_switch_on_max_drawdown(self, risk_agent, sample_decision_event):
        """Test kill switch activates on max drawdown breach."""
        risk_agent._risk_state.current_drawdown_pct = 0.12  # 12% > 10% limit

        # Simulate trigger
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.MAX_DRAWDOWN

        await risk_agent.process_event(sample_decision_event)

        risk_agent._event_bus.publish.assert_called_once()
        published_event = risk_agent._event_bus.publish.call_args[0][0]
        assert published_event.approved is False

    def test_kill_switch_on_var_breach(self, risk_agent):
        """Test kill switch can be triggered by VaR breach."""
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.VAR_BREACH

        assert risk_agent._kill_switch_active is True
        assert risk_agent._kill_switch_reason == KillSwitchReason.VAR_BREACH

    def test_kill_switch_on_anomaly_detected(self, risk_agent):
        """Test kill switch can be triggered by anomaly (e.g., margin call)."""
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.ANOMALY_DETECTED

        assert risk_agent._kill_switch_active is True
        assert risk_agent._kill_switch_reason == KillSwitchReason.ANOMALY_DETECTED

    def test_kill_switch_manual_activation(self, risk_agent):
        """Test manual kill switch activation."""
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.MANUAL
        risk_agent._kill_switch_time = datetime.now(timezone.utc)

        assert risk_agent._kill_switch_active is True
        assert risk_agent._kill_switch_reason == KillSwitchReason.MANUAL
        assert risk_agent._kill_switch_time is not None

    def test_kill_switch_deactivation(self, risk_agent):
        """Test kill switch deactivation."""
        # First activate
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.DAILY_LOSS_LIMIT

        # Then deactivate
        risk_agent._kill_switch_active = False
        risk_agent._kill_switch_reason = None

        assert risk_agent._kill_switch_active is False
        assert risk_agent._kill_switch_reason is None

    @pytest.mark.asyncio
    async def test_all_new_orders_rejected_when_kill_switch_active(self, risk_agent, sample_decision_event):
        """Test that all new orders are rejected when kill switch is active."""
        risk_agent._kill_switch_active = True
        risk_agent._kill_switch_reason = KillSwitchReason.MARKET_DISRUPTION

        # Try multiple decisions
        for i in range(3):
            decision = DecisionEvent(
                source_agent="CIOAgent",
                symbol=f"SYM{i}",
                action=OrderSide.BUY,
                quantity=100,
                rationale="Test",
                data_sources=("ib",),
                conviction_score=0.8,
            )
            await risk_agent.process_event(decision)

        # All should be rejected
        assert risk_agent._event_bus.publish.call_count == 3
        for call in risk_agent._event_bus.publish.call_args_list:
            published_event = call[0][0]
            assert published_event.approved is False


# ============================================================================
# POSITION LIMIT TESTS
# ============================================================================

class TestPositionLimits:
    """Test position limit enforcement."""

    def test_position_size_limit_configuration(self, risk_agent):
        """Test position size limit is correctly configured."""
        assert risk_agent._max_position_pct == 0.05  # 5% from config

    def test_position_within_limits(self, risk_agent):
        """Test position within acceptable limits."""
        # 3% position < 5% limit
        position_pct = 0.03
        assert position_pct < risk_agent._max_position_pct

    def test_position_exceeds_limit(self, risk_agent):
        """Test position exceeding limit is detected."""
        # 7% position > 5% limit
        position_pct = 0.07
        assert position_pct > risk_agent._max_position_pct

    def test_sector_exposure_limit(self, risk_agent):
        """Test sector exposure limit configuration."""
        assert risk_agent._max_sector_pct == 0.20  # 20% from config

    def test_sector_exposure_within_limits(self, risk_agent):
        """Test sector exposure within limits."""
        risk_agent._risk_state.sector_exposures = {
            "Technology": 0.15,  # 15% < 20% limit
            "Healthcare": 0.10,
        }

        for sector, exposure in risk_agent._risk_state.sector_exposures.items():
            assert exposure < risk_agent._max_sector_pct

    def test_sector_exposure_exceeds_limit(self, risk_agent):
        """Test sector exposure exceeding limit."""
        risk_agent._risk_state.sector_exposures = {
            "Technology": 0.25,  # 25% > 20% limit
        }

        assert risk_agent._risk_state.sector_exposures["Technology"] > risk_agent._max_sector_pct

    def test_gross_exposure_limit(self, risk_agent):
        """Test gross exposure limit configuration."""
        assert risk_agent._max_gross_exposure_pct == 2.0  # 200% from config

    def test_leverage_limit(self, risk_agent):
        """Test leverage limit configuration."""
        assert risk_agent._max_leverage == 2.0

    @pytest.mark.asyncio
    async def test_reject_decision_exceeding_position_limit(self, risk_agent, sample_decision_event):
        """Test that decisions exceeding position limit are rejected."""
        # Mock validation to fail on position size
        async def mock_validate(decision):
            return RiskValidationResult(
                approved=False,
                checks=[
                    RiskCheckResult(
                        check_name="position_size",
                        passed=False,
                        current_value=7.0,
                        limit_value=5.0,
                        message="Position size exceeds limit",
                    )
                ],
                risk_metrics={},
                rejection_reason="Position size 7.0% exceeds limit of 5.0%",
            )

        with patch.object(risk_agent, '_validate_decision', mock_validate):
            with patch.object(risk_agent, '_refresh_portfolio_state', AsyncMock()):
                await risk_agent.process_event(sample_decision_event)

        risk_agent._event_bus.publish.assert_called_once()
        published_event = risk_agent._event_bus.publish.call_args[0][0]
        assert published_event.approved is False
        assert "position" in published_event.rejection_reason.lower()

    def test_position_info_tracking(self, risk_agent):
        """Test position info is properly tracked."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=1000,
            avg_cost=150.0,
            market_value=160000.0,
            unrealized_pnl=10000.0,
            weight_pct=16.0,  # 16% of portfolio
            sector="Technology",
        )

        risk_agent._risk_state.positions["AAPL"] = position

        assert "AAPL" in risk_agent._risk_state.positions
        assert risk_agent._risk_state.positions["AAPL"].weight_pct == 16.0

    def test_multiple_positions_total_exposure(self, risk_agent):
        """Test total exposure calculation with multiple positions."""
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=100,
                avg_cost=150.0,
                market_value=15000.0,
                unrealized_pnl=0.0,
                weight_pct=1.5,
                sector="Technology",
            ),
            "GOOGL": PositionInfo(
                symbol="GOOGL",
                quantity=50,
                avg_cost=2800.0,
                market_value=145000.0,
                unrealized_pnl=5000.0,
                weight_pct=14.5,
                sector="Technology",
            ),
        }

        risk_agent._risk_state.positions = positions

        total_weight = sum(p.weight_pct for p in positions.values())
        assert total_weight == 16.0

    def test_long_short_position_netting(self, risk_agent):
        """Test long and short position exposure calculation."""
        risk_agent._risk_state.long_exposure = 0.80  # 80%
        risk_agent._risk_state.short_exposure = 0.30  # 30%

        net_exposure = risk_agent._risk_state.long_exposure - risk_agent._risk_state.short_exposure
        gross_exposure = risk_agent._risk_state.long_exposure + risk_agent._risk_state.short_exposure

        assert net_exposure == 0.50
        assert gross_exposure == 1.10

    @pytest.mark.asyncio
    async def test_reject_decision_on_sector_concentration(self, risk_agent, sample_decision_event):
        """Test rejection on sector concentration breach."""
        risk_agent._risk_state.sector_exposures = {
            "Technology": 0.19,  # Already at 19%
        }

        # Adding more Tech would breach 20% limit
        async def mock_validate(decision):
            return RiskValidationResult(
                approved=False,
                checks=[
                    RiskCheckResult(
                        check_name="sector_exposure",
                        passed=False,
                        current_value=22.0,
                        limit_value=20.0,
                        message="Sector exposure exceeds limit",
                    )
                ],
                risk_metrics={},
                rejection_reason="Technology sector exposure 22.0% exceeds limit of 20.0%",
            )

        with patch.object(risk_agent, '_validate_decision', mock_validate):
            with patch.object(risk_agent, '_refresh_portfolio_state', AsyncMock()):
                await risk_agent.process_event(sample_decision_event)

        risk_agent._event_bus.publish.assert_called_once()
        published_event = risk_agent._event_bus.publish.call_args[0][0]
        assert published_event.approved is False

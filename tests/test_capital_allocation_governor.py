"""
Tests for Capital Allocation Governor
=====================================

Tests cover:
- Strategy registration
- Regime-based allocation adjustments
- Drawdown-triggered capital reduction
- Rebalancing triggers
- Budget enforcement
"""

import pytest
from datetime import datetime, timezone, timedelta

from core.capital_allocation_governor import (
    CapitalAllocationGovernor,
    StrategyBudget,
    MarketRegime,
    DrawdownLevel,
    create_default_governor,
)


class TestStrategyBudget:
    """Tests for StrategyBudget dataclass."""

    def test_effective_allocation_no_adjustments(self):
        """Base allocation should be returned with no adjustments."""
        budget = StrategyBudget(
            strategy_name="TestStrategy",
            base_allocation_pct=10.0,
            current_allocation_pct=10.0,
        )
        assert budget.effective_allocation_pct == 10.0

    def test_effective_allocation_with_multipliers(self):
        """Multipliers should affect effective allocation."""
        budget = StrategyBudget(
            strategy_name="TestStrategy",
            base_allocation_pct=10.0,
            current_allocation_pct=10.0,
            regime_multiplier=1.5,
            drawdown_multiplier=0.8,
        )
        # 10.0 * 1.5 * 0.8 = 12.0
        assert budget.effective_allocation_pct == 12.0

    def test_effective_allocation_respects_max(self):
        """Effective allocation should not exceed max."""
        budget = StrategyBudget(
            strategy_name="TestStrategy",
            base_allocation_pct=20.0,
            current_allocation_pct=20.0,
            regime_multiplier=2.0,  # Would be 40%
            max_allocation_pct=25.0,
        )
        assert budget.effective_allocation_pct == 25.0

    def test_effective_allocation_respects_min(self):
        """Effective allocation should not go below min."""
        budget = StrategyBudget(
            strategy_name="TestStrategy",
            base_allocation_pct=10.0,
            current_allocation_pct=10.0,
            regime_multiplier=0.1,  # Would be 1%
            min_allocation_pct=5.0,
        )
        assert budget.effective_allocation_pct == 5.0

    def test_available_pct(self):
        """Available should be effective minus used."""
        budget = StrategyBudget(
            strategy_name="TestStrategy",
            base_allocation_pct=10.0,
            current_allocation_pct=10.0,
            current_usage_pct=3.0,
        )
        assert budget.available_pct == 7.0

    def test_available_pct_not_negative(self):
        """Available should never be negative."""
        budget = StrategyBudget(
            strategy_name="TestStrategy",
            base_allocation_pct=10.0,
            current_allocation_pct=10.0,
            current_usage_pct=15.0,  # Over-allocated
        )
        assert budget.available_pct == 0.0


class TestCapitalAllocationGovernor:
    """Tests for the Capital Allocation Governor."""

    @pytest.fixture
    def governor(self):
        """Create a governor for testing."""
        return CapitalAllocationGovernor(
            total_capital=1_000_000.0,
            rebalance_threshold_pct=5.0,
            min_rebalance_interval_minutes=60.0,
        )

    def test_register_strategy(self, governor):
        """Should register strategies correctly."""
        governor.register_strategy(
            "TestStrategy",
            base_allocation_pct=15.0,
            min_allocation_pct=5.0,
            max_allocation_pct=25.0,
        )

        assert "TestStrategy" in governor._budgets
        budget = governor._budgets["TestStrategy"]
        assert budget.base_allocation_pct == 15.0
        assert budget.min_allocation_pct == 5.0
        assert budget.max_allocation_pct == 25.0

    def test_update_regime_risk_on(self, governor):
        """Risk-on regime should increase momentum allocation."""
        governor.register_strategy("MomentumAgent", 20.0)
        governor.register_strategy("MacroAgent", 15.0)

        allocations = governor.update_regime(MarketRegime.RISK_ON)

        # MomentumAgent should have increased allocation
        assert allocations["MomentumAgent"] > 20.0
        # MacroAgent should have decreased
        assert allocations["MacroAgent"] < 15.0

    def test_update_regime_risk_off(self, governor):
        """Risk-off regime should favor defensive strategies."""
        governor.register_strategy("MomentumAgent", 20.0)
        governor.register_strategy("MacroAgent", 15.0)

        allocations = governor.update_regime(MarketRegime.RISK_OFF)

        # MacroAgent should have increased allocation
        assert allocations["MacroAgent"] > 15.0
        # MomentumAgent should have decreased
        assert allocations["MomentumAgent"] < 20.0

    def test_update_regime_tracks_changes(self, governor):
        """Regime changes should be tracked in stats."""
        governor.register_strategy("TestStrategy", 10.0)

        governor.update_regime(MarketRegime.RISK_ON)
        governor.update_regime(MarketRegime.VOLATILE)

        stats = governor.get_statistics()
        assert stats["regime_changes"] == 2

    def test_update_regime_same_no_change(self, governor):
        """Same regime should not increment counter."""
        governor.register_strategy("TestStrategy", 10.0)

        governor.update_regime(MarketRegime.RISK_ON)
        governor.update_regime(MarketRegime.RISK_ON)  # Same

        stats = governor.get_statistics()
        assert stats["regime_changes"] == 1

    def test_update_drawdown_normal(self, governor):
        """Normal drawdown should not reduce allocations."""
        governor.register_strategy("TestStrategy", 20.0)

        allocations = governor.update_drawdown(0.02)  # 2% drawdown

        assert allocations["TestStrategy"] == 20.0  # No reduction

    def test_update_drawdown_warning(self, governor):
        """Warning drawdown should reduce allocations by 20%."""
        governor.register_strategy("TestStrategy", 20.0)

        allocations = governor.update_drawdown(0.04)  # 4% drawdown

        assert allocations["TestStrategy"] == 16.0  # 20% reduction

    def test_update_drawdown_critical(self, governor):
        """Critical drawdown should significantly reduce allocations."""
        governor.register_strategy("TestStrategy", 20.0)

        allocations = governor.update_drawdown(0.09)  # 9% drawdown

        assert allocations["TestStrategy"] == 8.0  # 60% reduction

    def test_update_drawdown_severe(self, governor):
        """Severe drawdown should minimize allocations."""
        governor.register_strategy("TestStrategy", 20.0)

        allocations = governor.update_drawdown(0.12)  # 12% drawdown

        assert allocations["TestStrategy"] == 4.0  # 80% reduction

    def test_can_allocate_within_budget(self, governor):
        """Should allow allocation within budget."""
        governor.register_strategy("TestStrategy", 20.0)

        can_alloc, max_allowed, reason = governor.can_allocate("TestStrategy", 15.0)

        assert can_alloc is True
        assert max_allowed == 15.0
        assert reason == ""

    def test_can_allocate_exceeds_budget(self, governor):
        """Should reduce allocation to available budget."""
        governor.register_strategy("TestStrategy", 20.0)
        governor.update_strategy_usage("TestStrategy", 15.0)  # Using 15%

        can_alloc, max_allowed, reason = governor.can_allocate("TestStrategy", 10.0)

        assert can_alloc is True
        assert max_allowed == 5.0  # Only 5% available
        assert "Reduced" in reason

    def test_can_allocate_no_budget(self, governor):
        """Should reject allocation when no budget available."""
        governor.register_strategy("TestStrategy", 20.0)
        governor.update_strategy_usage("TestStrategy", 20.0)  # Fully used

        can_alloc, max_allowed, reason = governor.can_allocate("TestStrategy", 5.0)

        assert can_alloc is False
        assert max_allowed == 0.0
        assert "No budget" in reason

    def test_can_allocate_unregistered(self, governor):
        """Should reject allocation for unregistered strategy."""
        can_alloc, max_allowed, reason = governor.can_allocate("Unknown", 10.0)

        assert can_alloc is False
        assert "not registered" in reason

    def test_check_rebalance_not_needed(self, governor):
        """Should not trigger rebalance when drift is small."""
        governor.register_strategy("TestStrategy", 20.0)
        governor.update_strategy_usage("TestStrategy", 19.0)  # 1% drift

        needed, drift = governor.check_rebalance_needed()

        assert needed is False

    def test_check_rebalance_needed(self, governor):
        """Should trigger rebalance when drift exceeds threshold."""
        governor.register_strategy("TestStrategy", 20.0)
        governor.update_strategy_usage("TestStrategy", 10.0)  # 10% drift

        needed, drift = governor.check_rebalance_needed()

        assert needed is True
        assert drift["TestStrategy"] == 10.0

    def test_check_rebalance_interval(self, governor):
        """Should respect minimum rebalance interval."""
        governor.register_strategy("TestStrategy", 20.0)
        governor.update_strategy_usage("TestStrategy", 10.0)
        governor._last_rebalance_time = datetime.now(timezone.utc)

        needed, _ = governor.check_rebalance_needed()

        assert needed is False  # Too soon after last rebalance

    def test_calculate_rebalance_orders(self, governor):
        """Should calculate correct rebalance orders."""
        governor.register_strategy("Strategy1", 20.0)
        governor.register_strategy("Strategy2", 15.0)
        governor.update_strategy_usage("Strategy1", 10.0)
        governor.update_strategy_usage("Strategy2", 20.0)

        orders = governor.calculate_rebalance_orders()

        assert len(orders) == 2
        # Strategy1 needs increase
        s1_order = next(o for o in orders if o["strategy"] == "Strategy1")
        assert s1_order["action"] == "increase"
        assert s1_order["change_pct"] == 10.0

        # Strategy2 needs decrease
        s2_order = next(o for o in orders if o["strategy"] == "Strategy2")
        assert s2_order["action"] == "decrease"
        assert s2_order["change_pct"] == -5.0

    def test_set_total_capital(self, governor):
        """Should update total capital."""
        governor.set_total_capital(2_000_000.0)

        assert governor._total_capital == 2_000_000.0

    def test_set_total_capital_zero_safe(self, governor):
        """Should handle zero previous capital safely."""
        # Create new governor with 0 initial capital
        gov = CapitalAllocationGovernor(total_capital=0.0)
        gov.set_total_capital(1_000_000.0)

        assert gov._total_capital == 1_000_000.0

    def test_get_statistics(self, governor):
        """Should return comprehensive statistics."""
        governor.register_strategy("TestStrategy", 20.0)
        governor.update_regime(MarketRegime.RISK_ON)
        governor.update_drawdown(0.05)

        stats = governor.get_statistics()

        assert stats["total_capital"] == 1_000_000.0
        assert stats["current_regime"] == "risk_on"
        assert stats["drawdown_level"] == "elevated"
        assert stats["registered_strategies"] == 1

    def test_get_strategy_budgets(self, governor):
        """Should return detailed budget info."""
        governor.register_strategy("TestStrategy", 20.0, 5.0, 30.0)
        governor.update_strategy_usage("TestStrategy", 10.0)

        budgets = governor.get_strategy_budgets()

        assert len(budgets) == 1
        budget = budgets[0]
        assert budget["strategy"] == "TestStrategy"
        assert budget["base_allocation_pct"] == 20.0
        assert budget["current_usage_pct"] == 10.0
        assert budget["available_pct"] == 10.0


class TestCreateDefaultGovernor:
    """Tests for factory function."""

    def test_creates_with_default_strategies(self):
        """Should create governor with default strategy allocations."""
        governor = create_default_governor(total_capital=500_000.0)

        assert governor._total_capital == 500_000.0
        assert "MacroAgent" in governor._budgets
        assert "MomentumAgent" in governor._budgets
        assert "StatArbAgent" in governor._budgets

    def test_default_allocations_sum_reasonable(self):
        """Default allocations should sum to reasonable total."""
        governor = create_default_governor()

        total = sum(b.base_allocation_pct for b in governor._budgets.values())
        assert 90.0 <= total <= 110.0  # Allow some flexibility


class TestRegimeScenarios:
    """Integration tests for regime scenarios."""

    def test_full_cycle_risk_on_to_off(self):
        """Test full cycle from risk-on to risk-off."""
        governor = create_default_governor()

        # Start in risk-on
        alloc_risk_on = governor.update_regime(MarketRegime.RISK_ON)
        momentum_risk_on = alloc_risk_on["MomentumAgent"]

        # Move to risk-off
        alloc_risk_off = governor.update_regime(MarketRegime.RISK_OFF)
        momentum_risk_off = alloc_risk_off["MomentumAgent"]

        # Momentum should be lower in risk-off
        assert momentum_risk_off < momentum_risk_on

    def test_drawdown_escalation(self):
        """Test progressive drawdown escalation."""
        governor = create_default_governor()

        levels = []
        for dd in [0.02, 0.04, 0.06, 0.09, 0.12]:
            governor.update_drawdown(dd)
            levels.append(governor._drawdown_level)

        # Should escalate through levels
        assert levels[0] == DrawdownLevel.NORMAL
        assert levels[1] == DrawdownLevel.WARNING
        assert levels[2] == DrawdownLevel.ELEVATED
        assert levels[3] == DrawdownLevel.CRITICAL
        assert levels[4] == DrawdownLevel.SEVERE

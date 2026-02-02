"""
Cross-Strategy Risk Budget Manager
==================================

Implements risk budget allocation across strategies and portfolio rebalancing triggers
per Expert Review Issues #P3 and #P4.

Features:
- Cross-strategy risk budget allocation with various methods
- Risk consumption tracking per strategy
- Dynamic risk budget adjustment based on performance
- Portfolio rebalancing triggers (threshold-based, time-based, drift-based)
- Risk parity allocation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Risk budget allocation methods."""
    EQUAL = "equal"  # Equal risk budget to each strategy
    RISK_PARITY = "risk_parity"  # Proportional to inverse volatility
    PERFORMANCE_WEIGHTED = "performance_weighted"  # Based on Sharpe ratio
    FIXED = "fixed"  # Fixed percentages from config
    DRAWDOWN_ADJUSTED = "drawdown_adjusted"  # Reduce budget when in drawdown


class RebalanceTrigger(Enum):
    """Types of rebalancing triggers."""
    THRESHOLD = "threshold"  # Drift exceeds threshold
    TIME = "time"  # Regular time-based rebalancing
    DRAWDOWN = "drawdown"  # Strategy drawdown threshold
    PERFORMANCE = "performance"  # Performance deviation
    VOLATILITY = "volatility"  # Volatility regime change
    MANUAL = "manual"  # Manual trigger


@dataclass
class StrategyRiskBudget:
    """Risk budget allocation for a single strategy."""
    strategy: str
    target_allocation: float  # Target risk allocation (0-1)
    current_allocation: float  # Actual current risk consumption
    min_allocation: float = 0.05  # Minimum allowed allocation
    max_allocation: float = 0.40  # Maximum allowed allocation
    current_var: float = 0.0  # Current VaR consumption
    max_var: float = 0.0  # Maximum allowed VaR
    current_drawdown: float = 0.0  # Current drawdown
    max_drawdown: float = 0.10  # Maximum allowed drawdown
    volatility: float = 0.0  # Strategy volatility
    sharpe_ratio: float = 0.0  # Rolling Sharpe ratio
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_frozen: bool = False  # True if strategy is frozen due to limit breach
    freeze_reason: str = ""


@dataclass
class RebalanceEvent:
    """A rebalancing event triggered by the system."""
    timestamp: datetime
    trigger: RebalanceTrigger
    strategies_affected: list[str]
    old_allocations: dict[str, float]
    new_allocations: dict[str, float]
    reason: str


@dataclass
class RiskBudgetConfig:
    """Configuration for risk budget manager."""
    # Allocation method
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY

    # Total portfolio risk budget (as fraction of portfolio)
    total_risk_budget: float = 0.02  # 2% of portfolio

    # Rebalancing thresholds
    rebalance_drift_threshold: float = 0.20  # 20% drift from target
    rebalance_time_interval_hours: int = 24  # Daily rebalancing check

    # Drawdown limits
    strategy_max_drawdown: float = 0.10  # 10% max drawdown per strategy
    portfolio_max_drawdown: float = 0.15  # 15% max portfolio drawdown

    # Performance thresholds
    sharpe_freeze_threshold: float = -0.5  # Freeze strategy if Sharpe < -0.5
    sharpe_recovery_threshold: float = 0.0  # Unfreeze when Sharpe > 0

    # Volatility regime
    vol_scaling_enabled: bool = True
    high_vol_threshold: float = 0.30  # 30% annualized vol is "high"
    high_vol_reduction: float = 0.50  # Reduce budget by 50% in high vol


class RiskBudgetManager:
    """
    Cross-Strategy Risk Budget Manager.

    Manages risk allocation across multiple trading strategies,
    ensuring total risk stays within portfolio limits.
    """

    def __init__(
        self,
        config: RiskBudgetConfig | None = None,
        fixed_allocations: dict[str, float] | None = None
    ):
        """
        Initialize risk budget manager.

        Args:
            config: Risk budget configuration
            fixed_allocations: Fixed strategy allocations (for FIXED method)
        """
        self._config = config or RiskBudgetConfig()
        self._fixed_allocations = fixed_allocations or {}

        # Strategy budgets
        self._budgets: dict[str, StrategyRiskBudget] = {}

        # Rebalancing history
        self._rebalance_history: list[RebalanceEvent] = []
        self._last_rebalance: datetime | None = None

        # Portfolio-level tracking
        self._portfolio_var: float = 0.0
        self._portfolio_drawdown: float = 0.0
        self._portfolio_high_water_mark: float = 0.0

        logger.info(
            f"RiskBudgetManager initialized: "
            f"method={self._config.allocation_method.value}, "
            f"total_budget={self._config.total_risk_budget:.1%}"
        )

    def register_strategy(
        self,
        strategy: str,
        initial_allocation: float | None = None,
        min_allocation: float = 0.05,
        max_allocation: float = 0.40
    ) -> None:
        """
        Register a strategy for risk budget tracking.

        Args:
            strategy: Strategy name
            initial_allocation: Initial target allocation (optional)
            min_allocation: Minimum allowed allocation
            max_allocation: Maximum allowed allocation
        """
        if strategy in self._budgets:
            logger.warning(f"Strategy {strategy} already registered, updating")

        # Use fixed allocation if available, otherwise equal split
        if initial_allocation is None:
            if strategy in self._fixed_allocations:
                initial_allocation = self._fixed_allocations[strategy]
            else:
                # Equal split among registered strategies
                n_strategies = len(self._budgets) + 1
                initial_allocation = 1.0 / n_strategies

        budget = StrategyRiskBudget(
            strategy=strategy,
            target_allocation=initial_allocation,
            current_allocation=0.0,
            min_allocation=min_allocation,
            max_allocation=max_allocation,
            max_var=self._config.total_risk_budget * initial_allocation,
            max_drawdown=self._config.strategy_max_drawdown,
        )

        self._budgets[strategy] = budget

        # Recalculate allocations if not using fixed method
        if self._config.allocation_method != AllocationMethod.FIXED:
            self._recalculate_allocations()

        logger.info(
            f"Registered strategy {strategy}: "
            f"target={budget.target_allocation:.1%}, "
            f"max_var={budget.max_var:.2%}"
        )

    def update_strategy_metrics(
        self,
        strategy: str,
        current_var: float = 0.0,
        volatility: float = 0.0,
        sharpe_ratio: float = 0.0,
        current_drawdown: float = 0.0,
        pnl: float = 0.0
    ) -> None:
        """
        Update strategy metrics for risk budget calculation.

        Args:
            strategy: Strategy name
            current_var: Current VaR consumption
            volatility: Annualized volatility
            sharpe_ratio: Rolling Sharpe ratio
            current_drawdown: Current drawdown (as positive fraction)
            pnl: Latest P&L
        """
        if strategy not in self._budgets:
            logger.warning(f"Unknown strategy {strategy}, ignoring metrics")
            return

        budget = self._budgets[strategy]
        budget.current_var = current_var
        budget.volatility = volatility
        budget.sharpe_ratio = sharpe_ratio
        budget.current_drawdown = current_drawdown
        budget.last_update = datetime.now(timezone.utc)

        # Calculate current allocation as fraction of total budget
        if self._config.total_risk_budget > 0:
            budget.current_allocation = current_var / self._config.total_risk_budget

        # Check for freeze conditions
        self._check_freeze_conditions(budget)

    def _check_freeze_conditions(self, budget: StrategyRiskBudget) -> None:
        """Check if strategy should be frozen."""
        # Drawdown check
        if budget.current_drawdown > budget.max_drawdown:
            if not budget.is_frozen:
                budget.is_frozen = True
                budget.freeze_reason = f"Drawdown {budget.current_drawdown:.1%} > {budget.max_drawdown:.1%}"
                logger.warning(f"Strategy {budget.strategy} FROZEN: {budget.freeze_reason}")
            return

        # Sharpe check
        if budget.sharpe_ratio < self._config.sharpe_freeze_threshold:
            if not budget.is_frozen:
                budget.is_frozen = True
                budget.freeze_reason = f"Sharpe {budget.sharpe_ratio:.2f} < {self._config.sharpe_freeze_threshold:.2f}"
                logger.warning(f"Strategy {budget.strategy} FROZEN: {budget.freeze_reason}")
            return

        # VaR check
        if budget.current_var > budget.max_var * 1.1:  # 10% buffer
            if not budget.is_frozen:
                budget.is_frozen = True
                budget.freeze_reason = f"VaR {budget.current_var:.2%} > max {budget.max_var:.2%}"
                logger.warning(f"Strategy {budget.strategy} FROZEN: {budget.freeze_reason}")
            return

        # Check for unfreeze conditions
        if budget.is_frozen:
            can_unfreeze = (
                budget.current_drawdown < budget.max_drawdown * 0.8 and
                budget.sharpe_ratio > self._config.sharpe_recovery_threshold and
                budget.current_var < budget.max_var * 0.9
            )
            if can_unfreeze:
                budget.is_frozen = False
                budget.freeze_reason = ""
                logger.info(f"Strategy {budget.strategy} UNFROZEN")

    def _recalculate_allocations(self) -> None:
        """Recalculate target allocations based on method."""
        if len(self._budgets) == 0:
            return

        method = self._config.allocation_method

        if method == AllocationMethod.EQUAL:
            self._allocate_equal()
        elif method == AllocationMethod.RISK_PARITY:
            self._allocate_risk_parity()
        elif method == AllocationMethod.PERFORMANCE_WEIGHTED:
            self._allocate_performance_weighted()
        elif method == AllocationMethod.DRAWDOWN_ADJUSTED:
            self._allocate_drawdown_adjusted()
        elif method == AllocationMethod.FIXED:
            self._allocate_fixed()

        # Apply min/max constraints
        self._apply_allocation_constraints()

        # Update max VaR for each strategy
        for budget in self._budgets.values():
            budget.max_var = self._config.total_risk_budget * budget.target_allocation

    def _allocate_equal(self) -> None:
        """Equal allocation to all active strategies."""
        active_strategies = [s for s, b in self._budgets.items() if not b.is_frozen]
        if not active_strategies:
            return

        allocation = 1.0 / len(active_strategies)
        for strategy in active_strategies:
            self._budgets[strategy].target_allocation = allocation

        # Frozen strategies get 0
        for strategy, budget in self._budgets.items():
            if budget.is_frozen:
                budget.target_allocation = 0.0

    def _allocate_risk_parity(self) -> None:
        """
        Risk parity allocation - inverse volatility weighting.

        Allocates more to lower-volatility strategies.
        """
        active_budgets = {s: b for s, b in self._budgets.items() if not b.is_frozen}

        if not active_budgets:
            return

        # Calculate inverse volatility weights
        inv_vols = {}
        for strategy, budget in active_budgets.items():
            vol = max(budget.volatility, 0.01)  # Minimum 1% vol
            inv_vols[strategy] = 1.0 / vol

        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol > 0:
            for strategy, inv_vol in inv_vols.items():
                self._budgets[strategy].target_allocation = inv_vol / total_inv_vol

        # Frozen strategies get 0
        for strategy, budget in self._budgets.items():
            if budget.is_frozen:
                budget.target_allocation = 0.0

    def _allocate_performance_weighted(self) -> None:
        """
        Performance-weighted allocation based on Sharpe ratio.

        Allocates more to better-performing strategies.
        """
        active_budgets = {s: b for s, b in self._budgets.items() if not b.is_frozen}

        if not active_budgets:
            return

        # Use positive-shifted Sharpe for weighting
        sharpe_weights = {}
        min_sharpe = min(b.sharpe_ratio for b in active_budgets.values())
        shift = abs(min_sharpe) + 0.5 if min_sharpe < 0 else 0.5

        for strategy, budget in active_budgets.items():
            sharpe_weights[strategy] = max(budget.sharpe_ratio + shift, 0.1)

        total_weight = sum(sharpe_weights.values())

        if total_weight > 0:
            for strategy, weight in sharpe_weights.items():
                self._budgets[strategy].target_allocation = weight / total_weight

        # Frozen strategies get 0
        for strategy, budget in self._budgets.items():
            if budget.is_frozen:
                budget.target_allocation = 0.0

    def _allocate_drawdown_adjusted(self) -> None:
        """
        Drawdown-adjusted allocation.

        Reduces allocation to strategies in drawdown.
        """
        # Start with equal base
        self._allocate_equal()

        # Adjust based on drawdown
        for strategy, budget in self._budgets.items():
            if budget.is_frozen:
                continue

            # Reduce allocation based on drawdown severity
            if budget.current_drawdown > 0:
                dd_ratio = budget.current_drawdown / budget.max_drawdown
                reduction = min(dd_ratio * 0.5, 0.5)  # Max 50% reduction
                budget.target_allocation *= (1 - reduction)

        # Renormalize
        total = sum(b.target_allocation for b in self._budgets.values())
        if total > 0:
            for budget in self._budgets.values():
                budget.target_allocation /= total

    def _allocate_fixed(self) -> None:
        """Use fixed allocations from configuration."""
        for strategy, budget in self._budgets.items():
            if strategy in self._fixed_allocations:
                budget.target_allocation = self._fixed_allocations[strategy]
            else:
                # Fall back to equal split for unconfigured strategies
                budget.target_allocation = 1.0 / len(self._budgets)

    def _apply_allocation_constraints(self) -> None:
        """Apply min/max constraints and renormalize."""
        # Apply constraints
        for budget in self._budgets.values():
            if not budget.is_frozen:
                budget.target_allocation = max(
                    budget.min_allocation,
                    min(budget.max_allocation, budget.target_allocation)
                )

        # Renormalize to sum to 1.0
        total = sum(b.target_allocation for b in self._budgets.values())
        if total > 0:
            for budget in self._budgets.values():
                budget.target_allocation /= total

    def check_rebalance_triggers(self) -> list[RebalanceEvent]:
        """
        Check all rebalancing triggers and return any triggered events.

        Returns:
            List of triggered rebalance events
        """
        events = []

        # Check threshold trigger (drift from target)
        drift_event = self._check_drift_trigger()
        if drift_event:
            events.append(drift_event)

        # Check time-based trigger
        time_event = self._check_time_trigger()
        if time_event:
            events.append(time_event)

        # Check drawdown trigger
        dd_event = self._check_drawdown_trigger()
        if dd_event:
            events.append(dd_event)

        # Check volatility regime trigger
        vol_event = self._check_volatility_trigger()
        if vol_event:
            events.append(vol_event)

        return events

    def _check_drift_trigger(self) -> RebalanceEvent | None:
        """Check if allocation drift exceeds threshold."""
        affected = []
        for strategy, budget in self._budgets.items():
            if budget.target_allocation > 0:
                drift = abs(budget.current_allocation - budget.target_allocation)
                drift_pct = drift / budget.target_allocation
                if drift_pct > self._config.rebalance_drift_threshold:
                    affected.append(strategy)

        if affected:
            return self._create_rebalance_event(
                RebalanceTrigger.THRESHOLD,
                affected,
                f"Drift > {self._config.rebalance_drift_threshold:.0%}"
            )
        return None

    def _check_time_trigger(self) -> RebalanceEvent | None:
        """Check if time-based rebalancing is due."""
        if self._last_rebalance is None:
            return None

        hours_since = (datetime.now(timezone.utc) - self._last_rebalance).total_seconds() / 3600

        if hours_since >= self._config.rebalance_time_interval_hours:
            return self._create_rebalance_event(
                RebalanceTrigger.TIME,
                list(self._budgets.keys()),
                f"Time interval {hours_since:.1f}h >= {self._config.rebalance_time_interval_hours}h"
            )
        return None

    def _check_drawdown_trigger(self) -> RebalanceEvent | None:
        """Check for drawdown-triggered rebalancing."""
        affected = []
        for strategy, budget in self._budgets.items():
            if budget.current_drawdown > budget.max_drawdown * 0.8:  # 80% of max
                affected.append(strategy)

        if affected:
            return self._create_rebalance_event(
                RebalanceTrigger.DRAWDOWN,
                affected,
                f"Strategies approaching drawdown limit"
            )
        return None

    def _check_volatility_trigger(self) -> RebalanceEvent | None:
        """Check for volatility regime change trigger."""
        if not self._config.vol_scaling_enabled:
            return None

        # Calculate average portfolio volatility
        active_budgets = [b for b in self._budgets.values() if not b.is_frozen]
        if not active_budgets:
            return None

        avg_vol = sum(b.volatility * b.target_allocation for b in active_budgets)

        if avg_vol > self._config.high_vol_threshold:
            return self._create_rebalance_event(
                RebalanceTrigger.VOLATILITY,
                list(self._budgets.keys()),
                f"Portfolio vol {avg_vol:.1%} > {self._config.high_vol_threshold:.1%}"
            )
        return None

    def _create_rebalance_event(
        self,
        trigger: RebalanceTrigger,
        affected: list[str],
        reason: str
    ) -> RebalanceEvent:
        """Create a rebalancing event."""
        old_allocs = {s: b.target_allocation for s, b in self._budgets.items()}

        # Recalculate allocations
        self._recalculate_allocations()

        new_allocs = {s: b.target_allocation for s, b in self._budgets.items()}

        event = RebalanceEvent(
            timestamp=datetime.now(timezone.utc),
            trigger=trigger,
            strategies_affected=affected,
            old_allocations=old_allocs,
            new_allocations=new_allocs,
            reason=reason,
        )

        self._rebalance_history.append(event)
        self._last_rebalance = event.timestamp

        # Keep only last 100 events
        if len(self._rebalance_history) > 100:
            self._rebalance_history = self._rebalance_history[-100:]

        logger.info(
            f"Rebalance triggered: {trigger.value} - {reason}. "
            f"Affected: {affected}"
        )

        return event

    def trigger_manual_rebalance(self, reason: str = "Manual trigger") -> RebalanceEvent:
        """Manually trigger a rebalancing."""
        return self._create_rebalance_event(
            RebalanceTrigger.MANUAL,
            list(self._budgets.keys()),
            reason
        )

    def get_available_budget(self, strategy: str) -> float:
        """
        Get available risk budget for a strategy.

        Returns the maximum VaR that can be consumed.
        """
        if strategy not in self._budgets:
            return 0.0

        budget = self._budgets[strategy]

        if budget.is_frozen:
            return 0.0

        # Available = max - current
        available = budget.max_var - budget.current_var

        # Apply volatility scaling if enabled
        if self._config.vol_scaling_enabled:
            avg_vol = sum(
                b.volatility * b.target_allocation
                for b in self._budgets.values() if not b.is_frozen
            )
            if avg_vol > self._config.high_vol_threshold:
                available *= self._config.high_vol_reduction

        return max(0.0, available)

    def can_increase_position(self, strategy: str, additional_var: float) -> tuple[bool, str]:
        """
        Check if a strategy can increase its position.

        Args:
            strategy: Strategy name
            additional_var: Additional VaR from new position

        Returns:
            Tuple of (can_increase, reason)
        """
        if strategy not in self._budgets:
            return False, f"Unknown strategy: {strategy}"

        budget = self._budgets[strategy]

        if budget.is_frozen:
            return False, f"Strategy frozen: {budget.freeze_reason}"

        available = self.get_available_budget(strategy)

        if additional_var > available:
            return False, f"Insufficient budget: need {additional_var:.4f}, available {available:.4f}"

        return True, "OK"

    def get_budget(self, strategy: str) -> StrategyRiskBudget | None:
        """Get budget for a strategy."""
        return self._budgets.get(strategy)

    def get_all_budgets(self) -> dict[str, StrategyRiskBudget]:
        """Get all strategy budgets."""
        return dict(self._budgets)

    def get_portfolio_utilization(self) -> float:
        """Get total portfolio risk budget utilization."""
        total_var = sum(b.current_var for b in self._budgets.values())
        if self._config.total_risk_budget > 0:
            return total_var / self._config.total_risk_budget
        return 0.0

    def get_status(self) -> dict[str, Any]:
        """Get manager status for monitoring."""
        return {
            "allocation_method": self._config.allocation_method.value,
            "total_risk_budget": self._config.total_risk_budget,
            "portfolio_utilization": self.get_portfolio_utilization(),
            "strategies": {
                s: {
                    "target_allocation": b.target_allocation,
                    "current_allocation": b.current_allocation,
                    "current_var": b.current_var,
                    "max_var": b.max_var,
                    "available_budget": self.get_available_budget(s),
                    "is_frozen": b.is_frozen,
                    "freeze_reason": b.freeze_reason,
                    "sharpe": b.sharpe_ratio,
                    "drawdown": b.current_drawdown,
                }
                for s, b in self._budgets.items()
            },
            "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
            "rebalance_history_count": len(self._rebalance_history),
        }

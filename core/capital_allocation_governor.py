"""
Capital Allocation Governor
===========================

Manages capital allocation across strategies based on:
- Market regime (risk-on, risk-off, volatile, etc.)
- Current drawdown level
- Strategy performance and risk-adjusted returns
- Correlation between strategies

The governor enforces capital budgets and triggers rebalancing
when allocations drift too far from targets.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime for allocation adjustment."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    VOLATILE = "volatile"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    NEUTRAL = "neutral"


class DrawdownLevel(Enum):
    """Drawdown severity levels for capital reduction."""
    NORMAL = "normal"       # < 3%: Full allocation
    WARNING = "warning"     # 3-5%: Reduce by 20%
    ELEVATED = "elevated"   # 5-8%: Reduce by 40%
    CRITICAL = "critical"   # 8-10%: Reduce by 60%
    SEVERE = "severe"       # > 10%: Reduce by 80%


@dataclass
class StrategyBudget:
    """Capital budget for a strategy."""
    strategy_name: str
    base_allocation_pct: float  # Base % of total capital
    current_allocation_pct: float  # Current adjusted allocation
    regime_multiplier: float = 1.0
    drawdown_multiplier: float = 1.0
    performance_adjustment: float = 0.0
    min_allocation_pct: float = 0.0  # Floor
    max_allocation_pct: float = 25.0  # Ceiling
    current_usage_pct: float = 0.0  # How much is currently used
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def effective_allocation_pct(self) -> float:
        """Calculate effective allocation after all adjustments."""
        adjusted = self.base_allocation_pct * self.regime_multiplier * self.drawdown_multiplier
        adjusted += self.performance_adjustment
        return max(self.min_allocation_pct, min(self.max_allocation_pct, adjusted))

    @property
    def available_pct(self) -> float:
        """Available allocation (effective - used)."""
        return max(0, self.effective_allocation_pct - self.current_usage_pct)


@dataclass
class AllocationSnapshot:
    """Point-in-time snapshot of capital allocations."""
    timestamp: datetime
    total_capital: float
    regime: MarketRegime
    drawdown_level: DrawdownLevel
    drawdown_pct: float
    allocations: dict[str, float]  # strategy -> allocated %
    usage: dict[str, float]  # strategy -> used %


class CapitalAllocationGovernor:
    """
    Governs capital allocation across strategies.

    Features:
    - Regime-based allocation adjustments
    - Drawdown-triggered capital reduction
    - Performance-weighted rebalancing
    - Drift detection and rebalancing triggers
    - Capital budget enforcement
    """

    def __init__(
        self,
        total_capital: float = 1_000_000.0,
        rebalance_threshold_pct: float = 5.0,
        min_rebalance_interval_minutes: float = 60.0,
    ):
        """
        Initialize the Capital Allocation Governor.

        Args:
            total_capital: Total capital under management
            rebalance_threshold_pct: Trigger rebalance if drift exceeds this %
            min_rebalance_interval_minutes: Minimum time between rebalances
        """
        self._total_capital = total_capital
        self._rebalance_threshold_pct = rebalance_threshold_pct
        self._min_rebalance_interval = timedelta(minutes=min_rebalance_interval_minutes)

        # Current state
        self._current_regime = MarketRegime.NEUTRAL
        self._current_drawdown_pct = 0.0
        self._drawdown_level = DrawdownLevel.NORMAL

        # Strategy budgets
        self._budgets: dict[str, StrategyBudget] = {}

        # Regime-based allocation multipliers
        self._regime_multipliers = {
            MarketRegime.RISK_ON: {
                "MomentumAgent": 1.3,
                "StatArbAgent": 0.9,
                "MacroAgent": 0.8,
                "MarketMakingAgent": 1.2,
                "MACDvAgent": 1.2,
                "default": 1.0,
            },
            MarketRegime.RISK_OFF: {
                "MomentumAgent": 0.6,
                "StatArbAgent": 1.2,
                "MacroAgent": 1.5,
                "MarketMakingAgent": 0.7,
                "MACDvAgent": 0.8,
                "default": 0.8,
            },
            MarketRegime.VOLATILE: {
                "MomentumAgent": 0.5,
                "StatArbAgent": 0.7,
                "MacroAgent": 1.0,
                "MarketMakingAgent": 0.5,
                "MACDvAgent": 0.6,
                "default": 0.6,
            },
            MarketRegime.TRENDING: {
                "MomentumAgent": 1.4,
                "StatArbAgent": 0.8,
                "MacroAgent": 0.9,
                "MarketMakingAgent": 0.9,
                "MACDvAgent": 1.3,
                "default": 1.1,
            },
            MarketRegime.MEAN_REVERTING: {
                "MomentumAgent": 0.7,
                "StatArbAgent": 1.4,
                "MacroAgent": 1.0,
                "MarketMakingAgent": 1.3,
                "MACDvAgent": 0.8,
                "default": 1.0,
            },
            MarketRegime.NEUTRAL: {
                "default": 1.0,
            },
        }

        # Drawdown reduction factors
        self._drawdown_multipliers = {
            DrawdownLevel.NORMAL: 1.0,
            DrawdownLevel.WARNING: 0.80,
            DrawdownLevel.ELEVATED: 0.60,
            DrawdownLevel.CRITICAL: 0.40,
            DrawdownLevel.SEVERE: 0.20,
        }

        # Drawdown thresholds (% of capital)
        self._drawdown_thresholds = {
            DrawdownLevel.WARNING: 0.03,    # 3%
            DrawdownLevel.ELEVATED: 0.05,   # 5%
            DrawdownLevel.CRITICAL: 0.08,   # 8%
            DrawdownLevel.SEVERE: 0.10,     # 10%
        }

        # History tracking
        self._allocation_history: deque[AllocationSnapshot] = deque(maxlen=1000)
        self._rebalance_history: deque[dict] = deque(maxlen=100)
        self._last_rebalance_time: Optional[datetime] = None

        # Statistics
        self._stats = {
            "total_rebalances": 0,
            "regime_changes": 0,
            "drawdown_escalations": 0,
            "budget_violations_blocked": 0,
        }

        logger.info(
            f"CapitalAllocationGovernor initialized with ${total_capital:,.0f} capital, "
            f"{rebalance_threshold_pct}% rebalance threshold"
        )

    def register_strategy(
        self,
        strategy_name: str,
        base_allocation_pct: float,
        min_allocation_pct: float = 0.0,
        max_allocation_pct: float = 25.0,
    ) -> None:
        """
        Register a strategy with the governor.

        Args:
            strategy_name: Name of the strategy
            base_allocation_pct: Base allocation as % of total capital (0-100)
            min_allocation_pct: Minimum allocation (floor, 0-100)
            max_allocation_pct: Maximum allocation (ceiling, 0-100)

        Raises:
            ValueError: If allocation parameters are invalid
        """
        # CRITICAL FIX: Input validation for allocation parameters
        if not strategy_name or not strategy_name.strip():
            raise ValueError("strategy_name must be non-empty")
        if not (0.0 <= base_allocation_pct <= 100.0):
            raise ValueError(f"base_allocation_pct must be 0-100, got {base_allocation_pct}")
        if not (0.0 <= min_allocation_pct <= 100.0):
            raise ValueError(f"min_allocation_pct must be 0-100, got {min_allocation_pct}")
        if not (0.0 <= max_allocation_pct <= 100.0):
            raise ValueError(f"max_allocation_pct must be 0-100, got {max_allocation_pct}")
        if min_allocation_pct > max_allocation_pct:
            raise ValueError(f"min_allocation_pct ({min_allocation_pct}) cannot exceed max_allocation_pct ({max_allocation_pct})")
        if not (min_allocation_pct <= base_allocation_pct <= max_allocation_pct):
            raise ValueError(f"base_allocation_pct ({base_allocation_pct}) must be between min ({min_allocation_pct}) and max ({max_allocation_pct})")

        self._budgets[strategy_name] = StrategyBudget(
            strategy_name=strategy_name,
            base_allocation_pct=base_allocation_pct,
            current_allocation_pct=base_allocation_pct,
            min_allocation_pct=min_allocation_pct,
            max_allocation_pct=max_allocation_pct,
        )

        logger.info(
            f"Registered strategy {strategy_name}: {base_allocation_pct}% base allocation "
            f"(min={min_allocation_pct}%, max={max_allocation_pct}%)"
        )

    def update_regime(self, regime: MarketRegime) -> dict[str, float]:
        """
        Update market regime and recalculate allocations.

        Args:
            regime: New market regime

        Returns:
            Dict of strategy -> new effective allocation %
        """
        if regime == self._current_regime:
            return self.get_current_allocations()

        old_regime = self._current_regime
        self._current_regime = regime
        self._stats["regime_changes"] += 1

        logger.info(f"Regime changed: {old_regime.value} -> {regime.value}")

        # Update all regime multipliers
        regime_mults = self._regime_multipliers.get(regime, {})
        default_mult = regime_mults.get("default", 1.0)

        for strategy_name, budget in self._budgets.items():
            mult = regime_mults.get(strategy_name, default_mult)
            budget.regime_multiplier = mult
            budget.last_updated = datetime.now(timezone.utc)

        self._record_allocation_snapshot()
        return self.get_current_allocations()

    def update_drawdown(self, drawdown_pct: float) -> dict[str, float]:
        """
        Update current drawdown and adjust allocations.

        Args:
            drawdown_pct: Current drawdown as decimal (0.05 = 5%, must be 0.0-1.0)

        Returns:
            Dict of strategy -> new effective allocation %

        Raises:
            ValueError: If drawdown_pct is outside valid range
        """
        # CRITICAL FIX: Validate drawdown percentage
        if not isinstance(drawdown_pct, (int, float)):
            raise ValueError(f"drawdown_pct must be numeric, got {type(drawdown_pct)}")
        if drawdown_pct < 0.0:
            logger.warning(f"Negative drawdown ({drawdown_pct}) clamped to 0")
            drawdown_pct = 0.0
        if drawdown_pct > 1.0:
            logger.warning(f"Drawdown > 100% ({drawdown_pct*100:.1f}%) clamped to 1.0")
            drawdown_pct = 1.0

        self._current_drawdown_pct = drawdown_pct

        # Determine drawdown level
        old_level = self._drawdown_level
        new_level = DrawdownLevel.NORMAL

        for level, threshold in sorted(
            self._drawdown_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if drawdown_pct >= threshold:
                new_level = level
                break

        if new_level != old_level:
            self._drawdown_level = new_level
            self._stats["drawdown_escalations"] += 1

            logger.warning(
                f"Drawdown level changed: {old_level.value} -> {new_level.value} "
                f"(drawdown: {drawdown_pct*100:.1f}%)"
            )

        # Update all drawdown multipliers
        dd_mult = self._drawdown_multipliers.get(new_level, 1.0)
        for budget in self._budgets.values():
            budget.drawdown_multiplier = dd_mult
            budget.last_updated = datetime.now(timezone.utc)

        self._record_allocation_snapshot()
        return self.get_current_allocations()

    def update_strategy_usage(self, strategy_name: str, usage_pct: float) -> None:
        """
        Update how much of a strategy's budget is currently in use.

        Args:
            strategy_name: Name of the strategy
            usage_pct: Current usage as % of total capital
        """
        if strategy_name in self._budgets:
            self._budgets[strategy_name].current_usage_pct = usage_pct

    def update_performance_adjustment(
        self,
        strategy_name: str,
        adjustment_pct: float,
    ) -> None:
        """
        Apply performance-based allocation adjustment.

        Args:
            strategy_name: Name of the strategy
            adjustment_pct: Adjustment as % points (can be negative)
        """
        if strategy_name in self._budgets:
            self._budgets[strategy_name].performance_adjustment = adjustment_pct
            self._budgets[strategy_name].last_updated = datetime.now(timezone.utc)

    def can_allocate(
        self,
        strategy_name: str,
        requested_pct: float,
    ) -> tuple[bool, float, str]:
        """
        Check if a strategy can allocate requested capital.

        Args:
            strategy_name: Strategy requesting allocation
            requested_pct: Requested allocation as % of total capital

        Returns:
            Tuple of (allowed, max_allowed_pct, reason)
        """
        if strategy_name not in self._budgets:
            return False, 0.0, f"Strategy {strategy_name} not registered"

        budget = self._budgets[strategy_name]
        available = budget.available_pct

        if requested_pct <= available:
            return True, requested_pct, ""

        # Partial allocation allowed
        if available > 0:
            self._stats["budget_violations_blocked"] += 1
            return True, available, f"Reduced from {requested_pct:.1f}% to {available:.1f}% (budget limit)"

        self._stats["budget_violations_blocked"] += 1
        return False, 0.0, f"No budget available (used: {budget.current_usage_pct:.1f}%, effective: {budget.effective_allocation_pct:.1f}%)"

    def get_current_allocations(self) -> dict[str, float]:
        """Get current effective allocations for all strategies."""
        return {
            name: budget.effective_allocation_pct
            for name, budget in self._budgets.items()
        }

    def get_available_allocations(self) -> dict[str, float]:
        """Get available (unused) allocations for all strategies."""
        return {
            name: budget.available_pct
            for name, budget in self._budgets.items()
        }

    def check_rebalance_needed(self) -> tuple[bool, dict[str, float]]:
        """
        Check if rebalancing is needed based on allocation drift.

        Returns:
            Tuple of (rebalance_needed, drift_by_strategy)
        """
        # Check minimum interval
        now = datetime.now(timezone.utc)
        if self._last_rebalance_time:
            if now - self._last_rebalance_time < self._min_rebalance_interval:
                return False, {}

        # Calculate drift from target
        drift = {}
        max_drift = 0.0

        for name, budget in self._budgets.items():
            target = budget.effective_allocation_pct
            current = budget.current_usage_pct
            strategy_drift = abs(current - target)
            drift[name] = strategy_drift
            max_drift = max(max_drift, strategy_drift)

        rebalance_needed = max_drift >= self._rebalance_threshold_pct
        return rebalance_needed, drift

    def calculate_rebalance_orders(self) -> list[dict]:
        """
        Calculate what trades are needed to rebalance to targets.

        Returns:
            List of rebalance instructions
        """
        orders = []

        for name, budget in self._budgets.items():
            target = budget.effective_allocation_pct
            current = budget.current_usage_pct
            diff = target - current

            if abs(diff) >= 0.5:  # Only if >0.5% difference
                orders.append({
                    "strategy": name,
                    "action": "increase" if diff > 0 else "decrease",
                    "current_pct": current,
                    "target_pct": target,
                    "change_pct": diff,
                    "change_value": (diff / 100) * self._total_capital,
                })

        return orders

    def record_rebalance(self, orders_executed: list[dict]) -> None:
        """Record a rebalance event."""
        now = datetime.now(timezone.utc)
        self._last_rebalance_time = now
        self._stats["total_rebalances"] += 1

        record = {
            "timestamp": now.isoformat(),
            "regime": self._current_regime.value,
            "drawdown_level": self._drawdown_level.value,
            "drawdown_pct": self._current_drawdown_pct,
            "orders": orders_executed,
        }

        self._rebalance_history.append(record)

        logger.info(
            f"Rebalance recorded: {len(orders_executed)} adjustments in "
            f"{self._current_regime.value} regime"
        )

    def _record_allocation_snapshot(self) -> None:
        """Record current allocation state."""
        snapshot = AllocationSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_capital=self._total_capital,
            regime=self._current_regime,
            drawdown_level=self._drawdown_level,
            drawdown_pct=self._current_drawdown_pct,
            allocations={n: b.effective_allocation_pct for n, b in self._budgets.items()},
            usage={n: b.current_usage_pct for n, b in self._budgets.items()},
        )

        self._allocation_history.append(snapshot)

    def set_total_capital(self, capital: float) -> None:
        """Update total capital under management."""
        old_capital = self._total_capital
        self._total_capital = capital

        if old_capital > 0:
            change_pct = ((capital - old_capital) / old_capital) * 100
            logger.info(f"Capital updated: ${old_capital:,.0f} -> ${capital:,.0f} ({change_pct:+.1f}%)")

    def get_statistics(self) -> dict:
        """Get governor statistics."""
        return {
            **self._stats,
            "total_capital": self._total_capital,
            "current_regime": self._current_regime.value,
            "drawdown_level": self._drawdown_level.value,
            "drawdown_pct": self._current_drawdown_pct * 100,
            "registered_strategies": len(self._budgets),
            "total_allocated_pct": sum(b.effective_allocation_pct for b in self._budgets.values()),
            "total_used_pct": sum(b.current_usage_pct for b in self._budgets.values()),
            "last_rebalance": self._last_rebalance_time.isoformat() if self._last_rebalance_time else None,
        }

    def get_strategy_budgets(self) -> list[dict]:
        """Get detailed budget info for all strategies."""
        return [
            {
                "strategy": b.strategy_name,
                "base_allocation_pct": b.base_allocation_pct,
                "regime_multiplier": b.regime_multiplier,
                "drawdown_multiplier": b.drawdown_multiplier,
                "performance_adjustment": b.performance_adjustment,
                "effective_allocation_pct": b.effective_allocation_pct,
                "current_usage_pct": b.current_usage_pct,
                "available_pct": b.available_pct,
                "min_pct": b.min_allocation_pct,
                "max_pct": b.max_allocation_pct,
            }
            for b in self._budgets.values()
        ]

    def get_allocation_history(self, limit: int = 100) -> list[dict]:
        """Get recent allocation history."""
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "regime": s.regime.value,
                "drawdown_level": s.drawdown_level.value,
                "drawdown_pct": s.drawdown_pct * 100,
                "allocations": s.allocations,
                "usage": s.usage,
            }
            for s in list(self._allocation_history)[-limit:]
        ]


def create_default_governor(total_capital: float = 1_000_000.0) -> CapitalAllocationGovernor:
    """
    Create a governor with default strategy allocations.

    Default allocations:
    - MacroAgent: 15%
    - MomentumAgent: 20%
    - StatArbAgent: 20%
    - MarketMakingAgent: 10%
    - MACDvAgent: 15%
    - Other: 5% each

    Args:
        total_capital: Total capital under management

    Returns:
        Configured CapitalAllocationGovernor
    """
    governor = CapitalAllocationGovernor(total_capital=total_capital)

    # Core strategies
    governor.register_strategy("MacroAgent", 15.0, min_allocation_pct=5.0, max_allocation_pct=25.0)
    governor.register_strategy("MomentumAgent", 20.0, min_allocation_pct=5.0, max_allocation_pct=30.0)
    governor.register_strategy("StatArbAgent", 20.0, min_allocation_pct=5.0, max_allocation_pct=30.0)
    governor.register_strategy("MarketMakingAgent", 10.0, min_allocation_pct=0.0, max_allocation_pct=15.0)
    governor.register_strategy("MACDvAgent", 15.0, min_allocation_pct=5.0, max_allocation_pct=25.0)

    # Supplementary strategies
    governor.register_strategy("SessionAgent", 5.0, min_allocation_pct=0.0, max_allocation_pct=10.0)
    governor.register_strategy("IndexSpreadAgent", 5.0, min_allocation_pct=0.0, max_allocation_pct=10.0)
    governor.register_strategy("TTMSqueezeAgent", 5.0, min_allocation_pct=0.0, max_allocation_pct=10.0)
    governor.register_strategy("MeanReversionAgent", 5.0, min_allocation_pct=0.0, max_allocation_pct=10.0)

    return governor

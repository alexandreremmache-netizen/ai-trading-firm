# risk_budget

**Path**: `C:\Users\Alexa\ai-trading-firm\core\risk_budget.py`

## Overview

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

## Classes

### AllocationMethod

**Inherits from**: Enum

Risk budget allocation methods.

### RebalanceTrigger

**Inherits from**: Enum

Types of rebalancing triggers.

### StrategyRiskBudget

Risk budget allocation for a single strategy.

### RebalanceEvent

A rebalancing event triggered by the system.

### RiskBudgetConfig

Configuration for risk budget manager.

### RiskBudgetManager

Cross-Strategy Risk Budget Manager.

Manages risk allocation across multiple trading strategies,
ensuring total risk stays within portfolio limits.

#### Methods

##### `def __init__(self, config: , fixed_allocations: )`

Initialize risk budget manager.

Args:
    config: Risk budget configuration
    fixed_allocations: Fixed strategy allocations (for FIXED method)

##### `def register_strategy(self, strategy: str, initial_allocation: , min_allocation: float, max_allocation: float) -> None`

Register a strategy for risk budget tracking.

Args:
    strategy: Strategy name
    initial_allocation: Initial target allocation (optional)
    min_allocation: Minimum allowed allocation
    max_allocation: Maximum allowed allocation

##### `def update_strategy_metrics(self, strategy: str, current_var: float, volatility: float, sharpe_ratio: float, current_drawdown: float, pnl: float) -> None`

Update strategy metrics for risk budget calculation.

Args:
    strategy: Strategy name
    current_var: Current VaR consumption
    volatility: Annualized volatility
    sharpe_ratio: Rolling Sharpe ratio
    current_drawdown: Current drawdown (as positive fraction)
    pnl: Latest P&L

##### `def check_rebalance_triggers(self) -> list[RebalanceEvent]`

Check all rebalancing triggers and return any triggered events.

Returns:
    List of triggered rebalance events

##### `def trigger_manual_rebalance(self, reason: str) -> RebalanceEvent`

Manually trigger a rebalancing.

##### `def get_available_budget(self, strategy: str) -> float`

Get available risk budget for a strategy.

Returns the maximum VaR that can be consumed.

##### `def can_increase_position(self, strategy: str, additional_var: float) -> tuple[bool, str]`

Check if a strategy can increase its position.

Args:
    strategy: Strategy name
    additional_var: Additional VaR from new position

Returns:
    Tuple of (can_increase, reason)

##### `def get_budget(self, strategy: str)`

Get budget for a strategy.

##### `def get_all_budgets(self) -> dict[str, StrategyRiskBudget]`

Get all strategy budgets.

##### `def get_portfolio_utilization(self) -> float`

Get total portfolio risk budget utilization.

##### `def get_status(self) -> dict[str, Any]`

Get manager status for monitoring.

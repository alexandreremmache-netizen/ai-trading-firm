# risk_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\risk_agent.py`

## Overview

Risk Agent
==========

Validates all trading decisions against risk limits.
Implements kill-switch mechanism for emergency halt.

Responsibility: Risk validation ONLY.
Does NOT handle regulatory compliance (see ComplianceAgent).

## Classes

### KillSwitchReason

**Inherits from**: Enum

Reasons for activating kill-switch.

### DrawdownLevel

**Inherits from**: Enum

Tiered drawdown response levels.

### KillSwitchAction

**Inherits from**: Enum

Actions available during kill-switch activation.

### PositionInfo

Position information for a single instrument.

### GreeksState

Portfolio Greeks state with staleness tracking.

#### Methods

##### `def is_stale(self, max_age_seconds: float) -> bool`

Check if Greeks data is stale.

##### `def age_seconds(self) -> float`

Get age of Greeks data in seconds.

### PositionGreeks

Per-position Greeks for individual position risk limits (#R4).

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### LiquidityMetrics

Liquidity risk metrics.

### DrawdownRecoveryState

Tracks drawdown recovery metrics (#R11).

Monitors:
- Current drawdown start time
- Historical recovery times
- Recovery velocity

### MarginState

Intraday margin monitoring state (#R10).

Tracks margin requirements throughout the trading day,
not just at EOD.

#### Methods

##### `def is_warning(self) -> bool`

Check if margin is at warning level.

##### `def is_critical(self) -> bool`

Check if margin is at critical level.

##### `def is_margin_call(self) -> bool`

Check if in margin call territory.

### RiskState

Current risk state of the portfolio.

### RiskCheckResult

Result of a single risk check.

### RiskValidationResult

Complete result of risk validation.

### RiskAgent

**Inherits from**: ValidationAgent

Risk Management Agent.

Validates all trading decisions against:
1. Position size limits (5% max per position)
2. Sector concentration limits (20% max per sector)
3. Portfolio leverage (2x max)
4. VaR limits (2% at 95% confidence)
5. Daily loss limit (-3% triggers halt)
6. Maximum drawdown (-10% triggers halt)
7. Rate limits (anti-HFT: 10 orders/min, 100ms interval)

Implements KILL-SWITCH for emergency situations.

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger, broker: Optional[IBBroker])`

##### `async def initialize(self) -> None`

Initialize risk state from broker.

##### `def get_subscribed_events(self) -> list[EventType]`

Risk agent subscribes to decisions.

##### `async def process_event(self, event: Event) -> None`

Validate trading decisions.

##### `def get_concentration_metrics(self) -> dict`

Get comprehensive portfolio concentration metrics (#R8).

Returns metrics for monitoring:
- HHI (Herfindahl-Hirschman Index)
- Top N positions concentration
- Sector concentration breakdown
- Effective number of positions

##### `def get_drawdown_level(self) -> DrawdownLevel`

Get current drawdown level for external queries.

##### `def get_position_size_multiplier(self) -> float`

Get position size multiplier based on drawdown level.

Returns:
    1.0 for NORMAL/WARNING, configured reduction for REDUCE, 0.0 for HALT

##### `def get_cvar_alert_status(self) -> dict`

Get current CVaR alert status (#R13).

##### `def trigger_var_recalculation(self, reason: str) -> None`

Manually trigger VaR recalculation.

Useful for external systems that detect significant market events.

##### `def check_latency_breach(self, latency_ms: float) -> bool`

Check if latency exceeds threshold (MiFID II RTS 6).

Returns True if latency is acceptable, False if breached.

##### `async def activate_kill_switch_manual(self, authorized_by: str, reason: str, action: KillSwitchAction) -> bool`

Manually activate kill switch (requires authorization).

Args:
    authorized_by: Username of person activating
    reason: Reason for manual activation
    action: Action to take

Returns:
    True if successfully activated

##### `def deactivate_kill_switch(self, authorized_by: str, second_authorization: ) -> tuple[bool, str]`

Deactivate kill-switch (requires authorization, optionally dual).

MiFID II RTS 6 recommends dual authorization for reactivation
after emergency halt.

Args:
    authorized_by: Primary authorizing user
    second_authorization: Second authorizing user (if dual auth required)

Returns:
    Tuple of (success, message)

##### `def get_kill_switch_history(self) -> list[dict]`

Get audit trail of all kill switch activations.

##### `def add_daily_return(self, return_pct: float) -> None`

Add a daily return to history for VaR calculation.

##### `def update_greeks(self, delta: float, gamma: float, vega: float, theta: float, rho: float, update_time: ) -> None`

Update portfolio Greeks with timestamp.

Args:
    delta: Portfolio delta
    gamma: Portfolio gamma
    vega: Portfolio vega
    theta: Portfolio theta
    rho: Portfolio rho (optional)
    update_time: Time of Greeks calculation (defaults to now)

##### `def update_position_greeks(self, symbol: str, delta: float, gamma: float, vega: float, theta: float, rho: float, contracts: int, update_time: ) -> None`

Update Greeks for a specific position (#R4).

Args:
    symbol: Position symbol
    delta: Position delta
    gamma: Position gamma
    vega: Position vega
    theta: Position theta
    rho: Position rho (optional)
    contracts: Number of contracts
    update_time: Time of calculation (defaults to now)

##### `def get_position_greeks(self, symbol: str)`

Get Greeks for a specific position (#R4).

##### `def get_all_position_greeks(self) -> dict[str, PositionGreeks]`

Get Greeks for all positions (#R4).

##### `def clear_position_greeks(self, symbol: ) -> None`

Clear position Greeks data (#R4).

Args:
    symbol: Specific symbol to clear, or None to clear all

##### `def get_greeks_status(self) -> dict`

Get detailed Greeks status including staleness.

##### `def update_liquidity_data(self, symbol: str, avg_daily_volume: int) -> None`

Update ADV data for a symbol.

##### `def update_stress_test_result(self, pnl_impact: float, scenario_name: str) -> None`

Update worst stress test result.

##### `def set_var_calculator(self, var_calculator) -> None`

Set enhanced VaR calculator.

##### `def set_stress_tester(self, stress_tester) -> None`

Set stress tester.

##### `def set_correlation_manager(self, correlation_manager) -> None`

Set correlation manager.

##### `def set_risk_notifier(self, risk_notifier) -> None`

Set risk limit breach notifier (#R27).

##### `async def update_margin_state(self, initial_margin: float, maintenance_margin: float, available_margin: ) -> None`

Update margin state for intraday monitoring (#R10).

Should be called periodically (e.g., every trade or every minute)
to monitor margin utilization throughout the day.

Args:
    initial_margin: Required initial margin
    maintenance_margin: Required maintenance margin
    available_margin: Available margin (calculated if not provided)

##### `async def refresh_margin_from_broker(self) -> None`

Refresh margin state from broker (#R10).

Should be called periodically for real-time margin monitoring.

##### `def get_margin_status(self) -> dict`

Get current margin monitoring status (#R10).

##### `def reset_intraday_margin_tracking(self) -> None`

Reset intraday margin tracking (call at start of day).

##### `def get_drawdown_recovery_status(self) -> dict`

Get current drawdown recovery status (#R11).

Returns detailed metrics about current and historical drawdowns.

##### `async def run_stress_tests(self) -> None`

Run all stress tests and update state.

##### `def get_status(self) -> dict`

Get current risk agent status for monitoring.

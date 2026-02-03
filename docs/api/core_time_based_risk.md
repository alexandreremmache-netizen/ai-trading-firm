# time_based_risk

**Path**: `C:\Users\Alexa\ai-trading-firm\core\time_based_risk.py`

## Overview

Time-Based Risk Module
======================

Risk limits that are time-of-day aware (Issue #R19).
Overnight vs intraday risk differentiation (Issue #R22).

Features:
- Time-of-day dependent risk limits
- Market hours awareness
- Overnight position limits
- Intraday vs overnight risk metrics

## Classes

### TradingSession

**Inherits from**: str, Enum

Trading session periods.

### RiskLevel

**Inherits from**: str, Enum

Risk level classification.

### SessionRiskLimits

Risk limits for a trading session.

#### Methods

##### `def to_dict(self) -> dict`

### OvernightRiskMetrics

Overnight risk metrics (#R22).

#### Methods

##### `def to_dict(self) -> dict`

### IntradayRiskMetrics

Intraday risk metrics (#R22).

#### Methods

##### `def to_dict(self) -> dict`

### TimeBasedRiskManager

Manages time-of-day dependent risk limits (#R19).

Adjusts risk limits based on market session and time.

#### Methods

##### `def __init__(self, base_var_limit: float, base_position_limit: float, timezone_str: str)`

##### `def add_holiday(self, holiday: date) -> None`

Add a market holiday.

##### `def add_half_day(self, half_day: date) -> None`

Add a half-day (early close).

##### `def set_custom_limits(self, session: TradingSession, limits: SessionRiskLimits) -> None`

Override default limits for a session.

##### `def get_current_session(self, dt: ) -> TradingSession`

Get current trading session.

##### `def get_session_limits(self, session: ) -> SessionRiskLimits`

Get risk limits for a session.

##### `def get_effective_limits(self, dt: ) -> dict`

Get effective risk limits for current time.

##### `def check_limit(self, current_position: float, current_var: float, proposed_order_size: float, is_market_order: bool) -> dict`

Check if proposed action is within time-based limits.

Returns dict with 'allowed' and any violations.

##### `def get_time_to_session_change(self) -> timedelta`

Get time until next session change.

### OvernightRiskManager

Manages overnight vs intraday risk (#R22).

Differentiates risk treatment based on holding period.

#### Methods

##### `def __init__(self, overnight_var_multiplier: float, gap_risk_pct: float, max_overnight_exposure_pct: float, timezone_str: str)`

##### `def update_position(self, symbol: str, quantity: int, value: float, asset_class: str, beta: float) -> None`

Update position for overnight tracking.

##### `def record_trade(self, pnl: float) -> None`

Record a trade execution.

##### `def record_order(self) -> None`

Record an order submission.

##### `def record_exposure(self, exposure: float, var: float) -> None`

Record current exposure for intraday tracking.

##### `def calculate_overnight_risk(self, account_equity: float) -> OvernightRiskMetrics`

Calculate overnight risk metrics.

##### `def calculate_intraday_risk(self, current_exposure: float, current_var: float) -> IntradayRiskMetrics`

Calculate intraday risk metrics.

##### `def check_overnight_limits(self, account_equity: float) -> dict`

Check if overnight position limits are exceeded.

##### `def suggest_position_reduction(self, account_equity: float) -> list[dict]`

Suggest positions to reduce for overnight limits.

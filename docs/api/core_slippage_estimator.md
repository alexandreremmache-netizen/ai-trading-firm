# slippage_estimator

**Path**: `C:\Users\Alexa\ai-trading-firm\core\slippage_estimator.py`

## Overview

Slippage Estimation Module
==========================

Estimates expected slippage for signal generation (Issue #Q11).
Incorporates capacity constraints for strategy sizing (Issue #Q12).

Features:
- Market impact estimation
- Liquidity-based slippage
- Capacity constraints
- Size-adjusted signal strength

## Classes

### LiquidityTier

**Inherits from**: str, Enum

Asset liquidity classification.

### SlippageEstimate

Estimated slippage for a trade.

#### Methods

##### `def to_dict(self) -> dict`

### CapacityConstraints

Capacity limits for a strategy/symbol.

#### Methods

##### `def to_dict(self) -> dict`

### LiquidityProfile

Liquidity characteristics of an asset.

### SlippageEstimator

Estimates execution slippage for trading signals (#Q11).

Uses multiple models:
- Square-root market impact (Almgren-Chriss)
- Linear spread model
- Volatility adjustment

#### Methods

##### `def __init__(self, permanent_impact_fraction: float, volatility_risk_premium: float)`

##### `def update_liquidity_profile(self, symbol: str, adv: int, spread_bps: float, depth_shares: int, volatility_pct: float, tier: ) -> LiquidityProfile`

Update liquidity profile for a symbol.

##### `def update_price(self, symbol: str, price: float) -> None`

Update current price for a symbol.

##### `def estimate_slippage(self, symbol: str, side: str, quantity: int, price: , urgency: str) -> SlippageEstimate`

Estimate slippage for a potential trade.

Args:
    symbol: Trading symbol
    side: 'BUY' or 'SELL'
    quantity: Number of shares/contracts
    price: Current price (uses cached if not provided)
    urgency: 'immediate', 'normal', or 'patient'

Returns:
    SlippageEstimate with breakdown

### CapacityManager

Manages capacity constraints for strategies (#Q12).

Ensures strategies don't exceed sustainable position sizes.

#### Methods

##### `def __init__(self, default_max_adv_pct: float, default_max_order_adv_pct: float, default_max_daily_volume_pct: float)`

##### `def set_adv(self, symbol: str, adv: int) -> None`

Set average daily volume for a symbol.

##### `def set_custom_constraints(self, symbol: str, max_position_pct_adv: , max_order_pct_adv: , max_daily_volume_pct: ) -> None`

Set custom capacity constraints for a symbol.

##### `def update_position(self, symbol: str, quantity: int) -> None`

Update current position for a symbol.

##### `def record_execution(self, symbol: str, quantity: int) -> None`

Record executed volume.

##### `def get_constraints(self, symbol: str) -> CapacityConstraints`

Get capacity constraints for a symbol.

##### `def check_capacity(self, symbol: str, proposed_quantity: int) -> dict`

Check if proposed trade fits within capacity constraints.

Returns:
    Dict with 'allowed', 'max_allowed', and 'reasons'

##### `def get_adjusted_signal_size(self, symbol: str, desired_quantity: int, signal_strength: float) -> dict`

Get capacity-adjusted position size.

Scales down size if hitting capacity limits.

##### `def reset_daily_volume(self) -> None`

Reset daily volume counters (call at EOD).

##### `def get_all_constraints(self) -> dict[str, dict]`

Get constraints for all tracked symbols.

### SignalSlippageAdjuster

Adjusts signal strength based on expected slippage (#Q11).

Reduces signal strength when slippage would consume expected alpha.

#### Methods

##### `def __init__(self, slippage_estimator: SlippageEstimator, capacity_manager: CapacityManager, min_edge_after_slippage_bps: float)`

##### `def adjust_signal(self, symbol: str, direction: str, raw_strength: float, expected_alpha_bps: float, desired_quantity: int, price: float, urgency: str) -> dict`

Adjust signal strength based on execution costs and capacity.

Args:
    symbol: Trading symbol
    direction: 'LONG' or 'SHORT'
    raw_strength: Original signal strength (0-1)
    expected_alpha_bps: Expected alpha from the signal in bps
    desired_quantity: Desired position size
    price: Current price
    urgency: Execution urgency

Returns:
    Adjusted signal with all components

##### `def batch_adjust_signals(self, signals: list[dict]) -> list[dict]`

Adjust multiple signals considering portfolio-level capacity.

Args:
    signals: List of dicts with symbol, direction, strength, alpha, quantity, price

Returns:
    List of adjusted signals sorted by net alpha

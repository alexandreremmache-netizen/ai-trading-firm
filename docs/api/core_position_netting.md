# position_netting

**Path**: `C:\Users\Alexa\ai-trading-firm\core\position_netting.py`

## Overview

Position Netting
================

Aggregates and nets positions across multiple strategies to provide
a consolidated view of portfolio exposure.

Required for:
- Accurate risk calculation
- Margin optimization
- Cross-strategy position limits
- Portfolio-level exposure reporting

## Classes

### PositionSide

**Inherits from**: Enum

Position direction.

### StrategyPosition

Position held by a specific strategy.

#### Methods

##### `def side(self) -> PositionSide`

### NetPosition

Net position aggregated across all strategies.

Provides consolidated view of exposure regardless of which
strategy originated the position.

#### Methods

##### `def side(self) -> PositionSide`

##### `def gross_quantity(self) -> int`

Total absolute quantity across strategies (ignores netting).

##### `def netting_benefit(self) -> int`

Quantity reduced through netting (gross - net).

##### `def to_dict(self) -> dict[str, Any]`

### PositionNetter

Aggregates positions across strategies and provides net exposure.

Features:
- Net position calculation
- Gross vs net exposure
- Strategy attribution
- P&L attribution
- Position limit checking on net basis

#### Methods

##### `def __init__(self, config: )`

Initialize position netter.

Args:
    config: Configuration with:
        - use_fifo: Use FIFO for P&L (default: True)
        - track_history: Track position history (default: True)

##### `def update_position(self, strategy: str, symbol: str, quantity: int, avg_price: float) -> None`

Update position for a strategy.

Args:
    strategy: Strategy name
    symbol: Instrument symbol
    quantity: Position quantity (positive=long, negative=short, 0=flat)
    avg_price: Average entry price

##### `def update_market_price(self, symbol: str, price: float) -> None`

Update market price for P&L calculation.

##### `def get_net_position(self, symbol: str)`

Get net position for a symbol across all strategies.

Args:
    symbol: Instrument symbol

Returns:
    NetPosition or None if no positions

##### `def get_all_net_positions(self) -> dict[str, NetPosition]`

Get net positions for all symbols.

##### `def get_strategy_positions(self, strategy: str) -> dict[str, StrategyPosition]`

Get all positions for a specific strategy.

##### `def get_gross_exposure(self) -> float`

Get total gross exposure (sum of absolute position values).

##### `def get_net_exposure(self) -> float`

Get total net exposure (net long - net short).

##### `def get_long_exposure(self) -> float`

Get total long exposure.

##### `def get_short_exposure(self) -> float`

Get total short exposure (as positive number).

##### `def get_total_unrealized_pnl(self) -> float`

Get total unrealized P&L across all positions.

##### `def check_position_limit(self, symbol: str, new_quantity: int, strategy: str, max_net_position: int) -> tuple[bool, str]`

Check if adding a position would exceed limits on net basis.

Args:
    symbol: Instrument symbol
    new_quantity: Proposed new quantity for this strategy
    strategy: Strategy name
    max_net_position: Maximum allowed net position

Returns:
    Tuple of (allowed, reason)

##### `def get_netting_summary(self) -> dict[str, Any]`

Get summary of netting benefits.

##### `def get_status(self) -> dict[str, Any]`

Get netter status for monitoring.

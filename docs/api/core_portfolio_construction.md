# portfolio_construction

**Path**: `C:\Users\Alexa\ai-trading-firm\core\portfolio_construction.py`

## Overview

Portfolio Construction Module
=============================

Target portfolio construction (Issue #P14).
Trade list generation (Issue #P15).
Portfolio comparison tools (Issue #P18).

Features:
- Target weight calculation
- Trade list with rebalancing
- Portfolio comparison metrics
- Transaction cost optimization

## Classes

### RebalanceMethod

**Inherits from**: str, Enum

Rebalancing methodology.

### TradeReason

**Inherits from**: str, Enum

Reason for generating a trade.

### TargetPosition

Target position in a portfolio.

#### Methods

##### `def weight_difference(self) -> float`

Difference from target.

##### `def is_overweight(self) -> bool`

##### `def is_underweight(self) -> bool`

##### `def to_dict(self) -> dict`

### Trade

Individual trade in a trade list.

#### Methods

##### `def notional(self) -> float`

##### `def to_dict(self) -> dict`

### TradeList

Complete trade list for rebalancing.

#### Methods

##### `def to_dict(self) -> dict`

### PortfolioComparison

Comparison between two portfolios (#P18).

#### Methods

##### `def to_dict(self) -> dict`

### TargetPortfolioBuilder

Builds target portfolios (#P14).

Supports multiple construction methodologies.

#### Methods

##### `def __init__(self, min_position_weight: float, max_position_weight: float, round_lots: bool, lot_size: int)`

##### `def set_price(self, symbol: str, price: float) -> None`

Set price for a symbol.

##### `def set_holding(self, symbol: str, quantity: int) -> None`

Set current holding.

##### `def set_sector(self, symbol: str, sector: str) -> None`

Set sector for a symbol.

##### `def build_equal_weight(self, symbols: list[str], portfolio_value: float) -> list[TargetPosition]`

Build equal-weight portfolio.

##### `def build_market_cap_weight(self, symbols_with_mcap: dict[str, float], portfolio_value: float) -> list[TargetPosition]`

Build market-cap weighted portfolio.

##### `def build_custom_weight(self, target_weights: dict[str, float], portfolio_value: float) -> list[TargetPosition]`

Build portfolio with custom weights.

##### `def build_risk_parity(self, symbols_with_vol: dict[str, float], portfolio_value: float) -> list[TargetPosition]`

Build risk parity portfolio (equal risk contribution).

### TradeListGenerator

Generates trade lists from target portfolios (#P15).

Supports multiple rebalancing methods.

#### Methods

##### `def __init__(self, commission_per_share: float, commission_min: float, impact_rate: float, short_term_tax_rate: float, long_term_tax_rate: float)`

##### `def set_cost_basis(self, symbol: str, cost_per_share: float, is_short_term: bool) -> None`

Set cost basis for tax calculations.

##### `def generate_full_rebalance(self, targets: list[TargetPosition], available_cash: float) -> TradeList`

Generate trade list for full rebalance.

##### `def generate_threshold_rebalance(self, targets: list[TargetPosition], threshold: float) -> TradeList`

Generate trades only for positions outside threshold.

##### `def generate_tax_aware_rebalance(self, targets: list[TargetPosition], max_tax: float) -> TradeList`

Generate trades minimizing tax impact.

### PortfolioComparator

Compares portfolios (#P18).

Provides detailed comparison metrics.

#### Methods

##### `def compare(self, portfolio1: dict[str, float], portfolio2: dict[str, float], name1: str, name2: str, sectors: ) -> PortfolioComparison`

Compare two portfolios by weight.

Args:
    portfolio1: First portfolio weights
    portfolio2: Second portfolio weights
    name1: Name for first portfolio
    name2: Name for second portfolio
    sectors: Optional sector assignments

Returns:
    PortfolioComparison with detailed metrics

##### `def track_drift(self, original: dict[str, float], current: dict[str, float], name: str) -> dict`

Track portfolio drift from original weights.

Returns drift metrics and rebalance recommendations.

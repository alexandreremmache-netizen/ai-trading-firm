# margin_optimizer

**Path**: `C:\Users\Alexa\ai-trading-firm\core\margin_optimizer.py`

## Overview

Margin Optimization Module
==========================

Cross-margin benefit calculation and optimization (Issue #R14).
Risk contribution attribution by strategy (Issue #R16).

Features:
- Portfolio margin vs Reg-T comparison
- Cross-margin benefits for hedged positions
- Risk contribution by strategy
- Margin efficiency optimization

## Classes

### MarginType

**Inherits from**: str, Enum

Margin calculation methodology.

### PositionMargin

Margin requirement for a single position.

#### Methods

##### `def to_dict(self) -> dict`

### CrossMarginBenefit

Cross-margin benefit analysis.

#### Methods

##### `def to_dict(self) -> dict`

### StrategyRiskContribution

Risk contribution by strategy (#R16).

#### Methods

##### `def to_dict(self) -> dict`

### CrossMarginCalculator

Calculates cross-margin benefits (#R14).

Identifies margin savings from hedged positions and correlated assets.

#### Methods

##### `def __init__(self, correlation_threshold: float, hedge_ratio_tolerance: float)`

##### `def set_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None`

Set correlation between two symbols.

##### `def set_asset_type(self, symbol: str, asset_type: str) -> None`

Set asset type for a symbol.

##### `def set_beta(self, symbol: str, beta: float) -> None`

Set beta for hedge ratio calculation.

##### `def update_position(self, symbol: str, quantity: int, price: float, asset_type: ) -> None`

Update position data.

##### `def calculate_position_margins(self) -> dict[str, PositionMargin]`

Calculate margin for all positions.

##### `def find_cross_margin_benefits(self) -> list[CrossMarginBenefit]`

Find all cross-margin benefits in the portfolio.

Analyzes pairs of positions for hedging/correlation benefits.

##### `def calculate_portfolio_margin_summary(self) -> dict`

Calculate total portfolio margin with cross-benefits.

### RiskContributionAnalyzer

Analyzes risk contribution by strategy (#R16).

Decomposes portfolio risk into strategy components.

#### Methods

##### `def __init__(self)`

##### `def register_strategy_position(self, strategy: str, symbol: str, quantity: int, market_value: float, volatility: ) -> None`

Register a position under a strategy.

##### `def set_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None`

Set correlation between symbols.

##### `def set_strategy_performance(self, strategy: str, returns: list[float], sharpe_ratio: float) -> None`

Set historical performance for a strategy.

##### `def calculate_strategy_var(self, strategy: str, confidence: float) -> float`

Calculate VaR for a single strategy.

##### `def calculate_marginal_var(self, strategy: str, portfolio_var: float, confidence: float) -> float`

Calculate marginal VaR contribution of a strategy.

##### `def analyze_all_strategies(self, total_margin: float, confidence: float) -> list[StrategyRiskContribution]`

Analyze risk contribution for all strategies.

Returns list of StrategyRiskContribution sorted by VaR contribution.

##### `def get_risk_attribution_summary(self, total_margin: float, confidence: float) -> dict`

Generate full risk attribution summary.

##### `def clear_all(self) -> None`

Clear all data for reset.

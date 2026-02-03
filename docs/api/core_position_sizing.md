# position_sizing

**Path**: `C:\Users\Alexa\ai-trading-firm\core\position_sizing.py`

## Overview

Position Sizing Module
======================

Advanced position sizing using Kelly Criterion and related methods.
Provides optimal position sizing with risk management constraints.

## Classes

### SizingMethod

**Inherits from**: Enum

Position sizing methodology.

### PositionSizeResult

Result of position sizing calculation.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### StrategyStats

Statistics for a trading strategy (used in Kelly calculation).

IMPORTANT: avg_win and avg_loss must be RETURNS (percentages as decimals),
NOT dollar P&L values. For example:
- avg_win = 0.02 means 2% average winning trade
- avg_loss = 0.01 means 1% average losing trade

If you have dollar P&L, divide by position size to get returns.

#### Methods

##### `def __post_init__(self)`

Validate inputs are in expected ranges.

##### `def edge(self) -> float`

Calculate edge (expected return per trade).

##### `def kelly_fraction(self) -> float`

Calculate full Kelly fraction.

Kelly formula: f* = (bp - q) / b
where:
    b = avg_win / avg_loss (win/loss ratio)
    p = win_rate
    q = 1 - p (loss rate)

Returns the optimal fraction of capital to risk.

##### `def from_dollar_pnl(cls, wins: list[float], losses: list[float], position_sizes: list[float], volatility: float) -> StrategyStats`

Create StrategyStats from dollar P&L values.

Converts dollar P&L to returns for proper Kelly calculation.

Args:
    wins: List of winning trade P&L in dollars
    losses: List of losing trade P&L in dollars (positive numbers)
    position_sizes: List of position sizes for each trade
    volatility: Strategy volatility (default 15%)

Returns:
    StrategyStats with properly calculated returns

### PositionSizer

Advanced position sizing calculator.

Features:
- Kelly Criterion with variants
- Volatility targeting
- Correlation-adjusted sizing
- Risk limit constraints

#### Methods

##### `def __init__(self, config: )`

Initialize position sizer.

Args:
    config: Configuration with:
        - method: Default sizing method (default: "kelly")
        - use_half_kelly: Use half Kelly (default: True)
        - max_position_pct: Maximum position size (default: 10%)
        - min_position_pct: Minimum position size (default: 1%)
        - vol_target: Target volatility (default: 15%)
        - correlation_discount: Discount for correlated positions (default: True)

##### `def update_strategy_stats(self, strategy: str, stats: StrategyStats) -> None`

Update statistics for a strategy.

Args:
    strategy: Strategy name
    stats: Strategy statistics

##### `def update_correlations(self, correlations: dict[tuple[str, str], float]) -> None`

Update correlation data.

Args:
    correlations: Dictionary of (symbol1, symbol2) -> correlation

##### `def calculate_kelly_size(self, strategy: str, portfolio_value: float, kelly_variant: SizingMethod) -> PositionSizeResult`

Calculate position size using Kelly Criterion.

Args:
    strategy: Strategy name
    portfolio_value: Total portfolio value
    kelly_variant: Kelly variant to use

Returns:
    PositionSizeResult with calculated size

##### `def calculate_vol_target_size(self, symbol: str, portfolio_value: float, asset_volatility: float, target_vol: ) -> PositionSizeResult`

Calculate position size using volatility targeting.

Position size = (target_vol / asset_vol) * portfolio_value

Args:
    symbol: Asset symbol
    portfolio_value: Total portfolio value
    asset_volatility: Asset's annualized volatility
    target_vol: Target portfolio volatility (default: config value)

Returns:
    PositionSizeResult with calculated size

##### `def calculate_correlation_adjusted_size(self, symbol: str, base_size_pct: float, existing_positions: dict[str, float], portfolio_value: float) -> PositionSizeResult`

Adjust position size based on correlation with existing positions.

Reduces size if highly correlated with existing holdings.

Args:
    symbol: Symbol to size
    base_size_pct: Base position size before adjustment
    existing_positions: Current positions (symbol -> value)
    portfolio_value: Total portfolio value

Returns:
    PositionSizeResult with correlation-adjusted size

##### `def calculate_optimal_size(self, symbol: str, strategy: str, portfolio_value: float, asset_volatility: , existing_positions: , method: ) -> PositionSizeResult`

Calculate optimal position size using configured method.

This is the main entry point that combines all sizing logic.

Args:
    symbol: Asset symbol
    strategy: Strategy name
    portfolio_value: Total portfolio value
    asset_volatility: Asset volatility (for vol targeting)
    existing_positions: Current positions (for correlation adjustment)
    method: Override default method

Returns:
    PositionSizeResult with optimal size

##### `def calculate_contracts(self, position_value: float, price: float, multiplier: float) -> int`

Calculate number of contracts for a given position value.

Args:
    position_value: Target position value
    price: Current price
    multiplier: Contract multiplier

Returns:
    Number of contracts (rounded down)

##### `def get_strategy_stats(self, strategy: str)`

Get stored statistics for a strategy.

##### `def get_all_stats(self) -> dict[str, StrategyStats]`

Get all strategy statistics.

##### `def get_optimal_portfolio_weights(self, strategies: list[str], portfolio_value: float) -> dict[str, float]`

Calculate optimal weights for multiple strategies.

Uses normalized Kelly fractions.

Args:
    strategies: List of strategy names
    portfolio_value: Total portfolio value

Returns:
    Dictionary of strategy to weight (0-1)

##### `def get_status(self) -> dict[str, Any]`

Get sizer status for monitoring.

##### `def optimize_portfolio_mean_variance(self, symbols: list[str], expected_returns: dict[str, float], covariance_matrix: np.ndarray, portfolio_value: float, target_return: , risk_free_rate: float) -> dict[str, PositionSizeResult]`

Mean-variance portfolio optimization (Markowitz) (#P5).

Finds the optimal portfolio weights that maximize the Sharpe ratio
(or minimize variance for a target return).

Uses quadratic programming to solve:
- Max: (w'μ - rf) / sqrt(w'Σw)  [max Sharpe]
- Or: Min: w'Σw  s.t. w'μ = target  [min var for target return]

Args:
    symbols: List of asset symbols
    expected_returns: Expected returns by symbol (annual)
    covariance_matrix: Covariance matrix (NxN numpy array)
    portfolio_value: Total portfolio value
    target_return: Target portfolio return (None = max Sharpe)
    risk_free_rate: Risk-free rate for Sharpe calculation

Returns:
    Dictionary of symbol to PositionSizeResult with optimal weights

##### `def optimize_portfolio_risk_parity(self, symbols: list[str], covariance_matrix: np.ndarray, portfolio_value: float, risk_budgets: ) -> dict[str, PositionSizeResult]`

Risk parity portfolio optimization (#P5).

Allocates positions so each contributes equally to portfolio risk.
Risk contribution_i = w_i * (Σw)_i / σ_p

If risk_budgets provided, allocates according to those proportions
instead of equal risk.

Args:
    symbols: List of asset symbols
    covariance_matrix: Covariance matrix (NxN numpy array)
    portfolio_value: Total portfolio value
    risk_budgets: Optional risk budget allocation (sums to 1)

Returns:
    Dictionary of symbol to PositionSizeResult

##### `def optimize_portfolio_min_variance(self, symbols: list[str], covariance_matrix: np.ndarray, portfolio_value: float) -> dict[str, PositionSizeResult]`

Minimum variance portfolio optimization (#P5).

Finds the portfolio with lowest possible variance (risk).
Useful for defensive positioning.

Args:
    symbols: List of asset symbols
    covariance_matrix: Covariance matrix (NxN numpy array)
    portfolio_value: Total portfolio value

Returns:
    Dictionary of symbol to PositionSizeResult

##### `def get_efficient_frontier(self, symbols: list[str], expected_returns: dict[str, float], covariance_matrix: np.ndarray, n_points: int, risk_free_rate: float) -> list[dict]`

Calculate the efficient frontier (#P5).

Returns a series of optimal portfolios from minimum variance
to maximum return.

Args:
    symbols: List of asset symbols
    expected_returns: Expected returns by symbol
    covariance_matrix: Covariance matrix
    n_points: Number of points on the frontier
    risk_free_rate: Risk-free rate

Returns:
    List of dicts with return, risk, sharpe, weights for each point

# var_calculator

**Path**: `C:\Users\Alexa\ai-trading-firm\core\var_calculator.py`

## Overview

Value at Risk Calculator
========================

Comprehensive VaR calculation with multiple methodologies:
- Parametric (Variance-Covariance)
- Historical Simulation
- Monte Carlo Simulation
- Incremental and Marginal VaR

Designed for institutional-grade risk management.

## Classes

### VaRMethod

**Inherits from**: Enum

VaR calculation methodology.

### VaRResult

Result of VaR calculation.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### LiquidityProfile

Liquidity profile for a position (#R6).

Used to adjust VaR for liquidity risk factors:
- Bid-ask spread (immediate cost)
- Market impact (price movement from liquidating)
- Time to liquidate (days to exit position)

#### Methods

##### `def days_to_liquidate(self) -> float`

Estimate days to liquidate position.

Assumes max 10% of ADV per day participation rate.

##### `def market_impact_bps(self) -> float`

Estimate market impact in basis points.

Uses square-root market impact model:
Impact = sigma * sqrt(Q / ADV) * constant

Where constant is typically 0.5-1.0 for developed markets.

##### `def total_liquidity_cost_bps(self) -> float`

Total expected liquidity cost in basis points.

### LiquidityAdjustedVaRResult

Result of liquidity-adjusted VaR calculation (#R6).

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### IncrementalVaRResult

Result of incremental/marginal VaR calculation.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### VaRCalculator

Comprehensive Value at Risk calculator.

Supports multiple methodologies and provides detailed
risk decomposition for portfolio management.

#### Methods

##### `def __init__(self, config: )`

Initialize VaR calculator.

Args:
    config: Configuration with:
        - method: Default VaR method (default: "all")
        - confidence_level: Default confidence (default: 0.95)
        - horizon_days: Default horizon (default: 1)
        - monte_carlo_simulations: MC iterations (default: 10000)
        - decay_factor: EWMA decay for volatility (default: 0.94)

##### `def update_returns(self, returns_dict: dict[str, np.ndarray]) -> None`

Update returns data for VaR calculation.

Args:
    returns_dict: Dictionary mapping symbols to returns arrays

##### `def calculate_parametric_var(self, positions: dict[str, float], portfolio_value: float, confidence_level: , horizon_days: ) -> VaRResult`

Calculate Parametric (Variance-Covariance) VaR.

Assumes returns are normally distributed.

Args:
    positions: Dictionary of symbol to position value
    portfolio_value: Total portfolio value
    confidence_level: Confidence level (e.g., 0.95)
    horizon_days: Time horizon in days

Returns:
    VaRResult with parametric VaR

##### `def calculate_historical_var(self, positions: dict[str, float], portfolio_value: float, confidence_level: , horizon_days: ) -> VaRResult`

Calculate Historical Simulation VaR.

Uses actual historical returns distribution.

Args:
    positions: Dictionary of symbol to position value
    portfolio_value: Total portfolio value
    confidence_level: Confidence level
    horizon_days: Time horizon

Returns:
    VaRResult with historical VaR

##### `def calculate_monte_carlo_var(self, positions: dict[str, float], portfolio_value: float, confidence_level: , horizon_days: , n_simulations: ) -> VaRResult`

Calculate Monte Carlo VaR.

Simulates potential portfolio returns using the covariance structure.

Args:
    positions: Dictionary of symbol to position value
    portfolio_value: Total portfolio value
    confidence_level: Confidence level
    horizon_days: Time horizon
    n_simulations: Number of simulations

Returns:
    VaRResult with Monte Carlo VaR

##### `def calculate_all_methods(self, positions: dict[str, float], portfolio_value: float, confidence_level: , horizon_days: ) -> dict[VaRMethod, VaRResult]`

Calculate VaR using all methods.

Args:
    positions: Position values
    portfolio_value: Total portfolio value
    confidence_level: Confidence level
    horizon_days: Time horizon

Returns:
    Dictionary of method to VaRResult

##### `def calculate_incremental_var(self, positions: dict[str, float], portfolio_value: float, new_position_symbol: str, new_position_value: float, method: VaRMethod) -> IncrementalVaRResult`

Calculate incremental VaR for adding a new position.

Args:
    positions: Current positions
    portfolio_value: Current portfolio value
    new_position_symbol: Symbol to add
    new_position_value: Value of new position
    method: VaR calculation method

Returns:
    IncrementalVaRResult with decomposition

##### `def calculate_component_var(self, positions: dict[str, float], portfolio_value: float) -> dict[str, IncrementalVaRResult]`

Calculate component VaR for all positions.

Shows each position's contribution to total VaR.

Args:
    positions: Current positions
    portfolio_value: Portfolio value

Returns:
    Dictionary of symbol to IncrementalVaRResult

##### `def stress_test_var(self, positions: dict[str, float], portfolio_value: float, volatility_multiplier: float, correlation_override: ) -> VaRResult`

Calculate stressed VaR with increased volatility.

Args:
    positions: Current positions
    portfolio_value: Portfolio value
    volatility_multiplier: Factor to increase volatility
    correlation_override: Override all correlations to this value

Returns:
    VaRResult under stressed conditions

##### `def calculate_liquidity_adjusted_var(self, positions: dict[str, float], portfolio_value: float, liquidity_profiles: dict[str, LiquidityProfile], confidence_level: , horizon_days: , method: VaRMethod) -> LiquidityAdjustedVaRResult`

Calculate liquidity-adjusted VaR (#R6).

Incorporates three liquidity risk components:
1. Direct liquidation cost (spread + market impact)
2. Extended horizon VaR (for positions that take multiple days to liquidate)
3. Liquidity-volatility correlation (liquidity often dries up during stress)

Args:
    positions: Dictionary of symbol to position value
    portfolio_value: Total portfolio value
    liquidity_profiles: Liquidity profile for each position
    confidence_level: Confidence level
    horizon_days: Base time horizon
    method: VaR calculation method for base VaR

Returns:
    LiquidityAdjustedVaRResult with decomposition

##### `def update_liquidity_profile(self, symbol: str, adv: float, bid_ask_spread_bps: float, position_size: float, volatility: ) -> LiquidityProfile`

Create or update a liquidity profile for a symbol.

Args:
    symbol: Instrument symbol
    adv: Average daily volume
    bid_ask_spread_bps: Typical bid-ask spread in basis points
    position_size: Current position size in units
    volatility: Daily volatility (uses returns data if None)

Returns:
    LiquidityProfile instance

##### `def get_status(self) -> dict[str, Any]`

Get calculator status for monitoring.

##### `def calculate_jump_adjusted_var(self, positions: dict[str, float], portfolio_value: float, confidence_level: , horizon_days: , jump_intensity: float, jump_mean: float, jump_std: float, n_simulations: ) -> VaRResult`

Calculate VaR with jump-diffusion model for fat tails (#R9).

Uses Merton's jump-diffusion model to capture extreme events
that normal distributions miss. Returns are modeled as:

dS/S = μdt + σdW + J*dN

Where:
- dW is standard Brownian motion
- dN is Poisson process with intensity λ (jump_intensity)
- J is jump size ~ Normal(jump_mean, jump_std)

This addresses the "fat tails" problem where market crashes
are more frequent than normal distributions suggest.

Args:
    positions: Dictionary of symbol to position value
    portfolio_value: Total portfolio value
    confidence_level: Confidence level (e.g., 0.99 for jump risk)
    horizon_days: Time horizon
    jump_intensity: Expected jumps per year (λ, default 0.1 = 1 every 10 years)
    jump_mean: Average jump size (default -5% for crash bias)
    jump_std: Jump size volatility (default 10%)
    n_simulations: Number of Monte Carlo paths

Returns:
    VaRResult with jump-adjusted VaR and tail statistics

##### `def calculate_fat_tail_metrics(self, positions: dict[str, float], portfolio_value: float) -> dict[str, Any]`

Calculate comprehensive fat tail risk metrics (#R9).

Returns metrics that capture tail risk beyond standard VaR:
- Skewness and kurtosis
- Tail ratio (losses vs gains)
- Maximum drawdown in historical returns
- Jump risk premium

Args:
    positions: Position values
    portfolio_value: Total portfolio value

Returns:
    Dictionary of fat tail metrics

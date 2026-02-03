# options_advanced

**Path**: `C:\Users\Alexa\ai-trading-firm\core\options_advanced.py`

## Overview

Advanced Options Analytics Module

Addresses MEDIUM priority issues:
- #O13: Option market making support
- #O14: Option pricing model comparison
- #O15: Jump diffusion model (Merton)
- #O16: Stochastic volatility (Heston model)

Provides institutional-grade options pricing and market making capabilities.

## Classes

### OptionType

**Inherits from**: Enum

Option type enumeration.

### PricingModel

**Inherits from**: Enum

Available pricing models.

### QuoteSide

**Inherits from**: Enum

Quote side for market making.

### OptionContract

Represents an option contract.

#### Methods

##### `def time_to_expiry(self) -> float`

Calculate time to expiry in years.

### MarketData

Market data for option pricing.

### PricingResult

Result of option pricing.

### ModelComparisonResult

Comparison of multiple pricing models.

### Quote

Market making quote.

### InventoryPosition

Current inventory position for market making.

### BasePricingModel

**Inherits from**: ABC

Abstract base class for pricing models.

#### Methods

##### `def price(self, contract: OptionContract, market_data: MarketData) -> PricingResult`

Price an option contract.

##### `def model_type(self) -> PricingModel`

Return the model type.

##### `def implied_volatility(self, contract: OptionContract, market_data: MarketData, market_price: float, max_iterations: int, tolerance: float) -> float`

Calculate implied volatility from market price.

Uses Brent's method with fallback to Newton-Raphson for robustness.
Handles edge cases near expiration and extreme market prices.

### BlackScholesModel

**Inherits from**: BasePricingModel

Standard Black-Scholes-Merton pricing model.

#### Methods

##### `def model_type(self) -> PricingModel`

##### `def price(self, contract: OptionContract, market_data: MarketData) -> PricingResult`

Price option using Black-Scholes.

### JumpParameters

Parameters for jump diffusion model.

### MertonJumpDiffusionModel

**Inherits from**: BasePricingModel

Merton (1976) Jump Diffusion Model.

Extends Black-Scholes with compound Poisson jumps to capture
discontinuous price movements (crash risk, earnings jumps, etc.).

dS/S = (mu - lambda*k) dt + sigma dW + (J-1) dN

where:
- N is a Poisson process with intensity lambda
- J is the jump size (log-normal distributed)
- k = E[J-1] is the expected percentage jump size

#### Methods

##### `def __init__(self, max_terms: int)`

Initialize with maximum series expansion terms.

##### `def model_type(self) -> PricingModel`

##### `def price(self, contract: OptionContract, market_data: MarketData, jump_params: Optional[JumpParameters]) -> PricingResult`

Price option using Merton jump diffusion.

Uses series expansion of BS prices weighted by Poisson probabilities.

##### `def calibrate(self, contracts: List[OptionContract], market_data: MarketData, market_prices: List[float]) -> JumpParameters`

Calibrate jump parameters to market prices.

Minimizes sum of squared pricing errors.

### HestonParameters

Parameters for Heston stochastic volatility model.

#### Methods

##### `def feller_condition(self) -> bool`

Check if Feller condition is satisfied (ensures positive variance).

### HestonModel

**Inherits from**: BasePricingModel

Heston (1993) Stochastic Volatility Model.

dS = mu*S*dt + sqrt(v)*S*dW1
dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW2
dW1*dW2 = rho*dt

Captures:
- Volatility smile/skew
- Mean reversion of volatility
- Correlation between returns and volatility (leverage effect)

#### Methods

##### `def __init__(self, integration_points: int)`

Initialize with number of integration points for characteristic function.

##### `def model_type(self) -> PricingModel`

##### `def price(self, contract: OptionContract, market_data: MarketData, heston_params: Optional[HestonParameters]) -> PricingResult`

Price option using Heston model via characteristic function.

Uses Carr-Madan FFT approach or direct integration.

##### `def calibrate(self, contracts: List[OptionContract], market_data: MarketData, market_prices: List[float], initial_params: Optional[HestonParameters]) -> HestonParameters`

Calibrate Heston parameters to market prices.

Uses differential evolution for global optimization.

### PricingModelComparator

Compare multiple pricing models for option valuation.

Provides:
- Side-by-side model comparison
- Model selection based on market fit
- Parameter sensitivity analysis

#### Methods

##### `def __init__(self)`

Initialize with available models.

##### `def compare_models(self, contract: OptionContract, market_data: MarketData, market_price: Optional[float], model_params: Optional[Dict[PricingModel, Any]]) -> ModelComparisonResult`

Compare all available models for a single contract.

Args:
    contract: Option contract to price
    market_data: Market data for pricing
    market_price: Optional market price for comparison
    model_params: Optional model-specific parameters

Returns:
    ModelComparisonResult with prices from all models

##### `def sensitivity_analysis(self, contract: OptionContract, market_data: MarketData, param_name: str, param_range: np.ndarray, model: PricingModel) -> Dict[str, List[float]]`

Analyze sensitivity of model output to parameter changes.

Args:
    contract: Option contract
    market_data: Base market data
    param_name: Parameter to vary (spot, vol, rate, etc.)
    param_range: Range of parameter values
    model: Model to use

Returns:
    Dictionary of parameter values and corresponding prices/Greeks

##### `def generate_vol_surface(self, underlying: str, spot: float, rate: float, strikes: List[float], expiries: List[float], market_prices: Dict[Tuple[float, float], float], model: PricingModel) -> Dict[Tuple[float, float], float]`

Generate implied volatility surface from market prices.

Args:
    underlying: Underlying asset
    spot: Current spot price
    rate: Risk-free rate
    strikes: List of strikes
    expiries: List of expiries (in years)
    market_prices: Dict of (strike, expiry) -> price
    model: Model to use for IV calculation

Returns:
    Dictionary of (strike, expiry) -> implied vol

### MarketMakingParameters

Parameters for market making strategy.

### OptionMarketMaker

Option market making engine.

Provides:
- Two-way quote generation
- Inventory management
- Risk-based position limits
- Dynamic spread adjustment

#### Methods

##### `def __init__(self, pricing_model: BasePricingModel, params: Optional[MarketMakingParameters])`

Initialize market maker with pricing model and parameters.

##### `def generate_quote(self, contract: OptionContract, market_data: MarketData, side: QuoteSide) -> Quote`

Generate market making quote.

Args:
    contract: Option contract to quote
    market_data: Current market data
    side: Which side to quote

Returns:
    Quote with bid/ask prices and sizes

##### `def update_inventory(self, contract: OptionContract, quantity: int, price: float, pricing_result: PricingResult) -> None`

Update inventory after a trade.

##### `def get_portfolio_greeks(self) -> Dict[str, float]`

Get aggregate portfolio Greeks.

##### `def calculate_hedge_trades(self, market_data: MarketData) -> List[Dict[str, Any]]`

Calculate hedge trades to neutralize Greeks.

Returns list of suggested hedging trades.

## Functions

### `def create_pricing_suite() -> Dict[str, Any]`

Create a complete pricing suite with all models.

Returns:
    Dictionary containing all pricing components

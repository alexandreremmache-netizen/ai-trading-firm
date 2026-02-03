# options_vol_strategy

**Path**: `C:\Users\Alexa\ai-trading-firm\strategies\options_vol_strategy.py`

## Overview

Options Volatility Strategy
===========================

Implements options and volatility-based trading logic.

MATURITY: BETA
--------------
Status: Comprehensive Greeks and volatility surface implementation
- [x] Black-Scholes pricing with dividends
- [x] Full Greeks calculation (delta, gamma, theta, vega, rho)
- [x] Implied volatility (Newton-Raphson)
- [x] IV percentile ranking
- [x] Vol surface construction (#O4)
- [x] Skew analysis (risk reversal, butterfly) (#O5)
- [x] Early exercise boundary (binomial) (#O3)
- [x] Vanna/Volga adjustments (#O11)
- [x] Option validation (#O2)
- [x] Spread strategies (verticals, iron condors) (#O7)
- [x] Pin risk detection (#O8)
- [x] Assignment risk calculation (#O9)
- [x] Gamma scalping support (#O10)
- [ ] SABR model (TODO)
- [ ] Local volatility surface (TODO)
- [ ] Variance swap replication (TODO)

Production Readiness:
- Unit tests: Good coverage for Greeks
- Validation: Option contract validation implemented
- Greeks bounds checking: Implemented

Use in production: WITH CAUTION
- IV solver may not converge for extreme values
- Early exercise uses binomial tree approximation
- Verify Greeks against broker values before trading

## Classes

### OptionValidationError

**Inherits from**: Exception

Raised when option contract validation fails.

### OptionData

Option contract data with validation (#O2).

Validates strike prices and expiration dates to ensure contract integrity.

#### Methods

##### `def __post_init__(self)`

Validate option contract data (#O2).

##### `def validate(self) -> None`

Validate option contract parameters (#O2).

Raises:
    OptionValidationError: If validation fails

##### `def is_expired(self) -> bool`

Check if option is expired.

##### `def is_atm(self) -> bool`

Check if option is approximately at-the-money (within 2% of strike).

##### `def mid_price(self) -> float`

Get mid-market price.

##### `def spread_pct(self) -> float`

Get bid-ask spread as percentage of mid price.

##### `def create_validated(cls, symbol: str, underlying: str, strike: float, expiry_days: int, is_call: bool, bid: float, ask: float, implied_vol: float, delta: float, gamma: float, theta: float, vega: float) -> OptionData`

Factory method to create validated option data.

Returns:
    Validated OptionData instance

Raises:
    OptionValidationError: If validation fails

### VolSignal

Volatility strategy signal.

### OptionsVolStrategy

Options Volatility Strategy Implementation.

Implements:
1. IV percentile ranking
2. Volatility risk premium analysis
3. Skew trading
4. Term structure analysis

TODO: Implement proper models:
- SABR for smile dynamics
- Local volatility surface
- Variance swap replication
- Dispersion trading

#### Methods

##### `def __init__(self, config: dict[str, Any])`

##### `def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float) -> float`

Calculate Black-Scholes option price with dividend yield.

Args:
    S: Spot price
    K: Strike price
    T: Time to expiry in years
    r: Risk-free interest rate (annualized)
    sigma: Volatility (annualized)
    is_call: True for call, False for put
    q: Continuous dividend yield (annualized, e.g., 0.02 for 2%)

Returns:
    Option price

Note: For stocks with discrete dividends, convert to continuous yield:
    q = -ln(1 - PV(dividends)/S) / T

##### `def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float) -> dict[str, float]`

Calculate option Greeks with dividend yield.

Args:
    S: Spot price
    K: Strike price
    T: Time to expiry in years
    r: Risk-free rate
    sigma: Volatility
    is_call: True for call, False for put
    q: Continuous dividend yield

Returns:
    Dictionary with delta, gamma, theta, vega, rho

##### `def implied_volatility(self, price: float, S: float, K: float, T: float, r: float, is_call: bool, q: float, precision: float, max_iterations: int) -> float`

Calculate implied volatility using Newton-Raphson.

Args:
    price: Market price of the option
    S: Spot price
    K: Strike price
    T: Time to expiry in years
    r: Risk-free rate
    is_call: True for call, False for put
    q: Continuous dividend yield
    precision: Price precision for convergence
    max_iterations: Maximum iterations

Returns:
    Implied volatility

TODO: Use more robust method (Brent's, bisection with bounds).

##### `def calculate_iv_percentile(self, current_iv: float, iv_history: list[float]) -> float`

Calculate IV percentile rank.

##### `def analyze_vol_surface(self, options: list[OptionData], spot_price: float) -> dict[str, Any]`

Analyze volatility surface characteristics.

Returns analysis of:
- ATM IV
- Skew (25 delta put - 25 delta call IV)
- Term structure

##### `def generate_signal(self, underlying: str, current_iv: float, iv_history: list[float], realized_vol: float, options: )`

Generate volatility trading signal.

##### `def calculate_early_exercise_boundary(self, S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float, n_steps: int) -> dict`

Calculate early exercise boundary for American options (#O3).

Uses binomial tree approximation for early exercise premium.

Args:
    S: Spot price
    K: Strike price
    T: Time to expiry (years)
    r: Risk-free rate
    sigma: Volatility
    is_call: True for call, False for put
    q: Dividend yield
    n_steps: Number of time steps in binomial tree

Returns:
    Early exercise analysis including boundary and premium

##### `def should_exercise_early(self, option: OptionData, underlying_price: float, risk_free_rate: float, dividend_yield: float, days_to_ex_div: , expected_dividend: ) -> dict`

Determine if early exercise is optimal (#O3).

For calls: Exercise before ex-dividend if dividend > time value
For puts: Exercise if time value < interest earned on proceeds

Args:
    option: Option contract data
    underlying_price: Current underlying price
    risk_free_rate: Risk-free rate
    dividend_yield: Continuous dividend yield
    days_to_ex_div: Days until ex-dividend date
    expected_dividend: Expected dividend amount

Returns:
    Early exercise recommendation

##### `def build_vol_surface(self, options: list[OptionData], spot_price: float, risk_free_rate: float) -> dict`

Build implied volatility surface (#O4).

Creates a grid of IV by strike (moneyness) and expiry.

Args:
    options: List of option data
    spot_price: Current underlying price
    risk_free_rate: Risk-free rate

Returns:
    Volatility surface data structure

##### `def interpolate_vol(self, surface: dict, target_expiry: int, target_moneyness: float)`

Interpolate volatility from surface (#O4).

Uses bilinear interpolation for intermediate points.

Args:
    surface: Vol surface from build_vol_surface
    target_expiry: Target expiry in days
    target_moneyness: Target log-moneyness

Returns:
    Interpolated implied volatility

##### `def analyze_skew(self, options: list[OptionData], spot_price: float, expiry_days: ) -> dict`

Analyze volatility skew at given expiry (#O5).

Calculates:
- 25-delta risk reversal (put IV - call IV)
- Butterfly (wing average - ATM)
- Skew slope

Args:
    options: List of option data
    spot_price: Current underlying price
    expiry_days: Specific expiry or None for all

Returns:
    Skew analysis metrics

##### `def detect_skew_anomaly(self, current_skew: dict, historical_skew: list[dict], z_threshold: float) -> dict`

Detect anomalies in current skew vs history (#O5).

Args:
    current_skew: Current skew analysis
    historical_skew: List of historical skew analyses
    z_threshold: Z-score threshold for anomaly

Returns:
    Anomaly detection results

##### `def calculate_greeks_term_structure(self, options: list[OptionData], spot_price: float) -> dict`

Calculate Greeks term structure across expiries (#O6).

Shows how Greeks evolve with time to expiration.

Args:
    options: List of option data
    spot_price: Current underlying price

Returns:
    Greeks term structure by expiry

##### `def create_vertical_spread(self, options: list[OptionData], spread_type: str, target_delta: float, width_pct: float)`

Create vertical spread strategy (#O7).

Args:
    options: Available options
    spread_type: Type of spread
    target_delta: Delta for short leg
    width_pct: Spread width as % of underlying

Returns:
    Spread definition or None if not possible

##### `def create_iron_condor(self, options: list[OptionData], put_delta: float, call_delta: float, wing_width_pct: float)`

Create iron condor strategy (#O7).

Sells put spread below market, sells call spread above market.

Args:
    options: Available options
    put_delta: Delta for short put
    call_delta: Delta for short call
    wing_width_pct: Wing width as % of strike

Returns:
    Iron condor definition

##### `def detect_pin_risk(self, options: list[OptionData], spot_price: float, position_size: int, pin_threshold_pct: float) -> dict`

Detect pin risk near expiration (#O8).

Pin risk occurs when underlying settles near a strike at expiration,
making assignment uncertain.

Args:
    options: Option positions
    spot_price: Current underlying price
    position_size: Number of contracts (negative = short)
    pin_threshold_pct: % from strike to flag as at-risk

Returns:
    Pin risk analysis

##### `def calculate_assignment_risk(self, option: OptionData, spot_price: float, position_size: int, days_to_ex_div: , dividend_amount: ) -> dict`

Calculate assignment risk for short options (#O9).

Assignment is most likely for:
- Deep ITM options (high intrinsic value)
- Near expiration
- Before ex-dividend for calls

Args:
    option: Option contract
    spot_price: Current underlying price
    position_size: Number of contracts (negative = short)
    days_to_ex_div: Days to ex-dividend
    dividend_amount: Expected dividend

Returns:
    Assignment risk analysis

##### `def calculate_gamma_scalp_parameters(self, option: OptionData, position_size: int, spot_price: float, realized_vol: float, hedge_interval_seconds: int) -> dict`

Calculate parameters for gamma scalping strategy (#O10).

Gamma scalping profits from realized volatility exceeding implied
by delta-hedging an options position.

Args:
    option: Option contract
    position_size: Number of contracts (positive = long gamma)
    spot_price: Current underlying price
    realized_vol: Current realized volatility
    hedge_interval_seconds: Seconds between hedges

Returns:
    Gamma scalping parameters

##### `def calculate_delta_hedge(self, options: list[tuple[OptionData, int]], spot_price: float, current_hedge_shares: int) -> dict`

Calculate required delta hedge (#O10).

Args:
    options: List of (option, position) tuples
    spot_price: Current underlying price
    current_hedge_shares: Current hedge position

Returns:
    Hedge adjustment needed

##### `def calculate_vanna(self, S: float, K: float, T: float, r: float, sigma: float, q: float) -> float`

Calculate Vanna (dDelta/dVol or dVega/dSpot) (#O11).

Vanna measures sensitivity of delta to volatility changes.

Args:
    S: Spot price
    K: Strike
    T: Time to expiry (years)
    r: Risk-free rate
    sigma: Volatility
    q: Dividend yield

Returns:
    Vanna value

##### `def calculate_volga(self, S: float, K: float, T: float, r: float, sigma: float, q: float) -> float`

Calculate Volga (dVega/dVol, also called Vomma) (#O11).

Volga measures convexity of vega with respect to volatility.

Args:
    S: Spot price
    K: Strike
    T: Time to expiry (years)
    r: Risk-free rate
    sigma: Volatility
    q: Dividend yield

Returns:
    Volga value

##### `def apply_vanna_volga_adjustment(self, bs_price: float, S: float, K: float, T: float, r: float, atm_vol: float, put_25d_vol: float, call_25d_vol: float, q: float) -> dict`

Apply Vanna-Volga adjustment to Black-Scholes price (#O11).

Adjusts BS price to account for smile dynamics using
market-implied risk reversals and butterflies.

Args:
    bs_price: Black-Scholes price
    S: Spot price
    K: Strike
    T: Time to expiry
    r: Risk-free rate
    atm_vol: ATM implied volatility
    put_25d_vol: 25-delta put IV
    call_25d_vol: 25-delta call IV
    q: Dividend yield

Returns:
    Adjusted price and components

##### `def suggest_portfolio_hedges(self, portfolio_greeks: dict, available_options: list[OptionData], spot_price: float, hedge_targets: ) -> list[dict]`

Suggest hedging trades for portfolio Greeks (#O12).

Args:
    portfolio_greeks: Current portfolio Greeks {delta, gamma, theta, vega}
    available_options: Options available for hedging
    spot_price: Current underlying price
    hedge_targets: Target Greeks (default: neutralize all)

Returns:
    List of suggested hedge trades

##### `def calculate_hedge_cost(self, hedge_suggestions: list[dict], options: list[OptionData], spot_price: float) -> dict`

Calculate cost of implementing hedge suggestions (#O12).

Args:
    hedge_suggestions: List of suggested hedges
    options: Option universe
    spot_price: Current underlying price

Returns:
    Cost breakdown

# options_vol_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\options_vol_agent.py`

## Overview

Options Volatility Agent
========================

Generates signals based on options market analysis.
Monitors implied volatility, skew, and term structure.

Responsibility: Options/volatility signal generation ONLY.
Does NOT make trading decisions or send orders.

## Classes

### VolatilityState

State for volatility tracking per underlying.

#### Methods

##### `def __post_init__(self)`

### OptionsVolAgent

**Inherits from**: SignalAgent

Options Volatility Agent.

Analyzes options markets for volatility-based signals:
1. IV percentile ranking
2. Volatility risk premium (IV vs RV)
3. Skew analysis
4. Term structure

Signal output:
- Volatility regime (high/low/normal)
- Vol selling/buying opportunities
- Skew trades

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `async def initialize(self) -> None`

Initialize volatility tracking.

##### `async def process_event(self, event: Event) -> None`

Process market data and generate volatility signals.

##### `def calculate_black_scholes_iv(self, option_price: float, spot: float, strike: float, tte: float, rate: float, is_call: bool, dividend_yield: float, max_iterations: int, precision: float)`

Calculate implied volatility via Newton-Raphson method.

Uses Black-Scholes-Merton formula with continuous dividend yield
and iteratively solves for IV.

The dividend yield adjustment uses the Merton extension:
- S * exp(-q*T) replaces S in the formula
- d1 = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)

Args:
    option_price: Market price of the option
    spot: Current spot price of underlying
    strike: Strike price
    tte: Time to expiry in years
    rate: Risk-free rate (annualized)
    is_call: True for call, False for put
    dividend_yield: Continuous dividend yield (annualized)
    max_iterations: Max Newton-Raphson iterations
    precision: Convergence threshold

Returns:
    Implied volatility or None if convergence fails

##### `def calculate_delta(self, spot: float, strike: float, tte: float, sigma: float, rate: float, is_call: bool, dividend_yield: float) -> float`

Calculate option delta using Black-Scholes-Merton with dividend yield.

For dividend-paying stocks:
- Call delta = exp(-q*T) * N(d1)
- Put delta = exp(-q*T) * (N(d1) - 1)

Args:
    spot: Current spot price
    strike: Strike price
    tte: Time to expiry in years
    sigma: Volatility
    rate: Risk-free rate
    is_call: True for call, False for put
    dividend_yield: Continuous dividend yield (annualized)

Returns:
    Delta value (-1 to 1)

##### `def filter_options_by_criteria(self, options: list[dict], spot: float, rate: float) -> list[dict]`

Filter options based on configured delta range and DTE criteria.

Args:
    options: List of option dicts with keys: strike, tte, sigma, is_call, price
    spot: Current spot price
    rate: Risk-free rate

Returns:
    Filtered list of options meeting criteria

##### `def get_optimal_option_for_signal(self, state: VolatilityState, direction: SignalDirection, rate: float)`

Find optimal option for given signal direction based on config criteria.

For vol selling (SHORT signal): Look for high IV options with delta in range
For vol buying (LONG signal): Look for low IV options with delta in range

Returns:
    Dict with recommended option parameters or None

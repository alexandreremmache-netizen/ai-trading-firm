# stat_arb_strategy

**Path**: `C:\Users\Alexa\ai-trading-firm\strategies\stat_arb_strategy.py`

## Overview

Statistical Arbitrage Strategy
==============================

Implements pairs trading, mean reversion, and commodity spreads.

MATURITY: BETA
--------------
Status: Comprehensive implementation with commodity spreads
- [x] Cointegration testing (simplified ADF)
- [x] Hedge ratio estimation (OLS)
- [x] Half-life calculation (OU process)
- [x] Commodity spreads (crack, crush, inter-commodity)
- [x] Optimal lag selection for ADF (#Q3)
- [x] Dollar-neutral spread sizing
- [ ] Johansen cointegration test (TODO)
- [ ] Kalman filter for dynamic hedge ratio (TODO)
- [ ] Transaction cost modeling (TODO)

Production Readiness:
- Unit tests: Partial coverage
- Backtesting: Spread definitions validated historically
- Live testing: Not yet performed

Use in production: WITH CAUTION
- Commodity spread ratios are industry-standard
- ADF test is simplified; production should use statsmodels
- Monitor half-life for regime changes

Features:
- Pairs trading (cointegration-based)
- Commodity spreads (crack, crush, calendar)
- Contract specs integration for proper sizing

## Classes

### SpreadType

**Inherits from**: Enum

Types of spread trades.

### CommoditySpread

Definition of a commodity spread trade.

### SpreadAnalysis

Analysis results for a spread trade.

### PairAnalysis

Analysis results for a trading pair.

### StatArbStrategy

Statistical Arbitrage Strategy Implementation.

Implements:
1. Cointegration testing (Engle-Granger)
2. Hedge ratio estimation
3. Mean reversion signal generation

TODO: Implement proper models:
- Johansen cointegration test
- Kalman filter for dynamic hedge ratio
- Ornstein-Uhlenbeck for half-life
- Transaction cost modeling

#### Methods

##### `def __init__(self, config: dict[str, Any])`

##### `def test_cointegration(self, prices_a: np.ndarray, prices_b: np.ndarray) -> PairAnalysis`

Test for cointegration between two price series.

TODO: Implement proper cointegration tests:
- Augmented Dickey-Fuller on residuals
- Johansen test for multiple series

##### `def generate_signal(self, analysis: PairAnalysis) -> dict[str, Any]`

Generate trading signal from pair analysis.

Returns signal dict with direction and strength.

##### `def analyze_spread(self, spread_name: str, prices: dict[str, np.ndarray], lookback: )`

Analyze a commodity spread.

Args:
    spread_name: Name of predefined spread
    prices: Dictionary of symbol to price arrays
    lookback: Lookback period (default: self._lookback)

Returns:
    SpreadAnalysis or None if spread can't be computed

##### `def get_spread_legs(self, spread_name: str)`

Get the legs (symbols and ratios) for a spread.

##### `def calculate_spread_position_sizes(self, spread_name: str, notional_value: float, prices: dict[str, float], contract_specs: , round_to_integer: bool)`

Calculate position sizes for spread legs with proper multiplier handling.

Args:
    spread_name: Spread name
    notional_value: Total notional value to allocate
    prices: Current prices
    contract_specs: Contract specifications (multipliers)
    round_to_integer: Whether to round to integer contracts

Returns:
    Dictionary of symbol to number of contracts

Note on ratio interpretation:
- Ratios in spread definitions are CONTRACT ratios, not price ratios
- For example, gold/silver ratio of 1:-1.6 means:
  1 GC contract ($200k) vs 1.6 SI contracts (~$200k) = dollar-neutral

##### `def calculate_dollar_neutral_spread(self, spread_name: str, target_notional: float, prices: dict[str, float], contract_specs: )`

Calculate dollar-neutral position sizes for a spread.

This ensures each leg has approximately equal dollar exposure,
which may differ from the defined contract ratios for some spreads.

Args:
    spread_name: Spread name
    target_notional: Target notional per leg
    prices: Current prices
    contract_specs: Contract specifications (multipliers)

Returns:
    Dictionary of symbol to number of contracts

##### `def get_all_spreads(self) -> list[str]`

Get list of all available spread names.

##### `def get_spreads_by_type(self, spread_type: SpreadType) -> list[str]`

Get spreads of a specific type.

##### `def get_spread_info(self, spread_name: str)`

Get detailed spread information.

##### `def scan_all_spreads(self, prices: dict[str, np.ndarray]) -> list[SpreadAnalysis]`

Scan all spreads and return those with signals.

Args:
    prices: Price data for all symbols

Returns:
    List of SpreadAnalysis for spreads with active signals

##### `def get_seasonal_spreads(self, month: ) -> list[str]`

Get spreads that are seasonally favorable.

Args:
    month: Month to check (1-12), default is current month

Returns:
    List of spread names

# fx_analytics

**Path**: `C:\Users\Alexa\ai-trading-firm\core\fx_analytics.py`

## Overview

FX Analytics Module
===================

Advanced FX analytics including volatility smile, central bank detection, and carry trade.

Issues Addressed:
- #X10: Missing FX volatility smile data
- #X11: No central bank intervention detection
- #X12: FX fixing rates not tracked
- #X13: No carry trade optimization

## Classes

### FXVolPoint

Single point on FX volatility surface (#X10).

### FXVolSmile

FX volatility smile at single expiry (#X10).

#### Methods

##### `def skew(self) -> float`

Get smile skew (positive = call premium).

##### `def smile_curvature(self) -> float`

Get smile curvature (convexity).

##### `def get_vol_at_delta(self, delta: float) -> float`

Interpolate volatility at specific delta (#X10).

Uses polynomial fitting through ATM, RR, and BF quotes.

##### `def to_dict(self) -> dict`

### FXVolSurface

Complete FX volatility surface (#X10).

#### Methods

##### `def get_vol(self, delta: float, expiry_days: int) -> float`

Get interpolated volatility from surface (#X10).

Args:
    delta: Delta (0-1 for calls)
    expiry_days: Days to expiry

Returns:
    Interpolated implied volatility

##### `def to_dict(self) -> dict`

### FXVolSmileManager

FX volatility smile management (#X10).

Tracks and manages vol surfaces for FX pairs with:
- Standard tenor smiles (1W, 1M, 3M, 6M, 1Y)
- Delta interpolation
- Term structure

#### Methods

##### `def __init__(self)`

##### `def update_smile(self, smile: FXVolSmile) -> None`

Update smile for a pair/tenor.

##### `def get_surface(self, pair: str)`

Get vol surface for pair.

##### `def get_vol(self, pair: str, delta: float, expiry_days: int)`

Get interpolated vol from surface.

##### `def analyze_smile(self, pair: str, expiry_days: int) -> dict`

Analyze smile characteristics (#X10).

Returns analysis of smile shape and signals.

### InterventionType

**Inherits from**: str, Enum

Type of central bank intervention.

### InterventionSignal

Signal of potential central bank intervention (#X11).

#### Methods

##### `def to_dict(self) -> dict`

### InterventionEvent

Historical intervention event (#X11).

### CentralBankInterventionDetector

Central bank intervention detection (#X11).

Detects potential intervention through:
- Abnormal price movements
- Volume spikes
- Volatility patterns
- Historical intervention levels

#### Methods

##### `def __init__(self)`

##### `def update_price(self, pair: str, price: float) -> None`

Update price for intervention monitoring.

##### `def check_intervention_risk(self, pair: str, current_price: float)`

Check for intervention risk (#X11).

Args:
    pair: Currency pair
    current_price: Current spot price

Returns:
    InterventionSignal if risk detected

##### `def record_intervention(self, event: InterventionEvent) -> None`

Record historical intervention event.

##### `def get_intervention_history(self, pair: str) -> list[InterventionEvent]`

Get intervention history for pair.

### FixingType

**Inherits from**: str, Enum

Type of FX fixing rate.

### FXFixingRate

FX fixing rate (#X12).

#### Methods

##### `def to_dict(self) -> dict`

### FXFixingManager

FX fixing rate tracking (#X12).

Tracks official fixing rates for:
- Portfolio valuation
- Benchmark comparison
- Execution analysis

#### Methods

##### `def __init__(self)`

##### `def record_fixing(self, fixing: FXFixingRate) -> None`

Record fixing rate (#X12).

##### `def get_latest_fixing(self, pair: str, fixing_type: FixingType)`

Get latest fixing rate.

##### `def get_fixing_history(self, pair: str, fixing_type: FixingType, days: int) -> list[FXFixingRate]`

Get fixing rate history.

##### `def calculate_fixing_vs_spot(self, pair: str, spot_rate: float, fixing_type: FixingType) -> dict`

Compare current spot to last fixing (#X12).

Returns analysis useful for fixing orders.

##### `def get_upcoming_fixings(self) -> list[dict]`

Get upcoming fixing times.

### CarryTradeOpportunity

Carry trade opportunity (#X13).

#### Methods

##### `def to_dict(self) -> dict`

### CarryPortfolio

Optimized carry portfolio (#X13).

#### Methods

##### `def to_dict(self) -> dict`

### CarryTradeOptimizer

Carry trade optimization (#X13).

Optimizes carry trades considering:
- Interest rate differentials
- Currency volatility
- Correlations
- Risk-off sensitivity

#### Methods

##### `def __init__(self)`

##### `def update_rate(self, currency: str, rate: float) -> None`

Update interest rate for currency.

##### `def update_volatility(self, pair: str, vol: float) -> None`

Update volatility for pair.

##### `def update_correlation(self, pair1: str, pair2: str, corr: float) -> None`

Update correlation between pairs.

##### `def update_risk_correlation(self, pair: str, corr: float) -> None`

Update correlation to risk-off (e.g., VIX or S&P).

##### `def find_opportunities(self, min_carry_bps: float, max_volatility: float) -> list[CarryTradeOpportunity]`

Find carry trade opportunities (#X13).

Args:
    min_carry_bps: Minimum annual carry in bps
    max_volatility: Maximum acceptable volatility

Returns:
    List of opportunities sorted by score

##### `def optimize_portfolio(self, opportunities: list[CarryTradeOpportunity], max_positions: int, target_volatility: float) -> CarryPortfolio`

Optimize carry portfolio (#X13).

Uses simple diversification approach.

##### `def get_carry_dashboard(self) -> dict`

Get carry trade dashboard (#X13).

## Functions

### `def get_pip_multiplier(pair: str) -> int`

Get pip multiplier for FX pair.

For JPY pairs (USDJPY, EURJPY, etc.): 1 pip = 0.01, so multiplier = 100
For other pairs (EURUSD, GBPUSD, etc.): 1 pip = 0.0001, so multiplier = 10000

Args:
    pair: Currency pair (e.g., "USDJPY", "EURUSD")

Returns:
    Pip multiplier (100 for JPY pairs, 10000 for others)

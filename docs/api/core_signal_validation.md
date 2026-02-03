# signal_validation

**Path**: `C:\Users\Alexa\ai-trading-firm\core\signal_validation.py`

## Overview

Signal Validation Module
========================

Mean reversion signal validation (Issue #Q16).
Spread ratio validation for stat arb (Issue #Q17).

Features:
- Mean reversion signal quality checks
- Spread ratio statistical validation
- Signal confidence scoring
- Historical performance validation

## Classes

### ValidationResult

**Inherits from**: str, Enum

Signal validation result.

### MeanReversionValidation

Mean reversion signal validation result (#Q16).

#### Methods

##### `def to_dict(self) -> dict`

### SpreadRatioValidation

Spread ratio validation for stat arb (#Q17).

#### Methods

##### `def to_dict(self) -> dict`

### MeanReversionValidator

Validates mean reversion signals (#Q16).

Ensures signals have statistical support before trading.

#### Methods

##### `def __init__(self, min_adf_confidence: float, max_half_life_days: float, min_half_life_days: float, max_hurst_exponent: float, zscore_entry_threshold: float)`

##### `def update_price(self, symbol: str, price: float, volume: int) -> None`

Update price history for a symbol.

##### `def validate_signal(self, symbol: str, direction: str, strength: float, current_price: ) -> MeanReversionValidation`

Validate a mean reversion signal.

Args:
    symbol: Trading symbol
    direction: 'LONG' (expecting price to rise) or 'SHORT'
    strength: Signal strength (0-1)
    current_price: Optional current price

Returns:
    MeanReversionValidation with detailed checks

### SpreadRatioValidator

Validates spread ratios for stat arb (#Q17).

Ensures pairs trading signals are statistically sound.

#### Methods

##### `def __init__(self, min_cointegration_confidence: float, max_half_life_days: float, min_correlation: float, hedge_ratio_stability_threshold: float, beta_neutrality_threshold: float)`

##### `def update_price(self, symbol: str, price: float) -> None`

Update price for a symbol.

##### `def set_beta(self, symbol: str, beta: float) -> None`

Set market beta for a symbol.

##### `def validate_spread(self, symbol1: str, symbol2: str, hedge_ratio: float) -> SpreadRatioValidation`

Validate a pairs trading spread.

Args:
    symbol1: Long leg symbol
    symbol2: Short leg symbol
    hedge_ratio: Number of symbol2 shares per symbol1 share

Returns:
    SpreadRatioValidation with all checks

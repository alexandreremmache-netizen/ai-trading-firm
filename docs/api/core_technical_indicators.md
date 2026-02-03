# technical_indicators

**Path**: `C:\Users\Alexa\ai-trading-firm\core\technical_indicators.py`

## Overview

Technical Indicators Module
===========================

Comprehensive technical analysis indicators (Issues #Q13, #Q14, #Q15).

Features:
- ADX (Average Directional Index) trend strength
- Bollinger Bands with squeeze detection
- Volume-weighted indicators (VWAP, OBV, MFI)
- Additional momentum and trend indicators

## Classes

### OHLCV

Single OHLCV bar.

### ADXResult

ADX calculation result (Issue #Q13).

#### Methods

##### `def trend_strength(self) -> str`

Classify trend strength.

##### `def trend_direction(self) -> str`

Determine trend direction.

### ADXCalculator

Average Directional Index calculator (Issue #Q13).

ADX measures trend strength without regard to direction.
Values above 25 indicate trending market.

#### Methods

##### `def __init__(self, period: int)`

##### `def update(self, bar: OHLCV)`

Update with new bar and return ADX result.

Returns None until enough data is accumulated.

### BollingerBandsResult

Bollinger Bands result (Issue #Q14).

#### Methods

##### `def is_squeeze(self) -> bool`

Detect Bollinger squeeze (low volatility).

##### `def position(self) -> str`

Price position within bands.

### BollingerBandsCalculator

Bollinger Bands calculator (Issue #Q14).

Bands based on SMA and standard deviation.
Useful for volatility and mean reversion signals.

#### Methods

##### `def __init__(self, period: int, std_dev: float)`

##### `def update(self, close: float)`

Update with new close and return bands.

Returns None until enough data.

### VWAPResult

VWAP calculation result (Issue #Q15).

### VWAPCalculator

Volume Weighted Average Price calculator (Issue #Q15).

Intraday benchmark for institutional execution.
Resets daily.

#### Methods

##### `def __init__(self, use_bands: bool, band_std: float)`

##### `def reset(self) -> None`

Reset for new trading day.

##### `def update(self, bar: OHLCV)`

Update with new bar and return VWAP.

Automatically resets on new day.

### OBVResult

On Balance Volume result (Issue #Q15).

### OBVCalculator

On Balance Volume calculator (Issue #Q15).

Cumulative volume indicator for trend confirmation.

#### Methods

##### `def __init__(self, ema_period: int)`

##### `def update(self, close: float, volume: float)`

Update with new close/volume and return OBV.

### MFIResult

Money Flow Index result (Issue #Q15).

### MFICalculator

Money Flow Index calculator (Issue #Q15).

Volume-weighted RSI that incorporates price and volume.

#### Methods

##### `def __init__(self, period: int)`

##### `def update(self, bar: OHLCV)`

Update with new bar and return MFI.

### ATRResult

Average True Range result.

### ATRCalculator

Average True Range calculator.

Measures volatility.

#### Methods

##### `def __init__(self, period: int)`

##### `def update(self, bar: OHLCV)`

Update with new bar and return ATR.

### StochasticResult

Stochastic Oscillator result.

### StochasticCalculator

Stochastic Oscillator calculator.

Momentum indicator comparing close to high-low range.

#### Methods

##### `def __init__(self, k_period: int, d_period: int)`

##### `def update(self, bar: OHLCV)`

Update with new bar and return Stochastic.

### KeltnerChannelResult

Keltner Channel result.

### KeltnerChannelCalculator

Keltner Channel calculator.

EMA-based bands using ATR for width.

#### Methods

##### `def __init__(self, ema_period: int, atr_period: int, atr_mult: float)`

##### `def update(self, bar: OHLCV)`

Update with new bar and return Keltner Channel.

### TechnicalIndicatorSuite

Comprehensive technical indicator suite.

Combines multiple indicators for signal generation.

#### Methods

##### `def __init__(self, adx_period: int, bb_period: int, bb_std: float, mfi_period: int, atr_period: int)`

##### `def update(self, bar: OHLCV) -> dict[str, Any]`

Update all indicators with new bar.

Returns dictionary of indicator results.

##### `def get_composite_signal(self, bar: OHLCV) -> dict[str, Any]`

Get composite trading signal from all indicators.

Returns signal strength and direction.

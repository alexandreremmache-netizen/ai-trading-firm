# momentum_strategy

**Path**: `C:\Users\Alexa\ai-trading-firm\strategies\momentum_strategy.py`

## Overview

Momentum Strategy
=================

Implements momentum and trend-following logic.

MATURITY: BETA
--------------
Status: Core functionality implemented and tested
- [x] MA crossovers implemented
- [x] RSI calculation (Wilder's smoothing)
- [x] MACD with proper signal line
- [x] Stop-loss calculation (ATR-based)
- [ ] ADX trend strength (TODO)
- [ ] Volume-weighted indicators (TODO)
- [ ] Ichimoku Cloud (TODO)

Production Readiness:
- Unit tests: Partial coverage
- Backtesting: Not yet performed
- Live testing: Not yet performed

Use in production: WITH CAUTION - verify signals manually before trading

## Classes

### MomentumSignal

Momentum signal output.

### MomentumStrategy

Momentum Strategy Implementation.

Implements:
1. Moving average crossovers
2. RSI (Relative Strength Index)
3. MACD (Moving Average Convergence Divergence)
4. Rate of Change (ROC)

TODO: Implement more sophisticated models:
- ADX for trend strength
- Ichimoku Cloud
- Bollinger Bands
- Volume-weighted indicators

#### Methods

##### `def __init__(self, config: dict[str, Any])`

##### `def calculate_sma(self, prices: np.ndarray, period: int) -> float`

Calculate Simple Moving Average.

##### `def calculate_ema(self, prices: np.ndarray, period: int) -> float`

Calculate Exponential Moving Average (returns single value).

##### `def calculate_ema_series(self, prices: np.ndarray, period: int) -> np.ndarray`

Calculate full EMA series for MACD signal line calculation.

Returns array of EMA values, same length as input prices.

##### `def calculate_rsi(self, prices: np.ndarray, period: int) -> float`

Calculate Relative Strength Index using Wilder's smoothing method.

Wilder's smoothing is an exponential moving average with
smoothing factor = 1/period (as opposed to 2/(period+1) for standard EMA).

##### `def calculate_macd(self, prices: np.ndarray) -> tuple[float, float, float]`

Calculate MACD, Signal line, and Histogram.

MACD Line = Fast EMA (12) - Slow EMA (26)
Signal Line = 9-period EMA of MACD Line
Histogram = MACD Line - Signal Line

Returns (macd, signal, histogram).

##### `def calculate_roc(self, prices: np.ndarray, period: int) -> float`

Calculate Rate of Change.

##### `def calculate_atr(self, prices: np.ndarray, highs: , lows: , period: int) -> float`

P1-13: Calculate Average True Range for stop-loss placement.

If highs/lows not provided, estimates from price changes.

##### `def calculate_stop_loss(self, current_price: float, direction: str, atr: float) -> tuple[float, float]`

P1-13: Calculate stop-loss price and percentage.

Args:
    current_price: Entry price
    direction: "long" or "short"
    atr: Average True Range

Returns:
    (stop_price, stop_pct)

##### `def analyze(self, symbol: str, prices: np.ndarray) -> MomentumSignal`

Analyze price series and generate momentum signal.

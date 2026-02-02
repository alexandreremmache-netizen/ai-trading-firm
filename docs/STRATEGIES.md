# Trading Strategies Documentation

## Overview

The AI Trading Firm implements multiple trading strategies through specialized signal agents. Each strategy generates advisory signals that are aggregated by the CIO agent for decision-making.

## Strategy Architecture

```
+------------------------------------------------------------------+
|                       STRATEGY LAYER                              |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------+  +-------------+  +-------------+                |
|  |  Momentum   |  |  Stat Arb   |  | Options Vol |                |
|  |  Strategy   |  |  Strategy   |  |  Strategy   |                |
|  +-------------+  +-------------+  +-------------+                |
|         |               |               |                         |
|         v               v               v                         |
|  +-------------+  +-------------+  +-------------+                |
|  |   Momentum  |  |  StatArb    |  | OptionsVol  |                |
|  |    Agent    |  |   Agent     |  |   Agent     |                |
|  +-------------+  +-------------+  +-------------+                |
|                                                                   |
|  +-------------+  +-------------+                                 |
|  |   Macro     |  |Market Making|                                 |
|  |  Strategy   |  |  Strategy   |                                 |
|  +-------------+  +-------------+                                 |
|         |               |                                         |
|         v               v                                         |
|  +-------------+  +-------------+                                 |
|  |   Macro     |  |    MM       |                                 |
|  |   Agent     |  |   Agent     |                                 |
|  +-------------+  +-------------+                                 |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Momentum Strategy

### Description
The momentum strategy captures trends using technical indicators. It identifies securities with strong price momentum and generates signals in the direction of the trend.

### Implementation

**File**: `strategies/momentum_strategy.py`

### Indicators

#### Simple Moving Average (SMA)
```python
def calculate_sma(self, prices: np.ndarray, period: int) -> float:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0.0
    return np.mean(prices[-period:])
```

#### Exponential Moving Average (EMA)
```python
def calculate_ema(self, prices: np.ndarray, period: int) -> float:
    """
    Calculate Exponential Moving Average.
    Alpha = 2 / (period + 1)
    """
    alpha = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema
```

#### Relative Strength Index (RSI)
```python
def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
    """
    Calculate RSI using Wilder's smoothing method.
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
```

**Interpretation**:
- RSI > 70: Overbought (potential sell signal)
- RSI < 30: Oversold (potential buy signal)

#### MACD (Moving Average Convergence Divergence)
```python
def calculate_macd(self, prices: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate MACD, Signal line, and Histogram.
    MACD Line = 12-EMA - 26-EMA
    Signal Line = 9-EMA of MACD Line
    Histogram = MACD Line - Signal Line
    """
```

**Interpretation**:
- MACD crosses above signal: Bullish
- MACD crosses below signal: Bearish
- Histogram expanding: Strengthening trend

### Signal Generation Logic

```
1. Calculate fast EMA (10-period)
2. Calculate slow EMA (30-period)
3. Calculate RSI (14-period)
4. Calculate MACD and signal line

IF fast_ema > slow_ema AND rsi < 70 AND macd > signal:
    Signal = LONG
    Strength = normalized_macd_histogram
    Confidence = (1 - rsi/100) * trend_clarity

ELIF fast_ema < slow_ema AND rsi > 30 AND macd < signal:
    Signal = SHORT
    Strength = -normalized_macd_histogram
    Confidence = (rsi/100) * trend_clarity

ELSE:
    Signal = FLAT
```

### Configuration

```yaml
agents:
  momentum:
    enabled: true
    fast_period: 10        # Fast moving average period
    slow_period: 30        # Slow moving average period
    signal_period: 9       # MACD signal line period
    rsi_period: 14         # RSI calculation period
    rsi_overbought: 70     # RSI overbought threshold
    rsi_oversold: 30       # RSI oversold threshold
```

---

## Statistical Arbitrage Strategy

### Description
The statistical arbitrage strategy exploits price relationships between related securities. It includes pairs trading, commodity spreads, and mean reversion strategies.

### Implementation

**File**: `strategies/stat_arb_strategy.py`

### Pairs Trading

#### Cointegration Test
```python
def test_cointegration(self, series1: np.ndarray, series2: np.ndarray) -> tuple[float, bool]:
    """
    Test for cointegration using Augmented Dickey-Fuller test.
    Returns (p-value, is_cointegrated)
    """
```

#### Z-Score Calculation
```python
def calculate_zscore(self, spread: np.ndarray, lookback: int) -> float:
    """
    Calculate z-score of spread.
    z = (spread - mean) / std
    """
    mean = np.mean(spread[-lookback:])
    std = np.std(spread[-lookback:])
    return (spread[-1] - mean) / std if std > 0 else 0.0
```

### Signal Generation Logic

```
1. Calculate hedge ratio using OLS regression
2. Calculate spread = price1 - hedge_ratio * price2
3. Calculate z-score of spread

IF zscore > entry_threshold (2.0):
    Signal = SHORT spread (short asset1, long asset2)
    Strength = -zscore / 3.0  # Normalized
    Confidence = min(1.0, abs(zscore) / 4.0)

ELIF zscore < -entry_threshold (-2.0):
    Signal = LONG spread (long asset1, short asset2)
    Strength = -zscore / 3.0
    Confidence = min(1.0, abs(zscore) / 4.0)

ELIF abs(zscore) < exit_threshold (0.5):
    Signal = EXIT (close position)
```

### Commodity Spreads

#### Crack Spread (3:2:1)
```python
COMMODITY_SPREADS = {
    "3-2-1_crack": CommoditySpread(
        name="3:2:1 Crack Spread",
        spread_type=SpreadType.CRACK,
        legs={
            "CL": -3.0,   # Short 3 crude oil
            "RB": 2.0,    # Long 2 gasoline
            "HO": 1.0,    # Long 1 heating oil
        },
        description="Standard refinery margin",
        typical_range=(5.0, 35.0),  # USD per barrel
        seasonality=[2, 3, 4, 5],   # Feb-May
    ),
}
```

#### Crush Spread (Soybeans)
```python
"soybean_crush": CommoditySpread(
    name="Soybean Crush Spread",
    spread_type=SpreadType.CRUSH,
    legs={
        "ZS": -10.0,   # Short 10 soybean contracts
        "ZM": 12.0,    # Long 12 soybean meal contracts
        "ZL": 9.0,     # Long 9 soybean oil contracts
    },
    description="CBOT Board Crush: processing margin",
)
```

### Configuration

```yaml
agents:
  stat_arb:
    enabled: true
    lookback_days: 60              # Rolling window for calculations
    zscore_entry_threshold: 2.0    # Z-score to enter trade
    zscore_exit_threshold: 0.5     # Z-score to exit trade
    pairs:
      - ["AAPL", "MSFT"]           # Tech sector pair
      - ["GOOGL", "META"]          # Communication services
      - ["ES", "NQ"]               # Index futures
      - ["GC", "SI"]               # Precious metals
```

---

## Options Volatility Strategy

### Description
The options volatility strategy trades based on implied volatility levels and options Greeks. It aims to capture volatility premium by selling when IV is high and buying when IV is low.

### Implementation

**File**: `strategies/options_vol_strategy.py`

### Option Data Validation

```python
@dataclass
class OptionData:
    """Option contract data with validation."""
    symbol: str
    underlying: str
    strike: float
    expiry_days: int
    is_call: bool
    bid: float
    ask: float
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float

    def validate(self) -> None:
        """
        Validate option contract parameters.
        - Strike > 0 and within bounds
        - Expiry >= 0 and within bounds
        - Valid bid/ask spread
        - Greeks within theoretical bounds
        """
```

### Black-Scholes Greeks

```python
def calculate_delta(self, S, K, T, r, sigma, is_call: bool) -> float:
    """
    Calculate option delta.
    Call Delta = N(d1)
    Put Delta = N(d1) - 1
    """

def calculate_gamma(self, S, K, T, r, sigma) -> float:
    """
    Calculate option gamma.
    Gamma = N'(d1) / (S * sigma * sqrt(T))
    """

def calculate_vega(self, S, K, T, r, sigma) -> float:
    """
    Calculate option vega.
    Vega = S * N'(d1) * sqrt(T)
    """

def calculate_theta(self, S, K, T, r, sigma, is_call: bool) -> float:
    """
    Calculate option theta (time decay).
    """
```

### Signal Generation Logic

```
1. Calculate IV percentile (current IV vs. historical)
2. Identify options within delta range
3. Check days to expiration (DTE) constraints

IF iv_percentile > 80 AND dte in [min_dte, max_dte]:
    Signal = SELL VOLATILITY
    Strategy = Sell puts/calls or spreads
    Strength = (iv_percentile - 50) / 50
    Confidence = iv_percentile / 100

ELIF iv_percentile < 20 AND dte in [min_dte, max_dte]:
    Signal = BUY VOLATILITY
    Strategy = Buy straddles/strangles
    Strength = (50 - iv_percentile) / 50
    Confidence = (100 - iv_percentile) / 100
```

### Configuration

```yaml
agents:
  options_vol:
    enabled: true
    iv_percentile_threshold: 80   # IV percentile to trigger sell
    min_dte: 7                    # Minimum days to expiration
    max_dte: 45                   # Maximum days to expiration
    delta_range: [0.20, 0.40]     # Target delta range
```

---

## Macro Strategy

### Description
The macro strategy generates signals based on macroeconomic indicators. It provides a top-down view of market conditions to inform risk allocation.

### Implementation

**File**: `strategies/macro_strategy.py`

### Indicators

#### Yield Curve
- 2-year vs 10-year Treasury spread
- Inversion signals recession risk
- Steepening signals economic growth

#### VIX (Volatility Index)
- VIX > 25: High fear, risk-off
- VIX < 15: Low fear, risk-on
- VIX term structure for regime detection

#### Dollar Index (DXY)
- Strong dollar: Pressure on international earnings
- Weak dollar: Commodity rally potential

### Signal Generation Logic

```
1. Collect macro indicator readings
2. Determine market regime

IF vix > 25 OR yield_curve_inverted:
    Regime = RISK_OFF
    Equity Signal = SHORT
    Bond Signal = LONG
    Gold Signal = LONG

ELIF vix < 15 AND yield_curve_steepening:
    Regime = RISK_ON
    Equity Signal = LONG
    Bond Signal = NEUTRAL

ELIF vix > 20 AND vix_term_structure_inverted:
    Regime = VOLATILE
    Signal = REDUCE EXPOSURE
```

### Configuration

```yaml
agents:
  macro:
    enabled: true
    indicators:
      - "yield_curve"
      - "vix"
      - "dxy"
    rebalance_frequency: "daily"
```

---

## Market Making Strategy

### Description
The market making strategy provides liquidity by quoting bid and ask prices. It captures the spread while managing inventory risk.

### Implementation

**File**: `strategies/market_making_strategy.py`

### Quote Generation

```python
def generate_quotes(self, mid_price: float, spread_bps: float) -> tuple[float, float]:
    """
    Generate bid and ask quotes around mid-price.
    Adjust for inventory to manage directional risk.
    """
    half_spread = mid_price * (spread_bps / 10000) / 2

    # Skew quotes based on inventory
    inventory_skew = self.inventory * self.skew_factor

    bid = mid_price - half_spread - inventory_skew
    ask = mid_price + half_spread - inventory_skew

    return bid, ask
```

### Inventory Management

```python
def adjust_for_inventory(self, current_inventory: int) -> float:
    """
    Calculate spread adjustment based on inventory.
    Wider spread when inventory is high.
    """
    inventory_ratio = abs(current_inventory) / self.max_inventory
    spread_multiplier = 1.0 + (inventory_ratio * 0.5)
    return spread_multiplier
```

### Configuration

```yaml
agents:
  market_making:
    enabled: true
    spread_bps: 10           # Base spread in basis points
    max_inventory: 1000      # Maximum inventory limit
    quote_refresh_ms: 1000   # Quote update frequency
```

---

## Signal Aggregation

The CIO agent aggregates signals from all strategies using weighted averaging and correlation adjustment.

### Weight Calculation

```python
def _aggregate_signals(self, signals: dict[str, SignalEvent]) -> SignalAggregation:
    """
    Aggregate signals with dynamic weights.

    1. Apply base weights
    2. Apply regime adjustments
    3. Apply performance adjustments
    4. Apply correlation discounts
    """
```

### Default Weights

| Strategy | Weight |
|----------|--------|
| Macro | 15% |
| Statistical Arbitrage | 25% |
| Momentum | 25% |
| Market Making | 15% |
| Options Volatility | 20% |

### Dynamic Weight Adjustment

#### Regime-Based
```python
_regime_weights = {
    MarketRegime.RISK_ON: {
        "MomentumAgent": 1.3,      # Increase momentum
        "StatArbAgent": 0.9,
        "MacroAgent": 0.8,
    },
    MarketRegime.VOLATILE: {
        "OptionsVolAgent": 1.4,    # Increase options
        "MarketMakingAgent": 0.7,
        "MomentumAgent": 0.8,
    },
}
```

#### Performance-Based
Strategies with higher Sharpe ratios get increased weights.

#### Correlation Adjustment
Highly correlated signals are discounted to avoid overconfidence.

---

## Signal Quality Metrics

### Conviction Score
```
conviction = weighted_confidence * signal_agreement
```

### Effective Signal Count
Accounts for correlation between signals:
```
effective_n = 1 / sum(wi * wj * corr_ij)
```

### Minimum Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Conviction | 0.6 | Required for decision |
| Signal Count | 2 | Required for decision |
| Strength | 0.1 | Minimum signal strength |

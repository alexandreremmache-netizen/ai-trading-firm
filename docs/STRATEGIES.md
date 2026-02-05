# Trading Strategies Documentation

## Overview

The AI Trading Firm implements 13 trading strategies across multiple asset classes and styles. Each strategy is designed for specific market conditions and has been calibrated for institutional-grade risk management.

## Strategy Index

| Strategy | File | Style | Asset Classes | Maturity |
|----------|------|-------|---------------|----------|
| Momentum | `momentum_strategy.py` | Trend Following | All | BETA |
| Statistical Arbitrage | `stat_arb_strategy.py` | Mean Reversion | Equities, Futures | BETA |
| Macro | `macro_strategy.py` | Macro/Fundamental | All | ALPHA |
| Market Making | `market_making_strategy.py` | Neutral | Equities, Futures | ALPHA |
| Options Volatility | `options_vol_strategy.py` | Volatility | Options | BETA |
| Session-Based | `session_strategy.py` | Intraday | Futures, Equities | BETA |
| Index Spread | `index_spread_strategy.py` | Relative Value | Index Futures | BETA |
| TTM Squeeze | `ttm_squeeze_strategy.py` | Volatility Breakout | All | BETA |
| Event-Driven | `event_driven_strategy.py` | Event | All | BETA |
| Mean Reversion | `mean_reversion_strategy.py` | Mean Reversion | Equities | BETA |
| Ichimoku Cloud | `ichimoku_strategy.py` | Trend Following | All | BETA |
| Seasonality | `seasonality.py` | Calendar | Commodities | ALPHA |

---

## Strategy Architecture

```
+------------------------------------------------------------------+
|                       STRATEGY LAYER                              |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------+  +-------------+  +-------------+  +-------------+
|  |  Momentum   |  |  Stat Arb   |  | Options Vol |  |   Macro     |
|  |  Strategy   |  |  Strategy   |  |  Strategy   |  |  Strategy   |
|  +-------------+  +-------------+  +-------------+  +-------------+
|         |               |               |               |         |
|         v               v               v               v         |
|  +-------------+  +-------------+  +-------------+  +-------------+
|  |   Momentum  |  |  StatArb    |  | OptionsVol  |  |   Macro     |
|  |    Agent    |  |   Agent     |  |   Agent     |  |   Agent     |
|  +-------------+  +-------------+  +-------------+  +-------------+
|                                                                   |
|  +-------------+  +-------------+  +-------------+  +-------------+
|  |   Session   |  |Index Spread |  |TTM Squeeze  |  |Event Driven |
|  |  Strategy   |  |  Strategy   |  |  Strategy   |  |  Strategy   |
|  +-------------+  +-------------+  +-------------+  +-------------+
|         |               |               |               |         |
|         v               v               v               v         |
|  +-------------+  +-------------+  +-------------+  +-------------+
|  |  Session    |  |IndexSpread  |  |TTMSqueeze   |  |EventDriven  |
|  |   Agent     |  |   Agent     |  |   Agent     |  |   Agent     |
|  +-------------+  +-------------+  +-------------+  +-------------+
|                                                                   |
|  +-------------+  +-------------+  +-------------+                 |
|  |Mean Revert  |  | Sentiment*  |  |  Chart*     |  (* = LLM)     |
|  |  Strategy   |  |  Strategy   |  | Analysis    |                 |
|  +-------------+  +-------------+  +-------------+                 |
|                                                                   |
+------------------------------------------------------------------+
```

---

## 1. Momentum Strategy

**File**: `strategies/momentum_strategy.py`

**Style**: Trend Following

**Description**: Implements momentum and trend-following logic using multiple technical indicators with research-optimized parameters.

### Indicators Used

| Indicator | Default Period | Purpose |
|-----------|----------------|---------|
| Moving Average Crossover | Fast: 10, Slow: 30 | Trend direction |
| RSI | 14 | Overbought/oversold |
| MACD | 12, 26, 9 | Momentum confirmation |
| ADX | 14 | Trend strength |
| ATR | 14 | Volatility-based stops |

### Signal Generation

```python
# Long Signal
if fast_ma > slow_ma and rsi < rsi_overbought and macd_histogram > 0:
    direction = "long"
    strength = normalize(macd_histogram)

# Short Signal
if fast_ma < slow_ma and rsi > rsi_oversold and macd_histogram < 0:
    direction = "short"
    strength = normalize(abs(macd_histogram))
```

### Asset-Specific Parameters

```yaml
asset_configs:
  futures:
    fast_period: 9
    slow_period: 21
    trend_filter_period: 200
    use_ema: true
  forex:
    fast_period: 5
    slow_period: 13
    use_ema: true
  equities:
    fast_period: 20
    slow_period: 50
    use_ema: false  # SMA for equities
  commodities:
    fast_period: 20
    slow_period: 50
    use_ema: true
```

### RSI Overrides by Asset

| Asset | Overbought | Oversold | Notes |
|-------|------------|----------|-------|
| MNQ | 75 | 25 | Wider for volatile assets |
| M2K | 75 | 25 | |
| NG | 80 | 20 | Very volatile |
| SI | 75 | 25 | |
| GBPUSD | 85 | 15 | |
| GC | 60 | 40 | Use midband |

### MACD Settings by Timeframe

```yaml
macd_by_timeframe:
  scalping_1m_5m:
    fast: 3
    slow: 10
    signal: 16
  day_trading_5m_15m:
    fast: 5
    slow: 35
    signal: 5
  day_trading_15m_1h:
    fast: 8
    slow: 17
    signal: 9
  swing_daily:
    fast: 12
    slow: 26
    signal: 9
```

### Risk Management

- **Stop-Loss**: ATR-based (2.5x ATR default)
- **Take-Profit**: 2:1 risk-reward ratio
- **Position Sizing**: Volatility-scaled
- **Blackout Periods**: FOMC, NFP, CPI events

### Advanced Features

- 52-Week High/Low signal
- Dual Momentum (Antonacci method)
- Cross-Sectional Momentum
- Multi-Timeframe Analysis
- ADX trend filter (strong > 25, moderate > 20)
- Triple confirmation mode (RSI + MACD + Stochastic)
- Demand/Supply zone detection

---

## 2. Statistical Arbitrage Strategy

**File**: `strategies/stat_arb_strategy.py`

**Style**: Mean Reversion / Pairs Trading

**Description**: Implements pairs trading, commodity spreads, and statistical arbitrage using cointegration analysis and Kalman filters.

### Spread Types

| Type | Description | Example |
|------|-------------|---------|
| PAIRS | Simple pairs trade | AAPL/MSFT |
| CRACK | Crude to products | CL -> RB + HO |
| CRUSH | Soybeans to products | ZS -> ZM + ZL |
| CALENDAR | Same commodity, different months | CLZ5/CLF6 |
| INTER_COMMODITY | Related commodities | GC/SI |

### Signal Generation

```python
# Calculate z-score of spread
z_score = (current_spread - mean_spread) / std_spread

# Entry signals
if z_score > entry_threshold:  # Short spread
    direction = "short"
elif z_score < -entry_threshold:  # Long spread
    direction = "long"

# Exit signals
if abs(z_score) < exit_threshold:
    direction = "flat"
```

### Predefined Commodity Spreads

**3:2:1 Crack Spread**:
```yaml
name: "3:2:1 Crack Spread"
legs:
  CL: -3.0   # Short 3 crude
  RB: 2.0    # Long 2 gasoline
  HO: 1.0    # Long 1 heating oil
typical_range: [5.0, 35.0]  # USD per barrel
seasonality: [2, 3, 4, 5]   # Feb-May (pre-driving season)
margin_offset_pct: 75.0
storage_cost_annual_pct: 3.5
```

**Gasoline Crack Spread**:
```yaml
name: "Gasoline Crack Spread"
legs:
  CL: -1.0   # Short crude
  RB: 1.0    # Long gasoline
typical_range: [5.0, 40.0]
seasonality: [2, 3, 4, 5, 6]  # Feb-Jun
```

**Soybean Crush Spread**:
```yaml
name: "Soybean Crush Spread"
legs:
  ZS: -10.0  # Short soybeans
  ZM: 12.0   # Long soybean meal
  ZL: 9.0    # Long soybean oil
typical_range: [0.5, 2.0]
seasonality: [9, 10, 11]  # Sep-Nov
```

### Cointegration Methods

| Method | File | Use Case |
|--------|------|----------|
| OLS | Built-in | Static hedge ratio |
| Rolling OLS | Built-in | Time-varying hedge ratio |
| Kalman Filter | `core/kalman_filter.py` | Dynamic hedge ratio with noise filtering |
| Johansen Test | Built-in | Multiple cointegrated series |

### Configuration

```yaml
agents:
  stat_arb:
    enabled: true
    lookback_days: 60
    zscore_entry_threshold: 2.0
    zscore_exit_threshold: 0.5
    pairs:
      - ["AAPL", "MSFT"]
      - ["GOOGL", "META"]
      - ["ES", "NQ"]
      - ["GC", "SI"]
    divergence_enabled: true
    divergence_threshold: 2.0
    divergence_lookback: 20
    correlation_breakdown_threshold: 0.3
```

### Risk Management

- Half-life calculation for mean reversion timing
- Dollar-neutral position sizing
- Correlation breakdown detection
- Maximum holding period based on half-life
- Storage cost modeling for commodity spreads

---

## 3. Macro Strategy

**File**: `strategies/macro_strategy.py`

**Style**: Macro/Fundamental

**Description**: Generates signals based on macroeconomic indicators including yield curves, volatility indices, and currency strength.

### Indicators

| Indicator | Signal | Threshold |
|-----------|--------|-----------|
| VIX | Risk-off | > 25 |
| 2s10s Spread | Recession | < 0 (inverted) |
| DXY | USD strength | Relative to 200-day MA |
| Credit Spreads | Risk-off | Widening > 50bps/month |

### Regime Classification

```python
class MarketRegime(Enum):
    RISK_ON = "risk_on"      # Low VIX, positive yield curve
    RISK_OFF = "risk_off"    # High VIX, inverted curve
    VOLATILE = "volatile"    # VIX > 30
    TRENDING = "trending"    # ADX > 25
    MEAN_REVERTING = "mean_reverting"
    NEUTRAL = "neutral"
```

### Integration with HMM Regime Detection

```python
from core.hmm_regime import HMMRegimeDetector, create_hmm_detector

detector = create_hmm_detector(n_states=3)
detector.fit(returns_history)
current_regime = detector.predict_state(recent_returns)
```

### Signal Logic

- **Risk-Off Signal**: VIX > 25, yield curve inverts
- **Risk-On Signal**: VIX < 15, positive yield curve
- **USD Impact**: Strong USD reduces international equity exposure
- **Recession Probability**: Based on 2s10s spread analysis

---

## 4. Market Making Strategy

**File**: `strategies/market_making_strategy.py`

**Style**: Neutral / Market Making

**Description**: Implements optimal market making using the Avellaneda-Stoikov model for spread and inventory management.

### Avellaneda-Stoikov Model

The strategy calculates optimal bid/ask prices based on:
- Current inventory position
- Time horizon
- Risk aversion parameter (gamma)
- Market volatility

```python
# Reservation price (fair value adjusted for inventory)
reservation_price = mid_price - inventory * gamma * volatility**2 * T

# Optimal spread
optimal_spread = gamma * volatility**2 * T + (2/gamma) * log(1 + gamma/k)

# Bid/Ask prices
bid_price = reservation_price - optimal_spread / 2
ask_price = reservation_price + optimal_spread / 2
```

### Inventory Management

| Inventory Level | Action |
|-----------------|--------|
| Within limits | Quote normally |
| Near max long | Widen bid, narrow ask |
| Near max short | Narrow bid, widen ask |
| At limit | Cancel opposite side |

### Configuration

```yaml
agents:
  market_making:
    enabled: true
    spread_bps: 10
    max_inventory: 1000
    quote_refresh_ms: 1000
```

---

## 5. Options Volatility Strategy

**File**: `strategies/options_vol_strategy.py`

**Style**: Volatility Trading

**Description**: Trades options based on implied volatility analysis, volatility surface, and Greeks management.

**Note**: Options trading is currently disabled in the system configuration.

### Features

- IV percentile ranking (0-100)
- Volatility surface analysis
- Term structure analysis
- Delta-targeted positions
- Greeks monitoring and limits

### Signal Logic

```python
# IV Percentile signals
if iv_percentile > 80:
    # Sell volatility (short straddle/strangle)
    direction = "short_vol"
elif iv_percentile < 20:
    # Buy volatility (long straddle/strangle)
    direction = "long_vol"
```

### Greeks Limits

```yaml
greeks:
  max_portfolio_delta: 500
  max_portfolio_gamma: 100
  max_portfolio_vega: 50000
  max_portfolio_theta: -10000
```

### Configuration

```yaml
agents:
  options_vol:
    enabled: false  # Currently disabled
    iv_percentile_threshold: 80
    min_dte: 7
    max_dte: 45
    delta_range: [0.20, 0.40]
```

---

## 6. Session-Based Strategy

**File**: `strategies/session_strategy.py`

**Style**: Intraday / Opening Range Breakout

**Description**: Trades based on session-specific patterns, particularly the Opening Range Breakout (ORB) strategy.

### Opening Range Breakout

```python
# Track first N minutes of session
opening_high = max(prices[:opening_range_minutes])
opening_low = min(prices[:opening_range_minutes])
opening_range = opening_high - opening_low

# Breakout thresholds
breakout_threshold = opening_range * atr_multiplier

# Long signal
if price > opening_high + breakout_threshold:
    direction = "long"
    stop_loss = opening_low

# Short signal
if price < opening_low - breakout_threshold:
    direction = "short"
    stop_loss = opening_high
```

### Trading Sessions

| Session | Hours (UTC) | Best For |
|---------|-------------|----------|
| Asian | 00:00-08:00 | USD/JPY, Gold |
| London | 08:00-17:00 | EUR/USD, GBP/USD |
| NY Overlap | 13:00-17:00 | All majors |
| US | 14:30-21:00 | ES, NQ, Equities |

### Optimal Trading Hours

```yaml
trading_sessions:
  futures:
    es_mes:
      optimal_hours: ["09:30-11:30", "14:00-16:00"]  # ET
      avoid_hours: ["12:00-14:00"]  # Lunch lull
  commodities:
    cl_mcl:
      optimal_hours: ["09:00-14:30"]  # NYMEX pit overlap
    gc_mgc:
      optimal_hours: ["08:00-12:00", "13:30-17:00"]
```

### Configuration

```yaml
agents:
  session:
    enabled: true
    opening_range_minutes: 30
    breakout_threshold_atr: 0.5
    session_momentum_enabled: true
    symbols: ["ES", "NQ", "SPY", "QQQ"]
```

---

## 7. Index Spread Strategy

**File**: `strategies/index_spread_strategy.py`

**Style**: Relative Value / Spread Trading

**Description**: Trades the spread between related index futures (ES/NQ, MES/MNQ).

### Signal Logic

```python
# Calculate spread ratio
spread = log(price_MES) - hedge_ratio * log(price_MNQ)

# Z-score calculation
z_score = (spread - mean_spread) / std_spread

# Entry signals
if z_score > zscore_entry:
    # Short MES, Long MNQ
    direction = "short_spread"
elif z_score < -zscore_entry:
    # Long MES, Short MNQ
    direction = "long_spread"
```

### Configuration

```yaml
agents:
  index_spread:
    enabled: true
    spread_pairs:
      - ["MES", "MNQ"]
      - ["ES", "NQ"]
    zscore_entry: 2.0
    zscore_exit: 0.5
    lookback_periods: 20
```

---

## 8. TTM Squeeze Strategy

**File**: `strategies/ttm_squeeze_strategy.py`

**Style**: Volatility Breakout

**Description**: Identifies low volatility periods (squeeze) using Bollinger Bands inside Keltner Channels, then trades the breakout.

### Squeeze Detection

```python
# Bollinger Bands
bb_upper = sma + bb_mult * std
bb_lower = sma - bb_mult * std

# Keltner Channels
kc_upper = ema + kc_mult * atr
kc_lower = ema - kc_mult * atr

# Squeeze condition
squeeze_on = bb_lower > kc_lower and bb_upper < kc_upper
squeeze_off = not squeeze_on

# Momentum for direction
momentum = linreg_slope(close - midline)
```

### Signal Logic

- **Squeeze ON**: Low volatility, prepare for breakout
- **Squeeze OFF**: Volatility expanding, enter trade
- **Direction**: Based on momentum histogram

### Configuration

```yaml
agents:
  ttm_squeeze:
    enabled: true
    bb_length: 20
    bb_mult: 2.0
    kc_length: 20
    kc_mult: 1.5
    momentum_length: 12
    symbols: ["ES", "NQ", "SPY", "AAPL", "MSFT"]
```

---

## 9. Event-Driven Strategy

**File**: `strategies/event_driven_strategy.py`

**Style**: Event Trading

**Description**: Trades around major economic events (FOMC, NFP, CPI) based on volatility patterns and historical reactions.

### Supported Events

| Event | Pre-Event Window | Post-Event Window |
|-------|------------------|-------------------|
| FOMC | 24 hours | 4 hours |
| NFP | 12 hours | 2 hours |
| CPI | 12 hours | 2 hours |
| Earnings | 24 hours | 4 hours |

### Signal Logic

```python
# Pre-event: Position for expected volatility
if hours_to_event < pre_event_hours:
    if expected_move > historical_average * 1.5:
        direction = "long_vol"

# Post-event: Fade extreme moves
if hours_since_event < post_event_hours:
    if abs(move) > 2 * expected_move:
        direction = opposite(move_direction)
```

### Configuration

```yaml
agents:
  event_driven:
    enabled: true
    events: ["FOMC", "NFP", "CPI"]
    pre_event_hours: 24
    post_event_hours: 4
    min_volatility_increase: 1.5
```

---

## 10. Mean Reversion Strategy

**File**: `strategies/mean_reversion_strategy.py`

**Style**: Mean Reversion

**Description**: Identifies overextended price moves using RSI extremes, Bollinger Band touches, and z-scores.

### Signal Components

| Component | Long Signal | Short Signal |
|-----------|-------------|--------------|
| RSI | < 30 | > 70 |
| Bollinger Band | Touch lower band | Touch upper band |
| Z-Score | < -2.0 | > 2.0 |

### Signal Logic

```python
# Combine signals for confirmation
signals = []

if rsi < rsi_oversold:
    signals.append("long")
if price <= bb_lower:
    signals.append("long")
if z_score < -zscore_threshold:
    signals.append("long")

# Require multiple confirmations
if signals.count("long") >= 2:
    direction = "long"
```

### Configuration

```yaml
agents:
  mean_reversion:
    enabled: true
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
    bb_period: 20
    bb_std: 2.0
    zscore_threshold: 2.0
    lookback_periods: 50
    symbols: ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "META"]
```

---

## 11. Ichimoku Cloud Strategy

**File**: `strategies/ichimoku_strategy.py`

**Style**: Trend Following

**Description**: Implements the Ichimoku Kinko Hyo (One Glance Equilibrium Chart) indicator system.

### Ichimoku Components

| Component | Calculation | Purpose |
|-----------|-------------|---------|
| Tenkan-sen | (9-period high + low) / 2 | Conversion line (fast) |
| Kijun-sen | (26-period high + low) / 2 | Base line (slow) |
| Senkou Span A | (Tenkan + Kijun) / 2 | Leading cloud boundary |
| Senkou Span B | (52-period high + low) / 2 | Lagging cloud boundary |
| Chikou Span | Close shifted back 26 periods | Confirmation |

### Signal Logic

```python
# Strong bullish (5 conditions)
if (price > cloud_top and
    tenkan > kijun and
    chikou > price_26_ago and
    cloud_is_green and  # Span A > Span B
    price > kijun):
    direction = "strong_long"

# Strong bearish (5 conditions)
if (price < cloud_bottom and
    tenkan < kijun and
    chikou < price_26_ago and
    cloud_is_red and
    price < kijun):
    direction = "strong_short"

# Kumo breakout signals
if price crosses above cloud:
    direction = "long"
if price crosses below cloud:
    direction = "short"
```

---

## 12. Seasonality Strategy

**File**: `strategies/seasonality.py`

**Style**: Calendar Effects

**Description**: Trades based on historical seasonal patterns in commodities and indices.

### Seasonal Patterns

| Asset | Best Months | Pattern |
|-------|-------------|---------|
| Crude Oil | Feb-May | Pre-driving season demand |
| Natural Gas | Oct-Dec | Pre-winter heating demand |
| Gold | Jan-Feb, Aug | Jewelry/wedding demand |
| Equities | Nov-Apr | "Sell in May" effect |
| Soybeans | Sep-Nov | Harvest pressure |

### Turn-of-Month Effect

```python
def get_turn_of_month_multiplier(date):
    """
    Last 4 + first 3 trading days show higher returns
    due to institutional cash flows (pension rebalancing).
    """
    if date.day <= 3 or date.day >= last_day - 3:
        return 1.25  # 25% position size boost
    return 1.0
```

### Configuration

```yaml
seasonality:
  enabled: true
  min_win_rate: 0.55
  min_patterns: 1

position_sizing:
  use_turn_of_month_boost: true
  tom_multiplier: 1.25
  tom_asset_classes: ["equity", "future"]
```

---

## Strategy Risk Parameters

### Stop-Loss Configuration

```yaml
agents:
  exit_rules:
    enabled: true
    use_atr_stops: true
    stop_loss_atr_multiplier: 2.5
    default_stop_loss_pct: 5.0
    default_take_profit_pct: 10.0
    trailing_enabled: true
    trailing_activation_pct: 2.0
    trailing_distance_pct: 3.0
    time_exit_hours: 0  # 0 = disabled
```

### ATR Multiplier Scaling by Volatility

| Volatility Regime | Multiplier | Effect |
|-------------------|------------|--------|
| Low (ATR ratio < 0.7) | 0.8x | Tighter stops |
| Normal | 1.0x | Standard |
| High (ATR ratio > 1.5) | 1.3x | Wider stops |

### Asset-Specific ATR Overrides

```yaml
stop_loss_scaling:
  base_atr_multiplier: 2.5
  atr_overrides:
    MES: 1.5
    MNQ: 2.0
    CL: 2.5
    NG: 3.0   # Very volatile
    GC: 2.0
    EURUSD: 1.5
```

---

## Risk-Reward Requirements

All strategies must meet minimum risk-reward ratios:

```yaml
signals:
  min_risk_reward_ratio: 2.0
  rr_overrides:
    scalping: 1.5    # Higher win rate expected
    swing: 2.0       # Standard
    position: 3.0    # Longer holding period
```

---

## Strategy Weights in CIO

The CIO agent aggregates signals using configurable weights:

```yaml
agents:
  cio:
    signal_weight_macro: 0.10
    signal_weight_stat_arb: 0.12
    signal_weight_momentum: 0.15
    signal_weight_market_making: 0.08
    signal_weight_options_vol: 0.0  # Disabled
    signal_weight_sentiment: 0.13
    signal_weight_chart_analysis: 0.15
    signal_weight_forecasting: 0.15
    use_dynamic_weights: true
    performance_weight_factor: 0.3
    regime_weight_factor: 0.2
```

### Dynamic Weight Adjustment

Weights are adjusted based on:
- **Performance**: Higher Sharpe ratio increases weight
- **Regime**: Risk-on/off affects strategy allocations
- **Correlation**: Correlated signals are discounted

---

*Document Version: 2.0*
*Last Updated: February 2026*

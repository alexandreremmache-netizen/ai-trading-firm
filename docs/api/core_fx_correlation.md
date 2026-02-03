# fx_correlation

**Path**: `C:\Users\Alexa\ai-trading-firm\core\fx_correlation.py`

## Overview

FX Correlation Module
=====================

Addresses issues:
- #X14: FX market depth not utilized
- #X15: No FX correlation regime switching

Features:
- FX market depth analysis
- Correlation regime detection and switching
- Cross-pair correlation tracking
- Regime-based strategy adjustment

## Classes

### FXCorrelationRegime

**Inherits from**: str, Enum

FX correlation regime states.

### FXMarketDepth

FX market depth analysis (#X14).

Tracks order book depth for FX pairs.

#### Methods

##### `def mid_price(self)`

Calculate mid price.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### FXMarketDepthAnalyzer

Analyzes FX market depth (#X14).

Uses depth data for execution optimization and market insight.

#### Methods

##### `def __init__(self)`

##### `def update_depth(self, depth: FXMarketDepth) -> None`

Update depth for a pair.

##### `def get_current_depth(self, pair: str)`

Get most recent depth for pair.

##### `def estimate_market_impact(self, pair: str, size_millions: float, side: str)`

Estimate market impact for a trade.

Args:
    pair: FX pair
    size_millions: Trade size in millions
    side: 'buy' or 'sell'

Returns:
    Dict with impact estimates

##### `def get_depth_quality(self, pair: str) -> dict`

Assess depth quality for a pair.

### RegimeIndicators

Indicators used for regime detection.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### RegimeState

Current regime state with probabilities.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### FXCorrelationRegimeDetector

Detects FX correlation regime switches (#X15).

Monitors cross-pair correlations and market indicators
to identify regime changes.

#### Methods

##### `def __init__(self, correlation_window: int, regime_change_threshold: float)`

Initialize regime detector.

Args:
    correlation_window: Days for correlation calculation
    regime_change_threshold: Min probability change to trigger switch

##### `def update_return(self, pair: str, daily_return: float) -> None`

Update daily return for a pair.

##### `def update_indicators(self, vix: , dxy_change: ) -> None`

Update market indicators.

##### `def detect_regime(self) -> RegimeState`

Detect current correlation regime.

Returns:
    RegimeState with current regime and indicators

##### `def get_strategy_adjustments(self) -> dict`

Get strategy adjustments based on current regime.

Returns recommendations for position sizing, pair selection, etc.

##### `def get_regime_history(self, lookback_days: int) -> list[dict]`

Get regime history for analysis.

## Functions

### `def get_pip_multiplier(pair: str) -> int`

Get pip multiplier for FX pair.

For JPY pairs (USDJPY, EURJPY, etc.): 1 pip = 0.01, so multiplier = 100
For other pairs (EURUSD, GBPUSD, etc.): 1 pip = 0.0001, so multiplier = 10000

Args:
    pair: Currency pair (e.g., "USDJPY", "EURUSD")

Returns:
    Pip multiplier (100 for JPY pairs, 10000 for others)

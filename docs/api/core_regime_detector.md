# regime_detector

**Path**: `C:\Users\Alexa\ai-trading-firm\core\regime_detector.py`

## Overview

Regime Detection Module
=======================

Market regime detection for strategy switching (Issue #Q9).

Features:
- Volatility regime detection (low/normal/high/crisis)
- Trend regime identification (trending/mean-reverting/ranging)
- Correlation regime shifts
- Hidden Markov Model-based regime estimation
- VIX-based regime classification
- Multi-factor regime scoring

## Classes

### VolatilityRegime

**Inherits from**: str, Enum

Volatility regime classification.

### TrendRegime

**Inherits from**: str, Enum

Trend regime classification.

### MarketRegime

**Inherits from**: str, Enum

Overall market regime.

### CorrelationRegime

**Inherits from**: str, Enum

Correlation regime classification.

### RegimeState

Current regime state.

#### Methods

##### `def is_favorable_for_momentum(self) -> bool`

Check if regime favors momentum strategies.

##### `def is_favorable_for_mean_reversion(self) -> bool`

Check if regime favors mean reversion strategies.

##### `def is_risk_off(self) -> bool`

Check if market is in risk-off mode.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### RegimeTransition

Record of regime transition.

### RegimeDetector

Multi-factor regime detection engine.

Combines volatility, trend, and correlation signals
to identify the current market regime.

#### Methods

##### `def __init__(self, volatility_lookback: int, trend_lookback: int, correlation_lookback: int, regime_persistence: int)`

##### `def update_price(self, symbol: str, price: float, timestamp: datetime) -> None`

Update price history for a symbol.

##### `def update_vix(self, vix_value: float) -> None`

Update VIX history.

##### `def detect_regime(self) -> RegimeState`

Detect current market regime.

Returns updated RegimeState.

##### `def get_transitions(self, limit: int) -> list[RegimeTransition]`

Get recent regime transitions.

##### `def get_strategy_weights(self, strategies: list[str]) -> dict[str, float]`

Get recommended strategy weights based on current regime.

Args:
    strategies: List of strategy names

Returns:
    Dictionary of strategy -> weight (0-1)

### HMMRegimeModel

Hidden Markov Model for regime detection.

Simplified implementation without external dependencies.

#### Methods

##### `def __post_init__(self)`

##### `def forward_algorithm(self, observations: list[float]) -> list[list[float]]`

Run forward algorithm to get state probabilities.

Returns matrix of state probabilities for each time step.

##### `def predict_state(self, observations: list[float]) -> int`

Predict most likely current state.

##### `def get_state_probabilities(self, observations: list[float]) -> list[float]`

Get current state probabilities.

### RegimeAwareStrategyAllocator

Allocates capital to strategies based on regime.

Provides a unified interface for regime-based allocation.

#### Methods

##### `def __init__(self, detector: RegimeDetector, strategy_configs: )`

##### `def get_allocation(self, strategies: dict[str, str], total_capital: float, base_weights: ) -> dict[str, float]`

Get regime-adjusted capital allocation.

Args:
    strategies: Map of strategy name to type
    total_capital: Total capital to allocate
    base_weights: Optional base weights (before regime adjustment)

Returns:
    Dictionary of strategy -> allocated capital

# strategy_parameters

**Path**: `C:\Users\Alexa\ai-trading-firm\core\strategy_parameters.py`

## Overview

Strategy Parameters Module
==========================

Addresses issues:
- #Q20: Magic numbers in RSI calculation (70/30 thresholds)
- #Q21: No parameter sensitivity analysis

Features:
- Centralized strategy parameter definitions
- Configurable thresholds with validation
- Parameter sensitivity analysis framework
- Parameter optimization support

## Classes

### ParameterBounds

Defines valid bounds for a strategy parameter.

Eliminates magic numbers by providing named, validated parameters.

#### Methods

##### `def __post_init__(self) -> None`

Validate bounds configuration.

##### `def validate(self, value: float) -> bool`

Check if value is within bounds.

##### `def generate_range(self) -> list[float]`

Generate all valid values within bounds.

### RSIParameters

RSI indicator parameters with named constants (#Q20).

Replaces magic numbers 70/30 with configurable, documented thresholds.

#### Methods

##### `def __init__(self, overbought: , oversold: , period: )`

Initialize RSI parameters.

Args:
    overbought: Overbought threshold (default 70)
    oversold: Oversold threshold (default 30)
    period: Lookback period (default 14)

##### `def is_overbought(self, rsi_value: float) -> bool`

Check if RSI indicates overbought condition.

##### `def is_oversold(self, rsi_value: float) -> bool`

Check if RSI indicates oversold condition.

##### `def get_signal_strength(self, rsi_value: float) -> float`

Get signal strength from RSI value.

Returns:
    -1 to 1 where:
    - Negative = oversold (buy signal)
    - Positive = overbought (sell signal)
    - Near zero = neutral

##### `def for_trending_market(cls) -> RSIParameters`

Create parameters optimized for trending markets.

##### `def for_ranging_market(cls) -> RSIParameters`

Create parameters optimized for ranging/mean-reverting markets.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### MACDParameters

MACD indicator parameters.

#### Methods

##### `def __init__(self, fast_period: , slow_period: , signal_period: )`

##### `def to_dict(self) -> dict`

Convert to dictionary.

### BollingerBandsParameters

Bollinger Bands indicator parameters.

#### Methods

##### `def __init__(self, period: , std_dev: )`

##### `def to_dict(self) -> dict`

Convert to dictionary.

### SensitivityResult

Result of a single sensitivity test.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### SensitivityReport

Complete sensitivity analysis report.

#### Methods

##### `def get_most_sensitive_parameter(self, metric: str)`

Find parameter with highest sensitivity to given metric.

##### `def get_stability_score(self, parameter: str) -> float`

Get stability score for a parameter.

Higher score = more stable (less sensitive)
Range: 0-100

##### `def to_dict(self) -> dict`

Convert to dictionary.

### ParameterSensitivityAnalyzer

Parameter sensitivity analysis framework (#Q21).

Analyzes how strategy performance changes with parameter variations.
Helps identify:
- Overfitted parameters
- Stable vs unstable parameter regions
- Optimal parameter ranges

#### Methods

##### `def __init__(self, strategy_evaluator: Callable[, dict[str, float]])`

Initialize analyzer.

Args:
    strategy_evaluator: Function that takes parameter dict and returns
                       metrics dict (e.g., {"sharpe_ratio": 1.5, "return": 0.1})

##### `def analyze_parameter(self, baseline_params: dict[str, Any], parameter_name: str, test_values: list[float], metrics: ) -> list[SensitivityResult]`

Analyze sensitivity to a single parameter.

Args:
    baseline_params: Baseline parameter values
    parameter_name: Parameter to vary
    test_values: Values to test
    metrics: Which metrics to track (default: all)

Returns:
    List of sensitivity results

##### `def analyze_all_parameters(self, baseline_params: dict[str, Any], parameter_bounds: dict[str, ParameterBounds], num_samples: int, metrics: ) -> SensitivityReport`

Analyze sensitivity to all parameters.

Args:
    baseline_params: Baseline parameter values
    parameter_bounds: Bounds for each parameter
    num_samples: Number of values to test per parameter
    metrics: Which metrics to track

Returns:
    Complete sensitivity report

##### `def find_optimal_parameters(self, baseline_params: dict[str, Any], parameter_bounds: dict[str, ParameterBounds], objective_metric: str, num_iterations: int) -> dict[str, Any]`

Find optimal parameters using grid search.

Args:
    baseline_params: Starting parameters
    parameter_bounds: Search bounds
    objective_metric: Metric to optimize
    num_iterations: Max iterations

Returns:
    Optimal parameter set

### StrategyParameterSet

Complete parameter set for a trading strategy.

Centralizes all parameters with validation and defaults.

#### Methods

##### `def __init__(self, strategy_name: str, rsi: , macd: , bollinger: , custom_params: )`

##### `def to_dict(self) -> dict`

Convert all parameters to dictionary.

##### `def to_flat_dict(self) -> dict[str, Any]`

Convert to flat dictionary for optimization.

##### `def from_flat_dict(cls, strategy_name: str, params: dict[str, Any]) -> StrategyParameterSet`

Create from flat dictionary.

##### `def get_default_bounds(cls) -> dict[str, ParameterBounds]`

Get default parameter bounds for optimization.

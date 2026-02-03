# var_backtest

**Path**: `C:\Users\Alexa\ai-trading-firm\core\var_backtest.py`

## Overview

VaR Backtesting Module
======================

Addresses issue #R28: Historical VaR backtest missing.

Features:
- VaR model validation through backtesting
- Kupiec POF test for exception analysis
- Christoffersen independence test
- Traffic light framework (Basel)
- Backtesting reports and visualization data

## Classes

### VaRBacktestZone

**Inherits from**: str, Enum

Basel traffic light zones for VaR backtesting.

### VaRException

A single VaR exception (loss exceeded VaR).

#### Methods

##### `def exception_severity(self) -> float`

Severity as ratio of loss to VaR.

### KupiecTestResult

Result of Kupiec POF (Proportion of Failures) test.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### ChristoffersenTestResult

Result of Christoffersen independence test.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### VaRBacktestResult

Complete VaR backtest result.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### VaRBacktester

VaR model backtesting framework (#R28).

Validates VaR models using historical exceptions analysis
and statistical tests.

#### Methods

##### `def __init__(self, confidence_level: float, significance_level: float)`

Initialize backtester.

Args:
    confidence_level: VaR confidence level (e.g., 0.95 for 95%)
    significance_level: Statistical test significance level

##### `def add_observation(self, date: datetime, var_estimate: float, actual_return: float) -> None`

Add a single observation for backtesting.

Args:
    date: Observation date
    var_estimate: VaR estimate (as positive number)
    actual_return: Actual realized return (negative for loss)

##### `def add_observations_bulk(self, var_estimates: list[tuple[datetime, float]], actual_returns: list[tuple[datetime, float]]) -> None`

Add multiple observations at once.

##### `def run_backtest(self) -> VaRBacktestResult`

Run VaR backtest on accumulated observations.

Returns:
    VaRBacktestResult with comprehensive analysis

##### `def reset(self) -> None`

Reset accumulated observations.

### RollingVaRBacktester

Rolling window VaR backtester for ongoing model validation.

Maintains a rolling window of observations and periodically
runs backtests.

#### Methods

##### `def __init__(self, window_size: int, confidence_level: float, backtest_frequency: int)`

Initialize rolling backtester.

Args:
    window_size: Number of observations in rolling window
    confidence_level: VaR confidence level
    backtest_frequency: Days between automatic backtests

##### `def add_observation(self, date: datetime, var_estimate: float, actual_return: float)`

Add observation and potentially trigger backtest.

Returns:
    VaRBacktestResult if backtest was triggered, None otherwise

##### `def force_backtest(self)`

Force immediate backtest if enough data.

##### `def get_last_result(self)`

Get most recent backtest result.

##### `def get_exception_count(self, lookback_days: ) -> int`

Get exception count for recent period.

# correlation_manager

**Path**: `C:\Users\Alexa\ai-trading-firm\core\correlation_manager.py`

## Overview

Correlation Manager
===================

Manages correlation analysis for portfolio risk management.
Provides rolling correlation matrices, concentration metrics,
and regime change detection.

## Classes

### CorrelationRegime

**Inherits from**: Enum

Market correlation regime classification.

### StressIndicator

**Inherits from**: Enum

Market stress indicator type (#R7).

### CorrelationAlert

Alert for correlation regime change or threshold breach.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary for logging.

### CorrelationSnapshot

Point-in-time correlation analysis.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary for serialization.

### CorrelationManager

Manages correlation analysis for the trading system.

Features:
- Rolling correlation matrix calculation
- Herfindahl index (concentration) computation
- Effective diversification count (effective N)
- Correlation regime detection
- Alert generation on regime changes

#### Methods

##### `def __init__(self, config: )`

Initialize correlation manager.

Args:
    config: Configuration dictionary with:
        - lookback_days: Rolling window size (default: 60)
        - max_pairwise_correlation: Alert threshold (default: 0.85)
        - min_history_days: Minimum data for calculation (default: 20)
        - regime_change_threshold: Sensitivity for regime detection (default: 0.15)

##### `def add_returns(self, symbol: str, timestamp: datetime, returns: float) -> None`

Add returns observation for a symbol.

Args:
    symbol: Instrument symbol
    timestamp: Observation timestamp
    returns: Period returns (e.g., daily returns)

##### `def add_returns_batch(self, returns_df: pd.DataFrame, timestamp: ) -> None`

Add batch of returns from a DataFrame.

Args:
    returns_df: DataFrame with symbols as columns and returns as values
    timestamp: Optional timestamp (uses current time if not provided)

##### `def calculate_correlation_matrix(self, symbols: , lookback_days: )`

Calculate rolling correlation matrix.

Args:
    symbols: List of symbols to include (default: all available)
    lookback_days: Rolling window (default: configured lookback)

Returns:
    Correlation matrix as DataFrame, or None if insufficient data

##### `def calculate_snapshot(self, symbols: , weights: )`

Calculate complete correlation snapshot.

Args:
    symbols: Symbols to analyze (default: all)
    weights: Portfolio weights by symbol (for weighted calculations)

Returns:
    CorrelationSnapshot with all metrics, or None if insufficient data

##### `def get_pair_correlation(self, symbol1: str, symbol2: str)`

Get correlation between two specific symbols.

Args:
    symbol1: First symbol
    symbol2: Second symbol

Returns:
    Correlation coefficient, or None if not available

##### `def get_highly_correlated_pairs(self, threshold: ) -> list[tuple[str, str, float]]`

Get list of highly correlated pairs.

Args:
    threshold: Correlation threshold (default: configured max)

Returns:
    List of (symbol1, symbol2, correlation) tuples

##### `def get_diversification_benefit(self, weights: dict[str, float], volatilities: dict[str, float])`

Calculate diversification benefit.

Diversification benefit = 1 - (portfolio_vol / weighted_avg_vol)

Args:
    weights: Portfolio weights
    volatilities: Symbol volatilities

Returns:
    Diversification benefit (0 = no benefit, 1 = maximum benefit)

##### `def get_recent_alerts(self, hours: int) -> list[CorrelationAlert]`

Get recent correlation alerts.

Args:
    hours: Lookback period in hours

Returns:
    List of recent alerts

##### `def get_current_regime(self)`

Get current correlation regime.

##### `def get_status(self) -> dict[str, Any]`

Get manager status for monitoring.

##### `def update_vix(self, vix_value: float) -> None`

Update VIX value for stress detection (#R7).

Args:
    vix_value: Current VIX level

##### `def update_baseline_volatility(self, symbol: str, baseline_vol: float) -> None`

Update baseline volatility for stress detection (#R7).

Args:
    symbol: Instrument symbol
    baseline_vol: Long-term average volatility

##### `def check_volatility_stress(self, symbol: str, current_vol: float) -> bool`

Check if current volatility indicates stress (#R7).

Args:
    symbol: Instrument symbol
    current_vol: Current volatility

Returns:
    True if volatility indicates stress

##### `def signal_external_stress(self, stress_on: bool, reason: str) -> None`

Signal external stress condition (#R7).

Called by risk agent or external monitoring systems.

Args:
    stress_on: Whether stress mode should be active
    reason: Reason for stress signal

##### `def get_stress_adjusted_correlation_matrix(self, symbols: )`

Get correlation matrix adjusted for current stress conditions (#R7).

If in stress mode, returns stress-adjusted correlations with faster decay.
Otherwise, returns the normal correlation matrix.

Args:
    symbols: List of symbols to include

Returns:
    Correlation matrix appropriate for current market conditions

##### `def get_stressed_correlation_matrix(self, stress_multiplier: float, correlation_floor: float)`

Generate stressed correlation matrix for scenario analysis (#R7).

Simulates crisis conditions where correlations increase.

Args:
    stress_multiplier: Factor to increase correlations
    correlation_floor: Minimum correlation in stress scenario

Returns:
    Stressed correlation matrix

##### `def is_in_stress_mode(self) -> bool`

Check if currently in stress mode (#R7).

##### `def get_stress_duration_hours(self)`

Get duration of current stress period in hours (#R7).

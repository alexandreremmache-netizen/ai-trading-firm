# signal_decay

**Path**: `C:\Users\Alexa\ai-trading-firm\core\signal_decay.py`

## Overview

Signal Decay/Half-Life Module
=============================

Models signal strength decay over time (Issue #Q18).

Features:
- Exponential decay modeling
- Half-life calculation and calibration
- Signal freshness scoring
- Multi-timeframe decay
- Historical decay analysis

## Classes

### DecayModel

**Inherits from**: str, Enum

Signal decay model types.

### SignalDecayConfig

Configuration for signal decay.

### DecayedSignal

Signal with decay tracking.

#### Methods

##### `def age_minutes(self) -> float`

Get signal age in minutes.

##### `def freshness_score(self) -> float`

Get freshness score (0-1, higher is fresher).

##### `def is_expired(self) -> bool`

Check if signal has expired.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### SignalDecayCalculator

Calculates signal decay based on various models (#Q18).

Supports multiple decay models for different signal types.

#### Methods

##### `def __init__(self, default_config: )`

##### `def calculate_decay(self, original_strength: float, age_minutes: float, config: ) -> float`

Calculate decayed signal strength.

Args:
    original_strength: Initial signal strength (0-1)
    age_minutes: Signal age in minutes
    config: Decay configuration

Returns:
    Decayed strength (0-1)

##### `def estimate_half_life(self, signal_history: list[tuple[float, float]]) -> float`

Estimate half-life from historical signal performance.

Uses regression to fit exponential decay curve.

Args:
    signal_history: List of (age, observed_strength) pairs

Returns:
    Estimated half-life in minutes

### SignalDecayManager

Manages signal decay for multiple signals (#Q18).

Tracks active signals and applies decay over time.

#### Methods

##### `def __init__(self, calculator: , cleanup_interval_minutes: float)`

##### `def add_signal(self, symbol: str, direction: str, strength: float, source_strategy: str, config: , metadata: ) -> DecayedSignal`

Add a new signal to track.

Args:
    symbol: Trading symbol
    direction: 'LONG', 'SHORT', or 'NEUTRAL'
    strength: Signal strength (0-1)
    source_strategy: Strategy that generated the signal
    config: Optional decay configuration
    metadata: Additional signal metadata

Returns:
    The created DecayedSignal

##### `def update_signal(self, signal_id: str, new_strength: , refresh: bool)`

Update an existing signal.

Args:
    signal_id: Signal to update
    new_strength: Optional new strength value
    refresh: If True, reset decay timer

Returns:
    Updated signal or None if not found

##### `def get_signal(self, signal_id: str)`

Get a specific signal by ID.

##### `def get_signals_for_symbol(self, symbol: str, include_expired: bool) -> list[DecayedSignal]`

Get all signals for a symbol.

##### `def get_aggregate_signal(self, symbol: str) -> dict`

Get aggregated signal for a symbol.

Combines multiple signals with decay-weighted averaging.

##### `def cleanup_expired(self) -> int`

Remove expired signals.

##### `def record_signal_performance(self, strategy: str, age_at_action: float, effectiveness: float) -> None`

Record signal performance for half-life calibration.

Args:
    strategy: Strategy that generated the signal
    age_at_action: Age of signal when action was taken (minutes)
    effectiveness: How effective the signal was (0-1)

##### `def calibrate_half_life(self, strategy: str) -> float`

Calibrate half-life for a strategy based on historical performance.

##### `def get_statistics(self) -> dict`

Get signal manager statistics.

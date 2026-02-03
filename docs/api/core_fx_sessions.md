# fx_sessions

**Path**: `C:\Users\Alexa\ai-trading-firm\core\fx_sessions.py`

## Overview

FX Session Awareness Module
===========================

Forex trading session awareness and analytics (Issue #X9).

Features:
- Major FX session tracking (Sydney, Tokyo, London, New York)
- Session overlap detection
- Liquidity estimation by session
- Volatility patterns by session
- Best execution timing recommendations

## Classes

### FXSession

**Inherits from**: str, Enum

Major FX trading sessions.

### SessionOverlap

**Inherits from**: str, Enum

Session overlap periods.

### SessionTime

Trading session times.

#### Methods

##### `def is_open(self, utc_time: time) -> bool`

Check if session is open at given UTC time.

##### `def is_peak(self, utc_time: time) -> bool`

Check if in peak hours.

### SessionCharacteristics

Trading characteristics for a session.

### FXSessionState

Current FX session state.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### FXSessionManager

Manages FX session awareness and provides trading recommendations.

Tracks which sessions are active and their characteristics.

#### Methods

##### `def __init__(self)`

##### `def get_current_state(self, utc_now: ) -> FXSessionState`

Get current FX session state.

Args:
    utc_now: Current UTC time (defaults to now)

Returns:
    FXSessionState with active sessions and recommendations

##### `def get_best_execution_window(self, pair: str) -> dict`

Get best execution window for a currency pair.

Returns recommended times and expected conditions.

##### `def record_volatility(self, session: FXSession, volatility: float) -> None`

Record observed volatility for session.

##### `def record_spread(self, session: FXSession, spread: float) -> None`

Record observed spread for session.

##### `def get_session_stats(self) -> dict`

Get session statistics from recorded data.

##### `def is_trading_recommended(self, pair: str, utc_now: ) -> tuple[bool, str]`

Check if trading is recommended for a pair at current time.

Returns (recommended, reason).

### SessionVolatilityPattern

Volatility pattern by hour for a session.

## Functions

### `def get_expected_volatility_multiplier(utc_now: ) -> float`

Get expected volatility multiplier for current time.

## Constants

- `SESSION_TIMES`
- `SESSION_CHARACTERISTICS`
- `VOLATILITY_PATTERNS`

# options_backtest

**Path**: `C:\Users\Alexa\ai-trading-firm\core\options_backtest.py`

## Overview

Options Strategy Backtesting Module
===================================

Addresses issue #O18: No option strategy backtesting.

Features:
- Options strategy backtesting framework
- Multi-leg strategy support
- Greeks evolution tracking
- Volatility surface modeling
- Strategy P&L decomposition

## Classes

### OptionType

**Inherits from**: str, Enum

Option type.

### OptionStrategy

**Inherits from**: str, Enum

Common option strategies.

### OptionLeg

Single leg of an options position.

#### Methods

##### `def is_long(self) -> bool`

Check if leg is long.

##### `def notional(self) -> float`

Calculate notional value.

##### `def days_to_expiry(self, as_of: datetime) -> int`

Calculate days to expiry.

### OptionPosition

Multi-leg options position.

#### Methods

##### `def max_profit(self)`

Calculate maximum profit (if defined).

##### `def max_loss(self)`

Calculate maximum loss (if defined).

##### `def net_premium(self) -> float`

Net premium paid/received.

##### `def is_expired(self, as_of: datetime) -> bool`

Check if all legs expired.

### OptionGreeks

Greeks for a position.

### OptionBacktestBar

Single bar of options backtest data.

### OptionBacktestResult

Result of options strategy backtest.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### BlackScholes

Black-Scholes option pricing for backtesting.

#### Methods

##### `def price(spot: float, strike: float, time_to_expiry: float, volatility: float, rate: float, is_call: bool, dividend_yield: float) -> float`

Calculate option price using Black-Scholes.

##### `def delta(spot: float, strike: float, time_to_expiry: float, volatility: float, rate: float, is_call: bool, dividend_yield: float) -> float`

Calculate delta.

##### `def gamma(spot: float, strike: float, time_to_expiry: float, volatility: float, rate: float, dividend_yield: float) -> float`

Calculate gamma.

##### `def theta(spot: float, strike: float, time_to_expiry: float, volatility: float, rate: float, is_call: bool, dividend_yield: float) -> float`

Calculate theta (per day).

##### `def vega(spot: float, strike: float, time_to_expiry: float, volatility: float, rate: float, dividend_yield: float) -> float`

Calculate vega (per 1% vol move).

### OptionStrategyBacktester

Options strategy backtesting engine (#O18).

Supports multi-leg strategies with Greeks tracking.

#### Methods

##### `def __init__(self, initial_capital: float, risk_free_rate: float, dividend_yield: float)`

Initialize backtester.

Args:
    initial_capital: Starting capital
    risk_free_rate: Risk-free rate for pricing
    dividend_yield: Dividend yield for underlying

##### `def add_position(self, position: OptionPosition, bar: OptionBacktestBar) -> bool`

Add new position to portfolio.

##### `def close_position(self, position: OptionPosition, bar: OptionBacktestBar, reason: str) -> float`

Close a position and return P&L.

##### `def run_backtest(self, data: list[OptionBacktestBar], entry_signal: Callable[, ], exit_signal: Callable[, bool]) -> OptionBacktestResult`

Run options strategy backtest.

Args:
    data: Historical bar data
    entry_signal: Function returning position to enter or None
    exit_signal: Function returning True if should exit

Returns:
    OptionBacktestResult with metrics

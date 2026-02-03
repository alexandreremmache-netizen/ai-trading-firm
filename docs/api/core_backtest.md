# backtest

**Path**: `C:\Users\Alexa\ai-trading-firm\core\backtest.py`

## Overview

Backtesting Framework
=====================

Comprehensive backtesting engine for strategy validation (Issue #Q7).

Features:
- Historical data simulation
- Multiple strategy support
- Transaction cost modeling
- Slippage simulation
- Performance metrics calculation
- Walk-forward analysis support
- Multi-asset portfolio backtesting

## Classes

### BacktestMode

**Inherits from**: str, Enum

Backtest execution mode.

### FillModel

**Inherits from**: str, Enum

Order fill simulation model.

### Bar

Single OHLCV bar.

#### Methods

##### `def __post_init__(self)`

### BacktestOrder

Order in backtest.

### BacktestFill

Fill result in backtest.

### BacktestPosition

Position tracking in backtest.

#### Methods

##### `def update(self, fill: BacktestFill, current_price: float) -> float`

Update position from fill, return realized P&L.

### TransactionCostModel

Transaction cost model for realistic backtesting (Issue #Q10).

#### Methods

##### `def calculate_commission(self, quantity: int, price: float) -> float`

Calculate commission for trade.

##### `def calculate_slippage(self, quantity: int, price: float, volatility: float, adv: ) -> float`

Calculate expected slippage.

### BacktestMetrics

Comprehensive backtest performance metrics.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### BacktestStrategy

**Inherits from**: ABC

Abstract base class for backtestable strategies.

#### Methods

##### `def __init__(self, strategy_id: str, params: )`

##### `def on_bar(self, bar: Bar, position: , portfolio_value: float)`

Generate signal on new bar.

Returns order if action needed, None otherwise.

##### `def on_fill(self, fill: BacktestFill) -> None`

Called when order is filled.

##### `def on_start(self, start_date: datetime, initial_capital: float) -> None`

Called at backtest start.

##### `def on_end(self, end_date: datetime, final_value: float) -> None`

Called at backtest end.

### BacktestEngine

Main backtesting engine.

Supports:
- Multiple strategies
- Realistic execution simulation
- Transaction costs
- Performance analytics

#### Methods

##### `def __init__(self, initial_capital: float, fill_model: FillModel, cost_model: , risk_free_rate: float)`

##### `def add_strategy(self, strategy: BacktestStrategy) -> None`

Add strategy to backtest.

##### `def run(self, data: dict[str, list[Bar]], start_date: , end_date: ) -> BacktestMetrics`

Run backtest on historical data.

Args:
    data: Dictionary mapping symbol to list of bars
    start_date: Optional start date filter
    end_date: Optional end date filter

Returns:
    BacktestMetrics with performance results

### WalkForwardAnalyzer

Walk-forward analysis for strategy validation (Issue #Q8 partial).

Helps detect overfitting by validating on out-of-sample data.

#### Methods

##### `def __init__(self, train_period_days: int, test_period_days: int, step_days: int)`

##### `def generate_windows(self, start_date: datetime, end_date: datetime) -> Iterator[tuple[datetime, datetime, datetime, datetime]]`

Generate train/test windows.

Yields: (train_start, train_end, test_start, test_end)

##### `def run(self, engine_factory: Callable[, BacktestEngine], strategy_factory: Callable[, BacktestStrategy], data: dict[str, list[Bar]], param_optimizer: ) -> list[dict]`

Run walk-forward analysis.

Args:
    engine_factory: Function that creates new BacktestEngine
    strategy_factory: Function that creates strategy with params
    data: Historical data
    param_optimizer: Optional function to optimize params on train set

Returns:
    List of results per window

### BacktestComparison

Compare multiple backtest results.

#### Methods

##### `def add_result(self, name: str, metrics: BacktestMetrics) -> None`

Add backtest result.

##### `def get_comparison_table(self) -> list[dict]`

Get comparison as table rows.

##### `def get_best_by_sharpe(self)`

Get strategy name with best Sharpe ratio.

##### `def get_best_by_return(self)`

Get strategy name with best total return.

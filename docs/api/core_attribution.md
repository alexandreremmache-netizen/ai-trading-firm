# attribution

**Path**: `C:\Users\Alexa\ai-trading-firm\core\attribution.py`

## Overview

Performance Attribution
=======================

Tracks and attributes P&L to strategies, enabling performance analysis
and dynamic weight adjustment. Essential for institutional-grade
portfolio management and compliance.

## Classes

### TradeOutcome

**Inherits from**: Enum

Trade outcome classification.

### TradeRecord

Record of a single trade for attribution.

#### Methods

##### `def gross_pnl(self) -> float`

Gross P&L before costs.

##### `def net_pnl(self) -> float`

Net P&L after costs.

##### `def outcome(self) -> TradeOutcome`

Classify trade outcome.

##### `def holding_period_hours(self)`

Calculate holding period in hours.

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### StrategyMetrics

Aggregated metrics for a strategy.

#### Methods

##### `def annualization_factor(self) -> float`

Get annualization factor based on data frequency.

##### `def win_rate(self) -> float`

Calculate win rate.

##### `def profit_factor(self) -> float`

Calculate profit factor (gross wins / gross losses).

##### `def sharpe_ratio(self) -> float`

Calculate annualized Sharpe ratio using excess returns.

Uses proper annualization based on data frequency:
- daily: sqrt(252)
- hourly: sqrt(252 * 6.5)
- minute: sqrt(252 * 6.5 * 60)
etc.

##### `def sortino_ratio(self) -> float`

Calculate Sortino ratio using excess returns and downside deviation.

Uses proper annualization based on data frequency.

##### `def max_drawdown(self) -> float`

Calculate maximum drawdown.

##### `def expectancy(self) -> float`

Calculate trade expectancy (expected P&L per trade).

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### PerformanceAttribution

Comprehensive performance attribution system.

Features:
- Trade-to-strategy mapping
- P&L attribution by strategy
- Risk-adjusted metrics (Sharpe, Sortino)
- Win rate and profit factor tracking
- Rolling performance windows

#### Methods

##### `def __init__(self, config: )`

Initialize attribution system.

Args:
    config: Configuration with:
        - rolling_window_days: Window for rolling metrics (default: 30)
        - risk_free_rate: Annual risk-free rate (default: 0.05)

##### `def record_trade_entry(self, strategy: str, symbol: str, side: str, quantity: int, entry_price: float, commission: float, tags: ) -> str`

Record a new trade entry.

Args:
    strategy: Strategy name
    symbol: Instrument symbol
    side: "buy" or "sell"
    quantity: Number of units
    entry_price: Entry price
    commission: Entry commission
    tags: Optional tags for categorization

Returns:
    Trade ID for future reference

##### `def record_trade_exit(self, trade_id: str, exit_price: float, commission: float, slippage: float)`

Record trade exit and calculate P&L.

Args:
    trade_id: Trade ID from entry
    exit_price: Exit price
    commission: Exit commission
    slippage: Execution slippage

Returns:
    Closed TradeRecord, or None if not found

##### `def get_strategy_metrics(self, strategy: str)`

Get metrics for a specific strategy.

##### `def get_all_strategy_metrics(self) -> dict[str, StrategyMetrics]`

Get metrics for all strategies.

##### `def get_strategy_pnl(self, strategy: str) -> float`

Get total P&L for a strategy.

##### `def get_strategy_sharpe(self, strategy: str) -> float`

Get Sharpe ratio for a strategy.

##### `def get_strategy_win_rate(self, strategy: str) -> float`

Get win rate for a strategy.

##### `def record_portfolio_value(self, timestamp: datetime, value: float) -> None`

P1-15: Record portfolio NAV for TWR calculation.

Should be called daily (or at each valuation point).

##### `def record_cash_flow(self, timestamp: datetime, amount: float) -> None`

P1-15: Record external cash flow for MWR calculation.

Args:
    timestamp: When the cash flow occurred
    amount: Positive for deposits, negative for withdrawals

##### `def calculate_twr(self, start_date: ) -> float`

P1-15: Calculate Time-Weighted Return (TWR).

TWR eliminates the effect of cash flows, showing pure investment
performance. Used for comparing manager skill regardless of
deposit/withdrawal timing.

Formula: TWR = ((1 + r1) * (1 + r2) * ... * (1 + rn)) - 1
where rn is the sub-period return between cash flows.

Returns:
    Annualized TWR as decimal (e.g., 0.15 = 15%)

##### `def calculate_mwr(self, start_date: ) -> float`

P1-15: Calculate Money-Weighted Return (MWR / IRR).

MWR reflects actual investor experience including the timing
of deposits and withdrawals. Higher weight given to returns
when more capital was invested.

Uses Newton-Raphson iteration to solve for IRR.

Returns:
    Annualized MWR as decimal (e.g., 0.12 = 12%)

##### `def get_return_comparison(self) -> dict`

P1-15: Get TWR vs MWR comparison.

A large difference between TWR and MWR indicates poor timing
of deposits/withdrawals relative to market performance.

Returns:
    Dictionary with TWR, MWR, and difference analysis

##### `def get_rolling_metrics(self, strategy: str, days: )`

Calculate metrics for a rolling window.

Args:
    strategy: Strategy name
    days: Window size (default: configured rolling_window_days)

Returns:
    StrategyMetrics for the window, or None if insufficient data

##### `def get_pnl_attribution(self) -> dict[str, float]`

Get P&L attribution by strategy.

Returns:
    Dictionary mapping strategy to total P&L

##### `def get_pnl_contribution(self) -> dict[str, float]`

Get P&L contribution percentages by strategy.

Returns:
    Dictionary mapping strategy to contribution percentage

##### `def get_recommended_weights(self, method: str) -> dict[str, float]`

Calculate recommended strategy weights based on performance.

Args:
    method: Weighting method - "sharpe", "win_rate", "profit_factor", "equal"

Returns:
    Normalized weights by strategy

##### `def get_symbol_attribution(self) -> dict[str, float]`

Get P&L attribution by symbol.

##### `def get_open_trades(self, strategy: ) -> list[TradeRecord]`

Get open trades.

Args:
    strategy: Filter by strategy (optional)

Returns:
    List of open trades

##### `def get_trade_history(self, strategy: , symbol: , limit: int) -> list[TradeRecord]`

Get trade history.

Args:
    strategy: Filter by strategy (optional)
    symbol: Filter by symbol (optional)
    limit: Maximum trades to return

Returns:
    List of trade records

##### `def get_daily_pnl_series(self, days: int) -> list[tuple[datetime, float]]`

Get daily P&L time series.

##### `def get_portfolio_summary(self) -> dict[str, Any]`

Get portfolio-level summary.

##### `def get_status(self) -> dict[str, Any]`

Get attribution system status for monitoring.

##### `def export_to_dataframe(self)`

Export trades to pandas DataFrame.

### ExposureLimit

Limit definition for a sector or factor (#P6).

### SectorFactorExposureManager

Manages sector and factor exposure constraints (#P6).

Tracks and enforces limits on:
- Sector exposures (Technology, Healthcare, etc.)
- Factor exposures (Value, Momentum, Size, etc.)
- Geographic exposures

#### Methods

##### `def __init__(self, portfolio_value: float)`

##### `def set_sector(self, symbol: str, sector: str) -> None`

Set sector classification for a symbol.

##### `def set_limit(self, limit: ExposureLimit, limit_type: str) -> None`

Set an exposure limit.

##### `def update_portfolio_value(self, value: float) -> None`

Update portfolio value for percentage calculations.

##### `def calculate_sector_exposures(self, positions: dict[str, float]) -> dict[str, dict]`

Calculate current sector exposures (#P6).

Args:
    positions: Map of symbols to notional positions

Returns:
    Exposure by sector

##### `def check_exposure_limits(self, positions: dict[str, float]) -> list[dict]`

Check all exposure limits and return violations (#P6).

Args:
    positions: Map of symbols to notional positions

Returns:
    List of limit violations

##### `def get_exposure_summary(self) -> dict`

Get summary of all exposures for monitoring.

### CashManager

Manages portfolio cash and liquidity (#P7).

Handles:
- Cash balance tracking
- Minimum cash reserves
- Cash sweep logic
- T+2 settlement tracking

#### Methods

##### `def __init__(self, initial_cash: float, min_cash_reserve_pct: float, target_cash_pct: float)`

##### `def update_cash(self, amount: float, reason: str) -> float`

Update cash balance (#P7).

Args:
    amount: Cash change (positive = inflow, negative = outflow)
    reason: Reason for cash change

Returns:
    New cash balance

##### `def get_available_cash(self, portfolio_value: float) -> float`

Get cash available for trading (#P7).

Subtracts minimum reserve and pending settlements.

Args:
    portfolio_value: Total portfolio value

Returns:
    Available cash for new positions

##### `def add_pending_settlement(self, amount: float, settlement_date: datetime, trade_id: str) -> None`

Add pending settlement (#P7).

Args:
    amount: Settlement amount (positive = receive, negative = pay)
    settlement_date: Expected settlement date
    trade_id: Associated trade ID

##### `def process_settlements(self) -> list[dict]`

Process due settlements (#P7).

Returns:
    List of processed settlements

##### `def calculate_cash_sweep(self, portfolio_value: float) -> dict`

Calculate cash sweep to maintain target allocation (#P7).

Args:
    portfolio_value: Total portfolio value

Returns:
    Sweep recommendation

##### `def get_cash_status(self) -> dict`

Get cash management status.

### DividendRecord

Record of a dividend event (#P8).

#### Methods

##### `def __post_init__(self)`

### DividendManager

Manages dividend tracking and processing (#P8).

Handles:
- Ex-date tracking
- Dividend accrual
- Tax withholding
- DRIP (dividend reinvestment)

#### Methods

##### `def __init__(self, enable_drip: bool, default_tax_rate: float)`

##### `def add_upcoming_dividend(self, symbol: str, ex_date: datetime, record_date: datetime, pay_date: datetime, amount_per_share: float, shares_held: int, dividend_type: str) -> DividendRecord`

Register an upcoming dividend (#P8).

Args:
    symbol: Stock symbol
    ex_date: Ex-dividend date
    record_date: Record date
    pay_date: Payment date
    amount_per_share: Dividend per share
    shares_held: Number of shares held
    dividend_type: Type of dividend

Returns:
    Dividend record

##### `def process_ex_dates(self, current_positions: dict[str, int], as_of: ) -> list[DividendRecord]`

Process dividends going ex (#P8).

Args:
    current_positions: Current share positions
    as_of: Processing date (default: now)

Returns:
    Dividends going ex

##### `def process_payments(self, as_of: ) -> list[dict]`

Process dividend payments (#P8).

Args:
    as_of: Processing date (default: now)

Returns:
    Processed payments

##### `def get_dividend_forecast(self, days: int) -> dict`

Get forecast of upcoming dividends.

##### `def get_ytd_dividends(self) -> dict`

Get year-to-date dividend summary.

### CorporateActionType

**Inherits from**: Enum

Types of corporate actions (#P9).

### CorporateAction

Corporate action record (#P9).

### CorporateActionProcessor

Processes corporate actions (#P9).

Handles:
- Stock splits and reverse splits
- Mergers and acquisitions
- Spin-offs
- Symbol changes

#### Methods

##### `def __init__(self)`

##### `def add_corporate_action(self, action: CorporateAction) -> None`

Add a corporate action to process.

##### `def process_split(self, action: CorporateAction, current_shares: int, cost_basis: float) -> dict`

Process stock split (#P9).

Args:
    action: Split action
    current_shares: Current share count
    cost_basis: Current cost basis

Returns:
    Adjusted position details

##### `def process_spinoff(self, action: CorporateAction, parent_shares: int, parent_cost_basis: float) -> dict`

Process spin-off (#P9).

Args:
    action: Spin-off action
    parent_shares: Parent company shares held
    parent_cost_basis: Parent cost basis

Returns:
    New position and adjusted basis

##### `def process_merger(self, action: CorporateAction, target_shares: int, target_cost_basis: float) -> dict`

Process merger/acquisition (#P9).

Args:
    action: Merger action
    target_shares: Target company shares held
    target_cost_basis: Target cost basis

Returns:
    Conversion details

##### `def process_pending_actions(self, positions: dict[str, tuple[int, float]], as_of: ) -> list[dict]`

Process all pending corporate actions (#P9).

Args:
    positions: Current positions
    as_of: Processing date

Returns:
    List of processed action results

##### `def get_pending_actions(self) -> list[dict]`

Get list of pending corporate actions.

### TaxLot

Individual tax lot for a position (#P10).

#### Methods

##### `def __post_init__(self)`

##### `def is_long_term(self) -> bool`

Check if lot qualifies for long-term capital gains (>1 year).

##### `def adjusted_cost_basis(self) -> float`

Get cost basis adjusted for wash sale disallowance.

### TaxLotManager

Manages tax lots for cost basis tracking (#P10).

Supports:
- FIFO (First In First Out)
- LIFO (Last In First Out)
- Specific identification
- Average cost
- Highest cost
- Lowest cost

#### Methods

##### `def __init__(self, default_method: str)`

##### `def add_lot(self, symbol: str, purchase_date: datetime, quantity: int, cost_per_share: float) -> TaxLot`

Add a new tax lot (#P10).

Args:
    symbol: Stock symbol
    purchase_date: Purchase date
    quantity: Number of shares
    cost_per_share: Cost per share

Returns:
    Created tax lot

##### `def select_lots_for_sale(self, symbol: str, quantity: int, method: ) -> list[tuple[TaxLot, int]]`

Select lots for a sale using specified method (#P10).

Args:
    symbol: Stock symbol
    quantity: Shares to sell
    method: Selection method (fifo, lifo, hifo, lofo, specific)

Returns:
    List of (lot, shares_to_sell) tuples

##### `def execute_sale(self, symbol: str, quantity: int, sale_price: float, sale_date: datetime, method: ) -> dict`

Execute a sale and calculate gain/loss (#P10).

Args:
    symbol: Stock symbol
    quantity: Shares to sell
    sale_price: Sale price per share
    sale_date: Sale date
    method: Lot selection method

Returns:
    Sale details with gain/loss

##### `def get_lots_summary(self, symbol: str) -> dict`

Get summary of lots for a symbol.

### BrinsonAttributor

Brinson performance attribution model (#P11).

Decomposes portfolio return into:
- Allocation effect (sector weight decisions)
- Selection effect (security selection within sectors)
- Interaction effect (combined effect)

#### Methods

##### `def __init__(self)`

##### `def calculate_attribution(self, portfolio_weights: dict[str, float], portfolio_returns: dict[str, float], benchmark_weights: dict[str, float], benchmark_returns: dict[str, float]) -> dict`

Calculate Brinson attribution (#P11).

Args:
    portfolio_weights: Portfolio sector weights
    portfolio_returns: Portfolio sector returns
    benchmark_weights: Benchmark sector weights
    benchmark_returns: Benchmark sector returns

Returns:
    Attribution breakdown

##### `def get_cumulative_attribution(self, periods: int) -> dict`

Get cumulative attribution over multiple periods.

### BenchmarkData

Benchmark data point (#P12).

### BenchmarkTracker

Tracks portfolio performance against benchmarks (#P12).

Supports multiple benchmarks and calculates:
- Tracking error
- Information ratio
- Active return
- Beta and alpha

#### Methods

##### `def __init__(self)`

##### `def add_benchmark(self, name: str) -> None`

Add a benchmark to track.

##### `def set_active_benchmark(self, name: str) -> None`

Set the primary benchmark for comparison.

##### `def record_benchmark_value(self, benchmark: str, timestamp: datetime, value: float) -> None`

Record benchmark value.

##### `def record_portfolio_value(self, timestamp: datetime, value: float) -> None`

Record portfolio value.

##### `def calculate_tracking_error(self, benchmark: , lookback_days: int)`

Calculate tracking error vs benchmark (#P12).

Tracking error = std dev of (portfolio return - benchmark return)

Args:
    benchmark: Benchmark name (uses active if None)
    lookback_days: Days of history to use

Returns:
    Annualized tracking error

##### `def calculate_information_ratio(self, benchmark: , lookback_days: int)`

Calculate information ratio (#P12).

IR = Active Return / Tracking Error

Args:
    benchmark: Benchmark name
    lookback_days: Days of history

Returns:
    Information ratio

##### `def get_benchmark_comparison(self, benchmark: ) -> dict`

Get comparison of portfolio vs benchmark.

### PortfolioHeatMapGenerator

Generates heat map data for portfolio visualization (#P13).

Creates visualizations for:
- Sector/asset performance
- Risk contribution
- Correlation matrix
- P&L by position

#### Methods

##### `def generate_performance_heatmap(positions: dict[str, dict], group_by: str) -> dict`

Generate performance heat map data (#P13).

Args:
    positions: Position data with returns
    group_by: Grouping field (sector, asset_class, etc.)

Returns:
    Heat map data structure

##### `def generate_risk_contribution_heatmap(risk_contributions: dict[str, float], positions: dict[str, dict]) -> dict`

Generate risk contribution heat map (#P13).

Args:
    risk_contributions: Risk contribution by symbol
    positions: Position data for grouping

Returns:
    Heat map data structure

##### `def generate_correlation_heatmap(correlation_matrix: dict[str, dict[str, float]]) -> dict`

Generate correlation matrix heat map (#P13).

Args:
    correlation_matrix: Pairwise correlations

Returns:
    Heat map data structure

##### `def generate_pnl_heatmap(daily_pnl: dict[str, list[float]], dates: list[str]) -> dict`

Generate P&L calendar heat map (#P13).

Args:
    daily_pnl: Daily P&L by symbol
    dates: List of date strings

Returns:
    Heat map data structure for calendar view

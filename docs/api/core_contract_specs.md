# contract_specs

**Path**: `C:\Users\Alexa\ai-trading-firm\core\contract_specs.py`

## Overview

Contract Specifications
=======================

Futures contract specifications for proper position sizing, margin calculation,
and P&L computation. This module provides institutional-grade contract data.

Based on CME/CBOT/NYMEX/COMEX specifications.

## Classes

### AssetClass

**Inherits from**: Enum

Asset class categorization for futures.

### Exchange

**Inherits from**: Enum

Exchange identifiers.

### ContractSpec

Futures contract specification.

All values are per standard contract unless otherwise noted.

#### Methods

##### `def point_value(self) -> float`

Dollar value of one point move.

##### `def calculate_tick_value(self) -> float`

Calculate tick value from multiplier and tick size.

##### `def notional_value(self, price: float, contracts: int) -> float`

Calculate notional value of position.

##### `def margin_required(self, contracts: int, is_initial: bool) -> float`

Calculate margin required for position.

##### `def pnl_per_tick(self, contracts: int) -> float`

Calculate P&L per tick for given number of contracts.

##### `def price_to_ticks(self, price_change: float) -> int`

Convert price change to number of ticks.

##### `def ticks_to_price(self, ticks: int) -> float`

Convert number of ticks to price change.

### CurrencyConverter

Currency conversion for P&L calculation.

Handles conversion of P&L from non-USD quote currencies to USD
for proper portfolio aggregation.

#### Methods

##### `def __init__(self)`

Initialize with default exchange rates (should be updated with live rates).

##### `def update_rate(self, currency: str, rate_to_usd: float) -> None`

Update exchange rate for a currency.

Args:
    currency: Currency code (e.g., "EUR", "JPY")
    rate_to_usd: How many USD per 1 unit of currency

##### `def update_rates_from_market_data(self, fx_prices: dict[str, float]) -> None`

Update rates from FX market data.

Args:
    fx_prices: Dictionary of pair -> price (e.g., {"EURUSD": 1.08, "USDJPY": 150.0})

##### `def get_rate(self, currency: str) -> float`

Get exchange rate (USD per 1 unit of currency).

Args:
    currency: Currency code

Returns:
    Exchange rate to USD, or 1.0 if unknown

##### `def convert_to_usd(self, amount: float, from_currency: str) -> float`

Convert amount from foreign currency to USD.

Args:
    amount: Amount in foreign currency
    from_currency: Source currency code

Returns:
    Equivalent amount in USD

##### `def convert_pnl_to_usd(self, pnl: float, quote_currency: str, symbol: ) -> float`

Convert P&L from quote currency to USD.

Special handling for JPY pairs where tick values are in JPY.

Args:
    pnl: P&L in quote currency
    quote_currency: Quote currency of the pair
    symbol: Optional symbol for logging

Returns:
    P&L in USD

##### `def calculate_fx_position_value_usd(self, base_currency: str, quote_currency: str, base_amount: float, price: float) -> float`

Calculate USD value of an FX position.

Args:
    base_currency: Base currency (e.g., "EUR" in EUR/USD)
    quote_currency: Quote currency (e.g., "USD" in EUR/USD)
    base_amount: Amount of base currency
    price: Current price (quote per base)

Returns:
    Position value in USD

##### `def get_tick_value_usd(self, spec: ContractSpec) -> float`

Get tick value in USD for a contract.

Args:
    spec: Contract specification

Returns:
    Tick value in USD

##### `def get_fx_pip_value_usd(self, pair: str, lot_size: int, current_price: ) -> float`

Calculate FX pip value in USD (#X2).

Properly handles JPY pairs where pip = 0.01 (vs 0.0001 for other pairs)
and converts native pip value to USD.

Args:
    pair: FX pair (e.g., "USDJPY", "EURUSD")
    lot_size: Position size in base currency units (default: 100,000 = 1 standard lot)
    current_price: Current price (required for XXX/USD pairs, optional for USD/XXX)

Returns:
    Value of 1 pip move in USD

Examples:
    USDJPY @ 150.00:
        - 1 pip = 0.01 (JPY pairs)
        - Pip value = 100,000 * 0.01 = 1,000 JPY
        - In USD = 1,000 / 150 = $6.67

    EURUSD @ 1.0800:
        - 1 pip = 0.0001 (standard pairs)
        - Pip value = 100,000 * 0.0001 = 10 USD
        - In USD = $10.00 (already in USD)

##### `def get_fx_tick_value_usd(self, pair: str, lot_size: int, current_price: ) -> float`

Calculate FX tick value in USD (half pip for JPY pairs).

Most FX brokers quote JPY pairs in 0.5 pip increments (0.005),
so tick value = pip value / 2 for JPY pairs.

Args:
    pair: FX pair
    lot_size: Position size
    current_price: Current price for accurate conversion

Returns:
    Tick value in USD

##### `def get_status(self) -> dict[str, Any]`

Get converter status.

### FXSwapRate

FX swap/rollover rate for overnight positions.

#### Methods

##### `def get_rollover_cost(self, position_size: float, is_long: bool) -> float`

Calculate rollover cost/credit for a position.

Args:
    position_size: Position size in lots
    is_long: True if long position, False if short

Returns:
    Rollover amount (positive = credit, negative = cost)

### FXRolloverManager

Manages FX rollover/swap rates for overnight positions.

FX positions held overnight are subject to rollover (swap) rates based on
the interest rate differential between the two currencies. This manager
tracks swap rates and calculates rollover costs/credits.

Note: Wednesday rollovers typically include 3 days (Wed-Thu-Fri-Sat-Sun)
to account for T+2 settlement.

#### Methods

##### `def __init__(self)`

Initialize with default swap rates.

##### `def update_swap_rate(self, pair: str, long_rate: float, short_rate: float, source: str) -> None`

Update swap rate for a currency pair.

Args:
    pair: Currency pair (e.g., "EURUSD")
    long_rate: Rate for long positions
    short_rate: Rate for short positions
    source: Source of the rate data

##### `def get_swap_rate(self, pair: str)`

Get swap rate for a pair.

##### `def calculate_rollover(self, pair: str, position_size: float, is_long: bool, nights: int) -> float`

Calculate rollover cost/credit for a position.

Args:
    pair: Currency pair
    position_size: Position size in lots
    is_long: True if long, False if short
    nights: Number of nights (usually 1, except Wednesday = 3)

Returns:
    Rollover amount in pips/points

##### `def calculate_rollover_usd(self, pair: str, position_size_lots: float, is_long: bool, nights: int) -> float`

Calculate rollover cost/credit in USD.

Args:
    pair: Currency pair
    position_size_lots: Position size in lots
    is_long: True if long, False if short
    nights: Number of nights

Returns:
    Rollover amount in USD

##### `def is_triple_swap_day(self) -> bool`

Check if today is a triple swap day (Wednesday for T+2 settlement).

Returns:
    True if today's rollover includes 3 days

##### `def get_rollover_nights(self) -> int`

Get number of nights for today's rollover.

##### `def record_rollover(self, pair: str, position_size: float, is_long: bool, rollover_amount: float, nights: int) -> None`

Record a rollover event for tracking.

Args:
    pair: Currency pair
    position_size: Position size
    is_long: Position direction
    rollover_amount: Amount credited/charged
    nights: Number of nights

##### `def get_total_rollovers(self, start_date: ) -> float`

Get total rollovers since start date.

Args:
    start_date: Optional start date filter

Returns:
    Total rollover amount

##### `def get_status(self) -> dict[str, Any]`

Get manager status.

### ContractSpecsManager

Centralized manager for contract specifications.

Provides:
- Contract spec lookup with caching
- Validation of contract symbols
- Margin and position value calculations
- Asset class grouping

#### Methods

##### `def __init__(self, config: )`

Initialize contract specs manager.

Args:
    config: Optional configuration with margin buffer settings

##### `def get_spec(self, symbol: str)`

Get contract specification by symbol.

Args:
    symbol: Contract symbol (e.g., "ES", "CL", "GC")

Returns:
    ContractSpec if found, None otherwise

##### `def is_valid_symbol(self, symbol: str) -> bool`

Check if symbol is a valid futures contract.

##### `def get_all_symbols(self) -> list[str]`

Get list of all available contract symbols.

##### `def get_symbols_by_asset_class(self, asset_class: AssetClass) -> list[str]`

Get all symbols for a given asset class.

##### `def get_symbols_by_exchange(self, exchange: Exchange) -> list[str]`

Get all symbols traded on a given exchange.

##### `def calculate_position_value(self, symbol: str, price: float, contracts: int)`

Calculate notional value of a position.

Args:
    symbol: Contract symbol
    price: Current price
    contracts: Number of contracts (positive=long, negative=short)

Returns:
    Notional value in USD, or None if symbol not found

##### `def calculate_margin_requirement(self, symbol: str, contracts: int, is_initial: bool)`

Calculate margin requirement for a position.

Includes configured margin buffer and cross-currency conversion (#X4).

Args:
    symbol: Contract symbol
    contracts: Number of contracts
    is_initial: True for initial margin, False for maintenance

Returns:
    Margin required in USD, or None if symbol not found

##### `def calculate_portfolio_margin(self, positions: dict[str, int], is_initial: bool, use_netting: bool) -> tuple[float, dict[str, float]]`

Calculate total portfolio margin with cross-currency conversion (#X4).

Args:
    positions: Dictionary of symbol -> quantity
    is_initial: True for initial margin
    use_netting: Apply cross-margin benefits for correlated positions

Returns:
    Tuple of (total_margin_usd, margin_by_symbol)

##### `def calculate_pnl(self, symbol: str, entry_price: float, exit_price: float, contracts: int, convert_to_usd: bool)`

Calculate P&L for a trade.

Args:
    symbol: Contract symbol
    entry_price: Entry price
    exit_price: Exit price
    contracts: Number of contracts (positive=long, negative=short)
    convert_to_usd: Whether to convert non-USD P&L to USD

Returns:
    P&L in USD (if convert_to_usd=True) or native currency, or None if symbol not found

##### `def calculate_pnl_in_ticks(self, symbol: str, entry_price: float, exit_price: float, contracts: int, convert_to_usd: bool)`

Calculate P&L for a trade with tick breakdown.

Args:
    symbol: Contract symbol
    entry_price: Entry price
    exit_price: Exit price
    contracts: Number of contracts
    convert_to_usd: Whether to convert to USD

Returns:
    Tuple of (P&L in currency, number of ticks), or None if symbol not found

##### `def get_tick_value_usd(self, symbol: str)`

Get tick value in USD for a symbol.

Handles currency conversion for non-USD quoted contracts.

Args:
    symbol: Contract symbol

Returns:
    Tick value in USD, or None if symbol not found

##### `def round_to_tick(self, symbol: str, price: float)`

Round price to nearest valid tick.

Args:
    symbol: Contract symbol
    price: Price to round

Returns:
    Rounded price, or None if symbol not found

##### `def get_liquidity_tier(self, symbol: str) -> str`

Get liquidity tier for a contract.

Returns: "high", "medium", or "low"

##### `def get_summary(self) -> dict[str, Any]`

Get summary of all contract specifications.

##### `def to_dataframe(self)`

Export all contract specs to a pandas DataFrame.

Returns:
    DataFrame with contract specifications

### FXProductType

**Inherits from**: Enum

Type of FX product (#X5).

### FXForwardRate

FX forward rate with forward points (#X5).

#### Methods

##### `def forward_rate(self) -> float`

Calculate outright forward rate.

##### `def annualized_forward_premium(self) -> float`

Calculate annualized forward premium/discount.

### FXProductManager

Manages FX spot vs forward distinction (#X5).

Handles:
- Spot trade identification (T+2)
- Forward contract pricing
- Forward points calculation
- Tenor mapping

#### Methods

##### `def __init__(self)`

##### `def is_spot_trade(self, settlement_days: int) -> bool`

Check if trade settles as spot (#X5).

Args:
    settlement_days: Days to settlement

Returns:
    True if spot (T+2 or less)

##### `def get_product_type(self, settlement_days: int) -> FXProductType`

Determine FX product type based on settlement (#X5).

Args:
    settlement_days: Days to settlement

Returns:
    FX product type

##### `def update_spot_rate(self, pair: str, rate: float) -> None`

Update spot rate for a currency pair.

##### `def calculate_forward_rate(self, pair: str, spot_rate: float, base_rate: float, quote_rate: float, days: int) -> FXForwardRate`

Calculate forward rate from interest rate differential (#X5).

Forward rate = Spot * (1 + r_quote * t) / (1 + r_base * t)

Args:
    pair: Currency pair (e.g., "EURUSD")
    spot_rate: Current spot rate
    base_rate: Base currency interest rate (annualized)
    quote_rate: Quote currency interest rate (annualized)
    days: Days to settlement

Returns:
    Forward rate calculation

##### `def get_forward_curve(self, pair: str) -> list[dict]`

Get forward curve for a currency pair.

##### `def store_forward_curve(self, pair: str, rates: list[FXForwardRate]) -> None`

Store forward curve for a currency pair.

### PipValueCalculator

Calculates pip values with proper currency conversion (#X6).

Handles:
- Standard and JPY pairs
- Cross-rate pip values
- Account currency conversion

#### Methods

##### `def __init__(self, account_currency: str)`

##### `def update_rate(self, pair: str, rate: float) -> None`

Update exchange rate.

##### `def get_pip_size(self, pair: str) -> float`

Get pip size for a currency pair (#X6).

##### `def calculate_pip_value(self, pair: str, lot_size: int, account_currency: str) -> dict`

Calculate pip value in account currency (#X6).

For a standard lot:
- If quote currency = account currency: pip_value = pip_size * lot_size
- If quote currency != account currency: need conversion

Args:
    pair: Currency pair (e.g., "EURUSD", "USDJPY")
    lot_size: Position size (default: standard lot)
    account_currency: Account currency (default: instance currency)

Returns:
    Pip value calculation details

##### `def calculate_position_pnl(self, pair: str, entry_rate: float, current_rate: float, lot_size: int, is_long: bool) -> dict`

Calculate P&L for a position in account currency (#X6).

Args:
    pair: Currency pair
    entry_rate: Entry exchange rate
    current_rate: Current exchange rate
    lot_size: Position size
    is_long: True for long, False for short

Returns:
    P&L calculation

### TriangularArbitrageOpportunity

Detected triangular arbitrage opportunity (#X7).

### TriangularArbitrageDetector

Detects triangular arbitrage opportunities (#X7).

For currencies A, B, C:
If (A/B) * (B/C) != (A/C), arbitrage exists.

Common triangles:
- EUR/USD, USD/JPY, EUR/JPY
- GBP/USD, USD/CHF, GBP/CHF

#### Methods

##### `def __init__(self, threshold_bps: float)`

##### `def update_rate(self, pair: str, bid: float, ask: float) -> None`

Update bid/ask for a currency pair.

##### `def get_rate(self, pair: str, side: str)`

Get rate for pair (bid, ask, or mid).

##### `def check_triangle(self, currency_a: str, currency_b: str, currency_c: str)`

Check for arbitrage in a currency triangle (#X7).

Path 1: A -> B -> C -> A
Path 2: A -> C -> B -> A

Args:
    currency_a: First currency (e.g., "EUR")
    currency_b: Second currency (e.g., "USD")
    currency_c: Third currency (e.g., "JPY")

Returns:
    Arbitrage opportunity if found

##### `def scan_all_triangles(self, currencies: ) -> list[TriangularArbitrageOpportunity]`

Scan all possible currency triangles (#X7).

Args:
    currencies: List of currencies to check (default: major)

Returns:
    List of arbitrage opportunities

##### `def get_recent_opportunities(self, lookback_seconds: int) -> list[dict]`

Get recent arbitrage opportunities.

### WeekendGapRiskManager

Manages weekend gap risk for FX positions (#X8).

Weekend gaps occur because FX markets close Friday 5 PM ET
and reopen Sunday 5 PM ET, but news can move prices.

Handles:
- Gap risk estimation
- Position reduction recommendations
- Historical gap analysis

#### Methods

##### `def __init__(self, risk_tolerance: str)`

Initialize weekend gap risk manager.

Args:
    risk_tolerance: "conservative", "moderate", or "aggressive"

##### `def is_weekend_approaching(self, hours_to_close: float) -> bool`

Check if FX weekend close is approaching (#X8).

FX closes Friday 5 PM ET.

Args:
    hours_to_close: Hours before close to flag

Returns:
    True if within hours_to_close of weekend

##### `def estimate_gap_risk(self, pair: str, position_size: float, pip_value: float, use_historical_max: bool) -> dict`

Estimate potential weekend gap risk (#X8).

Args:
    pair: Currency pair
    position_size: Position size in lots
    pip_value: Pip value in account currency
    use_historical_max: Use max gap instead of average

Returns:
    Gap risk analysis

##### `def recommend_position_reduction(self, pair: str, current_position: float, max_weekend_risk: float, pip_value: float) -> dict`

Recommend position reduction for weekend (#X8).

Args:
    pair: Currency pair
    current_position: Current position size (lots)
    max_weekend_risk: Maximum acceptable weekend risk (in account currency)
    pip_value: Pip value in account currency

Returns:
    Reduction recommendation

##### `def record_actual_gap(self, pair: str, friday_close: float, sunday_open: float, gap_date: datetime) -> dict`

Record actual weekend gap for analysis (#X8).

Args:
    pair: Currency pair
    friday_close: Friday closing price
    sunday_open: Sunday opening price
    gap_date: Date of the gap (Sunday)

Returns:
    Gap analysis

##### `def get_gap_statistics(self, pair: ) -> dict`

Get statistics on recorded gaps.

##### `def get_pre_weekend_checklist(self, positions: dict[str, float], pip_values: dict[str, float]) -> list[dict]`

Generate pre-weekend risk checklist (#X8).

Args:
    positions: {pair: position_size} for all positions
    pip_values: {pair: pip_value} for all pairs

Returns:
    List of risk items to review

## Functions

### `def get_currency_converter() -> CurrencyConverter`

Get or create the global currency converter instance.

### `def get_fx_rollover_manager() -> FXRolloverManager`

Get or create the global FX rollover manager.

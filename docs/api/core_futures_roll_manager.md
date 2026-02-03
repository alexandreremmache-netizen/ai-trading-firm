# futures_roll_manager

**Path**: `C:\Users\Alexa\ai-trading-firm\core\futures_roll_manager.py`

## Overview

Futures Roll Manager
====================

Manages futures contract rolling for continuous exposure.
Handles roll schedules, roll date detection, price adjustment,
and roll notifications.

## Classes

### RollMethod

**Inherits from**: Enum

Method for rolling futures contracts.

### RollStatus

**Inherits from**: Enum

Current roll status for a contract.

### RollSchedule

Roll schedule configuration for a contract.

#### Methods

##### `def get_next_contract_month(self, current_month: str, current_year: int) -> tuple[str, int]`

Get next contract month after current.

Args:
    current_month: Current month code (e.g., "H")
    current_year: Current year

Returns:
    Tuple of (next_month_code, year)

### FNDViolationLevel

**Inherits from**: Enum

Level of FND (First Notice Date) violation risk.

### FNDStatus

First Notice Date status for a contract.

### ContractInfo

Information about a specific contract.

#### Methods

##### `def is_expired(self) -> bool`

Check if contract is expired.

##### `def days_to_expiry(self) -> int`

Days until expiry.

##### `def days_to_fnd(self)`

Days until First Notice Date.

##### `def is_physical_delivery(self) -> bool`

Check if this is a physical delivery contract.

##### `def is_past_fnd(self) -> bool`

Check if we're past First Notice Date.

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### RollEvent

Event representing a roll signal or completion.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary for logging.

### FuturesRollManager

Manages futures contract rolling.

Features:
- Automatic roll date detection
- Roll signal generation
- Continuous contract price adjustment
- Roll event notifications

#### Methods

##### `def __init__(self, config: )`

Initialize roll manager.

Args:
    config: Configuration with:
        - enabled: Enable auto-roll detection (default: True)
        - auto_roll: Automatically generate roll signals (default: True)
        - custom_schedules: Override default roll schedules

##### `def register_callback(self, callback: Callable[, None]) -> None`

Register callback for roll events.

##### `def set_current_contract(self, symbol: str, contract_code: str, expiry_date: date, first_notice_date: ) -> None`

Set the current active contract for a symbol.

Args:
    symbol: Base symbol (e.g., "ES", "NG", "RTY")
    contract_code: Full contract code (e.g., "ESH25", "NGF25", "RTYZ24")
    expiry_date: Contract expiry date
    first_notice_date: First notice date for physical delivery

##### `def check_roll_status(self, symbol: str) -> RollStatus`

Check if a contract needs to be rolled.

Args:
    symbol: Base symbol

Returns:
    Current roll status

##### `def complete_roll(self, symbol: str, new_contract_code: str, new_expiry_date: date, roll_price_adjustment: float, first_notice_date: ) -> RollEvent`

Complete a roll to a new contract.

Args:
    symbol: Base symbol
    new_contract_code: New contract code
    new_expiry_date: New expiry date
    roll_price_adjustment: Price adjustment for continuous pricing
    first_notice_date: First notice date for new contract

Returns:
    Roll completion event

##### `def get_continuous_price(self, symbol: str, raw_price: float) -> float`

Get continuous contract adjusted price.

Args:
    symbol: Contract symbol
    raw_price: Raw price from current contract

Returns:
    Adjusted continuous price

##### `def get_raw_price(self, symbol: str, continuous_price: float) -> float`

Convert continuous price back to raw price.

Args:
    symbol: Contract symbol
    continuous_price: Adjusted continuous price

Returns:
    Raw price in current contract

##### `def get_current_contract(self, symbol: str)`

Get current contract info for symbol.

##### `def get_roll_schedule(self, symbol: str)`

Get roll schedule for symbol.

##### `def get_contracts_approaching_roll(self) -> list[ContractInfo]`

Get list of contracts approaching roll.

##### `def get_roll_history(self, symbol: , limit: int) -> list[RollEvent]`

Get roll event history.

Args:
    symbol: Filter by symbol (optional)
    limit: Maximum events to return

Returns:
    List of roll events

##### `def check_all_contracts(self) -> dict[str, RollStatus]`

Check roll status for all tracked contracts.

Returns:
    Dictionary of symbol to roll status

##### `def check_fnd_status(self, symbol: str, has_position: bool, warning_days: int) -> FNDStatus`

Check First Notice Date status for a contract.

CRITICAL: Physical delivery contracts must be closed before FND
to avoid delivery obligations. This is especially important for
retail/prop traders who cannot take physical delivery.

Args:
    symbol: Contract symbol
    has_position: Whether we currently have a position
    warning_days: Days before FND to start warning

Returns:
    FNDStatus with violation level and required action

##### `def can_hold_position(self, symbol: str) -> tuple[bool, str]`

Check if it's safe to hold a position in this contract.

Enforces FND rules for physical delivery contracts.

Args:
    symbol: Contract symbol

Returns:
    Tuple of (can_hold, reason)

##### `def can_open_position(self, symbol: str) -> tuple[bool, str]`

Check if it's safe to open a new position in this contract.

More restrictive than can_hold_position - prevents opening
positions too close to FND.

Args:
    symbol: Contract symbol

Returns:
    Tuple of (can_open, reason)

##### `def get_fnd_violations(self, positions: dict[str, float]) -> list[FNDStatus]`

Check all positions for FND violations.

Args:
    positions: Dictionary of symbol -> position quantity

Returns:
    List of FNDStatus for contracts with violations/warnings

##### `def get_status(self) -> dict[str, Any]`

Get manager status for monitoring.

### BasisSpread

Tracks basis between spot and futures or between contract months (#F5).

#### Methods

##### `def __post_init__(self)`

### BasisTracker

Tracks basis and calendar spreads for futures (#F5).

Monitors:
- Spot vs front-month basis (contango/backwardation)
- Calendar spreads between contract months
- Roll yield opportunities

#### Methods

##### `def __init__(self)`

##### `def update_spot_price(self, symbol: str, price: float) -> None`

Update spot price for a symbol.

##### `def update_futures_price(self, symbol: str, contract: str, price: float) -> None`

Update futures price for a contract.

##### `def calculate_spot_basis(self, symbol: str, front_contract: str)`

Calculate basis between spot and front month (#F5).

Returns:
    BasisSpread or None if prices not available

##### `def calculate_calendar_spread(self, symbol: str, front_contract: str, back_contract: str, days_between: int)`

Calculate calendar spread between two contract months (#F5).

Args:
    symbol: Base symbol
    front_contract: Front month contract code
    back_contract: Back month contract code
    days_between: Approximate days between expiries

Returns:
    BasisSpread or None if prices not available

##### `def get_term_structure(self, symbol: str) -> dict`

Get full term structure for a symbol (#F5).

Returns prices and spreads across all contract months.

##### `def detect_roll_yield_opportunity(self, symbol: str, threshold_bps: float)`

Detect roll yield opportunities (#F5).

Identifies when calendar spread offers significant carry.

Args:
    symbol: Symbol to analyze
    threshold_bps: Minimum annualized spread in bps to flag

Returns:
    Opportunity details or None

##### `def get_basis_statistics(self, symbol: str, lookback: int) -> dict`

Get basis statistics for monitoring.

### DeliveryMonthValidator

Validates delivery month matches between internal specs and IB (#F6).

Prevents mismatches that could lead to wrong contract trading.

#### Methods

##### `def validate_contract_month(cls, symbol: str, month_code: str, source: str) -> dict`

Validate if month code is valid for symbol (#F6).

Args:
    symbol: Contract symbol
    month_code: Month code to validate
    source: Source of the month code

Returns:
    Validation result

##### `def validate_roll_schedule(cls, symbol: str) -> dict`

Validate roll schedule matches IB contract months (#F6).

Args:
    symbol: Contract symbol to validate

Returns:
    Validation result with any mismatches

##### `def validate_all_schedules(cls) -> list[dict]`

Validate all roll schedules against IB (#F6).

### ExpirationWarning

Warning about approaching contract expiration (#F7).

### ExpirationWarningSystem

Monitors and alerts on approaching expirations (#F7).

Provides tiered warnings:
- INFO: 14+ days out
- WARNING: 7-14 days out
- CRITICAL: <7 days out

#### Methods

##### `def __init__(self, warning_days: int, critical_days: int, fnd_buffer_days: int)`

##### `def check_expiration(self, symbol: str, contract_code: str, days_to_expiry: int, days_to_fnd: , position_size: int, is_physical_delivery: bool)`

Check if expiration warning should be issued (#F7).

Args:
    symbol: Base symbol
    contract_code: Full contract code
    days_to_expiry: Days until expiration
    days_to_fnd: Days until first notice (physical delivery)
    position_size: Current position size
    is_physical_delivery: Whether contract has physical delivery

Returns:
    ExpirationWarning if warning needed

##### `def check_all_positions(self, positions: dict[str, int], roll_manager: FuturesRollManager) -> list[ExpirationWarning]`

Check all positions for expiration warnings (#F7).

Args:
    positions: Map of contract codes to position sizes
    roll_manager: Roll manager for contract info

Returns:
    List of warnings

##### `def acknowledge_warning(self, contract_code: str) -> None`

Acknowledge a warning to suppress repeat alerts.

##### `def get_active_warnings(self, level: ) -> list[ExpirationWarning]`

Get active warnings, optionally filtered by level.

Args:
    level: Filter by level ("INFO", "WARNING", "CRITICAL") or None for all

Returns:
    List of active warnings

##### `def get_summary(self) -> dict`

Get warning summary for monitoring.

### SettlementPrice

Settlement price data for a contract (#F8).

### SettlementPriceManager

Manages settlement prices for futures (#F8).

Handles:
- Daily settlement prices for mark-to-market
- Final settlement prices for expiring contracts
- Settlement price history

#### Methods

##### `def __init__(self)`

##### `def record_daily_settlement(self, symbol: str, contract_code: str, settlement_date: date, settlement_price: float, volume: , open_interest: ) -> SettlementPrice`

Record daily settlement price (#F8).

Args:
    symbol: Base symbol
    contract_code: Full contract code
    settlement_date: Date of settlement
    settlement_price: Settlement price
    volume: Trading volume
    open_interest: Open interest

Returns:
    SettlementPrice record

##### `def record_final_settlement(self, symbol: str, contract_code: str, settlement_date: date, settlement_price: float) -> SettlementPrice`

Record final settlement price at expiration (#F8).

Args:
    symbol: Base symbol
    contract_code: Full contract code
    settlement_date: Final settlement date
    settlement_price: Final settlement price

Returns:
    SettlementPrice record

##### `def get_latest_settlement(self, contract_code: str)`

Get most recent settlement price for a contract.

##### `def get_final_settlement(self, contract_code: str)`

Get final settlement price if contract has expired.

##### `def calculate_mark_to_market(self, contract_code: str, position: int, entry_price: float)`

Calculate mark-to-market P&L using settlement price (#F8).

Args:
    contract_code: Contract code
    position: Number of contracts (positive=long, negative=short)
    entry_price: Average entry price

Returns:
    MTM calculation or None if no settlement

##### `def get_settlement_history(self, contract_code: str, lookback_days: int) -> list[dict]`

Get settlement price history for a contract.

##### `def calculate_variation_margin(self, positions: dict[str, tuple[int, float]]) -> dict`

Calculate total variation margin requirement (#F8).

Args:
    positions: Map of contract codes to (position, entry_price) tuples

Returns:
    Variation margin calculation

##### `def get_status(self) -> dict`

Get manager status for monitoring.

## Functions

### `def parse_contract_code(contract_code: str)`

Parse futures contract code into root symbol, month code, and year.

Handles multiple formats (#F2):
- "ESZ4"   -> ("ES", "Z", 2024)   2-letter symbol + month + 1-digit year
- "ESZ24"  -> ("ES", "Z", 2024)   2-letter symbol + month + 2-digit year
- "NGF25"  -> ("NG", "F", 2025)   2-letter symbol + month + 2-digit year
- "RTYZ4"  -> ("RTY", "Z", 2024)  3-letter symbol + month + 1-digit year
- "RTYZ24" -> ("RTY", "Z", 2024)  3-letter symbol + month + 2-digit year

Args:
    contract_code: Full contract code string

Returns:
    Tuple of (root_symbol, month_code, year) or None if parsing fails

### `def format_contract_code(root_symbol: str, month_code: str, year: int, two_digit_year: bool) -> str`

Format a contract code from components.

Args:
    root_symbol: Root symbol (e.g., "ES", "NG", "RTY")
    month_code: Month code (e.g., "Z", "F")
    year: Full year (e.g., 2024)
    two_digit_year: If True, use 2-digit year (e.g., "24"), else 1-digit

Returns:
    Formatted contract code (e.g., "ESZ4" or "ESZ24")

### `def calculate_roll_date(expiry_date: date, roll_days_before: int, method: RollMethod) -> date`

Calculate roll date for a contract.

Args:
    expiry_date: Contract expiry date
    roll_days_before: Days before expiry to roll
    method: Roll method

Returns:
    Recommended roll date

### `def get_contract_expiry(symbol: str, month_code: str, year: int)`

Estimate contract expiry date.

Note: This is an approximation. Use exchange data for production.

Args:
    symbol: Base symbol
    month_code: Contract month code
    year: Contract year

Returns:
    Estimated expiry date

### `def estimate_first_notice_date(symbol: str, contract_month: str, contract_year: int)`

P2-18: Automatically estimate First Notice Date (FND) for physical delivery contracts.

FND varies by exchange and commodity type:
- CME Energy (CL, NG): Last business day of month prior to delivery month
- CME Metals (GC, SI): Last business day of month prior to delivery month
- CME Grains (ZC, ZW, ZS): Last business day of month prior to delivery month
- COMEX (HG): 2 business days before 1st delivery day
- Cash-settled contracts (ES, NQ, etc.): No FND (returns None)

Args:
    symbol: Futures symbol (e.g., "CL", "GC")
    contract_month: Contract month code (F, G, H, J, K, M, N, Q, U, V, X, Z)
    contract_year: Contract year

Returns:
    Estimated FND date, or None for cash-settled contracts

### `def get_fnd_with_auto_estimate(symbol: str, contract_month: str, contract_year: int, manual_fnd: )`

P2-18: Get FND with automatic estimation fallback.

Args:
    symbol: Futures symbol
    contract_month: Contract month code
    contract_year: Contract year
    manual_fnd: Manually specified FND (takes precedence if provided)

Returns:
    FND date (manual if provided, else estimated, else None for cash-settled)

### `def get_complete_roll_calendar(year: int) -> list[dict]`

Generate complete roll calendar for a year (#F4).

Args:
    year: Calendar year

Returns:
    List of roll events with dates and symbols

### `def parse_contract_month(contract_code: str) -> tuple[int, str]`

Parse contract code to (year, month_code) for sorting.

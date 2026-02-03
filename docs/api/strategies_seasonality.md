# seasonality

**Path**: `C:\Users\Alexa\ai-trading-firm\strategies\seasonality.py`

## Overview

Seasonality Strategy
====================

Implements seasonal patterns for commodity and equity trading.
Based on historical calendar effects and production/consumption cycles.

MATURITY: PRODUCTION-READY
--------------------------
Status: Complete implementation with historical validation
- [x] Comprehensive seasonal patterns database
- [x] Active pattern detection
- [x] Signal generation with confidence scoring
- [x] Pattern confluence analysis
- [x] Historical pattern validation
- [x] Monthly bias calculation
- [x] Upcoming patterns scanning
- [x] Custom pattern support

Production Readiness:
- Unit tests: Good coverage
- Historical data: Patterns based on 10+ years of data
- Win rates: Based on documented historical performance

APPROVED FOR PRODUCTION USE
- Patterns are well-researched (driving season, planting, harvest, etc.)
- Win rates are conservative estimates
- Always combine with other signals for confirmation
- Monitor actual vs expected performance

Covered Markets:
- Energy: Natural Gas (NG), Crude Oil (CL), Gasoline (RB)
- Agriculture: Corn (ZC), Wheat (ZW), Soybeans (ZS)
- Metals: Gold (GC)
- Equities: ES/SPY (Santa Rally, Sell in May)

## Classes

### SeasonalPattern

**Inherits from**: Enum

Types of seasonal patterns.

### SeasonalStrength

**Inherits from**: Enum

Strength of seasonal signal.

### SeasonalWindow

Defines a seasonal trading window.

#### Methods

##### `def strength(self) -> SeasonalStrength`

Determine signal strength from win rate.

##### `def is_active(self, check_date: date) -> bool`

Check if this seasonal window is active on given date.

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### SeasonalSignal

Signal generated from seasonal analysis.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### SeasonalityStrategy

Seasonality-based trading strategy.

Generates signals based on historical seasonal patterns
in commodities and equity markets.

#### Methods

##### `def __init__(self, config: )`

Initialize seasonality strategy.

Args:
    config: Configuration with:
        - min_win_rate: Minimum win rate to generate signal (default: 0.55)
        - enabled_symbols: List of symbols to analyze (default: all)
        - min_patterns: Minimum confirming patterns (default: 1)

##### `def get_active_patterns(self, symbol: str, check_date: ) -> list[SeasonalWindow]`

Get active seasonal patterns for a symbol.

Args:
    symbol: Instrument symbol
    check_date: Date to check (default: today)

Returns:
    List of active SeasonalWindows

##### `def generate_signal(self, symbol: str, check_date: )`

Generate seasonal signal for a symbol.

Args:
    symbol: Instrument symbol
    check_date: Date to check

Returns:
    SeasonalSignal if patterns are active, None otherwise

##### `def scan_all_symbols(self, check_date: ) -> list[SeasonalSignal]`

Scan all symbols for active seasonal signals.

Args:
    check_date: Date to check

Returns:
    List of SeasonalSignals

##### `def get_upcoming_patterns(self, days_ahead: int) -> list[tuple[date, SeasonalWindow]]`

Get patterns starting within the next N days.

Args:
    days_ahead: Days to look ahead

Returns:
    List of (start_date, pattern) tuples

##### `def add_pattern(self, window: SeasonalWindow) -> None`

Add a custom seasonal pattern.

##### `def get_all_patterns(self, symbol: ) -> list[SeasonalWindow]`

Get all patterns, optionally filtered by symbol.

##### `def get_status(self) -> dict[str, Any]`

Get strategy status.

##### `def validate_pattern(self, pattern: SeasonalWindow, price_data: np.ndarray, dates: np.ndarray, lookback_years: int) -> dict[str, Any]`

Validate a seasonal pattern against historical price data.

Calculates actual win rate and returns based on historical performance.

Args:
    pattern: SeasonalWindow to validate
    price_data: Historical prices (close prices)
    dates: Corresponding dates as numpy datetime64 array
    lookback_years: Years of history to analyze

Returns:
    Dictionary with validation metrics

##### `def update_pattern_from_history(self, symbol: str, pattern_name: str, price_data: np.ndarray, dates: np.ndarray, lookback_years: int) -> bool`

Update pattern's win rate and avg return based on historical data.

Args:
    symbol: Symbol to update
    pattern_name: Name of pattern to update
    price_data: Historical prices
    dates: Corresponding dates
    lookback_years: Years to analyze

Returns:
    True if pattern was updated, False otherwise

##### `def calculate_pattern_confluence(self, symbol: str, check_date: ) -> dict[str, Any]`

Calculate pattern confluence score for a symbol.

Multiple overlapping patterns in same direction = stronger signal.
Conflicting patterns = weaker signal.

Args:
    symbol: Symbol to analyze
    check_date: Date to check

Returns:
    Dictionary with confluence analysis

##### `def get_monthly_bias(self, symbol: str) -> dict[int, dict[str, float]]`

Get monthly directional bias for a symbol based on all patterns.

Returns:
    Dictionary of month -> {"long": score, "short": score, "net": bias}

# portfolio_export

**Path**: `C:\Users\Alexa\ai-trading-firm\core\portfolio_export.py`

## Overview

Portfolio Export Module
=======================

Addresses issues:
- #P20: Portfolio metrics caching suboptimal
- #P21: No portfolio export formats (IBKR, etc.)

Features:
- Export portfolio to IBKR flex query format
- CSV/Excel export with multiple templates
- FIX protocol format
- Portfolio metrics caching with intelligent invalidation

## Classes

### ExportFormat

**Inherits from**: str, Enum

Portfolio export formats.

### PortfolioPosition

Position for export.

#### Methods

##### `def pnl_pct(self) -> float`

P&L as percentage.

### PortfolioSummary

Portfolio summary for export.

#### Methods

##### `def position_count(self) -> int`

Number of positions.

##### `def total_unrealized_pnl(self) -> float`

Total unrealized P&L.

### IBKRFlexExporter

Export to IBKR Flex Query format (#P21).

Generates XML compatible with IBKR's reporting tools.

#### Methods

##### `def __init__(self, account_id: str)`

##### `def export_positions(self, positions: list[PortfolioPosition], as_of_date: ) -> str`

Export positions to IBKR Flex XML format.

Args:
    positions: List of positions
    as_of_date: Report date

Returns:
    XML string

##### `def export_account_info(self, summary: PortfolioSummary) -> str`

Export account information to IBKR format.

### CSVPortfolioExporter

Export portfolio to CSV formats (#P21).

Supports multiple templates for different use cases.

#### Methods

##### `def __init__(self, delimiter: str)`

##### `def export_positions(self, positions: list[PortfolioPosition], template: str) -> str`

Export positions to CSV.

Templates:
- standard: Basic position info
- detailed: Full position data
- pnl: P&L focused
- reconciliation: For broker reconciliation

##### `def export_summary(self, summary: PortfolioSummary) -> str`

Export portfolio summary to CSV.

### FIXPortfolioExporter

Export portfolio in FIX protocol format (#P21).

Generates FIX-style messages for position reporting.

#### Methods

##### `def __init__(self, sender_comp_id: str, target_comp_id: str)`

##### `def export_position_report(self, position: PortfolioPosition) -> str`

Generate FIX Position Report (AP) message.

Args:
    position: Position to export

Returns:
    FIX message string

### CachedPortfolioMetrics

Cached portfolio metrics.

#### Methods

##### `def age_seconds(self) -> float`

Age of cache in seconds.

### PortfolioMetricsCache

Intelligent caching for portfolio metrics (#P20).

Features:
- Position-change aware invalidation
- Tiered TTL based on metric volatility
- Memory-efficient storage
- Thread-safe access

#### Methods

##### `def __init__(self, default_ttl: float)`

##### `def get(self, key: str)`

Get cached value if valid.

##### `def set(self, key: str, value: Any, ttl: , metric_type: ) -> None`

Store value in cache.

##### `def invalidate(self, key: str) -> bool`

Invalidate specific key.

##### `def invalidate_pattern(self, pattern: str) -> int`

Invalidate keys matching pattern prefix.

##### `def on_position_change(self) -> None`

Handle position change - invalidate relevant caches.

##### `def on_price_update(self, symbols: list[str]) -> None`

Handle price update - selective invalidation.

##### `def get_or_compute(self, key: str, compute_fn: Callable[, Any], ttl: , metric_type: ) -> Any`

Get from cache or compute with double-check pattern.

Uses double-check locking to prevent deadlock while avoiding
duplicate computation by multiple threads.

##### `def get_stats(self) -> dict`

Get cache statistics.

### PortfolioExporter

High-level portfolio export manager.

Supports multiple formats with caching.

#### Methods

##### `def __init__(self, account_id: str, output_dir: str)`

##### `def export(self, positions: list[PortfolioPosition], format: ExportFormat, filename: , template: str) -> str`

Export portfolio to file.

Returns:
    Path to exported file

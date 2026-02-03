# metrics_export

**Path**: `C:\Users\Alexa\ai-trading-firm\core\metrics_export.py`

## Overview

Strategy Performance Metrics Export
===================================

Addresses issue #Q23: No strategy performance metrics export.

Features:
- Export performance metrics to multiple formats (JSON, CSV, HTML, Excel)
- Standardized metric definitions
- Time-series export support
- Compliance-ready reporting formats

## Classes

### ExportFormat

**Inherits from**: str, Enum

Supported export formats.

### MetricCategory

**Inherits from**: str, Enum

Categories for organizing metrics.

### MetricDefinition

Definition of a performance metric.

Provides standardized descriptions and formatting for compliance reporting.

#### Methods

##### `def format_value(self, value: float) -> str`

Format value according to metric definition.

### PerformanceSnapshot

Snapshot of strategy performance at a point in time.

Supports time-series export of performance metrics.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### PerformanceMetricsExporter

Exports strategy performance metrics to various formats (#Q23).

Supports:
- Single snapshot export
- Time-series export
- Multi-strategy comparison
- Compliance-ready formats

#### Methods

##### `def __init__(self, output_dir: str, include_definitions: bool)`

Initialize exporter.

Args:
    output_dir: Directory for exported files
    include_definitions: Include metric definitions in exports

##### `def add_snapshot(self, snapshot: PerformanceSnapshot) -> None`

Add performance snapshot for time-series export.

##### `def export_metrics(self, metrics: dict[str, float], strategy_name: str, format: ExportFormat, filename: , metadata: ) -> str`

Export metrics to file.

Args:
    metrics: Dictionary of metric name to value
    strategy_name: Name of the strategy
    format: Export format
    filename: Optional custom filename
    metadata: Additional metadata to include

Returns:
    Path to exported file

##### `def export_time_series(self, format: ExportFormat, filename: , metrics_to_include: ) -> str`

Export time-series of performance snapshots.

Args:
    format: Export format
    filename: Optional custom filename
    metrics_to_include: Specific metrics to include (default: all)

Returns:
    Path to exported file

##### `def export_comparison(self, strategies: dict[str, dict[str, float]], format: ExportFormat, filename: ) -> str`

Export multi-strategy comparison.

Args:
    strategies: Dict mapping strategy name to metrics dict
    format: Export format
    filename: Optional custom filename

Returns:
    Path to exported file

## Constants

- `METRIC_DEFINITIONS`

# custom_reports

**Path**: `C:\Users\Alexa\ai-trading-firm\core\custom_reports.py`

## Overview

Custom Reporting Module
=======================

Flexible custom reporting framework for portfolio management.

Issues Addressed:
- #P19: Missing custom reporting
- #E30: No execution venue selection logic (integrated here)

## Classes

### ReportFormat

**Inherits from**: str, Enum

Output format for reports.

### ReportFrequency

**Inherits from**: str, Enum

Report generation frequency.

### ReportCategory

**Inherits from**: str, Enum

Report category.

### ReportDefinition

Definition of a custom report (#P19).

#### Methods

##### `def to_dict(self) -> dict`

### ReportExecution

Record of report execution (#P19).

#### Methods

##### `def to_dict(self) -> dict`

### CustomReportBuilder

Builder for custom reports (#P19).

Allows defining custom reports with:
- Multiple data sources
- Custom filters and transformations
- Flexible column definitions
- Grouping and aggregation
- Multiple output formats

#### Methods

##### `def __init__(self, report_name: str, category: ReportCategory)`

##### `def with_description(self, description: str) -> CustomReportBuilder`

Set report description.

##### `def with_frequency(self, frequency: ReportFrequency) -> CustomReportBuilder`

Set report frequency.

##### `def with_formats(self, formats: list[ReportFormat]) -> CustomReportBuilder`

Set output formats.

##### `def add_data_source(self, source: str) -> CustomReportBuilder`

Add data source.

##### `def add_filter(self, field: str, operator: str, value: Any) -> CustomReportBuilder`

Add filter condition.

##### `def add_column(self, name: str, source: str, transform: , format_spec: ) -> CustomReportBuilder`

Add column definition.

##### `def with_grouping(self, fields: list[str]) -> CustomReportBuilder`

Set grouping fields.

##### `def with_sorting(self, column: str, direction: str) -> CustomReportBuilder`

Add sorting.

##### `def with_aggregation(self, column: str, function: str) -> CustomReportBuilder`

Add aggregation.

##### `def build(self, created_by: str) -> ReportDefinition`

Build report definition.

### CustomReportEngine

Custom report execution engine (#P19).

Features:
- Report definition management
- Data source integration
- Report execution and scheduling
- Multiple output formats
- Execution history

#### Methods

##### `def __init__(self)`

##### `def register_data_handler(self, source_name: str, handler: Callable[, list[dict]]) -> None`

Register data source handler.

##### `def register_transform(self, name: str, func: Callable) -> None`

Register custom transform function.

##### `def save_report(self, report: ReportDefinition) -> None`

Save report definition.

##### `def get_report(self, report_id: str)`

Get report definition.

##### `def list_reports(self, category: ) -> list[ReportDefinition]`

List available reports.

##### `def execute_report(self, report_id: str, output_format: , override_filters: ) -> tuple[str, ReportExecution]`

Execute a report (#P19).

Args:
    report_id: Report to execute
    output_format: Override default format
    override_filters: Additional filters

Returns:
    (output_data, execution_record)

##### `def get_execution_history(self, report_id: , limit: int) -> list[ReportExecution]`

Get report execution history.

### VenueType

**Inherits from**: str, Enum

Execution venue types (#E30).

### OrderCharacteristic

**Inherits from**: str, Enum

Order characteristics for venue selection (#E30).

### VenueCharacteristics

Characteristics of an execution venue (#E30).

#### Methods

##### `def to_dict(self) -> dict`

### VenueSelectionResult

Result of venue selection (#E30).

#### Methods

##### `def to_dict(self) -> dict`

### ExecutionVenueSelector

Execution venue selection logic (#E30).

Features:
- Venue characteristic analysis
- Order-based venue matching
- Cost optimization
- Fill rate optimization
- Information leakage mitigation

#### Methods

##### `def __init__(self)`

##### `def register_venue(self, venue: VenueCharacteristics) -> None`

Register execution venue.

##### `def update_symbol_adv(self, symbol: str, adv: float) -> None`

Update average daily volume for symbol.

##### `def classify_order(self, symbol: str, quantity: int, price: float, is_urgent: bool, is_info_sensitive: bool) -> OrderCharacteristic`

Classify order by characteristics (#E30).

Args:
    symbol: Symbol
    quantity: Order quantity
    price: Current price
    is_urgent: Urgency flag
    is_info_sensitive: Information sensitivity

Returns:
    Order characteristic

##### `def select_venues(self, symbol: str, quantity: int, price: float, is_passive: bool, is_urgent: bool, is_info_sensitive: bool, max_venues: int) -> VenueSelectionResult`

Select optimal execution venues (#E30).

Args:
    symbol: Symbol to trade
    quantity: Order quantity
    price: Current price
    is_passive: Passive (make) vs aggressive (take)
    is_urgent: Urgency flag
    is_info_sensitive: Information sensitivity
    max_venues: Maximum venues to use

Returns:
    VenueSelectionResult with recommendations

##### `def record_execution(self, venue_id: str, symbol: str, quantity: int, fill_pct: float, fill_time_ms: float, price_improvement_bps: float) -> None`

Record execution for venue performance tracking.

##### `def get_venue_statistics(self, venue_id: str) -> dict`

Get venue performance statistics.

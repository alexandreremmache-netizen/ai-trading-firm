# dashboard_metrics

**Path**: `C:\Users\Alexa\ai-trading-firm\core\dashboard_metrics.py`

## Overview

Dashboard Metrics Module
========================

Addresses issues:
- #R26: Risk dashboard metrics incomplete
- #C41: Compliance dashboard metrics incomplete
- #E31: Execution statistics dashboard incomplete
- #O17: Option analytics dashboard incomplete

Features:
- Comprehensive real-time metrics for dashboards
- Risk metrics with alerts and thresholds
- Compliance status tracking
- Execution quality metrics
- Options analytics summary

## Classes

### AlertLevel

**Inherits from**: str, Enum

Alert severity levels.

### MetricStatus

**Inherits from**: str, Enum

Metric status indicators.

### DashboardAlert

Alert for dashboard display.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### RiskDashboardMetrics

Comprehensive risk metrics for dashboard display (#R26).

Provides real-time risk monitoring with thresholds and alerts.

#### Methods

##### `def get_alerts(self) -> list[DashboardAlert]`

Generate alerts based on current metrics.

##### `def get_status(self, metric_name: str) -> MetricStatus`

Get status indicator for a specific metric.

##### `def to_dict(self) -> dict`

Convert to dictionary for API/JSON.

### ComplianceDashboardMetrics

Compliance metrics for dashboard display (#C41).

Tracks regulatory compliance status and upcoming deadlines.

#### Methods

##### `def get_alerts(self) -> list[DashboardAlert]`

Generate compliance alerts.

##### `def calculate_compliance_score(self) -> float`

Calculate overall compliance score (0-100).

##### `def to_dict(self) -> dict`

Convert to dictionary.

### ExecutionDashboardMetrics

Execution statistics for dashboard display (#E31).

Tracks execution quality and order flow metrics.

#### Methods

##### `def get_alerts(self) -> list[DashboardAlert]`

Generate execution alerts.

##### `def calculate_execution_quality_score(self) -> float`

Calculate overall execution quality score (0-100).

##### `def to_dict(self) -> dict`

Convert to dictionary.

### OptionsDashboardMetrics

Options analytics for dashboard display (#O17).

Comprehensive options portfolio metrics and risk indicators.

#### Methods

##### `def get_alerts(self) -> list[DashboardAlert]`

Generate options-specific alerts.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### DashboardMetricsCollector

Collects and aggregates metrics for all dashboards.

Provides unified interface for dashboard data.

#### Methods

##### `def __init__(self)`

##### `def get_all_alerts(self) -> list[DashboardAlert]`

Get all alerts from all dashboards.

##### `def get_summary(self) -> dict`

Get summary for all dashboards.

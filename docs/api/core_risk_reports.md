# risk_reports

**Path**: `C:\Users\Alexa\ai-trading-firm\core\risk_reports.py`

## Overview

Risk Report Generation Module
=============================

Automated risk report generation (Issue #R25).

Features:
- Daily risk summary reports
- Position risk breakdown
- VaR/CVaR analysis
- Greeks exposure reports
- Limit utilization tracking
- Stress test summaries
- Export to multiple formats (JSON, CSV, HTML)

## Classes

### ReportType

**Inherits from**: str, Enum

Type of risk report.

### ReportFormat

**Inherits from**: str, Enum

Output format for reports.

### ReportSection

A section within a report.

### RiskReport

Complete risk report.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

##### `def add_section(self, section: ReportSection) -> None`

Add a section to the report.

### RiskReportGenerator

Generates various risk reports (#R25).

Provides comprehensive risk reporting for compliance and monitoring.

#### Methods

##### `def __init__(self, output_dir: str, firm_name: str)`

##### `def generate_daily_summary(self, risk_state: Any, positions: dict[str, Any], var_results: , stress_results: ) -> RiskReport`

Generate daily risk summary report.

Comprehensive overview of portfolio risk for the day.

##### `def generate_position_risk_report(self, positions: dict[str, Any], risk_contributions: ) -> RiskReport`

Generate detailed position risk breakdown report.

##### `def generate_limit_utilization_report(self, current_values: dict[str, float], limits: dict[str, float]) -> RiskReport`

Generate limit utilization report.

##### `def export_report(self, report: RiskReport, format: ReportFormat, filename: ) -> str`

Export report to file.

Returns the filepath of the exported report.

### ScheduledReportManager

Manages scheduled report generation.

#### Methods

##### `def __init__(self, generator: RiskReportGenerator)`

##### `def should_generate_daily_report(self) -> bool`

Check if daily report should be generated.

##### `def mark_daily_report_generated(self) -> None`

Mark that daily report was generated.

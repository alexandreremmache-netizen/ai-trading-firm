# regulatory_monitoring

**Path**: `C:\Users\Alexa\ai-trading-firm\core\regulatory_monitoring.py`

## Overview

Regulatory Change Monitoring and Compliance Report Templates.

This module provides automated monitoring of regulatory changes and
standardized templates for compliance reporting.

Addresses:
- #C42 - Regulatory change monitoring not automated
- #C43 - Compliance report templates outdated

## Classes

### RegulatorType

**Inherits from**: Enum

Types of financial regulators.

### RegulatoryDomain

**Inherits from**: Enum

Regulatory domains/categories.

### ChangeImpact

**Inherits from**: Enum

Impact level of regulatory changes.

### ChangeStatus

**Inherits from**: Enum

Status of regulatory change tracking.

### RegulatoryChange

Represents a regulatory change or update.

#### Methods

##### `def to_dict(self) -> Dict[str, Any]`

Convert to dictionary.

##### `def days_until_effective(self) -> int`

Calculate days until effective date.

##### `def is_overdue(self) -> bool`

Check if implementation is overdue.

### RegulatoryFeed

Configuration for a regulatory feed source.

### RegulatoryChangeMonitor

Monitors regulatory changes from various sources.

Provides automated tracking and alerting for regulatory updates
that may affect trading operations.

#### Methods

##### `def __init__(self)`

##### `def add_feed(self, feed: RegulatoryFeed)`

Add a regulatory feed source.

##### `def subscribe(self, domain: RegulatoryDomain, callback: Callable[, None])`

Subscribe to changes in a regulatory domain.

##### `def register_change(self, change: RegulatoryChange) -> bool`

Register a new regulatory change.

Returns True if this is a new change, False if duplicate.

##### `def update_status(self, change_id: str, status: ChangeStatus, notes: str)`

Update the status of a regulatory change.

##### `def get_pending_changes(self, impact_filter: Optional[ChangeImpact], domain_filter: Optional[RegulatoryDomain]) -> List[RegulatoryChange]`

Get list of pending regulatory changes.

##### `def get_overdue_changes(self) -> List[RegulatoryChange]`

Get list of overdue regulatory changes.

##### `def get_change_summary(self) -> Dict[str, Any]`

Get summary of all tracked regulatory changes.

### ReportFormat

**Inherits from**: Enum

Report output formats.

### ReportSection

Section within a compliance report.

#### Methods

##### `def to_dict(self) -> Dict[str, Any]`

Convert to dictionary.

### ComplianceReportTemplate

Template for compliance reports.

#### Methods

##### `def to_dict(self) -> Dict[str, Any]`

Convert to dictionary.

### ComplianceReportGenerator

Generates compliance reports using standardized templates.

Provides templates for various EU regulatory requirements including
MiFID II, EMIR, MAR, and internal compliance reporting.

#### Methods

##### `def __init__(self)`

##### `def get_template(self, template_id: str) -> Optional[ComplianceReportTemplate]`

Get a report template by ID.

##### `def list_templates(self, regulation_filter: Optional[str], frequency_filter: Optional[str]) -> List[ComplianceReportTemplate]`

List available report templates.

##### `def generate_report(self, template_id: str, data: Dict[str, Any], output_format: ReportFormat, report_date: Optional[datetime]) -> Dict[str, Any]`

Generate a compliance report from a template.

Args:
    template_id: ID of the template to use
    data: Data to populate the report
    output_format: Desired output format
    report_date: Date for the report (defaults to today)

Returns:
    Generated report content

##### `def export_template_catalog(self) -> str`

Export catalog of all templates as JSON.

### RegulatoryComplianceSystem

Unified system for regulatory change monitoring and compliance reporting.

Combines change monitoring with report generation for comprehensive
regulatory compliance management.

#### Methods

##### `def __init__(self)`

##### `def get_compliance_status(self) -> Dict[str, Any]`

Get overall compliance status.

##### `def generate_compliance_dashboard_data(self) -> Dict[str, Any]`

Generate data for compliance dashboard.

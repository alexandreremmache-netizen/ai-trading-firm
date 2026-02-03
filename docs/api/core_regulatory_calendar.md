# regulatory_calendar

**Path**: `C:\Users\Alexa\ai-trading-firm\core\regulatory_calendar.py`

## Overview

Regulatory Calendar Module
==========================

Regulatory reporting calendar maintenance (Issue #C34).
Compliance training records tracking (Issue #C40).
Gift and entertainment logging (Issue #C39).

Features:
- EU/AMF regulatory deadline tracking
- Automated reminder generation
- Compliance training management
- Gift/entertainment log with thresholds

## Classes

### ReportType

**Inherits from**: str, Enum

Types of regulatory reports.

### ReportFrequency

**Inherits from**: str, Enum

Report submission frequency.

### TrainingStatus

**Inherits from**: str, Enum

Training completion status.

### RegulatoryDeadline

Regulatory reporting deadline.

#### Methods

##### `def to_dict(self) -> dict`

### ScheduledReport

Instance of a scheduled report.

#### Methods

##### `def to_dict(self) -> dict`

### ComplianceTraining

Compliance training record (#C40).

#### Methods

##### `def to_dict(self) -> dict`

### TrainingRecord

Individual training completion record.

#### Methods

##### `def to_dict(self) -> dict`

### GiftEntertainmentEntry

Gift and entertainment log entry (#C39).

#### Methods

##### `def to_dict(self) -> dict`

### RegulatoryCalendar

Manages regulatory reporting deadlines (#C34).

Tracks all EU/AMF regulatory reporting requirements.

#### Methods

##### `def __init__(self, notification_callback: )`

##### `def add_custom_deadline(self, deadline: RegulatoryDeadline) -> None`

Add a custom regulatory deadline.

##### `def schedule_report(self, report_type: ReportType, period_start: date, period_end: date, deadline: datetime) -> ScheduledReport`

Schedule a specific report instance.

##### `def generate_schedule(self, start_date: date, end_date: date) -> list[ScheduledReport]`

Generate schedule for all reports in date range.

##### `def check_upcoming_deadlines(self, days_ahead: int) -> list[dict]`

Check for upcoming deadlines and generate alerts.

##### `def mark_submitted(self, report_id: str, confirmation_number: str) -> bool`

Mark a report as submitted.

##### `def get_calendar_view(self, month: int, year: int) -> dict`

Get calendar view of deadlines for a month.

##### `def get_compliance_status(self) -> dict`

Get overall compliance status.

### ComplianceTrainingManager

Manages compliance training records (#C40).

Tracks mandatory training completion and expiry.

#### Methods

##### `def __init__(self)`

##### `def add_training(self, training: ComplianceTraining) -> None`

Add a training course.

##### `def assign_training(self, training_id: str, employee_id: str, employee_name: str, due_date: date)`

Assign training to an employee.

##### `def complete_training(self, record_id: str, score: , passed: bool) -> bool`

Record training completion.

##### `def check_overdue(self) -> list[TrainingRecord]`

Find overdue training assignments.

##### `def get_employee_status(self, employee_id: str) -> dict`

Get training status for an employee.

##### `def get_overall_compliance(self) -> dict`

Get overall training compliance metrics.

### GiftEntertainmentLog

Gift and entertainment logging (#C39).

Tracks gifts/entertainment with approval workflows.

#### Methods

##### `def __init__(self, approval_threshold_eur: float, annual_limit_eur: float)`

##### `def log_entry(self, entry_type: str, employee_id: str, employee_name: str, counterparty_name: str, counterparty_company: str, description: str, value_eur: float, event_date: date, location: str) -> GiftEntertainmentEntry`

Log a gift/entertainment entry.

##### `def approve_entry(self, entry_id: str, approved_by: str) -> bool`

Approve a gift/entertainment entry.

##### `def get_employee_annual_total(self, employee_id: str, year: int) -> float`

Get total G&E value for employee in a year.

##### `def get_pending_approvals(self) -> list[GiftEntertainmentEntry]`

Get entries pending approval.

##### `def get_summary(self, year: ) -> dict`

Get G&E summary.

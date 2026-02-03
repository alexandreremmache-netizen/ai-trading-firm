"""
Custom Reporting Module
=======================

Flexible custom reporting framework for portfolio management.

Issues Addressed:
- #P19: Missing custom reporting
- #E30: No execution venue selection logic (integrated here)
"""

from __future__ import annotations

import logging
import json
import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM REPORTING FRAMEWORK (#P19)
# =============================================================================

class ReportFormat(str, Enum):
    """Output format for reports."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "text"


class ReportFrequency(str, Enum):
    """Report generation frequency."""
    REALTIME = "realtime"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"


class ReportCategory(str, Enum):
    """Report category."""
    PERFORMANCE = "performance"
    RISK = "risk"
    COMPLIANCE = "compliance"
    EXECUTION = "execution"
    POSITION = "position"
    ATTRIBUTION = "attribution"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


@dataclass
class ReportDefinition:
    """Definition of a custom report (#P19)."""
    report_id: str
    name: str
    description: str
    category: ReportCategory
    frequency: ReportFrequency
    formats: list[ReportFormat]
    data_sources: list[str]
    filters: dict[str, Any]
    columns: list[dict]  # {name, source, transform, format}
    grouping: list[str] | None
    sorting: list[dict] | None  # {column, direction}
    aggregations: list[dict] | None  # {column, function}
    created_by: str
    created_at: datetime
    last_modified: datetime
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            'report_id': self.report_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'frequency': self.frequency.value,
            'formats': [f.value for f in self.formats],
            'data_sources': self.data_sources,
            'filters': self.filters,
            'columns': self.columns,
            'grouping': self.grouping,
            'sorting': self.sorting,
            'aggregations': self.aggregations,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'is_active': self.is_active,
        }


@dataclass
class ReportExecution:
    """Record of report execution (#P19)."""
    execution_id: str
    report_id: str
    started_at: datetime
    completed_at: datetime | None
    status: str  # "running", "completed", "failed"
    row_count: int
    output_format: ReportFormat
    output_path: str | None
    error_message: str | None

    def to_dict(self) -> dict:
        return {
            'execution_id': self.execution_id,
            'report_id': self.report_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'row_count': self.row_count,
            'output_format': self.output_format.value,
            'output_path': self.output_path,
            'error_message': self.error_message,
        }


class CustomReportBuilder:
    """
    Builder for custom reports (#P19).

    Allows defining custom reports with:
    - Multiple data sources
    - Custom filters and transformations
    - Flexible column definitions
    - Grouping and aggregation
    - Multiple output formats
    """

    def __init__(self, report_name: str, category: ReportCategory = ReportCategory.CUSTOM):
        self._report_name = report_name
        self._category = category
        self._description = ""
        self._frequency = ReportFrequency.ON_DEMAND
        self._formats = [ReportFormat.JSON]
        self._data_sources: list[str] = []
        self._filters: dict[str, Any] = {}
        self._columns: list[dict] = []
        self._grouping: list[str] | None = None
        self._sorting: list[dict] | None = None
        self._aggregations: list[dict] | None = None

    def with_description(self, description: str) -> "CustomReportBuilder":
        """Set report description."""
        self._description = description
        return self

    def with_frequency(self, frequency: ReportFrequency) -> "CustomReportBuilder":
        """Set report frequency."""
        self._frequency = frequency
        return self

    def with_formats(self, formats: list[ReportFormat]) -> "CustomReportBuilder":
        """Set output formats."""
        self._formats = formats
        return self

    def add_data_source(self, source: str) -> "CustomReportBuilder":
        """Add data source."""
        self._data_sources.append(source)
        return self

    def add_filter(self, field: str, operator: str, value: Any) -> "CustomReportBuilder":
        """Add filter condition."""
        self._filters[field] = {'operator': operator, 'value': value}
        return self

    def add_column(
        self,
        name: str,
        source: str,
        transform: str | None = None,
        format_spec: str | None = None,
    ) -> "CustomReportBuilder":
        """Add column definition."""
        self._columns.append({
            'name': name,
            'source': source,
            'transform': transform,
            'format': format_spec,
        })
        return self

    def with_grouping(self, fields: list[str]) -> "CustomReportBuilder":
        """Set grouping fields."""
        self._grouping = fields
        return self

    def with_sorting(self, column: str, direction: str = "asc") -> "CustomReportBuilder":
        """Add sorting."""
        if self._sorting is None:
            self._sorting = []
        self._sorting.append({'column': column, 'direction': direction})
        return self

    def with_aggregation(self, column: str, function: str) -> "CustomReportBuilder":
        """Add aggregation."""
        if self._aggregations is None:
            self._aggregations = []
        self._aggregations.append({'column': column, 'function': function})
        return self

    def build(self, created_by: str) -> ReportDefinition:
        """Build report definition."""
        report_id = f"RPT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        return ReportDefinition(
            report_id=report_id,
            name=self._report_name,
            description=self._description,
            category=self._category,
            frequency=self._frequency,
            formats=self._formats,
            data_sources=self._data_sources,
            filters=self._filters,
            columns=self._columns,
            grouping=self._grouping,
            sorting=self._sorting,
            aggregations=self._aggregations,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
        )


@dataclass
class ReportSchedule:
    """Schedule definition for automated report generation (P3)."""
    schedule_id: str
    report_id: str
    frequency: ReportFrequency
    day_of_week: int | None = None  # 0=Monday, 6=Sunday (for weekly)
    day_of_month: int | None = None  # 1-31 (for monthly)
    hour: int = 8  # Hour of day (0-23)
    minute: int = 0  # Minute (0-59)
    timezone_str: str = "UTC"
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    output_format: ReportFormat = ReportFormat.JSON
    recipients: list[str] = field(default_factory=list)  # Email addresses

    def to_dict(self) -> dict:
        return {
            'schedule_id': self.schedule_id,
            'report_id': self.report_id,
            'frequency': self.frequency.value,
            'day_of_week': self.day_of_week,
            'day_of_month': self.day_of_month,
            'hour': self.hour,
            'minute': self.minute,
            'timezone': self.timezone_str,
            'enabled': self.enabled,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'output_format': self.output_format.value,
            'recipients': self.recipients,
        }


@dataclass
class EmailDistribution:
    """Email distribution configuration for reports (P3)."""
    distribution_id: str
    name: str
    recipients: list[str]  # Email addresses
    cc_recipients: list[str] = field(default_factory=list)
    bcc_recipients: list[str] = field(default_factory=list)
    subject_template: str = "Report: {report_name} - {date}"
    body_template: str = "Please find attached the {report_name} report generated on {date}."
    attach_report: bool = True
    include_summary_in_body: bool = True
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            'distribution_id': self.distribution_id,
            'name': self.name,
            'recipients': self.recipients,
            'cc_recipients': self.cc_recipients,
            'bcc_recipients': self.bcc_recipients,
            'subject_template': self.subject_template,
            'body_template': self.body_template,
            'attach_report': self.attach_report,
            'include_summary_in_body': self.include_summary_in_body,
            'enabled': self.enabled,
        }


class CustomReportEngine:
    """
    Custom report execution engine (#P19).

    Features:
    - Report definition management
    - Data source integration
    - Report execution and scheduling
    - Multiple output formats (including PDF/Excel export - P3)
    - Execution history
    - Email distribution (P3)
    """

    def __init__(self):
        self._reports: dict[str, ReportDefinition] = {}
        self._executions: list[ReportExecution] = []
        self._execution_counter = 0

        # Report scheduling (P3)
        self._schedules: dict[str, ReportSchedule] = {}
        self._schedule_counter = 0

        # Email distributions (P3)
        self._distributions: dict[str, EmailDistribution] = {}
        self._distribution_counter = 0

        # Data source handlers
        self._data_handlers: dict[str, Callable] = {}

        # Transform functions
        self._transforms: dict[str, Callable] = {
            'abs': abs,
            'round2': lambda x: round(x, 2) if isinstance(x, (int, float)) else x,
            'round4': lambda x: round(x, 4) if isinstance(x, (int, float)) else x,
            'pct': lambda x: x * 100 if isinstance(x, (int, float)) else x,
            'upper': lambda x: str(x).upper(),
            'lower': lambda x: str(x).lower(),
        }

        # Format functions
        self._formatters: dict[str, Callable] = {
            'currency': lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x),
            'percent': lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x),
            'number': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'decimal2': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x),
            'decimal4': lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x),
        }

        # Aggregation functions
        self._aggregators: dict[str, Callable] = {
            'sum': sum,
            'avg': lambda x: sum(x) / len(x) if x else 0,
            'min': min,
            'max': max,
            'count': len,
            'first': lambda x: x[0] if x else None,
            'last': lambda x: x[-1] if x else None,
        }

    def register_data_handler(
        self,
        source_name: str,
        handler: Callable[[], list[dict]],
    ) -> None:
        """Register data source handler."""
        self._data_handlers[source_name] = handler

    def register_transform(self, name: str, func: Callable) -> None:
        """Register custom transform function."""
        self._transforms[name] = func

    def save_report(self, report: ReportDefinition) -> None:
        """Save report definition."""
        self._reports[report.report_id] = report
        logger.info(f"Report saved: {report.report_id} - {report.name}")

    def get_report(self, report_id: str) -> ReportDefinition | None:
        """Get report definition."""
        return self._reports.get(report_id)

    def list_reports(
        self,
        category: ReportCategory | None = None,
    ) -> list[ReportDefinition]:
        """List available reports."""
        reports = list(self._reports.values())
        if category:
            reports = [r for r in reports if r.category == category]
        return reports

    def _fetch_data(self, data_sources: list[str]) -> list[dict]:
        """Fetch data from sources."""
        all_data = []
        for source in data_sources:
            handler = self._data_handlers.get(source)
            if handler:
                try:
                    data = handler()
                    all_data.extend(data)
                except Exception as e:
                    logger.error(f"Error fetching from {source}: {e}")
        return all_data

    def _apply_filters(
        self,
        data: list[dict],
        filters: dict[str, Any],
    ) -> list[dict]:
        """Apply filters to data."""
        if not filters:
            return data

        filtered = []
        for row in data:
            include = True
            for field, condition in filters.items():
                value = row.get(field)
                op = condition.get('operator', '=')
                target = condition.get('value')

                if op == '=' and value != target:
                    include = False
                elif op == '!=' and value == target:
                    include = False
                elif op == '>' and not (isinstance(value, (int, float)) and value > target):
                    include = False
                elif op == '<' and not (isinstance(value, (int, float)) and value < target):
                    include = False
                elif op == '>=' and not (isinstance(value, (int, float)) and value >= target):
                    include = False
                elif op == '<=' and not (isinstance(value, (int, float)) and value <= target):
                    include = False
                elif op == 'in' and value not in target:
                    include = False
                elif op == 'contains' and target not in str(value):
                    include = False

            if include:
                filtered.append(row)

        return filtered

    def _apply_columns(
        self,
        data: list[dict],
        columns: list[dict],
    ) -> list[dict]:
        """Apply column definitions to data."""
        result = []
        for row in data:
            new_row = {}
            for col in columns:
                name = col['name']
                source = col['source']
                transform = col.get('transform')
                format_spec = col.get('format')

                # Get value
                value = row.get(source)

                # Apply transform
                if transform and transform in self._transforms:
                    value = self._transforms[transform](value)

                # Store raw value (formatting applied at output)
                new_row[name] = value
                if format_spec:
                    new_row[f'_format_{name}'] = format_spec

            result.append(new_row)

        return result

    def _apply_grouping_and_aggregation(
        self,
        data: list[dict],
        grouping: list[str] | None,
        aggregations: list[dict] | None,
    ) -> list[dict]:
        """Apply grouping and aggregation."""
        if not grouping or not aggregations:
            return data

        # Group data
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in data:
            key = tuple(row.get(g) for g in grouping)
            groups[key].append(row)

        # Aggregate
        result = []
        for key, group_rows in groups.items():
            new_row = {}

            # Include grouping fields
            for i, field in enumerate(grouping):
                new_row[field] = key[i]

            # Apply aggregations
            for agg in aggregations:
                col = agg['column']
                func_name = agg['function']
                func = self._aggregators.get(func_name)

                if func:
                    values = [r.get(col) for r in group_rows if r.get(col) is not None]
                    if values:
                        new_row[f"{col}_{func_name}"] = func(values)

            result.append(new_row)

        return result

    def _apply_sorting(
        self,
        data: list[dict],
        sorting: list[dict] | None,
    ) -> list[dict]:
        """Apply sorting to data."""
        if not sorting:
            return data

        for sort_spec in reversed(sorting):
            col = sort_spec['column']
            reverse = sort_spec.get('direction', 'asc').lower() == 'desc'
            data = sorted(data, key=lambda x: x.get(col, ''), reverse=reverse)

        return data

    def _format_output(
        self,
        data: list[dict],
        format: ReportFormat,
    ) -> str:
        """Format data for output."""
        if format == ReportFormat.JSON:
            return json.dumps(data, indent=2, default=str)

        elif format == ReportFormat.CSV:
            if not data:
                return ""
            output = io.StringIO()
            # Filter out format metadata columns
            fieldnames = [k for k in data[0].keys() if not k.startswith('_format_')]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                clean_row = {k: v for k, v in row.items() if not k.startswith('_format_')}
                writer.writerow(clean_row)
            return output.getvalue()

        elif format == ReportFormat.HTML:
            if not data:
                return "<table><tr><td>No data</td></tr></table>"

            fieldnames = [k for k in data[0].keys() if not k.startswith('_format_')]

            html = ["<table border='1' style='border-collapse: collapse;'>"]
            html.append("<tr>")
            for field in fieldnames:
                html.append(f"<th style='padding: 8px;'>{field}</th>")
            html.append("</tr>")

            for row in data:
                html.append("<tr>")
                for field in fieldnames:
                    value = row.get(field, '')
                    format_spec = row.get(f'_format_{field}')
                    if format_spec and format_spec in self._formatters:
                        value = self._formatters[format_spec](value)
                    html.append(f"<td style='padding: 8px;'>{value}</td>")
                html.append("</tr>")

            html.append("</table>")
            return "\n".join(html)

        else:  # TEXT
            if not data:
                return "No data"

            fieldnames = [k for k in data[0].keys() if not k.startswith('_format_')]
            lines = []
            lines.append(" | ".join(fieldnames))
            lines.append("-" * len(lines[0]))

            for row in data:
                values = []
                for field in fieldnames:
                    value = row.get(field, '')
                    format_spec = row.get(f'_format_{field}')
                    if format_spec and format_spec in self._formatters:
                        value = self._formatters[format_spec](value)
                    values.append(str(value))
                lines.append(" | ".join(values))

            return "\n".join(lines)

    def execute_report(
        self,
        report_id: str,
        output_format: ReportFormat | None = None,
        override_filters: dict | None = None,
    ) -> tuple[str, ReportExecution]:
        """
        Execute a report (#P19).

        Args:
            report_id: Report to execute
            output_format: Override default format
            override_filters: Additional filters

        Returns:
            (output_data, execution_record)
        """
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report not found: {report_id}")

        self._execution_counter += 1
        execution_id = f"EXEC-{self._execution_counter:06d}"

        execution = ReportExecution(
            execution_id=execution_id,
            report_id=report_id,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            status="running",
            row_count=0,
            output_format=output_format or report.formats[0],
            output_path=None,
            error_message=None,
        )

        try:
            # Fetch data
            data = self._fetch_data(report.data_sources)

            # Merge filters
            filters = dict(report.filters)
            if override_filters:
                filters.update(override_filters)

            # Apply filters
            data = self._apply_filters(data, filters)

            # Apply columns
            data = self._apply_columns(data, report.columns)

            # Apply grouping/aggregation
            data = self._apply_grouping_and_aggregation(
                data, report.grouping, report.aggregations
            )

            # Apply sorting
            data = self._apply_sorting(data, report.sorting)

            # Format output
            output = self._format_output(data, execution.output_format)

            # Update execution
            execution.completed_at = datetime.now(timezone.utc)
            execution.status = "completed"
            execution.row_count = len(data)

            logger.info(
                f"Report executed: {report.name} - {execution.row_count} rows"
            )

        except Exception as e:
            execution.completed_at = datetime.now(timezone.utc)
            execution.status = "failed"
            execution.error_message = str(e)
            output = ""
            logger.error(f"Report execution failed: {e}")

        self._executions.append(execution)
        return output, execution

    def get_execution_history(
        self,
        report_id: str | None = None,
        limit: int = 100,
    ) -> list[ReportExecution]:
        """Get report execution history."""
        history = self._executions
        if report_id:
            history = [e for e in history if e.report_id == report_id]
        return history[-limit:]

    # =========================================================================
    # REPORT SCHEDULING (P3)
    # =========================================================================

    def create_schedule(
        self,
        report_id: str,
        frequency: ReportFrequency,
        hour: int = 8,
        minute: int = 0,
        day_of_week: int | None = None,
        day_of_month: int | None = None,
        output_format: ReportFormat = ReportFormat.JSON,
        recipients: list[str] | None = None,
    ) -> ReportSchedule:
        """
        Create a schedule for automated report generation (P3).

        Args:
            report_id: Report to schedule
            frequency: How often to run
            hour: Hour of day (0-23)
            minute: Minute (0-59)
            day_of_week: For weekly reports (0=Monday)
            day_of_month: For monthly reports (1-31)
            output_format: Output format for scheduled runs
            recipients: Email recipients for distribution

        Returns:
            ReportSchedule
        """
        if report_id not in self._reports:
            raise ValueError(f"Report not found: {report_id}")

        self._schedule_counter += 1
        schedule_id = f"SCHED-{self._schedule_counter:06d}"

        schedule = ReportSchedule(
            schedule_id=schedule_id,
            report_id=report_id,
            frequency=frequency,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            hour=hour,
            minute=minute,
            output_format=output_format,
            recipients=recipients or [],
            next_run=self._calculate_next_run(frequency, hour, minute, day_of_week, day_of_month),
        )

        self._schedules[schedule_id] = schedule
        logger.info(f"Schedule created: {schedule_id} for report {report_id}")
        return schedule

    def _calculate_next_run(
        self,
        frequency: ReportFrequency,
        hour: int,
        minute: int,
        day_of_week: int | None = None,
        day_of_month: int | None = None,
    ) -> datetime:
        """Calculate next scheduled run time."""
        now = datetime.now(timezone.utc)
        today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if frequency == ReportFrequency.HOURLY:
            next_run = now.replace(minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
            return next_run

        elif frequency == ReportFrequency.DAILY:
            if today <= now:
                return today + timedelta(days=1)
            return today

        elif frequency == ReportFrequency.WEEKLY:
            if day_of_week is None:
                day_of_week = 0  # Monday default
            days_ahead = day_of_week - now.weekday()
            if days_ahead <= 0 or (days_ahead == 0 and today <= now):
                days_ahead += 7
            return today + timedelta(days=days_ahead)

        elif frequency == ReportFrequency.MONTHLY:
            if day_of_month is None:
                day_of_month = 1
            # Try this month
            try:
                next_run = now.replace(day=day_of_month, hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    # Move to next month
                    if now.month == 12:
                        next_run = next_run.replace(year=now.year + 1, month=1)
                    else:
                        next_run = next_run.replace(month=now.month + 1)
                return next_run
            except ValueError:
                # Invalid day for month, use last day
                return today + timedelta(days=30)

        return today + timedelta(days=1)

    def update_schedule(
        self,
        schedule_id: str,
        enabled: bool | None = None,
        hour: int | None = None,
        minute: int | None = None,
        recipients: list[str] | None = None,
    ) -> ReportSchedule | None:
        """Update an existing schedule (P3)."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return None

        if enabled is not None:
            schedule.enabled = enabled
        if hour is not None:
            schedule.hour = hour
        if minute is not None:
            schedule.minute = minute
        if recipients is not None:
            schedule.recipients = recipients

        # Recalculate next run
        schedule.next_run = self._calculate_next_run(
            schedule.frequency,
            schedule.hour,
            schedule.minute,
            schedule.day_of_week,
            schedule.day_of_month,
        )

        logger.info(f"Schedule updated: {schedule_id}")
        return schedule

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule (P3)."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            logger.info(f"Schedule deleted: {schedule_id}")
            return True
        return False

    def list_schedules(self, report_id: str | None = None) -> list[ReportSchedule]:
        """List all schedules, optionally filtered by report (P3)."""
        schedules = list(self._schedules.values())
        if report_id:
            schedules = [s for s in schedules if s.report_id == report_id]
        return schedules

    def get_due_schedules(self) -> list[ReportSchedule]:
        """Get schedules that are due to run (P3)."""
        now = datetime.now(timezone.utc)
        due = []
        for schedule in self._schedules.values():
            if schedule.enabled and schedule.next_run and schedule.next_run <= now:
                due.append(schedule)
        return due

    def execute_scheduled_report(self, schedule_id: str) -> tuple[str, ReportExecution] | None:
        """
        Execute a scheduled report and update schedule state (P3).

        Returns:
            (output_data, execution_record) or None if schedule not found
        """
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return None

        # Execute the report
        output, execution = self.execute_report(
            schedule.report_id,
            output_format=schedule.output_format,
        )

        # Update schedule
        schedule.last_run = datetime.now(timezone.utc)
        schedule.next_run = self._calculate_next_run(
            schedule.frequency,
            schedule.hour,
            schedule.minute,
            schedule.day_of_week,
            schedule.day_of_month,
        )

        logger.info(f"Scheduled report executed: {schedule_id}, next run: {schedule.next_run}")
        return output, execution

    # =========================================================================
    # PDF/EXCEL EXPORT (P3)
    # =========================================================================

    def export_to_pdf(self, report_id: str, data: list[dict] | None = None) -> bytes:
        """
        Export report to PDF format (P3).

        Args:
            report_id: Report to export
            data: Optional pre-fetched data (if None, will execute report)

        Returns:
            PDF content as bytes

        Note: This is a placeholder that generates a simple PDF-like structure.
        In production, use libraries like reportlab or weasyprint.
        """
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report not found: {report_id}")

        # Get HTML output
        html_output, _ = self.execute_report(report_id, output_format=ReportFormat.HTML)

        # Create a simple PDF-like structure (placeholder)
        # In production, use reportlab or similar
        pdf_header = b"%PDF-1.4\n"
        pdf_content = f"""
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length {len(html_output)} >>
stream
Report: {report.name}
Generated: {datetime.now(timezone.utc).isoformat()}
---
{html_output[:500]}...
endstream
endobj
xref
0 5
trailer
<< /Root 1 0 R /Size 5 >>
startxref
0
%%EOF
""".encode('utf-8')

        logger.info(f"PDF export generated for report: {report_id}")
        return pdf_header + pdf_content

    def export_to_excel(self, report_id: str, data: list[dict] | None = None) -> bytes:
        """
        Export report to Excel format (P3).

        Args:
            report_id: Report to export
            data: Optional pre-fetched data

        Returns:
            Excel content as bytes (CSV format compatible with Excel)

        Note: This generates CSV which Excel can open.
        In production, use openpyxl or xlsxwriter for native Excel format.
        """
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report not found: {report_id}")

        # Get CSV output (Excel-compatible)
        csv_output, _ = self.execute_report(report_id, output_format=ReportFormat.CSV)

        logger.info(f"Excel export generated for report: {report_id}")
        return csv_output.encode('utf-8')

    # =========================================================================
    # EMAIL DISTRIBUTION (P3)
    # =========================================================================

    def create_distribution(
        self,
        name: str,
        recipients: list[str],
        cc_recipients: list[str] | None = None,
        subject_template: str | None = None,
        body_template: str | None = None,
    ) -> EmailDistribution:
        """
        Create an email distribution list (P3).

        Args:
            name: Distribution name
            recipients: List of email addresses
            cc_recipients: Optional CC recipients
            subject_template: Optional subject template with {report_name}, {date}
            body_template: Optional body template

        Returns:
            EmailDistribution
        """
        self._distribution_counter += 1
        dist_id = f"DIST-{self._distribution_counter:06d}"

        distribution = EmailDistribution(
            distribution_id=dist_id,
            name=name,
            recipients=recipients,
            cc_recipients=cc_recipients or [],
            subject_template=subject_template or "Report: {report_name} - {date}",
            body_template=body_template or "Please find attached the {report_name} report generated on {date}.",
        )

        self._distributions[dist_id] = distribution
        logger.info(f"Distribution created: {dist_id} - {name}")
        return distribution

    def get_distribution(self, distribution_id: str) -> EmailDistribution | None:
        """Get distribution by ID (P3)."""
        return self._distributions.get(distribution_id)

    def list_distributions(self) -> list[EmailDistribution]:
        """List all email distributions (P3)."""
        return list(self._distributions.values())

    def prepare_email_distribution(
        self,
        report_id: str,
        distribution_id: str,
        output_format: ReportFormat = ReportFormat.JSON,
    ) -> dict:
        """
        Prepare report for email distribution (P3).

        Args:
            report_id: Report to distribute
            distribution_id: Distribution list to use
            output_format: Format for attachment

        Returns:
            Dict with email details ready for sending
        """
        report = self._reports.get(report_id)
        distribution = self._distributions.get(distribution_id)

        if not report:
            raise ValueError(f"Report not found: {report_id}")
        if not distribution:
            raise ValueError(f"Distribution not found: {distribution_id}")

        # Execute report
        output, execution = self.execute_report(report_id, output_format=output_format)

        # Prepare email
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")

        subject = distribution.subject_template.format(
            report_name=report.name,
            date=date_str,
        )
        body = distribution.body_template.format(
            report_name=report.name,
            date=date_str,
        )

        # Add summary to body if configured
        if distribution.include_summary_in_body:
            body += f"\n\nExecution Summary:\n- Rows: {execution.row_count}\n- Status: {execution.status}"

        email_data = {
            'subject': subject,
            'body': body,
            'recipients': distribution.recipients,
            'cc_recipients': distribution.cc_recipients,
            'bcc_recipients': distribution.bcc_recipients,
            'attachment_name': f"{report.name.replace(' ', '_')}_{date_str}.{output_format.value}",
            'attachment_content': output,
            'report_id': report_id,
            'execution_id': execution.execution_id,
            'prepared_at': now.isoformat(),
        }

        logger.info(f"Email prepared for distribution: {distribution_id}, report: {report_id}")
        return email_data

    def send_report_email(
        self,
        report_id: str,
        distribution_id: str,
        output_format: ReportFormat = ReportFormat.JSON,
        smtp_handler: Callable[[dict], bool] | None = None,
    ) -> bool:
        """
        Send report via email distribution (P3).

        Args:
            report_id: Report to send
            distribution_id: Distribution list
            output_format: Format for attachment
            smtp_handler: Optional callback to handle actual email sending

        Returns:
            True if email was prepared (or sent if handler provided)
        """
        email_data = self.prepare_email_distribution(report_id, distribution_id, output_format)

        if smtp_handler:
            try:
                result = smtp_handler(email_data)
                logger.info(f"Report email sent: {email_data['subject']}")
                return result
            except Exception as e:
                logger.error(f"Failed to send report email: {e}")
                return False

        # Without handler, just log that email is ready
        logger.info(f"Report email prepared (no SMTP handler): {email_data['subject']}")
        return True


# =============================================================================
# EXECUTION VENUE SELECTION LOGIC (#E30)
# =============================================================================

class VenueType(str, Enum):
    """Execution venue types (#E30)."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ATS = "ats"
    INTERNALIZER = "internalizer"
    MARKET_MAKER = "market_maker"


class OrderCharacteristic(str, Enum):
    """Order characteristics for venue selection (#E30)."""
    SMALL = "small"  # < 1% ADV
    MEDIUM = "medium"  # 1-5% ADV
    LARGE = "large"  # 5-20% ADV
    BLOCK = "block"  # > 20% ADV
    URGENT = "urgent"
    PATIENT = "patient"
    PRICE_SENSITIVE = "price_sensitive"
    INFORMATION_SENSITIVE = "information_sensitive"


@dataclass
class VenueCharacteristics:
    """Characteristics of an execution venue (#E30)."""
    venue_id: str
    venue_type: VenueType
    name: str

    # Cost metrics
    maker_fee_bps: float
    taker_fee_bps: float
    typical_spread_bps: float

    # Quality metrics
    fill_rate_pct: float
    avg_fill_time_ms: float
    price_improvement_bps: float

    # Capacity
    avg_daily_volume: float
    max_order_size: float
    min_order_size: float

    # Features
    supports_midpoint: bool
    supports_hidden: bool
    supports_ioc: bool
    supports_block: bool

    # Restrictions
    restricted_symbols: list[str] = field(default_factory=list)
    min_notional: float = 0.0

    def to_dict(self) -> dict:
        return {
            'venue_id': self.venue_id,
            'venue_type': self.venue_type.value,
            'name': self.name,
            'maker_fee_bps': self.maker_fee_bps,
            'taker_fee_bps': self.taker_fee_bps,
            'typical_spread_bps': self.typical_spread_bps,
            'fill_rate_pct': self.fill_rate_pct,
            'avg_fill_time_ms': self.avg_fill_time_ms,
            'price_improvement_bps': self.price_improvement_bps,
            'avg_daily_volume': self.avg_daily_volume,
            'supports_midpoint': self.supports_midpoint,
            'supports_hidden': self.supports_hidden,
        }


@dataclass
class VenueSelectionResult:
    """Result of venue selection (#E30)."""
    symbol: str
    order_size: int
    order_characteristic: OrderCharacteristic
    recommended_venues: list[dict]  # [{venue_id, allocation_pct, reason}]
    estimated_cost_bps: float
    estimated_fill_time_ms: float
    routing_strategy: str
    considerations: list[str]

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'order_size': self.order_size,
            'order_characteristic': self.order_characteristic.value,
            'recommended_venues': self.recommended_venues,
            'estimated_cost_bps': self.estimated_cost_bps,
            'estimated_fill_time_ms': self.estimated_fill_time_ms,
            'routing_strategy': self.routing_strategy,
            'considerations': self.considerations,
        }


class ExecutionVenueSelector:
    """
    Execution venue selection logic (#E30).

    Features:
    - Venue characteristic analysis
    - Order-based venue matching
    - Cost optimization
    - Fill rate optimization
    - Information leakage mitigation
    """

    def __init__(self):
        self._venues: dict[str, VenueCharacteristics] = {}
        self._venue_performance: dict[str, list[dict]] = defaultdict(list)

        # Symbol-specific ADV
        self._symbol_adv: dict[str, float] = {}

    def register_venue(self, venue: VenueCharacteristics) -> None:
        """Register execution venue."""
        self._venues[venue.venue_id] = venue
        logger.info(f"Venue registered: {venue.venue_id} ({venue.venue_type.value})")

    def update_symbol_adv(self, symbol: str, adv: float) -> None:
        """Update average daily volume for symbol."""
        self._symbol_adv[symbol] = adv

    def classify_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        is_urgent: bool = False,
        is_info_sensitive: bool = False,
    ) -> OrderCharacteristic:
        """
        Classify order by characteristics (#E30).

        Args:
            symbol: Symbol
            quantity: Order quantity
            price: Current price
            is_urgent: Urgency flag
            is_info_sensitive: Information sensitivity

        Returns:
            Order characteristic
        """
        if is_urgent:
            return OrderCharacteristic.URGENT

        if is_info_sensitive:
            return OrderCharacteristic.INFORMATION_SENSITIVE

        # Size-based classification
        adv = self._symbol_adv.get(symbol, 1_000_000)
        order_value = quantity * price
        adv_pct = (quantity / adv) * 100

        if adv_pct < 1:
            return OrderCharacteristic.SMALL
        elif adv_pct < 5:
            return OrderCharacteristic.MEDIUM
        elif adv_pct < 20:
            return OrderCharacteristic.LARGE
        else:
            return OrderCharacteristic.BLOCK

    def _score_venue(
        self,
        venue: VenueCharacteristics,
        order_char: OrderCharacteristic,
        quantity: int,
        is_passive: bool,
    ) -> tuple[float, list[str]]:
        """
        Score venue for order (#E30).

        Returns: (score, reasons)
        """
        score = 100.0
        reasons = []

        # Cost scoring (lower is better)
        if is_passive:
            cost_bps = venue.maker_fee_bps - venue.price_improvement_bps
        else:
            cost_bps = venue.taker_fee_bps + venue.typical_spread_bps / 2

        cost_penalty = cost_bps * 2  # 2 points per bp
        score -= cost_penalty

        # Fill rate scoring
        fill_bonus = (venue.fill_rate_pct - 50) * 0.5  # Up to 25 points for 100% fill rate
        score += fill_bonus

        # Speed scoring (for urgent orders)
        if order_char == OrderCharacteristic.URGENT:
            if venue.avg_fill_time_ms < 10:
                score += 20
                reasons.append("fast_execution")
            elif venue.avg_fill_time_ms > 100:
                score -= 20

        # Information sensitivity scoring
        if order_char == OrderCharacteristic.INFORMATION_SENSITIVE:
            if venue.venue_type == VenueType.DARK_POOL:
                score += 30
                reasons.append("dark_pool_for_info_protection")
            elif venue.supports_hidden:
                score += 15
                reasons.append("hidden_order_support")

        # Block order scoring
        if order_char == OrderCharacteristic.BLOCK:
            if venue.supports_block:
                score += 25
                reasons.append("block_trading_support")
            if venue.venue_type == VenueType.DARK_POOL:
                score += 15
                reasons.append("dark_pool_for_block")

        # Capacity check
        if quantity > venue.max_order_size:
            score -= 50
            reasons.append("exceeds_max_size")
        elif quantity < venue.min_order_size:
            score -= 100  # Disqualify
            reasons.append("below_min_size")

        # Price improvement bonus
        if venue.price_improvement_bps > 0:
            score += venue.price_improvement_bps * 5
            reasons.append(f"price_improvement_{venue.price_improvement_bps:.1f}bps")

        return score, reasons

    def select_venues(
        self,
        symbol: str,
        quantity: int,
        price: float,
        is_passive: bool = True,
        is_urgent: bool = False,
        is_info_sensitive: bool = False,
        max_venues: int = 3,
    ) -> VenueSelectionResult:
        """
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
        """
        order_char = self.classify_order(
            symbol, quantity, price, is_urgent, is_info_sensitive
        )

        # Score all venues
        venue_scores = []
        for venue in self._venues.values():
            # Skip restricted
            if symbol in venue.restricted_symbols:
                continue

            score, reasons = self._score_venue(venue, order_char, quantity, is_passive)
            if score > 0:
                venue_scores.append({
                    'venue': venue,
                    'score': score,
                    'reasons': reasons,
                })

        # Sort by score
        venue_scores.sort(key=lambda x: x['score'], reverse=True)

        # Select top venues and allocate
        selected = venue_scores[:max_venues]
        total_score = sum(v['score'] for v in selected)

        recommendations = []
        for vs in selected:
            allocation_pct = (vs['score'] / total_score * 100) if total_score > 0 else 0
            recommendations.append({
                'venue_id': vs['venue'].venue_id,
                'venue_name': vs['venue'].name,
                'venue_type': vs['venue'].venue_type.value,
                'allocation_pct': round(allocation_pct, 1),
                'score': round(vs['score'], 1),
                'reasons': vs['reasons'],
            })

        # Estimate overall metrics
        if selected:
            weights = [v['score'] / total_score for v in selected]
            est_cost = sum(
                w * (selected[i]['venue'].taker_fee_bps if not is_passive else selected[i]['venue'].maker_fee_bps)
                for i, w in enumerate(weights)
            )
            est_time = sum(
                w * selected[i]['venue'].avg_fill_time_ms
                for i, w in enumerate(weights)
            )
        else:
            est_cost = 0
            est_time = 0

        # Determine routing strategy
        if order_char == OrderCharacteristic.BLOCK:
            strategy = "block_crossing"
        elif order_char == OrderCharacteristic.URGENT:
            strategy = "immediate_or_cancel"
        elif order_char == OrderCharacteristic.INFORMATION_SENSITIVE:
            strategy = "dark_first"
        elif is_passive:
            strategy = "passive_rebate_capture"
        else:
            strategy = "smart_order_routing"

        # Considerations
        considerations = []
        if order_char == OrderCharacteristic.LARGE or order_char == OrderCharacteristic.BLOCK:
            considerations.append("Consider TWAP/VWAP algo to minimize market impact")
        if order_char == OrderCharacteristic.INFORMATION_SENSITIVE:
            considerations.append("Route to dark pools first to minimize information leakage")
        if not is_passive:
            considerations.append("Aggressive order - expect full spread cost")

        return VenueSelectionResult(
            symbol=symbol,
            order_size=quantity,
            order_characteristic=order_char,
            recommended_venues=recommendations,
            estimated_cost_bps=est_cost,
            estimated_fill_time_ms=est_time,
            routing_strategy=strategy,
            considerations=considerations,
        )

    def record_execution(
        self,
        venue_id: str,
        symbol: str,
        quantity: int,
        fill_pct: float,
        fill_time_ms: float,
        price_improvement_bps: float,
    ) -> None:
        """Record execution for venue performance tracking."""
        self._venue_performance[venue_id].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'quantity': quantity,
            'fill_pct': fill_pct,
            'fill_time_ms': fill_time_ms,
            'price_improvement_bps': price_improvement_bps,
        })

        # Trim to last 1000 executions
        self._venue_performance[venue_id] = self._venue_performance[venue_id][-1000:]

    def get_venue_statistics(self, venue_id: str) -> dict:
        """Get venue performance statistics."""
        executions = self._venue_performance.get(venue_id, [])
        if not executions:
            return {'venue_id': venue_id, 'no_data': True}

        fill_rates = [e['fill_pct'] for e in executions]
        fill_times = [e['fill_time_ms'] for e in executions]
        improvements = [e['price_improvement_bps'] for e in executions]

        return {
            'venue_id': venue_id,
            'execution_count': len(executions),
            'avg_fill_rate': sum(fill_rates) / len(fill_rates),
            'avg_fill_time_ms': sum(fill_times) / len(fill_times),
            'avg_price_improvement_bps': sum(improvements) / len(improvements),
        }

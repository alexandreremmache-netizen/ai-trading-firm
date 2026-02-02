"""
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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Types of regulatory reports."""
    # MiFID II / MiFIR
    TRANSACTION_REPORT = "transaction_report"  # T+1
    RTS_27 = "rts_27_execution_quality"  # Quarterly
    RTS_28 = "rts_28_best_execution"  # Annual
    POSITION_REPORT = "position_report"  # Daily/Weekly

    # EMIR
    TRADE_REPOSITORY = "emir_trade_repository"  # T+1
    COLLATERAL_REPORT = "emir_collateral"  # Daily

    # MAR
    INSIDER_LIST = "mar_insider_list"  # As needed
    STOR = "stor_suspicious_transaction"  # ASAP

    # CSDR
    SETTLEMENT_DISCIPLINE = "csdr_settlement"  # Monthly

    # SFTR
    SECURITIES_FINANCING = "sftr"  # T+1

    # AMF Specific
    AMF_BDIF = "amf_bdif"  # Annual
    AMF_PSI = "amf_psi"  # Quarterly

    # General
    AML_CTF = "aml_ctf"  # As required
    KYC_REFRESH = "kyc_refresh"  # Periodic


class ReportFrequency(str, Enum):
    """Report submission frequency."""
    REAL_TIME = "real_time"
    DAILY = "daily"
    T_PLUS_1 = "t_plus_1"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    AD_HOC = "ad_hoc"


class TrainingStatus(str, Enum):
    """Training completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    OVERDUE = "overdue"


@dataclass
class RegulatoryDeadline:
    """Regulatory reporting deadline."""
    report_type: ReportType
    name: str
    description: str
    frequency: ReportFrequency

    # Deadline calculation
    deadline_rule: str  # e.g., "T+1", "15th of month", "last business day"
    deadline_time: str = "18:00"  # Local time
    timezone: str = "Europe/Paris"

    # Regulatory reference
    regulation: str = ""  # e.g., "MiFIR Art. 26"
    regulator: str = ""  # e.g., "AMF", "ESMA"

    # Alert settings
    warning_days: int = 3
    critical_days: int = 1

    # Submission details
    submission_portal: str = ""
    submission_format: str = ""

    def to_dict(self) -> dict:
        return {
            'report_type': self.report_type.value,
            'name': self.name,
            'description': self.description,
            'frequency': self.frequency.value,
            'deadline_rule': self.deadline_rule,
            'deadline_time': self.deadline_time,
            'regulation': self.regulation,
            'regulator': self.regulator,
            'warning_days': self.warning_days,
            'critical_days': self.critical_days,
        }


@dataclass
class ScheduledReport:
    """Instance of a scheduled report."""
    report_id: str
    report_type: ReportType
    period_start: date
    period_end: date
    deadline: datetime

    # Status
    status: str = "pending"  # pending, submitted, late, failed
    submitted_at: datetime | None = None
    confirmation_number: str = ""

    # Alerts sent
    warning_sent: bool = False
    critical_sent: bool = False

    def to_dict(self) -> dict:
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'deadline': self.deadline.isoformat(),
            'status': self.status,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'confirmation_number': self.confirmation_number,
        }


@dataclass
class ComplianceTraining:
    """Compliance training record (#C40)."""
    training_id: str
    title: str
    description: str
    category: str  # e.g., "MAR", "AML", "Best Execution"

    # Requirements
    mandatory: bool = True
    validity_months: int = 12
    target_audience: list[str] = field(default_factory=list)  # Roles

    # Regulatory basis
    regulation: str = ""

    def to_dict(self) -> dict:
        return {
            'training_id': self.training_id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'mandatory': self.mandatory,
            'validity_months': self.validity_months,
            'target_audience': self.target_audience,
            'regulation': self.regulation,
        }


@dataclass
class TrainingRecord:
    """Individual training completion record."""
    record_id: str
    training_id: str
    employee_id: str
    employee_name: str

    # Completion
    assigned_date: date
    due_date: date
    completed_date: date | None = None
    score: float | None = None  # Pass/fail score if applicable
    passed: bool = False

    # Status
    status: TrainingStatus = TrainingStatus.NOT_STARTED

    # Expiry
    expires_date: date | None = None

    def to_dict(self) -> dict:
        return {
            'record_id': self.record_id,
            'training_id': self.training_id,
            'employee_id': self.employee_id,
            'employee_name': self.employee_name,
            'assigned_date': self.assigned_date.isoformat(),
            'due_date': self.due_date.isoformat(),
            'completed_date': self.completed_date.isoformat() if self.completed_date else None,
            'score': self.score,
            'passed': self.passed,
            'status': self.status.value,
            'expires_date': self.expires_date.isoformat() if self.expires_date else None,
        }


@dataclass
class GiftEntertainmentEntry:
    """Gift and entertainment log entry (#C39)."""
    entry_id: str
    entry_type: str  # "gift_given", "gift_received", "entertainment_given", "entertainment_received"

    # Parties
    employee_id: str
    employee_name: str
    counterparty_name: str
    counterparty_company: str

    # Details
    description: str
    value_eur: float
    event_date: date
    location: str = ""

    # Approval
    requires_approval: bool = False
    approved: bool = False
    approved_by: str = ""
    approval_date: date | None = None

    # Compliance
    logged_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            'entry_id': self.entry_id,
            'entry_type': self.entry_type,
            'employee_id': self.employee_id,
            'employee_name': self.employee_name,
            'counterparty_name': self.counterparty_name,
            'counterparty_company': self.counterparty_company,
            'description': self.description,
            'value_eur': self.value_eur,
            'event_date': self.event_date.isoformat(),
            'location': self.location,
            'requires_approval': self.requires_approval,
            'approved': self.approved,
            'logged_at': self.logged_at.isoformat(),
        }


class RegulatoryCalendar:
    """
    Manages regulatory reporting deadlines (#C34).

    Tracks all EU/AMF regulatory reporting requirements.
    """

    # Standard EU regulatory deadlines
    STANDARD_DEADLINES = [
        RegulatoryDeadline(
            report_type=ReportType.TRANSACTION_REPORT,
            name="MiFIR Transaction Report",
            description="Art. 26 transaction reports to ARM",
            frequency=ReportFrequency.T_PLUS_1,
            deadline_rule="T+1 23:59",
            regulation="MiFIR Art. 26",
            regulator="ESMA/AMF",
            warning_days=0,  # Same day warning
            critical_days=0,
        ),
        RegulatoryDeadline(
            report_type=ReportType.TRADE_REPOSITORY,
            name="EMIR Trade Report",
            description="Derivative trade reporting to trade repository",
            frequency=ReportFrequency.T_PLUS_1,
            deadline_rule="T+1",
            regulation="EMIR Art. 9",
            regulator="ESMA",
            warning_days=0,
            critical_days=0,
        ),
        RegulatoryDeadline(
            report_type=ReportType.RTS_27,
            name="RTS 27 Execution Quality Report",
            description="Quarterly execution venue statistics",
            frequency=ReportFrequency.QUARTERLY,
            deadline_rule="3 months after quarter end",
            regulation="MiFID II RTS 27",
            regulator="ESMA/AMF",
            warning_days=14,
            critical_days=7,
        ),
        RegulatoryDeadline(
            report_type=ReportType.RTS_28,
            name="RTS 28 Best Execution Report",
            description="Annual top 5 venues report",
            frequency=ReportFrequency.ANNUAL,
            deadline_rule="April 30",
            regulation="MiFID II RTS 28",
            regulator="ESMA/AMF",
            warning_days=30,
            critical_days=14,
        ),
        RegulatoryDeadline(
            report_type=ReportType.AMF_BDIF,
            name="AMF BDIF Declaration",
            description="Annual declaration to AMF for French firms",
            frequency=ReportFrequency.ANNUAL,
            deadline_rule="March 31",
            regulation="CMF Art. L621-18-2",
            regulator="AMF",
            warning_days=30,
            critical_days=14,
        ),
        RegulatoryDeadline(
            report_type=ReportType.POSITION_REPORT,
            name="Position Report",
            description="Daily position reporting for commodities",
            frequency=ReportFrequency.DAILY,
            deadline_rule="T+0 18:00",
            regulation="MiFID II Art. 58",
            regulator="ESMA",
            warning_days=0,
            critical_days=0,
        ),
        RegulatoryDeadline(
            report_type=ReportType.SECURITIES_FINANCING,
            name="SFTR Report",
            description="Securities financing transaction report",
            frequency=ReportFrequency.T_PLUS_1,
            deadline_rule="T+1",
            regulation="SFTR Art. 4",
            regulator="ESMA",
            warning_days=0,
            critical_days=0,
        ),
        RegulatoryDeadline(
            report_type=ReportType.COLLATERAL_REPORT,
            name="EMIR Collateral Report",
            description="Daily collateral and valuation report",
            frequency=ReportFrequency.DAILY,
            deadline_rule="T+0",
            regulation="EMIR Art. 11",
            regulator="ESMA",
            warning_days=0,
            critical_days=0,
        ),
    ]

    def __init__(
        self,
        notification_callback: Callable[[str, str, dict], None] | None = None,
    ):
        self._notification_callback = notification_callback

        # Deadlines configuration
        self._deadlines: dict[ReportType, RegulatoryDeadline] = {
            d.report_type: d for d in self.STANDARD_DEADLINES
        }

        # Scheduled reports
        self._scheduled: dict[str, ScheduledReport] = {}
        self._report_counter = 0

        # Submission history
        self._submission_history: list[dict] = []

    def add_custom_deadline(self, deadline: RegulatoryDeadline) -> None:
        """Add a custom regulatory deadline."""
        self._deadlines[deadline.report_type] = deadline

    def schedule_report(
        self,
        report_type: ReportType,
        period_start: date,
        period_end: date,
        deadline: datetime,
    ) -> ScheduledReport:
        """Schedule a specific report instance."""
        self._report_counter += 1
        report_id = f"{report_type.value}_{period_end.isoformat()}_{self._report_counter}"

        report = ScheduledReport(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            deadline=deadline,
        )

        self._scheduled[report_id] = report
        return report

    def generate_schedule(
        self,
        start_date: date,
        end_date: date,
    ) -> list[ScheduledReport]:
        """Generate schedule for all reports in date range."""
        reports = []

        for report_type, deadline_config in self._deadlines.items():
            # Generate based on frequency
            current = start_date

            while current <= end_date:
                if deadline_config.frequency == ReportFrequency.DAILY:
                    period_start = current
                    period_end = current
                    deadline_date = current
                    current += timedelta(days=1)

                elif deadline_config.frequency == ReportFrequency.T_PLUS_1:
                    period_start = current
                    period_end = current
                    deadline_date = current + timedelta(days=1)
                    current += timedelta(days=1)

                elif deadline_config.frequency == ReportFrequency.WEEKLY:
                    period_start = current
                    period_end = current + timedelta(days=6)
                    deadline_date = period_end + timedelta(days=2)
                    current += timedelta(days=7)

                elif deadline_config.frequency == ReportFrequency.MONTHLY:
                    period_start = current.replace(day=1)
                    # End of month
                    next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
                    period_end = next_month - timedelta(days=1)
                    deadline_date = next_month + timedelta(days=14)
                    current = next_month

                elif deadline_config.frequency == ReportFrequency.QUARTERLY:
                    quarter_start_month = ((current.month - 1) // 3) * 3 + 1
                    period_start = current.replace(month=quarter_start_month, day=1)
                    quarter_end_month = quarter_start_month + 2
                    if quarter_end_month > 12:
                        period_end = current.replace(year=current.year+1, month=quarter_end_month-12, day=1) - timedelta(days=1)
                    else:
                        next_q = period_start.replace(month=quarter_start_month+3) if quarter_start_month <= 9 else period_start.replace(year=period_start.year+1, month=1)
                        period_end = next_q - timedelta(days=1)
                    deadline_date = period_end + timedelta(days=90)
                    current = period_end + timedelta(days=1)

                elif deadline_config.frequency == ReportFrequency.ANNUAL:
                    period_start = date(current.year, 1, 1)
                    period_end = date(current.year, 12, 31)
                    # Parse deadline rule
                    if "April" in deadline_config.deadline_rule:
                        deadline_date = date(current.year + 1, 4, 30)
                    elif "March" in deadline_config.deadline_rule:
                        deadline_date = date(current.year + 1, 3, 31)
                    else:
                        deadline_date = date(current.year + 1, 3, 31)
                    current = date(current.year + 1, 1, 1)

                else:
                    # Ad-hoc reports not auto-scheduled
                    break

                # Create deadline datetime
                deadline_dt = datetime.combine(
                    deadline_date,
                    datetime.strptime(deadline_config.deadline_time, "%H:%M").time(),
                    tzinfo=timezone.utc
                )

                report = self.schedule_report(
                    report_type=report_type,
                    period_start=period_start,
                    period_end=period_end,
                    deadline=deadline_dt,
                )
                reports.append(report)

                # Prevent infinite loops
                if current > end_date + timedelta(days=365):
                    break

        return reports

    def check_upcoming_deadlines(
        self,
        days_ahead: int = 7,
    ) -> list[dict]:
        """Check for upcoming deadlines and generate alerts."""
        now = datetime.now(timezone.utc)
        alerts = []

        for report in self._scheduled.values():
            if report.status == "submitted":
                continue

            days_until = (report.deadline - now).days

            deadline_config = self._deadlines.get(report.report_type)
            if deadline_config is None:
                continue

            # Critical alert
            if days_until <= deadline_config.critical_days and not report.critical_sent:
                alert = {
                    'level': 'critical',
                    'report_id': report.report_id,
                    'report_type': report.report_type.value,
                    'deadline': report.deadline.isoformat(),
                    'days_until': days_until,
                    'message': f"CRITICAL: {deadline_config.name} due in {days_until} days",
                }
                alerts.append(alert)
                report.critical_sent = True

                if self._notification_callback:
                    self._notification_callback('critical', 'regulatory_deadline', alert)

            # Warning alert
            elif days_until <= deadline_config.warning_days and not report.warning_sent:
                alert = {
                    'level': 'warning',
                    'report_id': report.report_id,
                    'report_type': report.report_type.value,
                    'deadline': report.deadline.isoformat(),
                    'days_until': days_until,
                    'message': f"WARNING: {deadline_config.name} due in {days_until} days",
                }
                alerts.append(alert)
                report.warning_sent = True

                if self._notification_callback:
                    self._notification_callback('warning', 'regulatory_deadline', alert)

            # Overdue
            elif days_until < 0 and report.status == "pending":
                report.status = "late"
                alert = {
                    'level': 'critical',
                    'report_id': report.report_id,
                    'report_type': report.report_type.value,
                    'deadline': report.deadline.isoformat(),
                    'days_overdue': abs(days_until),
                    'message': f"OVERDUE: {deadline_config.name} is {abs(days_until)} days late!",
                }
                alerts.append(alert)

                if self._notification_callback:
                    self._notification_callback('critical', 'regulatory_deadline', alert)

        return alerts

    def mark_submitted(
        self,
        report_id: str,
        confirmation_number: str = "",
    ) -> bool:
        """Mark a report as submitted."""
        report = self._scheduled.get(report_id)
        if report is None:
            return False

        report.status = "submitted"
        report.submitted_at = datetime.now(timezone.utc)
        report.confirmation_number = confirmation_number

        # Record in history
        self._submission_history.append({
            'report_id': report_id,
            'report_type': report.report_type.value,
            'submitted_at': report.submitted_at.isoformat(),
            'on_time': report.submitted_at <= report.deadline,
            'confirmation': confirmation_number,
        })

        logger.info(f"Report {report_id} marked as submitted: {confirmation_number}")
        return True

    def get_calendar_view(
        self,
        month: int,
        year: int,
    ) -> dict:
        """Get calendar view of deadlines for a month."""
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)

        by_date: dict[str, list[dict]] = {}

        for report in self._scheduled.values():
            deadline_date = report.deadline.date()
            if start <= deadline_date <= end:
                date_key = deadline_date.isoformat()
                if date_key not in by_date:
                    by_date[date_key] = []
                by_date[date_key].append(report.to_dict())

        return {
            'month': month,
            'year': year,
            'deadlines_by_date': by_date,
            'total_reports': sum(len(v) for v in by_date.values()),
        }

    def get_compliance_status(self) -> dict:
        """Get overall compliance status."""
        total = len(self._scheduled)
        submitted = sum(1 for r in self._scheduled.values() if r.status == "submitted")
        late = sum(1 for r in self._scheduled.values() if r.status == "late")
        pending = total - submitted - late

        return {
            'total_scheduled': total,
            'submitted_on_time': submitted,
            'late': late,
            'pending': pending,
            'compliance_rate': submitted / total * 100 if total > 0 else 100,
            'submission_history': self._submission_history[-20:],
        }


class ComplianceTrainingManager:
    """
    Manages compliance training records (#C40).

    Tracks mandatory training completion and expiry.
    """

    # Standard compliance trainings
    STANDARD_TRAININGS = [
        ComplianceTraining(
            training_id="MAR_BASIC",
            title="Market Abuse Regulation (MAR) Fundamentals",
            description="Understanding insider dealing, market manipulation, and reporting obligations",
            category="MAR",
            mandatory=True,
            validity_months=12,
            regulation="MAR 596/2014",
        ),
        ComplianceTraining(
            training_id="AML_KYC",
            title="Anti-Money Laundering & KYC",
            description="AML obligations, customer due diligence, suspicious activity reporting",
            category="AML",
            mandatory=True,
            validity_months=12,
            regulation="AMLD 2015/849",
        ),
        ComplianceTraining(
            training_id="MIFID_BEST_EX",
            title="MiFID II Best Execution",
            description="Best execution requirements and client order handling",
            category="Best Execution",
            mandatory=True,
            validity_months=12,
            regulation="MiFID II Art. 27",
        ),
        ComplianceTraining(
            training_id="DATA_PRIVACY",
            title="GDPR & Data Protection",
            description="Personal data handling, privacy rights, breach notification",
            category="Data Protection",
            mandatory=True,
            validity_months=24,
            regulation="GDPR 2016/679",
        ),
        ComplianceTraining(
            training_id="ETHICS_CODE",
            title="Code of Ethics & Conduct",
            description="Firm code of conduct, conflicts of interest, gifts policy",
            category="Ethics",
            mandatory=True,
            validity_months=12,
        ),
    ]

    def __init__(self):
        self._trainings: dict[str, ComplianceTraining] = {
            t.training_id: t for t in self.STANDARD_TRAININGS
        }
        self._records: dict[str, TrainingRecord] = {}
        self._record_counter = 0

    def add_training(self, training: ComplianceTraining) -> None:
        """Add a training course."""
        self._trainings[training.training_id] = training

    def assign_training(
        self,
        training_id: str,
        employee_id: str,
        employee_name: str,
        due_date: date,
    ) -> TrainingRecord | None:
        """Assign training to an employee."""
        training = self._trainings.get(training_id)
        if training is None:
            return None

        self._record_counter += 1
        record_id = f"TR_{employee_id}_{training_id}_{self._record_counter}"

        record = TrainingRecord(
            record_id=record_id,
            training_id=training_id,
            employee_id=employee_id,
            employee_name=employee_name,
            assigned_date=date.today(),
            due_date=due_date,
        )

        self._records[record_id] = record
        return record

    def complete_training(
        self,
        record_id: str,
        score: float | None = None,
        passed: bool = True,
    ) -> bool:
        """Record training completion."""
        record = self._records.get(record_id)
        if record is None:
            return False

        training = self._trainings.get(record.training_id)
        if training is None:
            return False

        record.completed_date = date.today()
        record.score = score
        record.passed = passed

        if passed:
            record.status = TrainingStatus.COMPLETED
            record.expires_date = date.today() + timedelta(days=training.validity_months * 30)
        else:
            record.status = TrainingStatus.NOT_STARTED  # Failed, needs retry

        logger.info(f"Training {record_id} completed: passed={passed}")
        return True

    def check_overdue(self) -> list[TrainingRecord]:
        """Find overdue training assignments."""
        today = date.today()
        overdue = []

        for record in self._records.values():
            if record.status in [TrainingStatus.NOT_STARTED, TrainingStatus.IN_PROGRESS]:
                if record.due_date < today:
                    record.status = TrainingStatus.OVERDUE
                    overdue.append(record)
            elif record.status == TrainingStatus.COMPLETED:
                if record.expires_date and record.expires_date < today:
                    record.status = TrainingStatus.EXPIRED
                    overdue.append(record)

        return overdue

    def get_employee_status(self, employee_id: str) -> dict:
        """Get training status for an employee."""
        employee_records = [
            r for r in self._records.values()
            if r.employee_id == employee_id
        ]

        by_status = {}
        for record in employee_records:
            status = record.status.value
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(record.to_dict())

        return {
            'employee_id': employee_id,
            'total_assigned': len(employee_records),
            'by_status': by_status,
            'compliance_rate': (
                sum(1 for r in employee_records if r.status == TrainingStatus.COMPLETED) /
                len(employee_records) * 100 if employee_records else 100
            ),
        }

    def get_overall_compliance(self) -> dict:
        """Get overall training compliance metrics."""
        total = len(self._records)
        completed = sum(1 for r in self._records.values() if r.status == TrainingStatus.COMPLETED)
        overdue = sum(1 for r in self._records.values() if r.status == TrainingStatus.OVERDUE)
        expired = sum(1 for r in self._records.values() if r.status == TrainingStatus.EXPIRED)

        return {
            'total_assignments': total,
            'completed': completed,
            'overdue': overdue,
            'expired': expired,
            'compliance_rate': completed / total * 100 if total > 0 else 100,
            'trainings_available': len(self._trainings),
        }


class GiftEntertainmentLog:
    """
    Gift and entertainment logging (#C39).

    Tracks gifts/entertainment with approval workflows.
    """

    def __init__(
        self,
        approval_threshold_eur: float = 100.0,
        annual_limit_eur: float = 500.0,
    ):
        self.approval_threshold = approval_threshold_eur
        self.annual_limit = annual_limit_eur

        self._entries: dict[str, GiftEntertainmentEntry] = {}
        self._entry_counter = 0

    def log_entry(
        self,
        entry_type: str,
        employee_id: str,
        employee_name: str,
        counterparty_name: str,
        counterparty_company: str,
        description: str,
        value_eur: float,
        event_date: date,
        location: str = "",
    ) -> GiftEntertainmentEntry:
        """Log a gift/entertainment entry."""
        self._entry_counter += 1
        entry_id = f"GE_{event_date.isoformat()}_{self._entry_counter}"

        requires_approval = value_eur >= self.approval_threshold

        entry = GiftEntertainmentEntry(
            entry_id=entry_id,
            entry_type=entry_type,
            employee_id=employee_id,
            employee_name=employee_name,
            counterparty_name=counterparty_name,
            counterparty_company=counterparty_company,
            description=description,
            value_eur=value_eur,
            event_date=event_date,
            location=location,
            requires_approval=requires_approval,
        )

        self._entries[entry_id] = entry

        # Check annual limit
        annual_total = self.get_employee_annual_total(employee_id, event_date.year)
        if annual_total > self.annual_limit:
            logger.warning(
                f"Employee {employee_id} has exceeded annual G&E limit: "
                f"€{annual_total:.2f} > €{self.annual_limit:.2f}"
            )

        logger.info(f"G&E entry logged: {entry_id} - €{value_eur:.2f}")
        return entry

    def approve_entry(
        self,
        entry_id: str,
        approved_by: str,
    ) -> bool:
        """Approve a gift/entertainment entry."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return False

        entry.approved = True
        entry.approved_by = approved_by
        entry.approval_date = date.today()

        logger.info(f"G&E entry {entry_id} approved by {approved_by}")
        return True

    def get_employee_annual_total(
        self,
        employee_id: str,
        year: int,
    ) -> float:
        """Get total G&E value for employee in a year."""
        return sum(
            e.value_eur for e in self._entries.values()
            if e.employee_id == employee_id and e.event_date.year == year
        )

    def get_pending_approvals(self) -> list[GiftEntertainmentEntry]:
        """Get entries pending approval."""
        return [
            e for e in self._entries.values()
            if e.requires_approval and not e.approved
        ]

    def get_summary(
        self,
        year: int | None = None,
    ) -> dict:
        """Get G&E summary."""
        entries = list(self._entries.values())
        if year:
            entries = [e for e in entries if e.event_date.year == year]

        total_value = sum(e.value_eur for e in entries)

        by_type = {}
        for e in entries:
            if e.entry_type not in by_type:
                by_type[e.entry_type] = {'count': 0, 'value': 0.0}
            by_type[e.entry_type]['count'] += 1
            by_type[e.entry_type]['value'] += e.value_eur

        return {
            'year': year or 'all',
            'total_entries': len(entries),
            'total_value_eur': total_value,
            'pending_approvals': len(self.get_pending_approvals()),
            'by_type': by_type,
            'approval_threshold_eur': self.approval_threshold,
            'annual_limit_eur': self.annual_limit,
        }

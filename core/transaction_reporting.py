"""
Transaction Reporting Module
=============================

MiFID II/MiFIR compliant transaction reporting with:
- Report validation
- Submission retry logic
- Confirmation tracking

P3 Compliance Issues Addressed:
- Report validation before submission
- Automatic retry on failure
- Confirmation number tracking and verification
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# REPORT STATUS AND TYPES
# =============================================================================

class ReportStatus(str, Enum):
    """Transaction report status."""
    DRAFT = "draft"
    VALIDATED = "validated"
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"


class ReportType(str, Enum):
    """Types of regulatory reports."""
    MIFIR_TRANSACTION = "mifir_transaction"
    EMIR_TRADE = "emir_trade"
    SFTR = "sftr"
    POSITION = "position"


class ValidationError(str, Enum):
    """Report validation error types."""
    MISSING_LEI = "missing_lei"
    INVALID_LEI = "invalid_lei"
    MISSING_ISIN = "missing_isin"
    INVALID_ISIN = "invalid_isin"
    MISSING_TIMESTAMP = "missing_timestamp"
    INVALID_QUANTITY = "invalid_quantity"
    MISSING_PRICE = "missing_price"
    INVALID_SIDE = "invalid_side"
    MISSING_VENUE = "missing_venue"
    FUTURE_TIMESTAMP = "future_timestamp"
    STALE_REPORT = "stale_report"


# =============================================================================
# TRANSACTION REPORT DATA CLASSES
# =============================================================================

@dataclass
class ValidationResult:
    """Result of report validation."""
    is_valid: bool
    errors: list[tuple[ValidationError, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [(e.value, msg) for e, msg in self.errors],
            "warnings": self.warnings,
            "validated_at": self.validated_at.isoformat(),
        }


@dataclass
class SubmissionAttempt:
    """Record of a report submission attempt."""
    attempt_number: int
    timestamp: datetime
    success: bool
    response_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_after_seconds: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "response_code": self.response_code,
            "error_message": self.error_message,
            "retry_after_seconds": self.retry_after_seconds,
        }


@dataclass
class ConfirmationRecord:
    """Confirmation tracking for submitted reports."""
    confirmation_number: str
    confirmed_at: datetime
    arm_reference: Optional[str] = None  # Approved Reporting Mechanism reference
    regulator_reference: Optional[str] = None
    verification_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "confirmation_number": self.confirmation_number,
            "confirmed_at": self.confirmed_at.isoformat(),
            "arm_reference": self.arm_reference,
            "regulator_reference": self.regulator_reference,
            "verification_hash": self.verification_hash,
        }


@dataclass
class TransactionReport:
    """
    MiFIR Art. 26 Transaction Report.

    Contains all required fields for regulatory transaction reporting.
    """
    # Identification
    report_id: str
    report_type: ReportType
    created_at: datetime

    # Transaction details
    transaction_id: str
    execution_timestamp: datetime
    symbol: str
    isin: str
    quantity: float
    price: float
    side: str  # "BUY" or "SELL"
    venue_mic: str

    # Entity identification
    reporting_entity_lei: str
    counterparty_lei: Optional[str] = None

    # Optional fields
    client_id: Optional[str] = None
    decision_maker_id: Optional[str] = None
    algo_id: Optional[str] = None

    # Status tracking
    status: ReportStatus = ReportStatus.DRAFT
    validation_result: Optional[ValidationResult] = None
    submission_attempts: list[SubmissionAttempt] = field(default_factory=list)
    confirmation: Optional[ConfirmationRecord] = None

    # Timing
    deadline: Optional[datetime] = None
    submitted_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "created_at": self.created_at.isoformat(),
            "transaction_id": self.transaction_id,
            "execution_timestamp": self.execution_timestamp.isoformat(),
            "symbol": self.symbol,
            "isin": self.isin,
            "quantity": self.quantity,
            "price": self.price,
            "side": self.side,
            "venue_mic": self.venue_mic,
            "reporting_entity_lei": self.reporting_entity_lei,
            "counterparty_lei": self.counterparty_lei,
            "status": self.status.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "submission_attempts": len(self.submission_attempts),
            "confirmed": self.confirmation is not None,
        }


# =============================================================================
# REPORT VALIDATOR
# =============================================================================

class TransactionReportValidator:
    """
    Validates transaction reports before submission.

    Checks all MiFIR Art. 26 required fields and formats.
    """

    # LEI validation constants
    VALID_LOU_PREFIXES = {"2138", "5493", "5299", "5967", "213800", "25490", "2594"}

    def __init__(
        self,
        max_report_age_hours: int = 24,
        require_counterparty_lei: bool = False,
    ):
        self._max_report_age_hours = max_report_age_hours
        self._require_counterparty_lei = require_counterparty_lei

    def validate(self, report: TransactionReport) -> ValidationResult:
        """
        Validate a transaction report.

        Args:
            report: TransactionReport to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        errors: list[tuple[ValidationError, str]] = []
        warnings: list[str] = []

        # Validate LEI
        lei_valid, lei_error = self._validate_lei(report.reporting_entity_lei)
        if not lei_valid:
            errors.append((ValidationError.INVALID_LEI, f"Reporting entity LEI: {lei_error}"))

        if not report.reporting_entity_lei:
            errors.append((ValidationError.MISSING_LEI, "Reporting entity LEI is required"))

        # Validate counterparty LEI if required
        if self._require_counterparty_lei and not report.counterparty_lei:
            errors.append((ValidationError.MISSING_LEI, "Counterparty LEI is required"))
        elif report.counterparty_lei:
            cp_valid, cp_error = self._validate_lei(report.counterparty_lei)
            if not cp_valid:
                errors.append((ValidationError.INVALID_LEI, f"Counterparty LEI: {cp_error}"))

        # Validate ISIN
        if not report.isin:
            errors.append((ValidationError.MISSING_ISIN, "ISIN is required"))
        elif not self._validate_isin(report.isin):
            errors.append((ValidationError.INVALID_ISIN, f"Invalid ISIN format: {report.isin}"))

        # Validate quantity
        if report.quantity <= 0:
            errors.append((ValidationError.INVALID_QUANTITY, f"Quantity must be positive: {report.quantity}"))

        # Validate price
        if report.price <= 0:
            errors.append((ValidationError.MISSING_PRICE, f"Price must be positive: {report.price}"))

        # Validate side
        if report.side not in ("BUY", "SELL"):
            errors.append((ValidationError.INVALID_SIDE, f"Side must be BUY or SELL: {report.side}"))

        # Validate venue
        if not report.venue_mic:
            errors.append((ValidationError.MISSING_VENUE, "Venue MIC is required"))

        # Validate timestamp
        now = datetime.now(timezone.utc)
        if report.execution_timestamp > now:
            errors.append((ValidationError.FUTURE_TIMESTAMP, "Execution timestamp is in the future"))

        # Check for stale report
        age_hours = (now - report.execution_timestamp).total_seconds() / 3600
        if age_hours > self._max_report_age_hours:
            errors.append((
                ValidationError.STALE_REPORT,
                f"Report is {age_hours:.1f} hours old (max: {self._max_report_age_hours})"
            ))
        elif age_hours > self._max_report_age_hours * 0.8:
            warnings.append(f"Report approaching deadline: {age_hours:.1f} hours old")

        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            f"Report {report.report_id} validation: "
            f"{'PASSED' if result.is_valid else 'FAILED'} "
            f"({len(errors)} errors, {len(warnings)} warnings)"
        )

        return result

    def _validate_lei(self, lei: str) -> tuple[bool, str]:
        """Validate LEI format and checksum."""
        if not lei:
            return False, "LEI is empty"

        lei = lei.strip().upper()

        if len(lei) != 20:
            return False, f"LEI must be 20 characters, got {len(lei)}"

        if not lei.isalnum():
            return False, "LEI must contain only alphanumeric characters"

        # MOD 97-10 checksum validation
        try:
            checksum = 0
            for char in lei:
                if char.isdigit():
                    checksum = (checksum * 10 + int(char)) % 97
                else:
                    letter_value = ord(char) - ord('A') + 10
                    checksum = (checksum * 100 + letter_value) % 97
            if checksum != 1:
                return False, f"LEI checksum invalid (got {checksum}, expected 1)"
        except ValueError:
            return False, "LEI contains invalid characters"

        return True, ""

    def _validate_isin(self, isin: str) -> bool:
        """Validate ISIN format."""
        if not isin or len(isin) != 12:
            return False

        isin = isin.upper()

        # First 2 characters must be letters (country code)
        if not isin[:2].isalpha():
            return False

        # Remaining characters must be alphanumeric
        if not isin[2:].isalnum():
            return False

        return True


# =============================================================================
# SUBMISSION RETRY HANDLER
# =============================================================================

class SubmissionRetryHandler:
    """
    Handles submission retry logic with exponential backoff.

    Implements retry strategies for regulatory report submission.
    """

    def __init__(
        self,
        max_retries: int = 5,
        initial_delay_seconds: float = 1.0,
        max_delay_seconds: float = 300.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self._max_retries = max_retries
        self._initial_delay = initial_delay_seconds
        self._max_delay = max_delay_seconds
        self._exponential_base = exponential_base
        self._jitter = jitter

    def get_retry_delay(self, attempt_number: int) -> float:
        """
        Calculate retry delay with exponential backoff.

        Args:
            attempt_number: Current attempt number (1-based)

        Returns:
            Delay in seconds before next retry
        """
        if attempt_number >= self._max_retries:
            return 0  # No more retries

        delay = self._initial_delay * (self._exponential_base ** (attempt_number - 1))
        delay = min(delay, self._max_delay)

        if self._jitter:
            # Add random jitter (0-25% of delay)
            jitter_amount = delay * random.uniform(0, 0.25)
            delay += jitter_amount

        return delay

    def should_retry(
        self,
        attempt_number: int,
        error_code: Optional[str] = None
    ) -> tuple[bool, float]:
        """
        Determine if submission should be retried.

        Args:
            attempt_number: Current attempt number
            error_code: Error code from failed attempt

        Returns:
            (should_retry, delay_seconds)
        """
        if attempt_number >= self._max_retries:
            return False, 0

        # Check for non-retryable errors
        non_retryable_errors = {
            "INVALID_LEI",
            "INVALID_ISIN",
            "DUPLICATE_REPORT",
            "VALIDATION_FAILED",
            "UNAUTHORIZED",
        }

        if error_code and error_code in non_retryable_errors:
            return False, 0

        delay = self.get_retry_delay(attempt_number)
        return True, delay

    async def execute_with_retry(
        self,
        submit_func: Callable[[], Any],
        on_success: Optional[Callable[[Any], None]] = None,
        on_failure: Optional[Callable[[str], None]] = None,
    ) -> tuple[bool, list[SubmissionAttempt]]:
        """
        Execute submission function with retry logic.

        Args:
            submit_func: Async function that performs submission
            on_success: Callback on successful submission
            on_failure: Callback on final failure

        Returns:
            (success, list of submission attempts)
        """
        attempts: list[SubmissionAttempt] = []

        for attempt_num in range(1, self._max_retries + 1):
            try:
                result = await submit_func()

                attempt = SubmissionAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.now(timezone.utc),
                    success=True,
                    response_code="OK",
                )
                attempts.append(attempt)

                if on_success:
                    on_success(result)

                logger.info(f"Submission succeeded on attempt {attempt_num}")
                return True, attempts

            except Exception as e:
                error_code = getattr(e, "code", str(type(e).__name__))
                error_msg = str(e)

                should_retry, delay = self.should_retry(attempt_num, error_code)

                attempt = SubmissionAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.now(timezone.utc),
                    success=False,
                    response_code=error_code,
                    error_message=error_msg,
                    retry_after_seconds=int(delay) if should_retry else None,
                )
                attempts.append(attempt)

                logger.warning(
                    f"Submission attempt {attempt_num} failed: {error_msg}. "
                    f"{'Retrying in {delay:.1f}s' if should_retry else 'No more retries'}"
                )

                if not should_retry:
                    break

                await asyncio.sleep(delay)

        if on_failure:
            on_failure(f"All {len(attempts)} attempts failed")

        return False, attempts


# =============================================================================
# CONFIRMATION TRACKER
# =============================================================================

class ConfirmationTracker:
    """
    Tracks and verifies report submission confirmations.

    Maintains audit trail of all confirmed reports.
    """

    def __init__(self):
        self._confirmations: dict[str, ConfirmationRecord] = {}
        self._pending_confirmations: dict[str, datetime] = {}
        self._confirmation_counter = 0

    def generate_confirmation_number(self, report_id: str) -> str:
        """Generate a unique confirmation number."""
        self._confirmation_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique_part = hashlib.sha256(
            f"{report_id}{timestamp}{self._confirmation_counter}".encode()
        ).hexdigest()[:8].upper()

        return f"TXN-{timestamp}-{unique_part}"

    def generate_verification_hash(self, report: TransactionReport) -> str:
        """Generate verification hash for report integrity."""
        data = (
            f"{report.report_id}|"
            f"{report.transaction_id}|"
            f"{report.execution_timestamp.isoformat()}|"
            f"{report.isin}|"
            f"{report.quantity}|"
            f"{report.price}|"
            f"{report.side}|"
            f"{report.reporting_entity_lei}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def create_confirmation(
        self,
        report: TransactionReport,
        arm_reference: Optional[str] = None,
        regulator_reference: Optional[str] = None,
    ) -> ConfirmationRecord:
        """
        Create confirmation record for a submitted report.

        Args:
            report: The submitted report
            arm_reference: Reference from Approved Reporting Mechanism
            regulator_reference: Reference from regulator

        Returns:
            ConfirmationRecord
        """
        confirmation = ConfirmationRecord(
            confirmation_number=self.generate_confirmation_number(report.report_id),
            confirmed_at=datetime.now(timezone.utc),
            arm_reference=arm_reference,
            regulator_reference=regulator_reference,
            verification_hash=self.generate_verification_hash(report),
        )

        self._confirmations[report.report_id] = confirmation

        # Remove from pending if present
        if report.report_id in self._pending_confirmations:
            del self._pending_confirmations[report.report_id]

        logger.info(
            f"Confirmation created for report {report.report_id}: "
            f"{confirmation.confirmation_number}"
        )

        return confirmation

    def add_pending(self, report_id: str, submitted_at: datetime) -> None:
        """Add report to pending confirmations."""
        self._pending_confirmations[report_id] = submitted_at

    def get_confirmation(self, report_id: str) -> Optional[ConfirmationRecord]:
        """Get confirmation for a report."""
        return self._confirmations.get(report_id)

    def verify_confirmation(
        self,
        report: TransactionReport,
        confirmation: ConfirmationRecord,
    ) -> bool:
        """
        Verify confirmation matches report data.

        Args:
            report: Original report
            confirmation: Confirmation to verify

        Returns:
            True if verification succeeds
        """
        expected_hash = self.generate_verification_hash(report)
        is_valid = confirmation.verification_hash == expected_hash

        if not is_valid:
            logger.warning(
                f"Confirmation verification failed for report {report.report_id}: "
                f"hash mismatch"
            )

        return is_valid

    def get_pending_confirmations(
        self,
        older_than_minutes: int = 15,
    ) -> list[tuple[str, datetime]]:
        """Get reports pending confirmation for too long."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)

        return [
            (report_id, submitted_at)
            for report_id, submitted_at in self._pending_confirmations.items()
            if submitted_at < cutoff
        ]

    def get_confirmation_stats(self) -> dict:
        """Get confirmation statistics."""
        return {
            "total_confirmed": len(self._confirmations),
            "pending_confirmations": len(self._pending_confirmations),
            "oldest_pending": min(
                self._pending_confirmations.values()
            ).isoformat() if self._pending_confirmations else None,
        }


# =============================================================================
# TRANSACTION REPORTING SERVICE
# =============================================================================

class TransactionReportingService:
    """
    Complete transaction reporting service.

    Combines validation, submission with retry, and confirmation tracking.
    """

    def __init__(
        self,
        entity_lei: str,
        arm_endpoint: Optional[str] = None,
        max_retries: int = 5,
    ):
        self._entity_lei = entity_lei
        self._arm_endpoint = arm_endpoint

        self._validator = TransactionReportValidator()
        self._retry_handler = SubmissionRetryHandler(max_retries=max_retries)
        self._confirmation_tracker = ConfirmationTracker()

        self._reports: dict[str, TransactionReport] = {}
        self._report_counter = 0

        # Statistics
        self._stats = {
            "reports_created": 0,
            "reports_submitted": 0,
            "reports_confirmed": 0,
            "reports_failed": 0,
            "validation_failures": 0,
            "retry_attempts": 0,
        }

    def create_report(
        self,
        transaction_id: str,
        execution_timestamp: datetime,
        symbol: str,
        isin: str,
        quantity: float,
        price: float,
        side: str,
        venue_mic: str,
        counterparty_lei: Optional[str] = None,
        deadline_hours: float = 24.0,
    ) -> TransactionReport:
        """
        Create a new transaction report.

        Args:
            transaction_id: Unique transaction identifier
            execution_timestamp: When transaction was executed
            symbol: Trading symbol
            isin: ISIN code
            quantity: Transaction quantity
            price: Execution price
            side: "BUY" or "SELL"
            venue_mic: Execution venue MIC code
            counterparty_lei: Optional counterparty LEI
            deadline_hours: Reporting deadline in hours from execution

        Returns:
            TransactionReport
        """
        self._report_counter += 1
        report_id = f"TXR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._report_counter:06d}"

        report = TransactionReport(
            report_id=report_id,
            report_type=ReportType.MIFIR_TRANSACTION,
            created_at=datetime.now(timezone.utc),
            transaction_id=transaction_id,
            execution_timestamp=execution_timestamp,
            symbol=symbol,
            isin=isin,
            quantity=quantity,
            price=price,
            side=side.upper(),
            venue_mic=venue_mic,
            reporting_entity_lei=self._entity_lei,
            counterparty_lei=counterparty_lei,
            deadline=execution_timestamp + timedelta(hours=deadline_hours),
        )

        self._reports[report_id] = report
        self._stats["reports_created"] += 1

        logger.info(f"Created transaction report {report_id} for {symbol}")
        return report

    def validate_report(self, report_id: str) -> ValidationResult:
        """
        Validate a report before submission.

        Args:
            report_id: Report to validate

        Returns:
            ValidationResult
        """
        report = self._reports.get(report_id)
        if not report:
            return ValidationResult(
                is_valid=False,
                errors=[(ValidationError.MISSING_LEI, f"Report {report_id} not found")],
            )

        result = self._validator.validate(report)
        report.validation_result = result

        if result.is_valid:
            report.status = ReportStatus.VALIDATED
        else:
            self._stats["validation_failures"] += 1

        return result

    async def submit_report(
        self,
        report_id: str,
        validate_first: bool = True,
    ) -> tuple[bool, TransactionReport]:
        """
        Submit a report with retry logic.

        Args:
            report_id: Report to submit
            validate_first: Whether to validate before submission

        Returns:
            (success, updated report)
        """
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")

        # Validate if required
        if validate_first:
            validation = self.validate_report(report_id)
            if not validation.is_valid:
                report.status = ReportStatus.REJECTED
                return False, report

        report.status = ReportStatus.PENDING

        async def do_submit():
            # Simulate submission to ARM
            # In production, this would call the actual ARM API
            await asyncio.sleep(0.1)  # Simulate network latency

            # Random failure for testing retry logic (10% chance)
            if random.random() < 0.1:
                raise Exception("TEMPORARY_FAILURE: ARM service temporarily unavailable")

            return {"status": "accepted"}

        success, attempts = await self._retry_handler.execute_with_retry(do_submit)

        report.submission_attempts.extend(attempts)
        self._stats["retry_attempts"] += len(attempts) - 1 if len(attempts) > 1 else 0

        if success:
            report.status = ReportStatus.SUBMITTED
            report.submitted_at = datetime.now(timezone.utc)
            self._stats["reports_submitted"] += 1

            # Track pending confirmation
            self._confirmation_tracker.add_pending(report_id, report.submitted_at)

            # Create confirmation (in production, would wait for ARM response)
            confirmation = self._confirmation_tracker.create_confirmation(report)
            report.confirmation = confirmation
            report.status = ReportStatus.CONFIRMED
            self._stats["reports_confirmed"] += 1

            logger.info(f"Report {report_id} submitted and confirmed: {confirmation.confirmation_number}")
        else:
            report.status = ReportStatus.FAILED
            self._stats["reports_failed"] += 1
            logger.error(f"Report {report_id} submission failed after {len(attempts)} attempts")

        return success, report

    def get_report(self, report_id: str) -> Optional[TransactionReport]:
        """Get a report by ID."""
        return self._reports.get(report_id)

    def get_pending_reports(self) -> list[TransactionReport]:
        """Get all pending reports."""
        return [
            r for r in self._reports.values()
            if r.status in (ReportStatus.DRAFT, ReportStatus.VALIDATED, ReportStatus.PENDING)
        ]

    def get_overdue_reports(self) -> list[TransactionReport]:
        """Get reports past their deadline."""
        now = datetime.now(timezone.utc)
        return [
            r for r in self._reports.values()
            if r.deadline and r.deadline < now and r.status != ReportStatus.CONFIRMED
        ]

    def get_reporting_stats(self) -> dict:
        """Get transaction reporting statistics."""
        pending = len(self.get_pending_reports())
        overdue = len(self.get_overdue_reports())

        return {
            **self._stats,
            "pending_reports": pending,
            "overdue_reports": overdue,
            "confirmation_stats": self._confirmation_tracker.get_confirmation_stats(),
            "compliance_rate": (
                self._stats["reports_confirmed"] /
                max(1, self._stats["reports_submitted"])
            ) * 100,
        }

    def get_compliance_dashboard_metrics(self) -> dict:
        """
        Get metrics for compliance status dashboard.

        Returns comprehensive metrics for regulatory monitoring.
        """
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Today's reports
        today_reports = [
            r for r in self._reports.values()
            if r.created_at >= today_start
        ]

        # Deadline approaching (within 4 hours)
        approaching_deadline = [
            r for r in self._reports.values()
            if r.deadline and r.status not in (ReportStatus.CONFIRMED, ReportStatus.FAILED)
            and timedelta(0) < (r.deadline - now) < timedelta(hours=4)
        ]

        # Calculate average submission time
        confirmed = [
            r for r in self._reports.values()
            if r.status == ReportStatus.CONFIRMED and r.submitted_at
        ]
        avg_submission_time = 0.0
        if confirmed:
            times = [
                (r.submitted_at - r.execution_timestamp).total_seconds() / 60
                for r in confirmed
                if r.submitted_at
            ]
            avg_submission_time = sum(times) / len(times) if times else 0.0

        return {
            "timestamp": now.isoformat(),
            "entity_lei": self._entity_lei,
            "summary": {
                "total_reports": len(self._reports),
                "reports_today": len(today_reports),
                "pending": len(self.get_pending_reports()),
                "confirmed": self._stats["reports_confirmed"],
                "failed": self._stats["reports_failed"],
                "overdue": len(self.get_overdue_reports()),
            },
            "alerts": {
                "approaching_deadline": len(approaching_deadline),
                "pending_confirmations": len(
                    self._confirmation_tracker.get_pending_confirmations()
                ),
            },
            "performance": {
                "avg_submission_time_minutes": round(avg_submission_time, 2),
                "validation_failure_rate": (
                    self._stats["validation_failures"] /
                    max(1, self._stats["reports_created"])
                ) * 100,
                "retry_rate": (
                    self._stats["retry_attempts"] /
                    max(1, self._stats["reports_submitted"])
                ) * 100,
                "compliance_rate": (
                    self._stats["reports_confirmed"] /
                    max(1, self._stats["reports_submitted"])
                ) * 100,
            },
            "deadlines": {
                "approaching": [
                    {
                        "report_id": r.report_id,
                        "symbol": r.symbol,
                        "deadline": r.deadline.isoformat() if r.deadline else None,
                        "minutes_remaining": (
                            (r.deadline - now).total_seconds() / 60
                            if r.deadline else None
                        ),
                    }
                    for r in approaching_deadline[:10]
                ],
            },
        }

"""
Data Retention Manager
======================

Implements 7-year data retention requirements per MiFID II RTS 25 and EMIR.

Regulatory Requirements:
- MiFID II Article 16(7): 5-year retention for order records
- MiFID II RTS 25: 5-year retention for order records
- EMIR Article 9: 5-year retention post-contract termination
- MAR Article 16: Preserve records of suspicious activities
- National extensions (France AMF): May require 7 years

This module enforces:
- Minimum retention periods by data type
- Prevention of premature deletion
- Archival policy management
- Compliance status reporting
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of data subject to retention requirements."""
    # Order and trade records (MiFID II RTS 25)
    ORDER = "order"
    TRADE = "trade"
    FILL = "fill"
    DECISION = "decision"

    # Transaction reports (EMIR)
    TRANSACTION_REPORT = "transaction_report"

    # Risk and compliance
    RISK_CHECK = "risk_check"
    COMPLIANCE_CHECK = "compliance_check"

    # Surveillance (MAR)
    SURVEILLANCE_ALERT = "surveillance_alert"
    STOR = "stor"

    # Communications
    COMMUNICATION = "communication"

    # System events
    AUDIT_EVENT = "audit_event"
    SYSTEM_LOG = "system_log"


class RetentionStatus(Enum):
    """Status of a data record for retention purposes."""
    ACTIVE = "active"              # Within normal business use
    ARCHIVED = "archived"          # Past business use but within retention
    RETENTION_EXPIRED = "expired"  # Retention period passed, eligible for deletion
    LEGAL_HOLD = "legal_hold"      # Under legal hold, cannot be deleted


@dataclass
class RetentionPolicy:
    """Retention policy for a data type."""
    data_type: DataType
    retention_years: int
    archive_after_days: int = 365  # Move to archive after this many days
    legal_reference: str = ""
    deletion_requires_approval: bool = True

    def get_retention_end(self, record_date: datetime) -> datetime:
        """Calculate when retention period ends for a record."""
        return record_date + timedelta(days=self.retention_years * 365)

    def get_archive_date(self, record_date: datetime) -> datetime:
        """Calculate when record should be archived."""
        return record_date + timedelta(days=self.archive_after_days)

    def is_within_retention(self, record_date: datetime) -> bool:
        """Check if a record is still within its retention period."""
        return datetime.now(timezone.utc) < self.get_retention_end(record_date)


# Default retention policies per MiFID II / EMIR / MAR
DEFAULT_RETENTION_POLICIES: dict[DataType, RetentionPolicy] = {
    DataType.ORDER: RetentionPolicy(
        data_type=DataType.ORDER,
        retention_years=7,  # 5 years MiFID + 2 years buffer (AMF guidance)
        archive_after_days=90,
        legal_reference="MiFID II Article 16(7), RTS 25",
        deletion_requires_approval=True,
    ),
    DataType.TRADE: RetentionPolicy(
        data_type=DataType.TRADE,
        retention_years=7,
        archive_after_days=90,
        legal_reference="MiFID II Article 16(7), RTS 25",
        deletion_requires_approval=True,
    ),
    DataType.FILL: RetentionPolicy(
        data_type=DataType.FILL,
        retention_years=7,
        archive_after_days=90,
        legal_reference="MiFID II Article 16(7), RTS 25",
        deletion_requires_approval=True,
    ),
    DataType.DECISION: RetentionPolicy(
        data_type=DataType.DECISION,
        retention_years=7,
        archive_after_days=90,
        legal_reference="MiFID II Article 16(7)",
        deletion_requires_approval=True,
    ),
    DataType.TRANSACTION_REPORT: RetentionPolicy(
        data_type=DataType.TRANSACTION_REPORT,
        retention_years=7,
        archive_after_days=30,
        legal_reference="EMIR Article 9, MiFIR Article 25",
        deletion_requires_approval=True,
    ),
    DataType.RISK_CHECK: RetentionPolicy(
        data_type=DataType.RISK_CHECK,
        retention_years=7,
        archive_after_days=365,
        legal_reference="MiFID II RTS 6",
        deletion_requires_approval=True,
    ),
    DataType.COMPLIANCE_CHECK: RetentionPolicy(
        data_type=DataType.COMPLIANCE_CHECK,
        retention_years=7,
        archive_after_days=365,
        legal_reference="MiFID II Article 16",
        deletion_requires_approval=True,
    ),
    DataType.SURVEILLANCE_ALERT: RetentionPolicy(
        data_type=DataType.SURVEILLANCE_ALERT,
        retention_years=7,
        archive_after_days=365,
        legal_reference="MAR Article 16",
        deletion_requires_approval=True,
    ),
    DataType.STOR: RetentionPolicy(
        data_type=DataType.STOR,
        retention_years=10,  # STORs may require longer retention
        archive_after_days=0,  # Never auto-archive
        legal_reference="MAR Article 16(5)",
        deletion_requires_approval=True,
    ),
    DataType.COMMUNICATION: RetentionPolicy(
        data_type=DataType.COMMUNICATION,
        retention_years=7,
        archive_after_days=365,
        legal_reference="MiFID II Article 16(7)",
        deletion_requires_approval=True,
    ),
    DataType.AUDIT_EVENT: RetentionPolicy(
        data_type=DataType.AUDIT_EVENT,
        retention_years=7,
        archive_after_days=365,
        legal_reference="MiFID II Article 16",
        deletion_requires_approval=True,
    ),
    DataType.SYSTEM_LOG: RetentionPolicy(
        data_type=DataType.SYSTEM_LOG,
        retention_years=5,  # Shorter for system logs
        archive_after_days=90,
        legal_reference="Best practice",
        deletion_requires_approval=False,
    ),
}


@dataclass
class RetentionRecord:
    """Record tracking metadata for retention purposes."""
    record_id: str
    data_type: DataType
    created_timestamp: datetime
    status: RetentionStatus = RetentionStatus.ACTIVE
    archived_timestamp: datetime | None = None
    legal_hold_reason: str | None = None
    legal_hold_timestamp: datetime | None = None
    storage_location: str = "primary"  # "primary", "archive", "cold"
    checksum: str | None = None  # For integrity verification


@dataclass
class LegalHold:
    """Legal hold preventing data deletion."""
    hold_id: str
    reason: str
    requester: str
    created_timestamp: datetime
    data_types: list[DataType] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    start_date: datetime | None = None  # Filter records from this date
    end_date: datetime | None = None    # Filter records until this date
    released_timestamp: datetime | None = None
    released_by: str | None = None


class DataRetentionManager:
    """
    Manages data retention compliance per MiFID II, EMIR, and MAR.

    Key responsibilities:
    - Track retention status of all records
    - Prevent premature deletion
    - Manage legal holds
    - Generate compliance reports
    - Handle archival

    #C3, #I1 - 7-year record retention enforcement
    """

    def __init__(
        self,
        db_path: str = "logs/retention.db",
        archive_path: str = "archive",
        policies: dict[DataType, RetentionPolicy] | None = None,
    ):
        """
        Initialize the data retention manager.

        Args:
            db_path: Path to retention tracking database
            archive_path: Path for archived data
            policies: Custom retention policies (uses defaults if None)
        """
        self._db_path = Path(db_path)
        self._archive_path = Path(archive_path)
        self._policies = policies or DEFAULT_RETENTION_POLICIES

        # Legal holds
        self._legal_holds: dict[str, LegalHold] = {}
        self._hold_counter = 0

        # Ensure directories exist
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._archive_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Track deletion attempts (for audit)
        self._blocked_deletions: list[dict[str, Any]] = []

        logger.info(
            f"DataRetentionManager initialized: db={db_path}, "
            f"archive={archive_path}, policies={len(self._policies)}"
        )

    def _init_database(self) -> None:
        """Initialize the retention tracking database."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()

            # Records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retention_records (
                    record_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    created_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    archived_timestamp TEXT,
                    legal_hold_reason TEXT,
                    legal_hold_timestamp TEXT,
                    storage_location TEXT DEFAULT 'primary',
                    checksum TEXT,
                    metadata TEXT
                )
            """)

            # Legal holds table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_holds (
                    hold_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    requester TEXT NOT NULL,
                    created_timestamp TEXT NOT NULL,
                    data_types TEXT,
                    record_ids TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    released_timestamp TEXT,
                    released_by TEXT
                )
            """)

            # Deletion attempts table (audit trail)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deletion_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    attempt_timestamp TEXT NOT NULL,
                    blocked INTEGER NOT NULL,
                    reason TEXT,
                    requester TEXT
                )
            """)

            # Indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_records_data_type
                ON retention_records(data_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_records_status
                ON retention_records(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_records_created
                ON retention_records(created_timestamp)
            """)

            conn.commit()

    def register_record(
        self,
        record_id: str,
        data_type: DataType,
        created_timestamp: datetime | None = None,
        checksum: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Register a new record for retention tracking.

        Args:
            record_id: Unique identifier for the record
            data_type: Type of data (order, trade, etc.)
            created_timestamp: When record was created (defaults to now)
            checksum: Optional checksum for integrity verification
            metadata: Additional metadata to store

        Returns:
            True if registered successfully
        """
        if created_timestamp is None:
            created_timestamp = datetime.now(timezone.utc)

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO retention_records
                    (record_id, data_type, created_timestamp, status,
                     storage_location, checksum, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_id,
                    data_type.value,
                    created_timestamp.isoformat(),
                    RetentionStatus.ACTIVE.value,
                    "primary",
                    checksum,
                    json.dumps(metadata) if metadata else None,
                ))
                conn.commit()

            logger.debug(f"Registered record {record_id} for retention ({data_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to register record {record_id}: {e}")
            return False

    def can_delete(
        self,
        record_id: str,
        requester: str = "system",
    ) -> tuple[bool, str]:
        """
        Check if a record can be deleted.

        Returns:
            Tuple of (can_delete, reason)
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data_type, created_timestamp, status, legal_hold_reason
                    FROM retention_records WHERE record_id = ?
                """, (record_id,))

                row = cursor.fetchone()

                if not row:
                    return True, "Record not tracked"

                data_type = DataType(row[0])
                created = datetime.fromisoformat(row[1])
                status = RetentionStatus(row[2])
                legal_hold_reason = row[3]

                # Check legal hold
                if status == RetentionStatus.LEGAL_HOLD:
                    self._log_deletion_attempt(record_id, True, f"Legal hold: {legal_hold_reason}", requester)
                    return False, f"Record under legal hold: {legal_hold_reason}"

                # Check active legal holds that might apply
                for hold in self._legal_holds.values():
                    if hold.released_timestamp is not None:
                        continue

                    # Check if hold applies to this data type
                    if data_type in hold.data_types or data_type == DataType.AUDIT_EVENT:
                        # Check date range
                        if hold.start_date and created < hold.start_date:
                            continue
                        if hold.end_date and created > hold.end_date:
                            continue

                        self._log_deletion_attempt(record_id, True, f"Active legal hold: {hold.hold_id}", requester)
                        return False, f"Active legal hold: {hold.hold_id} - {hold.reason}"

                # Check retention period
                policy = self._policies.get(data_type)
                if policy and policy.is_within_retention(created):
                    years_remaining = (policy.get_retention_end(created) - datetime.now(timezone.utc)).days / 365
                    self._log_deletion_attempt(
                        record_id, True,
                        f"Within retention period ({years_remaining:.1f} years remaining)",
                        requester
                    )
                    return False, f"Record within mandatory retention period ({policy.legal_reference})"

                # Can be deleted
                self._log_deletion_attempt(record_id, False, "Retention period expired", requester)
                return True, "Retention period expired"

        except Exception as e:
            logger.error(f"Error checking deletion for {record_id}: {e}")
            return False, f"Error checking retention: {e}"

    def _log_deletion_attempt(
        self,
        record_id: str,
        blocked: bool,
        reason: str,
        requester: str,
    ) -> None:
        """Log a deletion attempt for audit purposes."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO deletion_attempts
                    (record_id, attempt_timestamp, blocked, reason, requester)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    record_id,
                    datetime.now(timezone.utc).isoformat(),
                    1 if blocked else 0,
                    reason,
                    requester,
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log deletion attempt: {e}")

    def create_legal_hold(
        self,
        reason: str,
        requester: str,
        data_types: list[DataType] | None = None,
        record_ids: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        """
        Create a legal hold preventing deletion of matching records.

        Args:
            reason: Reason for the hold
            requester: Person/system creating the hold
            data_types: Types of data to hold (all if None)
            record_ids: Specific record IDs to hold
            start_date: Hold records from this date
            end_date: Hold records until this date

        Returns:
            Hold ID
        """
        self._hold_counter += 1
        hold_id = f"HOLD-{datetime.now().strftime('%Y%m%d')}-{self._hold_counter:04d}"

        hold = LegalHold(
            hold_id=hold_id,
            reason=reason,
            requester=requester,
            created_timestamp=datetime.now(timezone.utc),
            data_types=data_types or list(DataType),
            record_ids=record_ids or [],
            start_date=start_date,
            end_date=end_date,
        )

        self._legal_holds[hold_id] = hold

        # Store in database
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO legal_holds
                    (hold_id, reason, requester, created_timestamp,
                     data_types, record_ids, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hold_id,
                    reason,
                    requester,
                    hold.created_timestamp.isoformat(),
                    json.dumps([dt.value for dt in hold.data_types]),
                    json.dumps(hold.record_ids),
                    start_date.isoformat() if start_date else None,
                    end_date.isoformat() if end_date else None,
                ))

                # Mark specific records if provided
                if record_ids:
                    for record_id in record_ids:
                        cursor.execute("""
                            UPDATE retention_records
                            SET status = ?, legal_hold_reason = ?, legal_hold_timestamp = ?
                            WHERE record_id = ?
                        """, (
                            RetentionStatus.LEGAL_HOLD.value,
                            f"{hold_id}: {reason}",
                            datetime.now(timezone.utc).isoformat(),
                            record_id,
                        ))

                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store legal hold: {e}")

        logger.warning(
            f"LEGAL HOLD created: {hold_id} by {requester} - {reason} "
            f"(data_types={len(hold.data_types)}, records={len(hold.record_ids)})"
        )

        return hold_id

    def release_legal_hold(self, hold_id: str, released_by: str) -> bool:
        """
        Release a legal hold.

        Args:
            hold_id: ID of the hold to release
            released_by: Person/system releasing the hold

        Returns:
            True if released successfully
        """
        if hold_id not in self._legal_holds:
            logger.error(f"Legal hold {hold_id} not found")
            return False

        hold = self._legal_holds[hold_id]
        hold.released_timestamp = datetime.now(timezone.utc)
        hold.released_by = released_by

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE legal_holds
                    SET released_timestamp = ?, released_by = ?
                    WHERE hold_id = ?
                """, (
                    hold.released_timestamp.isoformat(),
                    released_by,
                    hold_id,
                ))

                # Update record statuses if specific records were held
                if hold.record_ids:
                    for record_id in hold.record_ids:
                        cursor.execute("""
                            UPDATE retention_records
                            SET status = 'active', legal_hold_reason = NULL, legal_hold_timestamp = NULL
                            WHERE record_id = ? AND legal_hold_reason LIKE ?
                        """, (record_id, f"{hold_id}%"))

                conn.commit()
        except Exception as e:
            logger.error(f"Failed to release legal hold: {e}")
            return False

        logger.info(f"Legal hold {hold_id} released by {released_by}")
        return True

    def archive_eligible_records(self) -> int:
        """
        Move eligible records to archive storage.

        Returns:
            Number of records archived
        """
        archived_count = 0
        now = datetime.now(timezone.utc)

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()

                for data_type, policy in self._policies.items():
                    archive_cutoff = (now - timedelta(days=policy.archive_after_days)).isoformat()

                    cursor.execute("""
                        SELECT record_id FROM retention_records
                        WHERE data_type = ?
                        AND status = 'active'
                        AND created_timestamp < ?
                        AND storage_location = 'primary'
                    """, (data_type.value, archive_cutoff))

                    for (record_id,) in cursor.fetchall():
                        # Move to archive (in production, would move actual data)
                        cursor.execute("""
                            UPDATE retention_records
                            SET status = 'archived',
                                archived_timestamp = ?,
                                storage_location = 'archive'
                            WHERE record_id = ?
                        """, (now.isoformat(), record_id))
                        archived_count += 1

                conn.commit()

        except Exception as e:
            logger.error(f"Error archiving records: {e}")

        if archived_count > 0:
            logger.info(f"Archived {archived_count} records")

        return archived_count

    def get_retention_status(self, record_id: str) -> dict[str, Any] | None:
        """
        Get detailed retention status for a record.

        Returns:
            Dictionary with retention details or None if not found
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data_type, created_timestamp, status, archived_timestamp,
                           legal_hold_reason, storage_location, metadata
                    FROM retention_records WHERE record_id = ?
                """, (record_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                data_type = DataType(row[0])
                created = datetime.fromisoformat(row[1])
                policy = self._policies.get(data_type)

                retention_end = policy.get_retention_end(created) if policy else None

                return {
                    "record_id": record_id,
                    "data_type": data_type.value,
                    "created_timestamp": row[1],
                    "status": row[2],
                    "archived_timestamp": row[3],
                    "legal_hold_reason": row[4],
                    "storage_location": row[5],
                    "retention_policy": {
                        "retention_years": policy.retention_years if policy else None,
                        "legal_reference": policy.legal_reference if policy else None,
                        "retention_end": retention_end.isoformat() if retention_end else None,
                        "days_until_expiry": (retention_end - datetime.now(timezone.utc)).days if retention_end else None,
                    },
                    "can_delete": self.can_delete(record_id)[0],
                }

        except Exception as e:
            logger.error(f"Error getting retention status: {e}")
            return None

    def get_compliance_report(self) -> dict[str, Any]:
        """
        Generate a compliance report for retention status.

        Returns:
            Comprehensive compliance report
        """
        report = {
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
            "policies": {},
            "record_counts": {},
            "status_breakdown": {},
            "legal_holds": {
                "active": 0,
                "released": 0,
                "total": len(self._legal_holds),
            },
            "deletion_attempts": {
                "total": 0,
                "blocked": 0,
            },
            "compliance_status": "COMPLIANT",
            "issues": [],
        }

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()

                # Policy summary
                for data_type, policy in self._policies.items():
                    report["policies"][data_type.value] = {
                        "retention_years": policy.retention_years,
                        "archive_after_days": policy.archive_after_days,
                        "legal_reference": policy.legal_reference,
                    }

                # Record counts by type
                cursor.execute("""
                    SELECT data_type, COUNT(*) FROM retention_records GROUP BY data_type
                """)
                for row in cursor.fetchall():
                    report["record_counts"][row[0]] = row[1]

                # Status breakdown
                cursor.execute("""
                    SELECT status, COUNT(*) FROM retention_records GROUP BY status
                """)
                for row in cursor.fetchall():
                    report["status_breakdown"][row[0]] = row[1]

                # Legal holds
                active_holds = [h for h in self._legal_holds.values() if h.released_timestamp is None]
                released_holds = [h for h in self._legal_holds.values() if h.released_timestamp is not None]
                report["legal_holds"]["active"] = len(active_holds)
                report["legal_holds"]["released"] = len(released_holds)
                report["legal_holds"]["details"] = [
                    {
                        "hold_id": h.hold_id,
                        "reason": h.reason,
                        "requester": h.requester,
                        "created": h.created_timestamp.isoformat(),
                    }
                    for h in active_holds
                ]

                # Deletion attempts
                cursor.execute("""
                    SELECT COUNT(*), SUM(blocked) FROM deletion_attempts
                """)
                row = cursor.fetchone()
                if row and row[0]:
                    report["deletion_attempts"]["total"] = row[0]
                    report["deletion_attempts"]["blocked"] = row[1] or 0

                # Check for compliance issues
                # Issue 1: Records past retention with no deletion
                now = datetime.now(timezone.utc)
                for data_type, policy in self._policies.items():
                    cutoff = (now - timedelta(days=policy.retention_years * 365)).isoformat()
                    cursor.execute("""
                        SELECT COUNT(*) FROM retention_records
                        WHERE data_type = ? AND created_timestamp < ? AND status != 'legal_hold'
                    """, (data_type.value, cutoff))
                    expired_count = cursor.fetchone()[0]
                    if expired_count > 0:
                        report["issues"].append({
                            "type": "expired_records",
                            "data_type": data_type.value,
                            "count": expired_count,
                            "severity": "LOW",
                            "description": f"{expired_count} records past retention period (may be deleted)",
                        })

                # Issue 2: Records approaching expiry
                for data_type, policy in self._policies.items():
                    warning_date = (now - timedelta(days=(policy.retention_years * 365) - 90)).isoformat()
                    cutoff = (now - timedelta(days=policy.retention_years * 365)).isoformat()
                    cursor.execute("""
                        SELECT COUNT(*) FROM retention_records
                        WHERE data_type = ? AND created_timestamp < ? AND created_timestamp > ?
                    """, (data_type.value, warning_date, cutoff))
                    warning_count = cursor.fetchone()[0]
                    if warning_count > 0:
                        report["issues"].append({
                            "type": "approaching_expiry",
                            "data_type": data_type.value,
                            "count": warning_count,
                            "severity": "INFO",
                            "description": f"{warning_count} records expiring within 90 days",
                        })

                if report["issues"]:
                    # Check if any critical issues
                    critical_issues = [i for i in report["issues"] if i["severity"] == "CRITICAL"]
                    if critical_issues:
                        report["compliance_status"] = "NON_COMPLIANT"
                    else:
                        report["compliance_status"] = "COMPLIANT_WITH_WARNINGS"

        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            report["compliance_status"] = "UNKNOWN"
            report["issues"].append({
                "type": "error",
                "severity": "CRITICAL",
                "description": f"Error generating report: {e}",
            })

        return report

    def get_records_by_date_range(
        self,
        data_type: DataType | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Query records by date range for audit purposes."""
        records = []

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM retention_records WHERE 1=1"
                params = []

                if data_type:
                    query += " AND data_type = ?"
                    params.append(data_type.value)

                if start_date:
                    query += " AND created_timestamp >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND created_timestamp <= ?"
                    params.append(end_date.isoformat())

                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]

                for row in cursor.fetchall():
                    records.append(dict(zip(columns, row)))

        except Exception as e:
            logger.error(f"Error querying records: {e}")

        return records

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for monitoring."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM retention_records")
                total_records = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM retention_records WHERE status = 'legal_hold'")
                held_records = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM retention_records WHERE status = 'archived'")
                archived_records = cursor.fetchone()[0]

                active_holds = len([h for h in self._legal_holds.values() if h.released_timestamp is None])

                return {
                    "total_records": total_records,
                    "active_records": total_records - archived_records,
                    "archived_records": archived_records,
                    "held_records": held_records,
                    "active_legal_holds": active_holds,
                    "policies_count": len(self._policies),
                    "default_retention_years": 7,
                }

        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {"error": str(e)}


# Integration helper for AuditLogger
def create_retention_aware_logger(
    audit_logger,
    retention_manager: DataRetentionManager,
) -> None:
    """
    Patch an AuditLogger to automatically register records with RetentionManager.

    This ensures all logged events are tracked for retention compliance.
    """
    original_write = audit_logger._write_entry

    def retention_aware_write(file_path, entry):
        # Call original write
        original_write(file_path, entry)

        # Map entry type to data type
        entry_type_map = {
            "trade": DataType.TRADE,
            "decision": DataType.DECISION,
            "compliance_check": DataType.COMPLIANCE_CHECK,
            "risk_alert": DataType.RISK_CHECK,
            "agent_surveillance_alert": DataType.SURVEILLANCE_ALERT,
            "agent_stor_created": DataType.STOR,
            "agent_stor_submitted": DataType.STOR,
        }

        data_type = entry_type_map.get(entry.entry_type)
        if not data_type:
            # Default to audit event
            data_type = DataType.AUDIT_EVENT

        # Register with retention manager
        record_id = entry.event_id or f"{entry.entry_type}-{entry.timestamp}"
        retention_manager.register_record(
            record_id=record_id,
            data_type=data_type,
            created_timestamp=datetime.fromisoformat(entry.timestamp),
            metadata={"entry_type": entry.entry_type, "agent": entry.agent_name},
        )

    audit_logger._write_entry = retention_aware_write
    logger.info("AuditLogger patched for retention tracking")

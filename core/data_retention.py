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

        except sqlite3.IntegrityError as e:
            logger.warning(f"Record {record_id} already exists in retention tracking: {e}")
            return False
        except sqlite3.OperationalError as e:
            logger.exception(f"Database operational error registering record {record_id}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error registering record {record_id}")
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

        except sqlite3.OperationalError as e:
            logger.exception(f"Database error checking deletion for {record_id}")
            return False, f"Database error checking retention: {e}"
        except ValueError as e:
            logger.error(f"Invalid data for record {record_id}: {e}")
            return False, f"Invalid record data: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error checking deletion for {record_id}")
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
        except sqlite3.OperationalError as e:
            logger.exception(f"Database error logging deletion attempt for {record_id}")
        except Exception as e:
            logger.exception(f"Unexpected error logging deletion attempt for {record_id}")

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
        hold_id = f"HOLD-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._hold_counter:04d}"

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


# =============================================================================
# P3: Configurable Retention Policies Per Data Type
# =============================================================================

@dataclass
class ConfigurableRetentionPolicy:
    """
    Extended retention policy with per-data-type configuration.

    P3 Enhancement: Supports:
    - Override default retention periods
    - Configure archive timing per type
    - Set custom storage locations
    - Define compliance requirements
    """
    data_type: DataType
    retention_years: int
    archive_after_days: int = 365
    legal_reference: str = ""
    deletion_requires_approval: bool = True
    # P3 extensions
    cold_storage_after_days: int = 730  # Move to cold storage after this
    cold_storage_location: str = "cold"
    encryption_required: bool = True
    compression_enabled: bool = True
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def get_cold_storage_date(self, record_date: datetime) -> datetime:
        """Calculate when record should be moved to cold storage."""
        return record_date + timedelta(days=self.cold_storage_after_days)

    def should_move_to_cold_storage(self, record_date: datetime) -> bool:
        """Check if a record should be moved to cold storage."""
        return datetime.now(timezone.utc) > self.get_cold_storage_date(record_date)


class RetentionPolicyManager:
    """
    Manages configurable retention policies per data type.

    P3 Enhancement: Provides:
    - Runtime policy configuration
    - Policy validation
    - Policy export/import
    - Compliance checking
    """

    def __init__(self):
        """Initialize with default policies."""
        self._policies: dict[DataType, ConfigurableRetentionPolicy] = {}
        self._load_default_policies()

    def _load_default_policies(self) -> None:
        """Load default retention policies."""
        for data_type, policy in DEFAULT_RETENTION_POLICIES.items():
            self._policies[data_type] = ConfigurableRetentionPolicy(
                data_type=data_type,
                retention_years=policy.retention_years,
                archive_after_days=policy.archive_after_days,
                legal_reference=policy.legal_reference,
                deletion_requires_approval=policy.deletion_requires_approval,
            )

    def set_policy(
        self,
        data_type: DataType,
        retention_years: int | None = None,
        archive_after_days: int | None = None,
        cold_storage_after_days: int | None = None,
        **kwargs,
    ) -> ConfigurableRetentionPolicy:
        """
        Set or update a retention policy for a data type.

        Args:
            data_type: Type of data
            retention_years: Override retention period
            archive_after_days: Override archive timing
            cold_storage_after_days: When to move to cold storage
            **kwargs: Additional policy attributes

        Returns:
            Updated policy
        """
        if data_type not in self._policies:
            # Create new policy with defaults
            default = DEFAULT_RETENTION_POLICIES.get(data_type)
            if default:
                self._policies[data_type] = ConfigurableRetentionPolicy(
                    data_type=data_type,
                    retention_years=default.retention_years,
                    archive_after_days=default.archive_after_days,
                    legal_reference=default.legal_reference,
                    deletion_requires_approval=default.deletion_requires_approval,
                )
            else:
                self._policies[data_type] = ConfigurableRetentionPolicy(
                    data_type=data_type,
                    retention_years=7,  # Default to 7 years
                )

        policy = self._policies[data_type]

        # Update provided fields
        if retention_years is not None:
            policy.retention_years = retention_years
        if archive_after_days is not None:
            policy.archive_after_days = archive_after_days
        if cold_storage_after_days is not None:
            policy.cold_storage_after_days = cold_storage_after_days

        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        logger.info(f"Updated retention policy for {data_type.value}: {retention_years or policy.retention_years} years")
        return policy

    def get_policy(self, data_type: DataType) -> ConfigurableRetentionPolicy | None:
        """Get policy for a data type."""
        return self._policies.get(data_type)

    def get_all_policies(self) -> dict[DataType, ConfigurableRetentionPolicy]:
        """Get all policies."""
        return self._policies.copy()

    def validate_policy(self, policy: ConfigurableRetentionPolicy) -> list[str]:
        """
        Validate a policy meets minimum requirements.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check minimum retention for regulated data
        min_retention = {
            DataType.ORDER: 5,
            DataType.TRADE: 5,
            DataType.FILL: 5,
            DataType.DECISION: 5,
            DataType.TRANSACTION_REPORT: 5,
            DataType.STOR: 10,
        }

        min_years = min_retention.get(policy.data_type, 0)
        if policy.retention_years < min_years:
            errors.append(
                f"Retention for {policy.data_type.value} must be at least {min_years} years "
                f"per {policy.legal_reference or 'regulatory requirements'}"
            )

        # Cold storage should be before retention expiry
        cold_storage_years = policy.cold_storage_after_days / 365
        if cold_storage_years > policy.retention_years:
            errors.append(
                f"Cold storage transition ({cold_storage_years:.1f} years) exceeds "
                f"retention period ({policy.retention_years} years)"
            )

        # Archive should be before cold storage
        archive_years = policy.archive_after_days / 365
        if archive_years > cold_storage_years:
            errors.append(
                f"Archive timing ({archive_years:.1f} years) exceeds "
                f"cold storage timing ({cold_storage_years:.1f} years)"
            )

        return errors

    def export_policies(self) -> dict[str, Any]:
        """Export policies as dictionary for serialization."""
        return {
            data_type.value: {
                "retention_years": p.retention_years,
                "archive_after_days": p.archive_after_days,
                "cold_storage_after_days": p.cold_storage_after_days,
                "cold_storage_location": p.cold_storage_location,
                "legal_reference": p.legal_reference,
                "deletion_requires_approval": p.deletion_requires_approval,
                "encryption_required": p.encryption_required,
                "compression_enabled": p.compression_enabled,
            }
            for data_type, p in self._policies.items()
        }

    def import_policies(self, data: dict[str, Any]) -> int:
        """
        Import policies from dictionary.

        Args:
            data: Dictionary of policy configurations

        Returns:
            Number of policies imported
        """
        imported = 0
        for type_name, config in data.items():
            try:
                data_type = DataType(type_name)
                self.set_policy(
                    data_type=data_type,
                    retention_years=config.get("retention_years"),
                    archive_after_days=config.get("archive_after_days"),
                    cold_storage_after_days=config.get("cold_storage_after_days"),
                    cold_storage_location=config.get("cold_storage_location", "cold"),
                    encryption_required=config.get("encryption_required", True),
                    compression_enabled=config.get("compression_enabled", True),
                )
                imported += 1
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to import policy for {type_name}: {e}")

        return imported


# =============================================================================
# P3: Archival to Cold Storage
# =============================================================================

class ColdStorageManager:
    """
    Manages archival to cold storage for long-term retention.

    P3 Enhancement: Supports:
    - Tiered storage (primary -> archive -> cold)
    - Configurable cold storage backends
    - Compression and encryption
    - Retrieval from cold storage
    """

    def __init__(
        self,
        primary_path: str = "data",
        archive_path: str = "archive",
        cold_storage_path: str = "cold_storage",
        compression_enabled: bool = True,
        encryption_key: bytes | None = None,
    ):
        """
        Initialize cold storage manager.

        Args:
            primary_path: Path for primary data
            archive_path: Path for archived data
            cold_storage_path: Path for cold storage
            compression_enabled: Enable compression for cold storage
            encryption_key: Optional encryption key (32 bytes for AES-256)
        """
        self._primary_path = Path(primary_path)
        self._archive_path = Path(archive_path)
        self._cold_storage_path = Path(cold_storage_path)
        self._compression_enabled = compression_enabled
        self._encryption_key = encryption_key

        # Create directories
        for path in [self._primary_path, self._archive_path, self._cold_storage_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Track movements
        self._movements: list[dict[str, Any]] = []

        logger.info(f"ColdStorageManager initialized: cold_storage={cold_storage_path}")

    def move_to_cold_storage(
        self,
        record_id: str,
        data: bytes,
        data_type: DataType,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Move data to cold storage.

        Args:
            record_id: Unique record identifier
            data: Raw data bytes
            data_type: Type of data
            metadata: Additional metadata to store

        Returns:
            Path to cold storage file
        """
        import gzip
        import hashlib

        # Prepare metadata
        cold_metadata = {
            "record_id": record_id,
            "data_type": data_type.value,
            "original_size": len(data),
            "moved_timestamp": datetime.now(timezone.utc).isoformat(),
            "compressed": self._compression_enabled,
            "encrypted": self._encryption_key is not None,
            **(metadata or {}),
        }

        # Calculate checksum before any transformation
        checksum = hashlib.sha256(data).hexdigest()
        cold_metadata["checksum"] = checksum

        # Compress if enabled
        if self._compression_enabled:
            data = gzip.compress(data)
            cold_metadata["compressed_size"] = len(data)

        # Encrypt if key provided (simplified - in production use proper encryption)
        if self._encryption_key:
            # Simple XOR for demo - use proper AES in production
            data = bytes(b ^ self._encryption_key[i % len(self._encryption_key)] for i, b in enumerate(data))

        # Create cold storage file path
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m")
        cold_dir = self._cold_storage_path / date_prefix / data_type.value
        cold_dir.mkdir(parents=True, exist_ok=True)

        file_path = cold_dir / f"{record_id}.cold"
        metadata_path = cold_dir / f"{record_id}.meta.json"

        # Write data and metadata
        with open(file_path, "wb") as f:
            f.write(data)

        with open(metadata_path, "w") as f:
            json.dump(cold_metadata, f, indent=2)

        # Track movement
        self._movements.append({
            "record_id": record_id,
            "data_type": data_type.value,
            "cold_path": str(file_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_size": cold_metadata["original_size"],
            "final_size": len(data),
        })

        logger.info(f"Moved {record_id} to cold storage: {file_path}")
        return str(file_path)

    def retrieve_from_cold_storage(
        self,
        record_id: str,
        data_type: DataType,
    ) -> tuple[bytes | None, dict[str, Any] | None]:
        """
        Retrieve data from cold storage.

        Args:
            record_id: Record identifier
            data_type: Type of data

        Returns:
            Tuple of (data bytes, metadata) or (None, None) if not found
        """
        import gzip
        import hashlib

        # Search for file (check recent date prefixes first)
        now = datetime.now(timezone.utc)
        for months_back in range(120):  # Look back up to 10 years
            check_date = now - timedelta(days=months_back * 30)
            date_prefix = check_date.strftime("%Y/%m")
            cold_dir = self._cold_storage_path / date_prefix / data_type.value

            file_path = cold_dir / f"{record_id}.cold"
            metadata_path = cold_dir / f"{record_id}.meta.json"

            if file_path.exists():
                break
        else:
            logger.warning(f"Cold storage file not found for {record_id}")
            return None, None

        # Read metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Read data
        with open(file_path, "rb") as f:
            data = f.read()

        # Decrypt if needed
        if metadata.get("encrypted") and self._encryption_key:
            data = bytes(b ^ self._encryption_key[i % len(self._encryption_key)] for i, b in enumerate(data))

        # Decompress if needed
        if metadata.get("compressed"):
            data = gzip.decompress(data)

        # Verify checksum
        checksum = hashlib.sha256(data).hexdigest()
        if checksum != metadata.get("checksum"):
            logger.error(f"Checksum mismatch for {record_id}! Data may be corrupted.")
            return None, metadata

        logger.info(f"Retrieved {record_id} from cold storage")
        return data, metadata

    def get_cold_storage_stats(self) -> dict[str, Any]:
        """Get statistics about cold storage."""
        stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "by_data_type": {},
            "by_year": {},
        }

        for year_dir in self._cold_storage_path.iterdir():
            if not year_dir.is_dir():
                continue

            year = year_dir.name
            stats["by_year"][year] = {"files": 0, "size": 0}

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for type_dir in month_dir.iterdir():
                    if not type_dir.is_dir():
                        continue

                    data_type = type_dir.name
                    if data_type not in stats["by_data_type"]:
                        stats["by_data_type"][data_type] = {"files": 0, "size": 0}

                    for file in type_dir.glob("*.cold"):
                        size = file.stat().st_size
                        stats["total_files"] += 1
                        stats["total_size_bytes"] += size
                        stats["by_year"][year]["files"] += 1
                        stats["by_year"][year]["size"] += size
                        stats["by_data_type"][data_type]["files"] += 1
                        stats["by_data_type"][data_type]["size"] += size

        return stats

    def get_recent_movements(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent cold storage movements."""
        return self._movements[-limit:]


# =============================================================================
# P3: Retention Compliance Reporting
# =============================================================================

@dataclass
class ComplianceMetric:
    """Single compliance metric."""
    name: str
    value: float | int | str
    status: str  # "pass", "warning", "fail"
    threshold: float | int | str | None = None
    description: str = ""


class RetentionComplianceReporter:
    """
    Generates detailed compliance reports for data retention.

    P3 Enhancement: Provides:
    - Detailed compliance metrics
    - Regulatory requirement mapping
    - Trend analysis
    - Export in multiple formats
    """

    def __init__(
        self,
        retention_manager: DataRetentionManager,
        policy_manager: RetentionPolicyManager | None = None,
    ):
        """Initialize compliance reporter."""
        self._retention_manager = retention_manager
        self._policy_manager = policy_manager or RetentionPolicyManager()
        self._report_history: list[dict[str, Any]] = []

    def generate_compliance_report(
        self,
        include_recommendations: bool = True,
        include_trend_analysis: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive compliance report.

        Args:
            include_recommendations: Include improvement recommendations
            include_trend_analysis: Include trend analysis from history

        Returns:
            Complete compliance report
        """
        report = {
            "report_id": f"CR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "metrics": [],
            "regulatory_compliance": {},
            "policy_review": {},
            "issues": [],
            "recommendations": [],
        }

        # Get base compliance report from retention manager
        base_report = self._retention_manager.get_compliance_report()
        report["summary"] = {
            "overall_status": base_report.get("compliance_status", "UNKNOWN"),
            "total_records": sum(base_report.get("record_counts", {}).values()),
            "active_legal_holds": base_report.get("legal_holds", {}).get("active", 0),
            "blocked_deletions": base_report.get("deletion_attempts", {}).get("blocked", 0),
        }

        # Generate metrics
        report["metrics"] = self._calculate_metrics(base_report)

        # Regulatory compliance mapping
        report["regulatory_compliance"] = self._assess_regulatory_compliance(base_report)

        # Policy review
        report["policy_review"] = self._review_policies()

        # Issues
        report["issues"] = base_report.get("issues", [])

        # Recommendations
        if include_recommendations:
            report["recommendations"] = self._generate_recommendations(base_report)

        # Trend analysis
        if include_trend_analysis and self._report_history:
            report["trend_analysis"] = self._analyze_trends()

        # Store in history
        self._report_history.append({
            "timestamp": report["generated_timestamp"],
            "status": report["summary"]["overall_status"],
            "total_records": report["summary"]["total_records"],
            "issues_count": len(report["issues"]),
        })
        # Keep last 365 days of history
        if len(self._report_history) > 365:
            self._report_history = self._report_history[-365:]

        return report

    def _calculate_metrics(self, base_report: dict[str, Any]) -> list[dict[str, Any]]:
        """Calculate compliance metrics."""
        metrics = []

        # Record coverage metric
        record_counts = base_report.get("record_counts", {})
        total_records = sum(record_counts.values())
        tracked_types = len(record_counts)
        total_types = len(DataType)

        metrics.append({
            "name": "data_type_coverage",
            "value": tracked_types / total_types * 100 if total_types > 0 else 0,
            "status": "pass" if tracked_types >= total_types * 0.8 else "warning",
            "threshold": "80%",
            "description": f"Percentage of data types with retention tracking ({tracked_types}/{total_types})",
        })

        # Legal hold effectiveness
        legal_holds = base_report.get("legal_holds", {})
        active_holds = legal_holds.get("active", 0)
        metrics.append({
            "name": "active_legal_holds",
            "value": active_holds,
            "status": "pass",  # Informational
            "description": "Number of active legal holds",
        })

        # Deletion protection rate
        deletion_attempts = base_report.get("deletion_attempts", {})
        total_attempts = deletion_attempts.get("total", 0)
        blocked = deletion_attempts.get("blocked", 0)
        protection_rate = (blocked / total_attempts * 100) if total_attempts > 0 else 100

        metrics.append({
            "name": "deletion_protection_rate",
            "value": protection_rate,
            "status": "pass" if protection_rate >= 95 else "warning",
            "threshold": "95%",
            "description": "Percentage of premature deletion attempts blocked",
        })

        # Status distribution
        status_breakdown = base_report.get("status_breakdown", {})
        archived_pct = (status_breakdown.get("archived", 0) / total_records * 100) if total_records > 0 else 0
        metrics.append({
            "name": "archive_rate",
            "value": archived_pct,
            "status": "pass",
            "description": "Percentage of records in archive storage",
        })

        return metrics

    def _assess_regulatory_compliance(self, base_report: dict[str, Any]) -> dict[str, Any]:
        """Assess compliance with specific regulations."""
        compliance = {
            "mifid_ii": {
                "status": "COMPLIANT",
                "requirements": [],
            },
            "emir": {
                "status": "COMPLIANT",
                "requirements": [],
            },
            "mar": {
                "status": "COMPLIANT",
                "requirements": [],
            },
        }

        # MiFID II - 5 year retention for orders/trades
        policies = base_report.get("policies", {})

        mifid_types = ["order", "trade", "fill", "decision"]
        for data_type in mifid_types:
            policy = policies.get(data_type, {})
            retention_years = policy.get("retention_years", 0)

            req = {
                "data_type": data_type,
                "required_years": 5,
                "configured_years": retention_years,
                "compliant": retention_years >= 5,
            }
            compliance["mifid_ii"]["requirements"].append(req)

            if not req["compliant"]:
                compliance["mifid_ii"]["status"] = "NON_COMPLIANT"

        # EMIR - 5 years post-contract for transaction reports
        tr_policy = policies.get("transaction_report", {})
        tr_years = tr_policy.get("retention_years", 0)
        compliance["emir"]["requirements"].append({
            "data_type": "transaction_report",
            "required_years": 5,
            "configured_years": tr_years,
            "compliant": tr_years >= 5,
        })
        if tr_years < 5:
            compliance["emir"]["status"] = "NON_COMPLIANT"

        # MAR - Preserve surveillance alerts and STORs
        for data_type in ["surveillance_alert", "stor"]:
            policy = policies.get(data_type, {})
            retention_years = policy.get("retention_years", 0)
            min_years = 10 if data_type == "stor" else 5

            req = {
                "data_type": data_type,
                "required_years": min_years,
                "configured_years": retention_years,
                "compliant": retention_years >= min_years,
            }
            compliance["mar"]["requirements"].append(req)

            if not req["compliant"]:
                compliance["mar"]["status"] = "NON_COMPLIANT"

        return compliance

    def _review_policies(self) -> dict[str, Any]:
        """Review configured policies."""
        review = {
            "total_policies": 0,
            "validation_errors": [],
            "policies": {},
        }

        for data_type, policy in self._policy_manager.get_all_policies().items():
            errors = self._policy_manager.validate_policy(policy)

            review["policies"][data_type.value] = {
                "retention_years": policy.retention_years,
                "archive_after_days": policy.archive_after_days,
                "cold_storage_after_days": policy.cold_storage_after_days,
                "valid": len(errors) == 0,
                "errors": errors,
            }
            review["total_policies"] += 1
            review["validation_errors"].extend(errors)

        return review

    def _generate_recommendations(self, base_report: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate improvement recommendations."""
        recommendations = []

        # Check for policies below recommended retention
        policies = base_report.get("policies", {})
        for data_type, policy in policies.items():
            retention_years = policy.get("retention_years", 0)
            if retention_years < 7:
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "retention",
                    "data_type": data_type,
                    "recommendation": f"Consider increasing retention for {data_type} from {retention_years} to 7 years "
                                    f"to align with French AMF guidance",
                })

        # Check for high deletion attempt rates
        deletion_attempts = base_report.get("deletion_attempts", {})
        if deletion_attempts.get("total", 0) > 100:
            blocked_rate = deletion_attempts.get("blocked", 0) / deletion_attempts["total"]
            if blocked_rate > 0.5:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "process",
                    "recommendation": "High rate of blocked deletions detected. "
                                    "Consider reviewing data lifecycle processes to prevent premature deletion attempts.",
                })

        # Check for expired records not deleted
        issues = base_report.get("issues", [])
        expired_issues = [i for i in issues if i.get("type") == "expired_records"]
        for issue in expired_issues:
            if issue.get("count", 0) > 1000:
                recommendations.append({
                    "priority": "LOW",
                    "category": "cleanup",
                    "data_type": issue.get("data_type"),
                    "recommendation": f"Consider scheduling cleanup for {issue.get('count')} expired "
                                    f"{issue.get('data_type')} records to optimize storage.",
                })

        return recommendations

    def _analyze_trends(self) -> dict[str, Any]:
        """Analyze trends from historical reports."""
        if len(self._report_history) < 2:
            return {"available": False, "message": "Insufficient history for trend analysis"}

        recent = self._report_history[-30:]  # Last 30 reports

        # Calculate trend metrics
        record_counts = [r.get("total_records", 0) for r in recent]
        issue_counts = [r.get("issues_count", 0) for r in recent]

        return {
            "available": True,
            "period_days": len(recent),
            "record_growth": {
                "start": record_counts[0] if record_counts else 0,
                "end": record_counts[-1] if record_counts else 0,
                "change_pct": ((record_counts[-1] - record_counts[0]) / record_counts[0] * 100)
                            if record_counts and record_counts[0] > 0 else 0,
            },
            "issue_trend": {
                "average_issues": sum(issue_counts) / len(issue_counts) if issue_counts else 0,
                "recent_issues": issue_counts[-1] if issue_counts else 0,
                "trend": "improving" if issue_counts and issue_counts[-1] < issue_counts[0] else "stable",
            },
        }

    def export_report_csv(self, report: dict[str, Any]) -> str:
        """Export report as CSV format."""
        lines = []

        # Header
        lines.append("AI Trading Firm - Data Retention Compliance Report")
        lines.append(f"Generated: {report.get('generated_timestamp', '')}")
        lines.append(f"Report ID: {report.get('report_id', '')}")
        lines.append("")

        # Summary
        summary = report.get("summary", {})
        lines.append("SUMMARY")
        lines.append(f"Overall Status,{summary.get('overall_status', '')}")
        lines.append(f"Total Records,{summary.get('total_records', 0)}")
        lines.append(f"Active Legal Holds,{summary.get('active_legal_holds', 0)}")
        lines.append("")

        # Metrics
        lines.append("COMPLIANCE METRICS")
        lines.append("Metric,Value,Status,Threshold,Description")
        for metric in report.get("metrics", []):
            lines.append(
                f"{metric.get('name', '')},{metric.get('value', '')},"
                f"{metric.get('status', '')},{metric.get('threshold', '')},"
                f"\"{metric.get('description', '')}\""
            )
        lines.append("")

        # Issues
        lines.append("ISSUES")
        lines.append("Type,Data Type,Count,Severity,Description")
        for issue in report.get("issues", []):
            lines.append(
                f"{issue.get('type', '')},{issue.get('data_type', '')},"
                f"{issue.get('count', '')},{issue.get('severity', '')},"
                f"\"{issue.get('description', '')}\""
            )

        return "\n".join(lines)

    def export_report_json(self, report: dict[str, Any]) -> str:
        """Export report as JSON format."""
        return json.dumps(report, indent=2, default=str)

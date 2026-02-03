# data_retention

**Path**: `C:\Users\Alexa\ai-trading-firm\core\data_retention.py`

## Overview

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

## Classes

### DataType

**Inherits from**: Enum

Types of data subject to retention requirements.

### RetentionStatus

**Inherits from**: Enum

Status of a data record for retention purposes.

### RetentionPolicy

Retention policy for a data type.

#### Methods

##### `def get_retention_end(self, record_date: datetime) -> datetime`

Calculate when retention period ends for a record.

##### `def get_archive_date(self, record_date: datetime) -> datetime`

Calculate when record should be archived.

##### `def is_within_retention(self, record_date: datetime) -> bool`

Check if a record is still within its retention period.

### RetentionRecord

Record tracking metadata for retention purposes.

### LegalHold

Legal hold preventing data deletion.

### DataRetentionManager

Manages data retention compliance per MiFID II, EMIR, and MAR.

Key responsibilities:
- Track retention status of all records
- Prevent premature deletion
- Manage legal holds
- Generate compliance reports
- Handle archival

#C3, #I1 - 7-year record retention enforcement

#### Methods

##### `def __init__(self, db_path: str, archive_path: str, policies: )`

Initialize the data retention manager.

Args:
    db_path: Path to retention tracking database
    archive_path: Path for archived data
    policies: Custom retention policies (uses defaults if None)

##### `def register_record(self, record_id: str, data_type: DataType, created_timestamp: , checksum: , metadata: ) -> bool`

Register a new record for retention tracking.

Args:
    record_id: Unique identifier for the record
    data_type: Type of data (order, trade, etc.)
    created_timestamp: When record was created (defaults to now)
    checksum: Optional checksum for integrity verification
    metadata: Additional metadata to store

Returns:
    True if registered successfully

##### `def can_delete(self, record_id: str, requester: str) -> tuple[bool, str]`

Check if a record can be deleted.

Returns:
    Tuple of (can_delete, reason)

##### `def create_legal_hold(self, reason: str, requester: str, data_types: , record_ids: , start_date: , end_date: ) -> str`

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

##### `def release_legal_hold(self, hold_id: str, released_by: str) -> bool`

Release a legal hold.

Args:
    hold_id: ID of the hold to release
    released_by: Person/system releasing the hold

Returns:
    True if released successfully

##### `def archive_eligible_records(self) -> int`

Move eligible records to archive storage.

Returns:
    Number of records archived

##### `def get_retention_status(self, record_id: str)`

Get detailed retention status for a record.

Returns:
    Dictionary with retention details or None if not found

##### `def get_compliance_report(self) -> dict[str, Any]`

Generate a compliance report for retention status.

Returns:
    Comprehensive compliance report

##### `def get_records_by_date_range(self, data_type: , start_date: , end_date: ) -> list[dict[str, Any]]`

Query records by date range for audit purposes.

##### `def get_summary(self) -> dict[str, Any]`

Get summary statistics for monitoring.

## Functions

### `def create_retention_aware_logger(audit_logger, retention_manager: DataRetentionManager) -> None`

Patch an AuditLogger to automatically register records with RetentionManager.

This ensures all logged events are tracked for retention compliance.

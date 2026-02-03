# regulatory_compliance

**Path**: `C:\Users\Alexa\ai-trading-firm\core\regulatory_compliance.py`

## Overview

Regulatory Compliance Module
============================

Implements EU/MiFID II/MAR compliance requirements.

Issues addressed:
- #C5: RTS 25 Order record keeping
- #C6: RTS 6 Algo trading kill switch
- #C7: MAR Art 16 Market abuse thresholds
- #C8: RTS 27 Best execution reporting
- #C9: RTS 28 Venue analysis
- #C10: EMIR Trade repository
- #C11: SFTR Securities financing
- #C12: MiFIR Art 26 Transaction reference
- #C13: RTS 24 Order ID format
- #C14-C17: Market abuse detection tuning
- #C18: RTS 6 Pre-trade risk controls
- #C19: Per-venue position limits
- #C20: Short selling locate
- #C21: Dark pool reporting
- #C22: Systematic internaliser
- #C23: TCA format
- #C24: Order execution policy
- #C25: Client categorization
- #C26: Cross-border reporting
- #C27: Clock synchronization
- #C28: Audit log rotation
- #C29: GDPR data handling
- #C30: Access control logs
- #C31: Change management
- #C32: Disaster recovery

## Classes

### RTS25OrderRecord

Complete order record per MiFID II RTS 25 (#C5).

Contains all 65 required fields for order record keeping.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary for storage/transmission.

##### `def validate_required_fields(cls, record: dict) -> list[str]`

Validate all required RTS 25 fields are present.

### RTS25RecordKeeper

RTS 25 compliant order record keeper (#C5).

Maintains complete order records with all required fields.

#### Methods

##### `def __init__(self, firm_id: str, country_code: str)`

##### `def create_order_record(self, order_id: str, client_id: str, instrument_isin: str, instrument_name: str, side: str, order_type: str, quantity: float, limit_price: , algo_id: , client_lei: ) -> RTS25OrderRecord`

Create a new RTS 25 compliant order record (#C5).

##### `def update_order_status(self, order_id: str, new_status: str, executed_qty: float, reason: ) -> None`

Update order status with timestamp (#C5).

##### `def get_record(self, order_id: str)`

Get order record.

##### `def export_records(self, start_date: datetime, end_date: datetime) -> list[dict]`

Export records for regulatory reporting.

### KillSwitchEventType

**Inherits from**: Enum

Types of kill switch events (#C6).

### KillSwitchAuditRecord

Audit record for kill switch events (#C6).

### KillSwitchAuditor

RTS 6 compliant kill switch audit trail (#C6).

Tracks all kill switch activations with required detail.

#### Methods

##### `def __init__(self)`

##### `def record_activation(self, triggered_by: str, reason: str, affected_algos: list[str], affected_orders: list[str], cancellation_latency_ms: float, success: bool, failure_reason: ) -> KillSwitchAuditRecord`

Record kill switch activation (#C6).

##### `def record_test(self, triggered_by: str, test_orders: list[str], latency_ms: float, success: bool) -> KillSwitchAuditRecord`

Record kill switch test (#C6).

##### `def is_test_overdue(self) -> bool`

Check if kill switch test is overdue.

##### `def get_audit_trail(self, days: int) -> list[dict]`

Get audit trail for reporting.

### MarketAbuseThresholds

Configurable market abuse detection thresholds (#C7, #C14-C17).

Calibrated for retail trading volumes.

### MarketAbuseDetector

MAR Art 16 compliant market abuse detection (#C7).

Configurable thresholds for different trading profiles.

#### Methods

##### `def __init__(self, thresholds: )`

##### `def configure_thresholds(self, thresholds: MarketAbuseThresholds) -> None`

Update detection thresholds (#C7).

##### `def record_order(self, order: dict) -> None`

Record order for analysis.

##### `def detect_wash_trading(self, client_id: str) -> list[dict]`

Detect potential wash trading (#C14).

Wash trading: buying and selling same security to create
misleading appearance of activity.

##### `def detect_spoofing(self, client_id: str) -> list[dict]`

Detect potential spoofing (#C15).

Spoofing: placing orders with intent to cancel before execution
to manipulate prices.

##### `def detect_layering(self, client_id: str, symbol: str) -> list[dict]`

Detect potential layering (#C16).

Layering: placing multiple orders at different price levels
with intent to cancel.

##### `def detect_quote_stuffing(self, client_id: str) -> list[dict]`

Detect potential quote stuffing (#C17).

Quote stuffing: submitting and cancelling large numbers of orders
to slow down other participants.

##### `def get_alerts(self, severity: ) -> list[dict]`

Get detected alerts.

### RTS27Report

RTS 27 Best Execution Report (#C8).

Published quarterly by execution venues.

### RTS28Report

RTS 28 Venue Analysis Report (#C9).

Published annually by investment firms.

### BestExecutionReporter

RTS 27/28 Best Execution Reporter (#C8, #C9).

Generates compliant execution reports.

#### Methods

##### `def __init__(self, firm_lei: str)`

##### `def record_execution(self, venue_mic: str, instrument_class: str, execution_time_ms: float, size: float, price: float, side: str, spread_at_execution_bps: float, price_improvement_bps: float) -> None`

Record an execution for reporting.

##### `def generate_rts28_report(self, year: int) -> RTS28Report`

Generate RTS 28 annual report (#C9).

### EMIRTradeReport

EMIR trade report for derivatives (#C10).

### EMIRReporter

EMIR Trade Repository Reporter (#C10).

Reports derivative trades to trade repository.

#### Methods

##### `def __init__(self, firm_lei: str, trade_repository_url: str)`

##### `def generate_uti(self, trade_id: str) -> str`

Generate Unique Transaction Identifier.

##### `def create_report(self, trade_id: str, other_party_lei: , side: str, trade_date: datetime, notional: float, currency: str, product_type: str, maturity_date: ) -> EMIRTradeReport`

Create EMIR trade report (#C10).

##### `def submit_reports(self) -> dict`

Submit pending reports to trade repository.

### TransactionReferenceGenerator

MiFIR Art 26 Transaction Reference Generator (#C12).

Generates compliant transaction reference numbers.

#### Methods

##### `def __init__(self, firm_id: str)`

##### `def generate_transaction_ref(self) -> str`

Generate MiFIR compliant transaction reference (#C12).

Format: FIRM-YYYYMMDD-NNNNNNNN

### RTS24OrderIDGenerator

RTS 24 Order ID Generator (#C13).

Generates compliant order identifiers.

#### Methods

##### `def __init__(self, venue_mic: str)`

##### `def generate_order_id(self) -> str`

Generate RTS 24 compliant order ID (#C13).

Format: MIC-TIMESTAMP-SEQUENCE

### PreTradeRiskLimits

Pre-trade risk control limits per RTS 6 (#C18).

### PreTradeRiskController

RTS 6 Pre-Trade Risk Controls (#C18).

Validates orders before submission.

#### Methods

##### `def __init__(self, limits: )`

##### `def validate_order(self, symbol: str, side: str, quantity: int, price: float, reference_price: ) -> tuple[bool, list[str]]`

Validate order against pre-trade limits (#C18).

Returns:
    (is_valid, list of rejection reasons)

##### `def update_position(self, symbol: str, value: float) -> None`

Update position tracking.

##### `def reset_daily_limits(self) -> None`

Reset daily limits (call at start of day).

### VenuePositionLimits

Per-venue position limits (#C19).

Tracks positions by venue as required by regulations.

#### Methods

##### `def __init__(self)`

##### `def set_limit(self, venue_mic: str, symbol: str, max_position: float) -> None`

Set position limit for symbol at venue.

##### `def update_position(self, venue_mic: str, symbol: str, position: float) -> None`

Update position at venue.

##### `def check_limit(self, venue_mic: str, symbol: str, additional: float) -> tuple[bool, ]`

Check if position would exceed venue limit (#C19).

Returns:
    (within_limit, error_message)

##### `def get_venue_utilization(self, venue_mic: str) -> dict[str, float]`

Get position utilization by symbol at venue.

### LocateRecord

Short selling locate record (#C20).

### ShortSellingLocator

Short selling locate requirements (#C20).

Tracks locates for short selling compliance.

#### Methods

##### `def __init__(self)`

##### `def request_locate(self, symbol: str, quantity: int, source: str, validity_hours: int) -> LocateRecord`

Request a locate for short selling (#C20).

##### `def use_locate(self, symbol: str, quantity: int) -> tuple[bool, ]`

Use locate for short sale (#C20).

Returns:
    (success, error_message)

##### `def get_available_locates(self, symbol: str) -> int`

Get available locate quantity for symbol.

##### `def expire_locates(self) -> int`

Expire old locates.

### ClientCategory

**Inherits from**: Enum

MiFID II client categories (#C25).

### ClientRecord

Client categorization record (#C25).

### ClientCategorizer

MiFID II Client Categorization (#C25).

Tracks client categories and opt-up/opt-down requests.

#### Methods

##### `def __init__(self)`

##### `def categorize_client(self, client_id: str, category: ClientCategory, lei: ) -> ClientRecord`

Categorize a client (#C25).

##### `def request_opt_up(self, client_id: str) -> bool`

Request opt-up from retail to professional.

##### `def approve_opt_up(self, client_id: str) -> bool`

Approve opt-up request.

##### `def get_client_category(self, client_id: str)`

Get client's current category.

##### `def get_clients_by_category(self, category: ClientCategory) -> list[str]`

Get all clients in a category.

### ClockSynchronizer

RTS 25 Clock Synchronization (#C27).

Ensures system clocks meet regulatory requirements.

#### Methods

##### `def __init__(self, activity_type: str)`

##### `def record_sync(self, source: str, drift_microseconds: float) -> None`

Record clock synchronization event (#C27).

##### `def is_compliant(self) -> tuple[bool, ]`

Check if clock synchronization is compliant (#C27).

Returns:
    (is_compliant, reason_if_not)

##### `def get_sync_report(self) -> dict`

Get clock synchronization report.

### AuditLogRotator

Audit log rotation policy (#C28).

Manages log rotation while maintaining compliance requirements.

#### Methods

##### `def __init__(self, log_dir: str, retention_days: int, rotation_size_mb: int, compression: bool)`

##### `def rotate_if_needed(self, current_log_path: str)`

Rotate log file if size threshold exceeded (#C28).

Returns:
    Path to rotated file or None

##### `def purge_old_logs(self) -> int`

Purge logs older than retention period (#C28).

Returns:
    Number of files purged

##### `def get_rotation_policy(self) -> dict`

Get current rotation policy.

### AccessEventType

**Inherits from**: Enum

Types of access events (#C30).

### AccessLogEntry

Access control log entry (#C30).

### AccessControlLogger

Access control logging (#C30).

Logs all access events for compliance.

#### Methods

##### `def __init__(self)`

##### `def log_access(self, event_type: AccessEventType, user_id: str, ip_address: str, action: str, success: bool, resource: , details: ) -> AccessLogEntry`

Log an access event (#C30).

##### `def get_user_activity(self, user_id: str, days: int) -> list[dict]`

Get activity log for a user.

##### `def get_failed_logins(self, hours: int) -> list[dict]`

Get failed login attempts.

### ChangeRecord

Change management record (#C31).

### ChangeManagementAuditor

Change management audit trail (#C31).

Tracks all system changes for compliance.

#### Methods

##### `def __init__(self)`

##### `def record_change(self, change_type: str, description: str, requested_by: str, implemented_by: str, affected_systems: list[str], approved_by: , rollback_plan: ) -> ChangeRecord`

Record a system change (#C31).

##### `def get_recent_changes(self, days: int) -> list[dict]`

Get recent changes.

##### `def get_changes_by_system(self, system: str) -> list[dict]`

Get changes affecting a specific system.

### DRTestRecord

Disaster recovery test record (#C32).

### DisasterRecoveryDocumentor

Disaster recovery documentation (#C32).

Automated DR documentation and test tracking.

#### Methods

##### `def __init__(self, rto_minutes: int, rpo_minutes: int)`

##### `def record_dr_test(self, test_type: str, scenario: str, rto_actual: , rpo_actual: , success: bool, issues: list[str], remediation: list[str]) -> DRTestRecord`

Record a DR test (#C32).

##### `def generate_dr_report(self) -> dict`

Generate DR documentation report (#C32).

##### `def is_test_overdue(self) -> bool`

Check if DR test is overdue.

### RegulatoryComplianceManager

Unified regulatory compliance manager.

Provides single interface to all compliance modules.

#### Methods

##### `def __init__(self, firm_lei: str, country_code: str)`

##### `def get_compliance_status(self) -> dict`

Get overall compliance status.

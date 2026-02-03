# logger

**Path**: `C:\Users\Alexa\ai-trading-firm\core\logger.py`

## Overview

Audit Logger
============

Comprehensive logging for regulatory compliance (EU/AMF).
All decisions must be logged with timestamp, data sources, rationale, and responsible agent.

## Classes

### AuditEntry

Single audit log entry.

#### Methods

##### `def to_json(self) -> str`

Convert to JSON string.

### AuditLogger

Audit logger for regulatory compliance.

Requirements (EU/AMF):
- All decisions logged with timestamp
- Data sources recorded
- Rationale documented
- Responsible agent identified
- Retention: 7 years (MiFID II)

Features:
- Rotating file handlers to prevent unbounded log growth
- Configurable max file size and backup count
- Both size-based and time-based rotation options

#### Methods

##### `def __init__(self, audit_file: str, trade_file: str, decision_file: str, max_bytes: int, backup_count: int, log_max_bytes: int, log_backup_count: int)`

##### `def get_log_stats(self) -> dict[str, Any]`

Get statistics about log files.

##### `def log_event(self, event: Event) -> None`

Log any event.

##### `def log_decision(self, agent_name: str, decision_id: str, symbol: str, action: str, quantity: int, rationale: str, data_sources: list[str], contributing_signals: list[str], conviction_score: float) -> None`

Log a trading decision with full compliance data.

This is a critical audit function - all decisions MUST be logged.

##### `def log_trade(self, agent_name: str, order_id: str, symbol: str, side: str, quantity: int, price: float, commission: float, decision_id: str) -> None`

Log a trade execution.

Links back to the original decision for audit trail.

##### `def log_risk_alert(self, agent_name: str, alert_type: str, severity: str, message: str, current_value: float, threshold_value: float, halt_trading: bool) -> None`

Log a risk alert.

##### `def log_compliance_check(self, decision_id: str, agent_name: str, approved: bool, checks: list[dict[str, Any]], rejection_code: , rejection_reason: ) -> None`

Log a compliance check result.

Required for EU/AMF regulatory compliance - all compliance
decisions must be fully documented.

##### `def log_agent_event(self, agent_name: str, event_type: str, details: dict[str, Any]) -> None`

Log an agent lifecycle event.

##### `def log_system_event(self, event_type: str, details: dict[str, Any]) -> None`

Log a system-level event.

##### `def get_decisions(self, start_date: , end_date: , symbol: ) -> list[dict]`

Query decision history for audit.

Returns decisions matching the criteria.

##### `def get_trades(self, start_date: , end_date: ) -> list[dict]`

Query trade history for audit.

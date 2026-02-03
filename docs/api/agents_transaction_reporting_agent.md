# transaction_reporting_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\transaction_reporting_agent.py`

## Overview

Transaction Reporting Agent
===========================

Implements ESMA RTS 22/23 transaction reporting requirements.
Generates reports within 15 minutes of execution per MiFID II.

Features:
- LEI validation
- AMF BDIF format support
- 65 required field population
- Report queue management

## Classes

### ReportStatus

**Inherits from**: Enum

Status of a transaction report.

### TransactionType

**Inherits from**: Enum

Transaction type codes per ESMA RTS.

### BuySellIndicator

**Inherits from**: Enum

Buy/Sell indicator.

### LEIInfo

Legal Entity Identifier information.

### TransactionReport

Transaction report per ESMA RTS 22/23.

Contains the 65 required fields for MiFID II transaction reporting.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

##### `def to_bdif_format(self) -> dict[str, Any]`

Convert to AMF BDIF (Banque de France Data Input Format).

Returns structured format for French regulator submission.

### TransactionReportingAgent

**Inherits from**: BaseAgent

Transaction reporting agent for MiFID II compliance.

Generates and submits transaction reports within 15 minutes
of execution as required by ESMA RTS 22/23.

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus, audit_logger, reporting_config: )`

Initialize transaction reporting agent.

Args:
    config: Agent configuration
    event_bus: Event bus
    audit_logger: Audit logger
    reporting_config: Reporting-specific configuration

##### `async def initialize(self) -> None`

Initialize the agent.

##### `async def stop(self) -> None`

Stop the agent.

##### `def get_subscribed_events(self) -> list[EventType]`

Subscribe to fill events.

##### `async def process_event(self, event: Event) -> None`

Process incoming events.

##### `def set_firm_lei(self, lei: str) -> bool`

Set the firm's LEI.

Args:
    lei: Legal Entity Identifier

Returns:
    True if valid and set

##### `def add_isin_mapping(self, symbol: str, isin: str) -> None`

Add symbol to ISIN mapping.

##### `def get_pending_count(self) -> int`

Get number of pending reports.

##### `def get_submitted_reports(self, hours: int, limit: int) -> list[TransactionReport]`

Get recently submitted reports.

##### `def get_report_by_id(self, report_id: str)`

Get a specific report by ID.

##### `def generate_daily_summary(self, report_date: ) -> dict[str, Any]`

Generate daily reporting summary.

Args:
    report_date: Date to summarize (default: today)

Returns:
    Summary dictionary

##### `def get_status(self) -> dict[str, Any]`

Get agent status for monitoring.

# surveillance_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\surveillance_agent.py`

## Overview

Market Surveillance Agent
=========================

Implements market abuse surveillance per MAR 2014/596/EU.
Detects potential manipulation patterns for compliance.

Monitors for:
- Wash trading
- Spoofing
- Quote stuffing
- Layering

## Classes

### SurveillanceAlertType

**Inherits from**: Enum

Types of surveillance alerts.

### AlertSeverity

**Inherits from**: Enum

Alert severity levels.

### STORStatus

**Inherits from**: Enum

STOR submission status per MAR Article 16.

### SurveillanceAlert

Market abuse surveillance alert.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary for logging.

### STORReport

Suspicious Transaction and Order Report (STOR) per MAR Article 16 (#C2).

Required fields per ESMA MAR Guidelines (2016/1452):
- Reporting entity details
- Suspicious order/transaction details
- Description of suspicious behavior
- Supporting documentation

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary for submission.

##### `def to_xml(self) -> str`

Convert to XML format for NCA submission.

Format follows ESMA MAR STOR reporting schema.

### OrderRecord

Record of an order for surveillance analysis.

### SurveillanceAgent

**Inherits from**: BaseAgent

Market abuse surveillance agent.

Monitors trading activity for potential market manipulation
patterns as required by MAR 2014/596/EU.

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus, audit_logger, surveillance_config: )`

Initialize surveillance agent.

Args:
    config: Agent configuration
    event_bus: Event bus for subscriptions
    audit_logger: Audit logger
    surveillance_config: Surveillance-specific configuration

##### `async def initialize(self) -> None`

Initialize the agent.

##### `def set_compliance_notifier(self, compliance_notifier) -> None`

Set compliance officer notifier (#C33).

##### `def get_subscribed_events(self) -> list[EventType]`

Subscribe to order and fill events.

##### `async def process_event(self, event: Event) -> None`

Process incoming events for surveillance.

##### `def record_order_cancelled(self, order_id: str) -> None`

Record an order cancellation.

##### `def submit_stor(self, stor_id: str) -> bool`

Submit a STOR to the National Competent Authority (#C2).

Per MAR Article 16(1), submission must be made "without delay"
to the competent authority of the most relevant market.

In production, this would submit to:
- AMF (France): https://stor.amf-france.org/
- FCA (UK): https://stor.fca.org.uk/
- BaFin (Germany), etc.

Returns:
    True if submission successful

##### `def get_stor_by_id(self, stor_id: str)`

Get a STOR by its ID.

##### `def get_pending_stors(self) -> list[STORReport]`

Get STORs that need to be submitted.

##### `def get_stor_summary(self) -> dict[str, Any]`

Get summary of STOR reports (#C2).

##### `def get_alerts(self, hours: int, alert_type: , severity: ) -> list[SurveillanceAlert]`

Get recent surveillance alerts.

Args:
    hours: Lookback period
    alert_type: Filter by alert type
    severity: Filter by severity

Returns:
    List of alerts

##### `def get_symbol_metrics(self, symbol: str) -> dict[str, Any]`

Get surveillance metrics for a symbol.

##### `def get_status(self) -> dict[str, Any]`

Get agent status for monitoring.

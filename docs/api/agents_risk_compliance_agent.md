# risk_compliance_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\risk_compliance_agent.py`

## Overview

Risk & Compliance Agent
=======================

Validates all trading decisions before execution.
Enforces risk limits and regulatory compliance (EU/AMF).

Responsibility: Risk validation and compliance checking ONLY.
Does NOT make trading decisions or send orders.

## Classes

### RiskState

Current risk state of the portfolio.

#### Methods

##### `def __post_init__(self)`

### RiskComplianceAgent

**Inherits from**: ValidationAgent

Risk & Compliance Agent.

Validates all trading decisions against:
1. Position limits
2. Portfolio risk limits (VaR, drawdown)
3. Rate limits (anti-HFT)
4. Regulatory compliance (EU/AMF)

EVERY decision must pass through this agent.

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger, broker: )`

##### `async def initialize(self) -> None`

Initialize risk state from broker.

##### `def get_subscribed_events(self) -> list[EventType]`

Risk agent subscribes to decisions.

##### `async def process_event(self, event: Event) -> None`

Validate trading decisions.

Every decision must pass ALL checks before execution.

##### `def resume_trading(self) -> None`

Resume trading after halt (requires manual intervention).

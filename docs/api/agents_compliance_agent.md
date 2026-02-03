# compliance_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\compliance_agent.py`

## Overview

Compliance Agent
================

Regulatory compliance validation for EU/AMF framework.
Validates all trading decisions against regulatory requirements.

Responsibility: Regulatory compliance ONLY.
Does NOT handle risk limits (see RiskAgent).

## Classes

### RejectionCode

**Inherits from**: Enum

Compliance rejection codes.

### BlackoutType

**Inherits from**: Enum

Types of blackout periods.

### BlackoutEvent

A blackout event for a symbol.

### ComplianceCheckResult

Result of a single compliance check.

### ComplianceValidationResult

Complete result of compliance validation.

### ComplianceAgent

**Inherits from**: ValidationAgent

Compliance Agent for EU/AMF Regulatory Framework.

Validates all trading decisions against:
1. Blackout periods (earnings, M&A, corporate actions)
2. Material Non-Public Information (MNPI) detection
3. Restricted instruments (sanctions, embargos)
4. Market hours
5. Short Selling Regulation (SSR)
6. Declaration thresholds (5%, 10%, etc.)
7. Data source validation

Ensures full audit trail per CLAUDE.md requirements.

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `def set_compliance_notifier(self, compliance_notifier) -> None`

Set compliance officer notifier (#C33).

##### `async def initialize(self) -> None`

Initialize compliance agent.

##### `def get_subscribed_events(self) -> list[EventType]`

Compliance agent subscribes to decisions and validated decisions.

##### `async def process_event(self, event: Event) -> None`

Validate decisions for regulatory compliance.

##### `def add_blackout_event(self, event: BlackoutEvent) -> None`

Add a blackout event to the calendar.

##### `def add_restricted_instrument(self, symbol: str) -> None`

Add instrument to restricted list.

##### `def activate_ssr(self, symbol: str) -> None`

Activate Short Selling Regulation for a symbol.

##### `def deactivate_ssr(self, symbol: str) -> None`

Deactivate Short Selling Regulation for a symbol.

##### `def suspend_trading(self, symbol: str) -> None`

Suspend trading for a symbol.

##### `def resume_trading(self, symbol: str) -> None`

Resume trading for a symbol.

##### `def validate_entity_lei(self) -> ComplianceCheckResult`

Validate the entity's LEI for transaction reporting.

Required for ESMA transaction reporting (RTS 22/23).

##### `def validate_counterparty_lei(self, lei: str) -> ComplianceCheckResult`

Validate a counterparty's LEI.

Args:
    lei: The counterparty's LEI to validate

Returns:
    ComplianceCheckResult with validation status

##### `def set_entity_lei(self, lei: str) -> bool`

Set the entity's LEI after validation.

Returns True if LEI is valid and was set.

##### `def add_isin_mapping(self, symbol: str, isin: str) -> bool`

Add an ISIN mapping for a symbol.

Args:
    symbol: Trading symbol (e.g., "AAPL")
    isin: ISIN code (e.g., "US0378331005")

Returns:
    True if ISIN is valid and was added

##### `def get_isin(self, symbol: str)`

Get ISIN for a symbol.

##### `def get_status(self) -> dict`

Get current compliance agent status for monitoring.

## Functions

### `def validate_lei(lei: str, strict: bool) -> tuple[bool, str]`

Validate a Legal Entity Identifier (LEI) per ISO 17442 standard.

LEI format:
- 20 characters total
- Characters 1-4: LOU (Local Operating Unit) prefix
- Characters 5-6: Reserved (usually "00")
- Characters 7-18: Entity-specific part
- Characters 19-20: Check digits (MOD 97-10)

Args:
    lei: The LEI string to validate
    strict: If True, also check for placeholder patterns

Returns:
    Tuple of (is_valid, error_message)

### `async def validate_lei_against_gleif(lei: str) -> tuple[bool, str, dict]`

Validate LEI against GLEIF (Global LEI Foundation) API.

This provides authoritative validation that the LEI:
- Exists in the global registry
- Is currently active (not lapsed/retired)
- Belongs to the expected entity

Args:
    lei: The LEI to validate

Returns:
    Tuple of (is_valid, error_message, entity_data)

Note:
    Requires network access to GLEIF API.
    For production, implement proper caching and rate limiting.

### `def calculate_lei_check_digits(lei_without_check: str) -> str`

Calculate the check digits for an LEI (for generating LEIs).

Args:
    lei_without_check: First 18 characters of LEI

Returns:
    2-character check digit string

## Constants

- `VALID_LOU_PREFIXES`
- `PLACEHOLDER_LEI_PATTERNS`

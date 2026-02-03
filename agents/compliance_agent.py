"""
Compliance Agent
================

Regulatory compliance validation for EU/AMF framework.
Validates all trading decisions against regulatory requirements.

Responsibility: Regulatory compliance ONLY.
Does NOT handle risk limits (see RiskAgent).
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time
from typing import TYPE_CHECKING, Optional, Set, Callable, Any
from enum import Enum
from zoneinfo import ZoneInfo

from core.agent_base import ValidationAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    ValidatedDecisionEvent,
    RiskAlertEvent,
    RiskAlertSeverity,
    OrderSide,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


# =============================================================================
# LEI VALIDATION (ISO 17442)
# =============================================================================

# Known valid LOU (Local Operating Unit) prefixes
# Source: GLEIF (https://www.gleif.org/en/about-lei/lei-issuing-organizations)
VALID_LOU_PREFIXES = {
    "2138",  # Bundesanzeiger Verlag (Germany)
    "5493",  # DTCC (USA)
    "5299",  # INSEE (France)
    "5967",  # NCCR (China)
    "213800",  # LSE/London Stock Exchange (UK)
    "25490",  # TAKASBANK (Turkey)
    "2594",  # Tokyo Stock Exchange (Japan)
    "506700",  # Hungarian National Bank
    "529900",  # ANNA-ESMA (European)
    "549300",  # SEC (USA)
    "635400",  # Russian NSD
    "724500",  # Spanish CNMV
    "815600",  # Bloomberg Finance
    "894500",  # MUFG (Japan)
    "959800",  # Iceland
    "969500",  # Paris Stock Exchange
}

# Known placeholder/test LEI patterns to reject
PLACEHOLDER_LEI_PATTERNS = {
    "PLACEHOLDER",
    "TEST",
    "DUMMY",
    "EXAMPLE",
    "XXXX",
    "0000000000",
}


def validate_lei(lei: str, strict: bool = True) -> tuple[bool, str]:
    """
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
    """
    if not lei:
        return False, "LEI is empty"

    # Remove whitespace and convert to uppercase
    lei = lei.strip().upper()

    # Check for placeholder patterns (critical issue from expert review)
    if strict:
        for pattern in PLACEHOLDER_LEI_PATTERNS:
            if pattern in lei:
                return False, f"LEI appears to be a placeholder (contains '{pattern}')"

    # Check length
    if len(lei) != 20:
        return False, f"LEI must be 20 characters, got {len(lei)}"

    # Check alphanumeric
    if not lei.isalnum():
        return False, "LEI must contain only alphanumeric characters"

    # Check LOU prefix (first 4 characters)
    lou_prefix = lei[:4]
    if strict and lou_prefix not in VALID_LOU_PREFIXES:
        # Check if first 6 characters match (some LOUs have 6-char prefixes)
        if lei[:6] not in VALID_LOU_PREFIXES:
            return False, f"LEI LOU prefix '{lou_prefix}' not in known valid list - GLEIF verification required"

    # Characters 5-6 should be "00" for standard LEIs
    if strict and lei[4:6] != "00":
        logger.warning(
            f"LEI characters 5-6 are '{lei[4:6]}' instead of '00' - "
            "may indicate special LEI type"
        )

    # MOD 97-10 checksum validation (ISO 7064)
    # Convert letters to numbers: A=10, B=11, ..., Z=35
    # Use incremental modulo calculation to avoid creating large integers
    # and improve performance for the 20+ digit numbers involved
    try:
        checksum = 0
        for char in lei:
            if char.isdigit():
                # For single digit, multiply current checksum by 10 and add digit
                checksum = (checksum * 10 + int(char)) % 97
            else:
                # Letters convert to 2-digit numbers (A=10, B=11, ..., Z=35)
                # so we need to multiply by 100 (shift two decimal places)
                letter_value = ord(char) - ord('A') + 10
                checksum = (checksum * 100 + letter_value) % 97
        if checksum != 1:
            return False, f"LEI checksum invalid (got {checksum}, expected 1)"
    except ValueError:
        return False, "LEI contains invalid characters"

    return True, ""


async def validate_lei_against_gleif(lei: str) -> tuple[bool, str, dict]:
    """
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
    """
    import aiohttp

    # First do local validation
    is_valid, error_msg = validate_lei(lei, strict=True)
    if not is_valid:
        return False, error_msg, {}

    # GLEIF API endpoint
    gleif_url = f"https://api.gleif.org/api/v1/lei-records/{lei}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(gleif_url, timeout=10) as response:
                if response.status == 404:
                    return False, "LEI not found in GLEIF registry", {}

                if response.status != 200:
                    logger.warning(f"GLEIF API returned status {response.status}")
                    # Fall back to local validation only
                    return True, "GLEIF validation unavailable, local validation passed", {}

                data = await response.json()

                # Extract relevant fields
                attributes = data.get("data", {}).get("attributes", {})
                entity = attributes.get("entity", {})
                registration = attributes.get("registration", {})

                entity_data = {
                    "legal_name": entity.get("legalName", {}).get("name"),
                    "status": entity.get("status"),
                    "jurisdiction": entity.get("jurisdiction"),
                    "registration_status": registration.get("status"),
                    "next_renewal_date": registration.get("nextRenewalDate"),
                    "managing_lou": registration.get("managingLou"),
                }

                # Check if LEI is active
                if registration.get("status") != "ISSUED":
                    return (
                        False,
                        f"LEI status is '{registration.get('status')}' (not ISSUED)",
                        entity_data,
                    )

                return True, "", entity_data

    except aiohttp.ClientError as e:
        logger.warning(f"GLEIF API request failed: {e}")
        # Fall back to local validation
        return True, "GLEIF validation unavailable, local validation passed", {}
    except Exception as e:
        logger.error(f"Unexpected error validating LEI against GLEIF: {e}")
        return True, "GLEIF validation error, local validation passed", {}


def calculate_lei_check_digits(lei_without_check: str) -> str:
    """
    Calculate the check digits for an LEI (for generating LEIs).

    Args:
        lei_without_check: First 18 characters of LEI

    Returns:
        2-character check digit string
    """
    if len(lei_without_check) != 18:
        raise ValueError("LEI base must be 18 characters")

    # Append "00" as placeholder for check digits
    lei_with_placeholder = lei_without_check.upper() + "00"

    # Convert to numeric
    numeric_lei = ""
    for char in lei_with_placeholder:
        if char.isdigit():
            numeric_lei += char
        else:
            numeric_lei += str(ord(char) - ord('A') + 10)

    # Calculate check digits: 98 - (number mod 97)
    remainder = int(numeric_lei) % 97
    check_digits = 98 - remainder

    return f"{check_digits:02d}"


class RejectionCode(Enum):
    """Compliance rejection codes."""
    BLACKOUT_PERIOD = "BLACKOUT_PERIOD"
    MNPI_DETECTED = "MNPI_DETECTED"
    RESTRICTED_INSTRUMENT = "RESTRICTED_INSTRUMENT"
    MARKET_CLOSED = "MARKET_CLOSED"
    SSR_RESTRICTION = "SSR_RESTRICTION"
    TRADING_SUSPENDED = "TRADING_SUSPENDED"
    THRESHOLD_BREACH = "THRESHOLD_BREACH"
    UNAPPROVED_SOURCE = "UNAPPROVED_SOURCE"
    NO_BORROW_AVAILABLE = "NO_BORROW_AVAILABLE"
    INVALID_LEI = "INVALID_LEI"
    INVALID_ISIN = "INVALID_ISIN"


class BlackoutType(Enum):
    """Types of blackout periods."""
    EARNINGS = "earnings"
    MA = "merger_acquisition"
    CAPITAL_INCREASE = "capital_increase"
    REGULATORY = "regulatory"


@dataclass
class BlackoutEvent:
    """A blackout event for a symbol."""
    symbol: str
    blackout_type: BlackoutType
    event_date: datetime
    blackout_start: datetime
    blackout_end: datetime
    description: str


@dataclass
class ComplianceCheckResult:
    """Result of a single compliance check."""
    check_name: str
    passed: bool
    code: Optional[RejectionCode] = None
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class ComplianceValidationResult:
    """Complete result of compliance validation."""
    approved: bool
    checks: list[ComplianceCheckResult]
    rejection_code: Optional[RejectionCode] = None
    rejection_reason: Optional[str] = None


class ComplianceAgent(ValidationAgent):
    """
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
    """

    # Approved data sources per CLAUDE.md
    APPROVED_SOURCES: Set[str] = {
        "bloomberg", "reuters", "interactive_brokers", "ib",
        "sec_edgar", "amf_bdif", "euronext", "company_ir",
        "yahoo_finance", "refinitiv",
    }

    # Suspicious patterns in data content
    SUSPICIOUS_PATTERNS = [
        r"leak(?:ed)?", r"insider", r"confidential",
        r"not\s+(?:yet\s+)?public", r"embargo", r"pre[_-]?release",
        r"private\s+information", r"undisclosed",
    ]

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._jurisdiction = config.parameters.get("jurisdiction", "EU")

        # Restricted instruments (sanctions, etc.)
        self._restricted_instruments: Set[str] = set(
            config.parameters.get("restricted_instruments", [])
        )

        # Blackout calendar
        self._blackout_events: dict[str, list[BlackoutEvent]] = {}

        # Short Selling Restriction active symbols
        self._ssr_active: Set[str] = set()

        # Trading suspended symbols
        self._suspended_symbols: Set[str] = set()

        # Market hours by exchange timezone
        # US markets use Eastern Time (ET)
        self._market_timezone = ZoneInfo("America/New_York")
        self._market_open_time = time(9, 30)  # 9:30 AM ET
        self._market_close_time = time(16, 0)  # 4:00 PM ET

        # Extended hours (pre-market and after-hours)
        self._premarket_open = time(4, 0)  # 4:00 AM ET
        self._afterhours_close = time(20, 0)  # 8:00 PM ET
        self._allow_extended_hours = config.parameters.get("allow_extended_hours", False)

        # Declaration thresholds (EU)
        self._declaration_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75]

        # Current positions for threshold checking
        self._current_positions: dict[str, float] = {}  # symbol -> weight

        # Decision cache for lookup (decision_id -> decision details)
        # Use OrderedDict for O(1) cache eviction (PERF-P1-002)
        self._decision_cache: OrderedDict[str, dict] = OrderedDict()
        self._decision_cache_max_size: int = 1000

        # LEI for this entity (should be configured)
        self._entity_lei = config.parameters.get("entity_lei", "")

        # ISIN mappings (symbol -> ISIN)
        self._isin_mappings: dict[str, str] = config.parameters.get("isin_mappings", {})

        # Monitoring
        self._check_latencies: list[float] = []
        self._rejections_today: list[tuple[datetime, RejectionCode, str]] = []

        # Pre-trade compliance checks caching (P2)
        pretrade_cache_config = config.parameters.get("pretrade_cache", {})
        self._pretrade_cache_enabled = pretrade_cache_config.get("enabled", True)
        self._pretrade_cache_ttl_seconds = pretrade_cache_config.get("ttl_seconds", 300)  # 5 min default
        self._pretrade_cache: dict[str, tuple[ComplianceCheckResult, datetime]] = {}  # key -> (result, timestamp)
        self._pretrade_cache_hits = 0
        self._pretrade_cache_misses = 0

        # Regulatory calendar integration (P2)
        reg_calendar_config = config.parameters.get("regulatory_calendar", {})
        self._regulatory_calendar_enabled = reg_calendar_config.get("enabled", True)
        self._regulatory_events: list[dict] = []  # List of upcoming regulatory events
        self._regulatory_deadlines: list[dict] = []  # List of upcoming deadlines
        self._regulatory_holidays: set[str] = set()  # YYYY-MM-DD format

        # Audit report scheduling (P2)
        audit_schedule_config = config.parameters.get("audit_schedule", {})
        self._audit_schedule_enabled = audit_schedule_config.get("enabled", True)
        self._scheduled_reports: list[dict] = []  # List of scheduled reports
        self._last_report_times: dict[str, datetime] = {}  # report_type -> last run time
        self._report_retention_days = audit_schedule_config.get("retention_days", 365)

        # Compliance notifier for officer notifications (MON-012)
        # Callable signature: (severity: str, alert_type: str, message: str, details: dict) -> None
        self._compliance_notifier: Optional[Callable[[str, str, str, dict], Any]] = None

    def set_compliance_notifier(self, compliance_notifier: Callable[[str, str, str, dict], Any]) -> None:
        """
        Set compliance officer notifier for real-time violation alerts (MON-012).

        Args:
            compliance_notifier: Callable that accepts (severity, alert_type, message, details)
        """
        self._compliance_notifier = compliance_notifier
        logger.info("ComplianceNotifier connected to ComplianceAgent for real-time alerts")

    async def _notify_compliance_violation(
        self,
        severity: str,
        alert_type: str,
        message: str,
        details: dict
    ) -> None:
        """
        Send real-time notification for compliance violations (MON-012).

        Args:
            severity: Alert severity (CRITICAL, HIGH, MEDIUM, LOW)
            alert_type: Type of violation (e.g., MNPI, BLACKOUT, SSR)
            message: Human-readable message
            details: Additional context for the violation
        """
        if self._compliance_notifier is not None:
            try:
                result = self._compliance_notifier(severity, alert_type, message, details)
                # Handle async notifiers
                if hasattr(result, '__await__'):
                    await result
                logger.debug(f"Compliance notification sent: {alert_type} - {severity}")
            except Exception as e:
                logger.error(f"Failed to send compliance notification: {e}")
        else:
            logger.warning(
                f"Compliance violation not notified (no notifier configured): "
                f"{alert_type} - {message}"
            )

    async def initialize(self) -> None:
        """Initialize compliance agent."""
        logger.info(f"ComplianceAgent initializing (jurisdiction={self._jurisdiction})")

        # P0-3: CRITICAL - Validate entity LEI at startup (MiFID II requirement)
        # LEI is mandatory for transaction reporting and compliance
        if not self._entity_lei:
            error_msg = (
                "Entity LEI is not configured. "
                "LEI (Legal Entity Identifier) is MANDATORY for MiFID II compliance and transaction reporting. "
                "Configure 'entity_lei' parameter in ComplianceAgent config."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # Validate LEI format and authenticity
        lei_validation_result = self.validate_entity_lei()
        if not lei_validation_result.passed:
            error_msg = (
                f"Entity LEI validation failed: {lei_validation_result.message}. "
                f"Cannot start ComplianceAgent without valid LEI (MiFID II requirement)."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Entity LEI validated successfully: {self._entity_lei[:4]}...{self._entity_lei[-4:]}")

        # Load blackout calendar
        await self._load_blackout_calendar()

        # Load restricted list
        await self._load_restricted_list()

        logger.info(f"ComplianceAgent initialized: "
                   f"{len(self._restricted_instruments)} restricted instruments, "
                   f"{sum(len(v) for v in self._blackout_events.values())} blackout events")

    def get_subscribed_events(self) -> list[EventType]:
        """Compliance agent subscribes to decisions and validated decisions."""
        # Subscribe to DECISION to cache them, and VALIDATED_DECISION to process after risk
        return [EventType.DECISION, EventType.VALIDATED_DECISION]

    async def process_event(self, event: Event) -> None:
        """Validate decisions for regulatory compliance."""
        # Import DecisionEvent here to avoid circular imports
        from core.events import DecisionEvent

        # Cache decisions for later lookup
        if isinstance(event, DecisionEvent):
            self._cache_decision(event)
            return

        if not isinstance(event, ValidatedDecisionEvent):
            return

        # Only process approved decisions from risk agent
        if not event.approved:
            return

        start_time = datetime.now(timezone.utc)

        # Get the original decision details
        # In production, would fetch from event store
        original_decision = await self._get_original_decision(event.original_decision_id)
        if not original_decision:
            logger.error(f"Could not find original decision {event.original_decision_id}")
            return

        # Run compliance checks
        result = await self._validate_compliance(original_decision, event)

        # Track latency
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        self._check_latencies.append(latency_ms)
        if len(self._check_latencies) > 1000:
            self._check_latencies = self._check_latencies[-1000:]

        # Create compliance-validated event
        compliance_event = ValidatedDecisionEvent(
            source_agent=self.name,
            original_decision_id=event.original_decision_id,
            approved=result.approved,
            adjusted_quantity=event.adjusted_quantity,
            rejection_reason=result.rejection_reason,
            risk_metrics=event.risk_metrics,
            compliance_checks=tuple(
                c.check_name for c in result.checks if c.passed
            ) + event.compliance_checks,
        )

        # Publish result
        await self._event_bus.publish(compliance_event)

        # Log for audit (REQUIRED per CLAUDE.md)
        self._audit_logger.log_compliance_check(
            decision_id=event.original_decision_id,
            agent_name=self.name,
            approved=result.approved,
            checks=[{
                "name": c.check_name,
                "passed": c.passed,
                "code": c.code.value if c.code else None,
                "message": c.message,
            } for c in result.checks],
            rejection_code=result.rejection_code.value if result.rejection_code else None,
            rejection_reason=result.rejection_reason,
        )

        if result.approved:
            logger.info(f"Decision {event.original_decision_id} COMPLIANCE APPROVED "
                       f"(latency={latency_ms:.1f}ms)")
        else:
            logger.warning(f"Decision {event.original_decision_id} COMPLIANCE REJECTED: "
                          f"{result.rejection_code.value if result.rejection_code else 'unknown'}")
            self._rejections_today.append((
                datetime.now(timezone.utc),
                result.rejection_code,
                original_decision.get("symbol", "unknown")
            ))

    async def _validate_compliance(
        self,
        decision: dict,
        risk_event: ValidatedDecisionEvent
    ) -> ComplianceValidationResult:
        """Run all compliance checks."""
        checks: list[ComplianceCheckResult] = []
        symbol = decision.get("symbol", "")

        # 1. Restricted instruments
        restricted_check = self._check_restricted_instrument(symbol)
        checks.append(restricted_check)
        if not restricted_check.passed:
            # MON-012: Send real-time notification for restricted instrument violation
            await self._notify_compliance_violation(
                severity="CRITICAL",
                alert_type="RESTRICTED_INSTRUMENT",
                message=restricted_check.message,
                details={
                    "symbol": symbol,
                    "decision_id": decision.get("decision_id"),
                    "code": restricted_check.code.value if restricted_check.code else None,
                }
            )
            return ComplianceValidationResult(
                approved=False,
                checks=checks,
                rejection_code=restricted_check.code,
                rejection_reason=restricted_check.message
            )

        # 2. Blackout period
        blackout_check = await self._check_blackout_period(symbol)
        checks.append(blackout_check)
        if not blackout_check.passed:
            # MON-012: Send real-time notification for blackout violation
            await self._notify_compliance_violation(
                severity="HIGH",
                alert_type="BLACKOUT_PERIOD",
                message=blackout_check.message,
                details={
                    "symbol": symbol,
                    "decision_id": decision.get("decision_id"),
                    "blackout_details": blackout_check.details,
                }
            )
            return ComplianceValidationResult(
                approved=False,
                checks=checks,
                rejection_code=blackout_check.code,
                rejection_reason=blackout_check.message
            )

        # 3. MNPI detection
        mnpi_check = await self._check_mnpi(decision)
        checks.append(mnpi_check)
        if not mnpi_check.passed:
            # This is serious - alert compliance officer
            await self._alert_mnpi_detected(decision, mnpi_check)
            # MON-012: Send CRITICAL real-time notification for MNPI
            await self._notify_compliance_violation(
                severity="CRITICAL",
                alert_type="MNPI_DETECTED",
                message=mnpi_check.message,
                details={
                    "symbol": symbol,
                    "decision_id": decision.get("decision_id"),
                    "rationale": decision.get("rationale", "")[:200],
                    "data_sources": decision.get("data_sources", []),
                    "pattern_detected": mnpi_check.details.get("pattern"),
                }
            )
            return ComplianceValidationResult(
                approved=False,
                checks=checks,
                rejection_code=mnpi_check.code,
                rejection_reason=mnpi_check.message
            )

        # 4. Market hours
        market_check = self._check_market_hours(symbol)
        checks.append(market_check)
        if not market_check.passed:
            return ComplianceValidationResult(
                approved=False,
                checks=checks,
                rejection_code=market_check.code,
                rejection_reason=market_check.message
            )

        # 5. Trading suspended
        suspended_check = self._check_trading_suspended(symbol)
        checks.append(suspended_check)
        if not suspended_check.passed:
            # MON-012: Send real-time notification for suspended trading
            await self._notify_compliance_violation(
                severity="HIGH",
                alert_type="TRADING_SUSPENDED",
                message=suspended_check.message,
                details={
                    "symbol": symbol,
                    "decision_id": decision.get("decision_id"),
                }
            )
            return ComplianceValidationResult(
                approved=False,
                checks=checks,
                rejection_code=suspended_check.code,
                rejection_reason=suspended_check.message
            )

        # 6. Short Selling Regulation (if applicable)
        if decision.get("action") == "sell" and decision.get("is_short", False):
            ssr_check = await self._check_ssr(symbol)
            checks.append(ssr_check)
            if not ssr_check.passed:
                # MON-012: Send real-time notification for SSR violation
                await self._notify_compliance_violation(
                    severity="HIGH",
                    alert_type="SSR_RESTRICTION",
                    message=ssr_check.message,
                    details={
                        "symbol": symbol,
                        "decision_id": decision.get("decision_id"),
                        "action": "short_sell",
                    }
                )
                return ComplianceValidationResult(
                    approved=False,
                    checks=checks,
                    rejection_code=ssr_check.code,
                    rejection_reason=ssr_check.message
                )

        # 7. Declaration threshold check
        threshold_check = await self._check_declaration_threshold(symbol, decision)
        checks.append(threshold_check)
        # Threshold check doesn't reject but flags for notification

        # 8. Data source validation
        source_check = self._check_data_sources(decision)
        checks.append(source_check)
        if not source_check.passed:
            # MON-012: Send real-time notification for unapproved data source
            await self._notify_compliance_violation(
                severity="MEDIUM",
                alert_type="UNAPPROVED_SOURCE",
                message=source_check.message,
                details={
                    "symbol": symbol,
                    "decision_id": decision.get("decision_id"),
                    "sources": decision.get("data_sources", []),
                }
            )
            return ComplianceValidationResult(
                approved=False,
                checks=checks,
                rejection_code=source_check.code,
                rejection_reason=source_check.message
            )

        # All checks passed
        return ComplianceValidationResult(
            approved=True,
            checks=checks,
        )

    def _check_restricted_instrument(self, symbol: str) -> ComplianceCheckResult:
        """Check if instrument is on restricted list."""
        if symbol in self._restricted_instruments:
            return ComplianceCheckResult(
                check_name="restricted_instrument",
                passed=False,
                code=RejectionCode.RESTRICTED_INSTRUMENT,
                message=f"Instrument {symbol} is on restricted list"
            )

        return ComplianceCheckResult(
            check_name="restricted_instrument",
            passed=True,
        )

    async def _check_blackout_period(self, symbol: str) -> ComplianceCheckResult:
        """Check if symbol is in blackout period."""
        now = datetime.now(timezone.utc)
        events = self._blackout_events.get(symbol, [])

        for event in events:
            if event.blackout_start <= now <= event.blackout_end:
                return ComplianceCheckResult(
                    check_name="blackout_period",
                    passed=False,
                    code=RejectionCode.BLACKOUT_PERIOD,
                    message=f"{symbol} in {event.blackout_type.value} blackout until {event.blackout_end.date()}",
                    details={
                        "blackout_type": event.blackout_type.value,
                        "event_date": event.event_date.isoformat(),
                        "blackout_end": event.blackout_end.isoformat(),
                        "description": event.description,
                    }
                )

        return ComplianceCheckResult(
            check_name="blackout_period",
            passed=True,
        )

    async def _check_mnpi(self, decision: dict) -> ComplianceCheckResult:
        """Check for Material Non-Public Information."""
        data_sources = decision.get("data_sources", [])
        rationale = decision.get("rationale", "")

        # Check data sources
        for source in data_sources:
            source_name = source if isinstance(source, str) else source.get("provider", "")
            source_lower = source_name.lower()

            if source_lower not in self.APPROVED_SOURCES:
                return ComplianceCheckResult(
                    check_name="mnpi_detection",
                    passed=False,
                    code=RejectionCode.UNAPPROVED_SOURCE,
                    message=f"Unapproved data source: {source_name}",
                    details={"source": source_name}
                )

        # Scan rationale for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, rationale, re.IGNORECASE):
                return ComplianceCheckResult(
                    check_name="mnpi_detection",
                    passed=False,
                    code=RejectionCode.MNPI_DETECTED,
                    message=f"Suspicious pattern detected in rationale: {pattern}",
                    details={"pattern": pattern}
                )

        return ComplianceCheckResult(
            check_name="mnpi_detection",
            passed=True,
        )

    def _check_market_hours(self, symbol: str) -> ComplianceCheckResult:
        """
        Check if market is open with proper timezone handling.

        Converts UTC time to market timezone (ET for US markets) before checking.
        Supports extended hours trading if configured.
        """
        # Get current time in UTC
        now_utc = datetime.now(timezone.utc)

        # Convert to market timezone (Eastern Time for US markets)
        now_et = now_utc.astimezone(self._market_timezone)

        current_time = now_et.time()
        is_weekday = now_et.weekday() < 5

        # Check for US market holidays (simplified - would use holiday calendar in production)
        # Major US market holidays: New Year's, MLK, Presidents Day, Good Friday,
        # Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas

        # Check regular trading hours
        regular_hours = (
            is_weekday and
            self._market_open_time <= current_time <= self._market_close_time
        )

        if regular_hours:
            return ComplianceCheckResult(
                check_name="market_hours",
                passed=True,
                details={
                    "market_time": now_et.strftime("%H:%M:%S %Z"),
                    "session": "regular"
                }
            )

        # Check extended hours if allowed
        if self._allow_extended_hours and is_weekday:
            premarket = self._premarket_open <= current_time < self._market_open_time
            afterhours = self._market_close_time < current_time <= self._afterhours_close

            if premarket or afterhours:
                session = "premarket" if premarket else "afterhours"
                return ComplianceCheckResult(
                    check_name="market_hours",
                    passed=True,
                    message=f"Trading in {session} session",
                    details={
                        "market_time": now_et.strftime("%H:%M:%S %Z"),
                        "session": session
                    }
                )

        # Market is closed
        return ComplianceCheckResult(
            check_name="market_hours",
            passed=False,
            code=RejectionCode.MARKET_CLOSED,
            message=f"Market closed (ET time: {now_et.strftime('%H:%M:%S %Z')}, weekday: {is_weekday})",
            details={
                "market_time": now_et.strftime("%H:%M:%S %Z"),
                "utc_time": now_utc.strftime("%H:%M:%S UTC"),
                "is_weekday": is_weekday,
                "regular_hours": f"{self._market_open_time}-{self._market_close_time} ET"
            }
        )

    def _check_trading_suspended(self, symbol: str) -> ComplianceCheckResult:
        """Check if trading is suspended for symbol."""
        if symbol in self._suspended_symbols:
            return ComplianceCheckResult(
                check_name="trading_suspended",
                passed=False,
                code=RejectionCode.TRADING_SUSPENDED,
                message=f"Trading suspended for {symbol}"
            )

        return ComplianceCheckResult(
            check_name="trading_suspended",
            passed=True,
        )

    async def _check_ssr(self, symbol: str) -> ComplianceCheckResult:
        """Check Short Selling Regulation restrictions."""
        if symbol in self._ssr_active:
            return ComplianceCheckResult(
                check_name="ssr_restriction",
                passed=False,
                code=RejectionCode.SSR_RESTRICTION,
                message=f"Short selling restricted for {symbol} (SSR active)"
            )

        return ComplianceCheckResult(
            check_name="ssr_restriction",
            passed=True,
        )

    async def _check_declaration_threshold(
        self, symbol: str, decision: dict
    ) -> ComplianceCheckResult:
        """Check if order would trigger declaration threshold."""
        current_weight = self._current_positions.get(symbol, 0.0)
        quantity = decision.get("quantity", 0)
        price = decision.get("limit_price", 100.0)
        nav = 1_000_000  # Would get from portfolio

        order_value = quantity * price
        order_weight = order_value / nav
        new_weight = current_weight + order_weight

        # Check if crossing a threshold
        crossed_thresholds = []
        for threshold in self._declaration_thresholds:
            if current_weight < threshold <= new_weight:
                crossed_thresholds.append(threshold)

        if crossed_thresholds:
            return ComplianceCheckResult(
                check_name="declaration_threshold",
                passed=True,  # Don't reject, but flag
                message=f"Order will cross threshold(s): {[f'{t*100}%' for t in crossed_thresholds]}",
                details={
                    "thresholds_crossed": crossed_thresholds,
                    "current_weight": current_weight,
                    "new_weight": new_weight,
                    "notification_required": True,
                }
            )

        return ComplianceCheckResult(
            check_name="declaration_threshold",
            passed=True,
        )

    def _check_data_sources(self, decision: dict) -> ComplianceCheckResult:
        """Validate that all data sources are approved."""
        data_sources = decision.get("data_sources", [])

        if not data_sources:
            return ComplianceCheckResult(
                check_name="data_sources",
                passed=False,
                code=RejectionCode.UNAPPROVED_SOURCE,
                message="No data sources specified (required for compliance)"
            )

        for source in data_sources:
            source_name = source if isinstance(source, str) else source.get("provider", "")
            if source_name.lower() not in self.APPROVED_SOURCES:
                return ComplianceCheckResult(
                    check_name="data_sources",
                    passed=False,
                    code=RejectionCode.UNAPPROVED_SOURCE,
                    message=f"Unapproved data source: {source_name}"
                )

        return ComplianceCheckResult(
            check_name="data_sources",
            passed=True,
            details={"sources": data_sources}
        )

    async def _alert_mnpi_detected(
        self, decision: dict, check_result: ComplianceCheckResult
    ) -> None:
        """Alert compliance officer about potential MNPI."""
        alert = RiskAlertEvent(
            source_agent=self.name,
            severity=RiskAlertSeverity.CRITICAL,
            alert_type="mnpi_detected",
            message=f"Potential MNPI detected: {check_result.message}",
            affected_symbols=(decision.get("symbol", ""),),
            halt_trading=False,  # Don't halt, but alert
        )

        await self._event_bus.publish(alert)

        logger.critical(f"MNPI ALERT: {check_result.message} for {decision.get('symbol')}")

    def _cache_decision(self, decision) -> None:
        """
        Cache a decision event for later compliance lookup.

        Uses OrderedDict for O(1) eviction of oldest entries (PERF-P1-002).

        Args:
            decision: DecisionEvent to cache
        """
        # Store decision details in a dict format for easy access
        decision_dict = {
            "decision_id": decision.event_id,
            "symbol": decision.symbol,
            "action": decision.action.value if decision.action else None,
            "quantity": decision.quantity,
            "limit_price": decision.limit_price,
            "stop_price": decision.stop_price,
            "rationale": decision.rationale,
            "data_sources": list(decision.data_sources),
            "contributing_signals": list(decision.contributing_signals),
            "conviction_score": decision.conviction_score,
            "timestamp": decision.timestamp.isoformat(),
            "is_short": decision.action and decision.action.value == "sell",
        }

        # If key exists, move to end (most recent)
        if decision.event_id in self._decision_cache:
            self._decision_cache.move_to_end(decision.event_id)

        self._decision_cache[decision.event_id] = decision_dict

        # Efficient O(1) eviction using OrderedDict.popitem(last=False)
        # Remove oldest entries if over max size (PERF-P1-002 fix)
        while len(self._decision_cache) > self._decision_cache_max_size:
            self._decision_cache.popitem(last=False)

        logger.debug(f"Decision cached for compliance: {decision.event_id[:8]} ({decision.symbol})")

    async def _get_original_decision(self, decision_id: str) -> Optional[dict]:
        """
        Get original decision details from cache.

        This is used to retrieve the full decision context when
        performing compliance checks on a validated decision.

        Args:
            decision_id: The ID of the decision to retrieve

        Returns:
            Decision dict if found, None otherwise
        """
        # First, try the local cache
        if decision_id in self._decision_cache:
            return self._decision_cache[decision_id]

        # If not in cache, log warning and return None
        # In production, could also try fetching from audit logs
        logger.warning(
            f"Decision {decision_id} not found in cache. "
            f"Ensure ComplianceAgent subscribes to DECISION events."
        )

        # Try to fetch from audit logger as fallback
        try:
            decisions = self._audit_logger.get_decisions()
            for d in decisions:
                if d.get("event_id") == decision_id:
                    return d.get("details", {})
        except Exception as e:
            logger.debug(f"Could not fetch decision from audit log: {e}")

        return None

    async def _load_blackout_calendar(self) -> None:
        """Load blackout calendar from external source."""
        # In production, would load from Bloomberg, Reuters, etc.
        # For now, create some sample events

        # Example: Add earnings blackouts
        sample_events = [
            BlackoutEvent(
                symbol="AAPL",
                blackout_type=BlackoutType.EARNINGS,
                event_date=datetime(2024, 1, 25, tzinfo=timezone.utc),
                blackout_start=datetime(2024, 1, 25, tzinfo=timezone.utc) - timedelta(days=30),
                blackout_end=datetime(2024, 1, 25, tzinfo=timezone.utc) + timedelta(days=1),
                description="Q1 2024 Earnings"
            ),
        ]

        for event in sample_events:
            if event.symbol not in self._blackout_events:
                self._blackout_events[event.symbol] = []
            self._blackout_events[event.symbol].append(event)

    async def _load_restricted_list(self) -> None:
        """Load restricted instruments list."""
        # In production, would load from compliance database
        # Includes sanctioned entities, etc.
        pass

    def add_blackout_event(self, event: BlackoutEvent) -> None:
        """Add a blackout event to the calendar."""
        if event.symbol not in self._blackout_events:
            self._blackout_events[event.symbol] = []
        self._blackout_events[event.symbol].append(event)
        logger.info(f"Added blackout event for {event.symbol}: {event.description}")

    def add_restricted_instrument(self, symbol: str) -> None:
        """Add instrument to restricted list."""
        self._restricted_instruments.add(symbol)
        logger.warning(f"Added {symbol} to restricted list")

    def activate_ssr(self, symbol: str) -> None:
        """Activate Short Selling Regulation for a symbol."""
        self._ssr_active.add(symbol)
        logger.warning(f"SSR activated for {symbol}")

    def deactivate_ssr(self, symbol: str) -> None:
        """Deactivate Short Selling Regulation for a symbol."""
        self._ssr_active.discard(symbol)
        logger.info(f"SSR deactivated for {symbol}")

    def suspend_trading(self, symbol: str) -> None:
        """Suspend trading for a symbol."""
        self._suspended_symbols.add(symbol)
        logger.critical(f"Trading suspended for {symbol}")

    def resume_trading(self, symbol: str) -> None:
        """Resume trading for a symbol."""
        self._suspended_symbols.discard(symbol)
        logger.info(f"Trading resumed for {symbol}")

    def validate_entity_lei(self) -> ComplianceCheckResult:
        """
        Validate the entity's LEI for transaction reporting.

        Required for ESMA transaction reporting (RTS 22/23).
        """
        if not self._entity_lei:
            return ComplianceCheckResult(
                check_name="entity_lei",
                passed=False,
                code=RejectionCode.INVALID_LEI,
                message="Entity LEI not configured (required for transaction reporting)"
            )

        is_valid, error_msg = validate_lei(self._entity_lei)

        if not is_valid:
            return ComplianceCheckResult(
                check_name="entity_lei",
                passed=False,
                code=RejectionCode.INVALID_LEI,
                message=f"Invalid entity LEI: {error_msg}"
            )

        return ComplianceCheckResult(
            check_name="entity_lei",
            passed=True,
            details={"lei": self._entity_lei}
        )

    def validate_counterparty_lei(self, lei: str) -> ComplianceCheckResult:
        """
        Validate a counterparty's LEI.

        Args:
            lei: The counterparty's LEI to validate

        Returns:
            ComplianceCheckResult with validation status
        """
        is_valid, error_msg = validate_lei(lei)

        if not is_valid:
            return ComplianceCheckResult(
                check_name="counterparty_lei",
                passed=False,
                code=RejectionCode.INVALID_LEI,
                message=f"Invalid counterparty LEI: {error_msg}",
                details={"lei": lei}
            )

        return ComplianceCheckResult(
            check_name="counterparty_lei",
            passed=True,
            details={"lei": lei}
        )

    def set_entity_lei(self, lei: str) -> bool:
        """
        Set the entity's LEI after validation.

        Returns True if LEI is valid and was set.
        """
        is_valid, error_msg = validate_lei(lei)
        if is_valid:
            self._entity_lei = lei
            logger.info(f"Entity LEI set: {lei}")
            return True
        else:
            logger.error(f"Invalid LEI rejected: {error_msg}")
            return False

    def add_isin_mapping(self, symbol: str, isin: str) -> bool:
        """
        Add an ISIN mapping for a symbol.

        Args:
            symbol: Trading symbol (e.g., "AAPL")
            isin: ISIN code (e.g., "US0378331005")

        Returns:
            True if ISIN is valid and was added
        """
        is_valid, error_msg = self._validate_isin(isin)
        if is_valid:
            self._isin_mappings[symbol] = isin
            logger.info(f"ISIN mapping added: {symbol} -> {isin}")
            return True
        else:
            logger.error(f"Invalid ISIN rejected for {symbol}: {error_msg}")
            return False

    def get_isin(self, symbol: str) -> str | None:
        """Get ISIN for a symbol."""
        return self._isin_mappings.get(symbol)

    def _validate_isin(self, isin: str) -> tuple[bool, str]:
        """
        Validate an ISIN (International Securities Identification Number).

        ISIN format:
        - 12 characters total
        - Characters 1-2: Country code (ISO 3166-1 alpha-2)
        - Characters 3-11: NSIN (National Securities Identifying Number)
        - Character 12: Check digit (Luhn algorithm)
        """
        if not isin:
            return False, "ISIN is empty"

        isin = isin.strip().upper()

        if len(isin) != 12:
            return False, f"ISIN must be 12 characters, got {len(isin)}"

        # First 2 characters must be letters (country code)
        if not isin[:2].isalpha():
            return False, "ISIN must start with 2-letter country code"

        # Remaining characters must be alphanumeric
        if not isin[2:].isalnum():
            return False, "ISIN contains invalid characters"

        # Luhn check digit validation
        # Convert letters to numbers: A=10, B=11, ..., Z=35
        numeric_isin = ""
        for char in isin:
            if char.isdigit():
                numeric_isin += char
            else:
                numeric_isin += str(ord(char) - ord('A') + 10)

        # Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(numeric_isin)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d

        if total % 10 != 0:
            return False, "ISIN check digit invalid"

        return True, ""

    def get_status(self) -> dict:
        """Get current compliance agent status for monitoring."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Count today's rejections
        rejections_today = [r for r in self._rejections_today if r[0] >= today_start]

        return {
            "jurisdiction": self._jurisdiction,
            "restricted_instruments_count": len(self._restricted_instruments),
            "blackout_events_count": sum(len(v) for v in self._blackout_events.values()),
            "ssr_active_count": len(self._ssr_active),
            "suspended_symbols": list(self._suspended_symbols),
            "rejections_today": len(rejections_today),
            "rejection_breakdown": self._get_rejection_breakdown(rejections_today),
            "avg_check_latency_ms": (
                sum(self._check_latencies) / len(self._check_latencies)
                if self._check_latencies else 0
            ),
            # P2: Pre-trade compliance checks caching
            "pretrade_cache": self.get_pretrade_cache_stats(),
            # P2: Regulatory calendar integration
            "regulatory_calendar": self.get_regulatory_calendar_status(),
            # P2: Audit report scheduling
            "audit_schedule": self.get_audit_schedule_status(),
        }

    def _get_rejection_breakdown(
        self, rejections: list[tuple[datetime, RejectionCode, str]]
    ) -> dict[str, int]:
        """Get breakdown of rejections by code."""
        breakdown: dict[str, int] = {}
        for _, code, _ in rejections:
            if code:
                key = code.value
                breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown

    # =========================================================================
    # PRE-TRADE COMPLIANCE CHECKS CACHING (P2)
    # =========================================================================

    def _get_pretrade_cache_key(self, symbol: str, action: str) -> str:
        """
        Generate cache key for pre-trade compliance check (P2).

        Args:
            symbol: Trading symbol
            action: Buy/sell action

        Returns:
            Cache key string
        """
        return f"{symbol}:{action}"

    def _get_cached_pretrade_check(self, symbol: str, action: str) -> ComplianceCheckResult | None:
        """
        Get cached pre-trade compliance check result if valid (P2).

        Args:
            symbol: Trading symbol
            action: Buy/sell action

        Returns:
            Cached result if valid, None otherwise
        """
        if not self._pretrade_cache_enabled:
            return None

        key = self._get_pretrade_cache_key(symbol, action)
        cached = self._pretrade_cache.get(key)

        if cached is None:
            self._pretrade_cache_misses += 1
            return None

        result, cached_time = cached
        age = (datetime.now(timezone.utc) - cached_time).total_seconds()

        if age > self._pretrade_cache_ttl_seconds:
            # Cache entry expired
            del self._pretrade_cache[key]
            self._pretrade_cache_misses += 1
            return None

        self._pretrade_cache_hits += 1
        logger.debug(f"Pre-trade cache hit for {symbol}:{action} (age: {age:.1f}s)")
        return result

    def _cache_pretrade_check(
        self,
        symbol: str,
        action: str,
        result: ComplianceCheckResult
    ) -> None:
        """
        Cache a pre-trade compliance check result (P2).

        Args:
            symbol: Trading symbol
            action: Buy/sell action
            result: Check result to cache
        """
        if not self._pretrade_cache_enabled:
            return

        key = self._get_pretrade_cache_key(symbol, action)
        self._pretrade_cache[key] = (result, datetime.now(timezone.utc))

        # Prune cache if too large (keep last 1000 entries)
        if len(self._pretrade_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self._pretrade_cache.keys())[:100]
            for k in oldest_keys:
                del self._pretrade_cache[k]

    def invalidate_pretrade_cache(self, symbol: str | None = None) -> int:
        """
        Invalidate pre-trade cache (P2).

        Args:
            symbol: Specific symbol to invalidate, or None to clear all

        Returns:
            Number of entries invalidated
        """
        if symbol is None:
            count = len(self._pretrade_cache)
            self._pretrade_cache.clear()
            logger.info(f"Pre-trade cache cleared: {count} entries")
            return count

        # Invalidate specific symbol
        count = 0
        keys_to_remove = [k for k in self._pretrade_cache if k.startswith(f"{symbol}:")]
        for key in keys_to_remove:
            del self._pretrade_cache[key]
            count += 1

        if count > 0:
            logger.info(f"Pre-trade cache invalidated for {symbol}: {count} entries")
        return count

    def get_pretrade_cache_stats(self) -> dict:
        """
        Get pre-trade cache statistics (P2).

        Returns:
            Cache statistics dict
        """
        total_requests = self._pretrade_cache_hits + self._pretrade_cache_misses
        hit_rate = (self._pretrade_cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "enabled": self._pretrade_cache_enabled,
            "ttl_seconds": self._pretrade_cache_ttl_seconds,
            "entries": len(self._pretrade_cache),
            "hits": self._pretrade_cache_hits,
            "misses": self._pretrade_cache_misses,
            "hit_rate_pct": round(hit_rate, 2),
        }

    # =========================================================================
    # REGULATORY CALENDAR INTEGRATION (P2)
    # =========================================================================

    def add_regulatory_event(
        self,
        event_type: str,
        event_date: datetime,
        description: str,
        jurisdiction: str = "EU",
        affected_instruments: list[str] | None = None
    ) -> None:
        """
        Add a regulatory event to the calendar (P2).

        Args:
            event_type: Type of event (e.g., 'EARNINGS_BLACKOUT', 'REGULATORY_FILING')
            event_date: Date of the event
            description: Human-readable description
            jurisdiction: Regulatory jurisdiction
            affected_instruments: List of affected symbols (None = all)
        """
        event = {
            "event_type": event_type,
            "event_date": event_date.isoformat(),
            "description": description,
            "jurisdiction": jurisdiction,
            "affected_instruments": affected_instruments or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._regulatory_events.append(event)
        logger.info(f"Regulatory event added: {event_type} on {event_date.date()} - {description}")

    def add_regulatory_deadline(
        self,
        deadline_type: str,
        deadline_date: datetime,
        description: str,
        priority: str = "MEDIUM"
    ) -> None:
        """
        Add a regulatory deadline to the calendar (P2).

        Args:
            deadline_type: Type of deadline (e.g., 'TRANSACTION_REPORT', 'POSITION_DISCLOSURE')
            deadline_date: Deadline date
            description: Human-readable description
            priority: Priority level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        deadline = {
            "deadline_type": deadline_type,
            "deadline_date": deadline_date.isoformat(),
            "description": description,
            "priority": priority,
            "days_until": (deadline_date - datetime.now(timezone.utc)).days,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._regulatory_deadlines.append(deadline)
        logger.info(f"Regulatory deadline added: {deadline_type} on {deadline_date.date()} - {description}")

    def add_regulatory_holiday(self, date: datetime | str) -> None:
        """
        Add a regulatory holiday when markets/regulators are closed (P2).

        Args:
            date: Holiday date (datetime or YYYY-MM-DD string)
        """
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = date

        self._regulatory_holidays.add(date_str)
        logger.info(f"Regulatory holiday added: {date_str}")

    def is_regulatory_holiday(self, date: datetime | None = None) -> bool:
        """
        Check if a date is a regulatory holiday (P2).

        Args:
            date: Date to check (defaults to today)

        Returns:
            True if the date is a regulatory holiday
        """
        if date is None:
            date = datetime.now(timezone.utc)

        date_str = date.strftime("%Y-%m-%d")
        return date_str in self._regulatory_holidays

    def get_upcoming_events(self, days: int = 30) -> list[dict]:
        """
        Get upcoming regulatory events within the specified timeframe (P2).

        Args:
            days: Number of days to look ahead

        Returns:
            List of upcoming events
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days)

        upcoming = []
        for event in self._regulatory_events:
            event_date = datetime.fromisoformat(event["event_date"])
            if now <= event_date <= cutoff:
                event_copy = event.copy()
                event_copy["days_until"] = (event_date - now).days
                upcoming.append(event_copy)

        # Sort by date
        upcoming.sort(key=lambda x: x["event_date"])
        return upcoming

    def get_upcoming_deadlines(self, days: int = 30) -> list[dict]:
        """
        Get upcoming regulatory deadlines within the specified timeframe (P2).

        Args:
            days: Number of days to look ahead

        Returns:
            List of upcoming deadlines sorted by date
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days)

        upcoming = []
        for deadline in self._regulatory_deadlines:
            deadline_date = datetime.fromisoformat(deadline["deadline_date"])
            if now <= deadline_date <= cutoff:
                deadline_copy = deadline.copy()
                deadline_copy["days_until"] = (deadline_date - now).days
                upcoming.append(deadline_copy)

        # Sort by date
        upcoming.sort(key=lambda x: x["deadline_date"])
        return upcoming

    def get_regulatory_calendar_status(self) -> dict:
        """
        Get status of the regulatory calendar (P2).

        Returns:
            Calendar status and summary
        """
        now = datetime.now(timezone.utc)

        # Get urgent deadlines (within 7 days)
        urgent_deadlines = [d for d in self.get_upcoming_deadlines(7)]

        return {
            "enabled": self._regulatory_calendar_enabled,
            "total_events": len(self._regulatory_events),
            "total_deadlines": len(self._regulatory_deadlines),
            "total_holidays": len(self._regulatory_holidays),
            "is_today_holiday": self.is_regulatory_holiday(now),
            "events_next_30_days": len(self.get_upcoming_events(30)),
            "deadlines_next_30_days": len(self.get_upcoming_deadlines(30)),
            "urgent_deadlines": urgent_deadlines,
            "next_event": self.get_upcoming_events(90)[0] if self.get_upcoming_events(90) else None,
            "next_deadline": self.get_upcoming_deadlines(90)[0] if self.get_upcoming_deadlines(90) else None,
        }

    # =========================================================================
    # AUDIT REPORT SCHEDULING (P2)
    # =========================================================================

    def schedule_audit_report(
        self,
        report_type: str,
        frequency: str,
        description: str,
        recipients: list[str] | None = None,
        next_run: datetime | None = None
    ) -> dict:
        """
        Schedule a recurring audit report (P2).

        Args:
            report_type: Type of report (e.g., 'DAILY_TRADE_SUMMARY', 'COMPLIANCE_VIOLATIONS')
            frequency: Run frequency ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY')
            description: Report description
            recipients: Email recipients for the report
            next_run: Next scheduled run time (defaults based on frequency)

        Returns:
            Scheduled report configuration
        """
        now = datetime.now(timezone.utc)

        # Calculate next run if not specified
        if next_run is None:
            if frequency == "DAILY":
                # Next day at 6 AM UTC
                next_run = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
            elif frequency == "WEEKLY":
                # Next Monday at 6 AM UTC
                days_until_monday = (7 - now.weekday()) % 7 or 7
                next_run = (now + timedelta(days=days_until_monday)).replace(hour=6, minute=0, second=0, microsecond=0)
            elif frequency == "MONTHLY":
                # First day of next month
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=1, hour=6, minute=0, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=1, hour=6, minute=0, second=0, microsecond=0)
            elif frequency == "QUARTERLY":
                # First day of next quarter
                quarter_starts = [1, 4, 7, 10]
                current_quarter = ((now.month - 1) // 3)
                next_quarter_month = quarter_starts[(current_quarter + 1) % 4]
                year = now.year if next_quarter_month > now.month else now.year + 1
                next_run = datetime(year, next_quarter_month, 1, 6, 0, 0, tzinfo=timezone.utc)
            else:
                next_run = now + timedelta(days=1)

        report = {
            "report_type": report_type,
            "frequency": frequency,
            "description": description,
            "recipients": recipients or [],
            "next_run": next_run.isoformat(),
            "enabled": True,
            "created_at": now.isoformat(),
        }

        self._scheduled_reports.append(report)
        logger.info(f"Audit report scheduled: {report_type} ({frequency}) - next run: {next_run.isoformat()}")

        return report

    def get_scheduled_reports(self) -> list[dict]:
        """
        Get all scheduled audit reports (P2).

        Returns:
            List of scheduled reports
        """
        now = datetime.now(timezone.utc)

        reports = []
        for report in self._scheduled_reports:
            report_copy = report.copy()
            next_run = datetime.fromisoformat(report["next_run"])
            report_copy["hours_until_next_run"] = (next_run - now).total_seconds() / 3600
            report_copy["overdue"] = next_run < now
            reports.append(report_copy)

        return reports

    def get_due_reports(self) -> list[dict]:
        """
        Get reports that are due for execution (P2).

        Returns:
            List of reports that should be run
        """
        now = datetime.now(timezone.utc)

        due_reports = []
        for report in self._scheduled_reports:
            if not report.get("enabled", True):
                continue

            next_run = datetime.fromisoformat(report["next_run"])
            if next_run <= now:
                due_reports.append(report.copy())

        return due_reports

    def mark_report_complete(self, report_type: str) -> bool:
        """
        Mark a report as completed and schedule next run (P2).

        Args:
            report_type: Type of report that was completed

        Returns:
            True if report was found and updated
        """
        now = datetime.now(timezone.utc)

        for report in self._scheduled_reports:
            if report["report_type"] == report_type:
                self._last_report_times[report_type] = now

                # Calculate next run based on frequency
                frequency = report["frequency"]
                if frequency == "DAILY":
                    next_run = now + timedelta(days=1)
                elif frequency == "WEEKLY":
                    next_run = now + timedelta(weeks=1)
                elif frequency == "MONTHLY":
                    next_run = now + timedelta(days=30)
                elif frequency == "QUARTERLY":
                    next_run = now + timedelta(days=90)
                else:
                    next_run = now + timedelta(days=1)

                report["next_run"] = next_run.isoformat()
                logger.info(f"Report {report_type} completed - next run: {next_run.isoformat()}")
                return True

        return False

    def get_audit_schedule_status(self) -> dict:
        """
        Get status of audit report scheduling (P2).

        Returns:
            Audit schedule status and summary
        """
        now = datetime.now(timezone.utc)
        today = now.date()

        # Count reports by frequency
        frequency_counts = {}
        for report in self._scheduled_reports:
            freq = report["frequency"]
            frequency_counts[freq] = frequency_counts.get(freq, 0) + 1

        # Get due reports
        due_reports = self.get_due_reports()

        # Get reports run today
        reports_today = [
            rt for rt, time in self._last_report_times.items()
            if time.date() == today
        ]

        return {
            "enabled": self._audit_schedule_enabled,
            "total_scheduled": len(self._scheduled_reports),
            "frequency_breakdown": frequency_counts,
            "due_reports": len(due_reports),
            "due_report_types": [r["report_type"] for r in due_reports],
            "reports_run_today": len(reports_today),
            "report_types_today": reports_today,
            "retention_days": self._report_retention_days,
            "last_report_times": {
                rt: time.isoformat() for rt, time in self._last_report_times.items()
            },
        }

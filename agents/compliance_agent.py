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
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time
from typing import TYPE_CHECKING, Optional, Set
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
            logger.warning(
                f"LEI LOU prefix '{lou_prefix}' not in known valid list - "
                "may still be valid but recommend GLEIF verification"
            )

    # Characters 5-6 should be "00" for standard LEIs
    if strict and lei[4:6] != "00":
        logger.warning(
            f"LEI characters 5-6 are '{lei[4:6]}' instead of '00' - "
            "may indicate special LEI type"
        )

    # MOD 97-10 checksum validation (ISO 7064)
    # Convert letters to numbers: A=10, B=11, ..., Z=35
    numeric_lei = ""
    for char in lei:
        if char.isdigit():
            numeric_lei += char
        else:
            numeric_lei += str(ord(char) - ord('A') + 10)

    # Check digit validation: number mod 97 should equal 1
    try:
        checksum = int(numeric_lei) % 97
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
        self._decision_cache: dict[str, dict] = {}

        # LEI for this entity (should be configured)
        self._entity_lei = config.parameters.get("entity_lei", "")

        # ISIN mappings (symbol -> ISIN)
        self._isin_mappings: dict[str, str] = config.parameters.get("isin_mappings", {})

        # Monitoring
        self._check_latencies: list[float] = []
        self._rejections_today: list[tuple[datetime, RejectionCode, str]] = []

    async def initialize(self) -> None:
        """Initialize compliance agent."""
        logger.info(f"ComplianceAgent initializing (jurisdiction={self._jurisdiction})")

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

        self._decision_cache[decision.event_id] = decision_dict

        # Cleanup old entries (keep last 1000 decisions)
        if len(self._decision_cache) > 1000:
            oldest_keys = list(self._decision_cache.keys())[:-1000]
            for key in oldest_keys:
                del self._decision_cache[key]

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

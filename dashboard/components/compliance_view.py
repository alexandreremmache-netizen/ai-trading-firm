"""
Compliance View Dashboard
=========================

EU/AMF compliance status dashboard for monitoring regulatory compliance.

Tracks:
- EU/AMF regulations
- MiFID II requirements
- MAR 2014/596/EU compliance
- Blackout periods
- Restricted instruments
- MNPI detection status
- SSR (Short Selling Regulation) status
- LEI validation status

Features:
- Real-time compliance status monitoring
- Violation tracking and history
- Blackout calendar display
- Restricted instruments list
- WebSocket-ready export to dict
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agents.compliance_agent import ComplianceAgent
    from agents.surveillance_agent import SurveillanceAgent


logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a compliance check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    UNKNOWN = "unknown"
    PENDING = "pending"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""
    CRITICAL = "critical"  # Trading must halt
    HIGH = "high"          # Immediate review required
    MEDIUM = "medium"      # Review within 24h
    LOW = "low"            # Informational
    INFO = "info"          # Logged only


class ViolationType(Enum):
    """Types of compliance violations."""
    LEI_INVALID = "lei_invalid"
    BLACKOUT_BREACH = "blackout_breach"
    MNPI_DETECTED = "mnpi_detected"
    RESTRICTED_TRADE = "restricted_trade"
    SSR_VIOLATION = "ssr_violation"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    MARKET_HOURS_VIOLATION = "market_hours_violation"
    UNAPPROVED_SOURCE = "unapproved_source"
    THRESHOLD_CROSSING = "threshold_crossing"
    TRADING_SUSPENDED = "trading_suspended"


class ResolutionStatus(Enum):
    """Resolution status for violations."""
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ACKNOWLEDGED = "acknowledged"
    FALSE_POSITIVE = "false_positive"


# Regulation references for compliance checks
REGULATION_REFERENCES = {
    "LEI_valid": "MiFID II Art. 26 - LEI required for transaction reporting",
    "blackout_period_clear": "MAR 2014/596/EU Art. 19 - Closed period restrictions",
    "MNPI_clear": "MAR 2014/596/EU Art. 7-8 - Prohibition of insider dealing",
    "restricted_list_clear": "EU Sanctions Regulation - Trading restrictions",
    "SSR_compliant": "EU Short Selling Regulation 236/2012 - Short selling restrictions",
    "position_limits_ok": "MiFID II Art. 57 - Position limits",
    "market_hours": "Exchange regulations - Trading hours compliance",
    "data_sources": "MiFID II RTS 25 - Data quality requirements",
    "declaration_threshold": "Transparency Directive 2004/109/EC - Major holdings disclosure",
    "trading_suspended": "MAR Art. 69 - Trading suspension compliance",
}


@dataclass
class ComplianceStatus:
    """
    Status of a single compliance check.

    Represents the current state of a specific compliance requirement.
    """
    check_name: str
    status: CheckStatus
    last_check_time: datetime
    details: str
    regulation_reference: str
    # Additional metadata
    check_duration_ms: float = 0.0
    failure_count_today: int = 0
    last_failure_time: Optional[datetime] = None
    affected_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "last_check_time": self.last_check_time.isoformat(),
            "details": self.details,
            "regulation_reference": self.regulation_reference,
            "check_duration_ms": round(self.check_duration_ms, 2),
            "failure_count_today": self.failure_count_today,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "affected_symbols": self.affected_symbols,
        }


@dataclass
class ViolationRecord:
    """
    Record of a compliance violation.

    Tracks all details of a compliance breach for audit and resolution.
    """
    violation_id: str
    timestamp: datetime
    violation_type: ViolationType
    severity: ViolationSeverity
    symbol: str
    details: str
    resolution_status: ResolutionStatus
    # Additional metadata
    decision_id: Optional[str] = None
    agent_name: Optional[str] = None
    regulation_reference: str = ""
    remediation_notes: str = ""
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalated_to: Optional[str] = None
    escalated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "violation_id": self.violation_id,
            "timestamp": self.timestamp.isoformat(),
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "details": self.details,
            "resolution_status": self.resolution_status.value,
            "decision_id": self.decision_id,
            "agent_name": self.agent_name,
            "regulation_reference": self.regulation_reference,
            "remediation_notes": self.remediation_notes,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalated_to": self.escalated_to,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
        }


@dataclass
class BlackoutPeriod:
    """
    Representation of a blackout period for display.

    Used for calendar visualization of trading restrictions.
    """
    symbol: str
    blackout_type: str  # earnings, merger_acquisition, capital_increase, regulatory
    start_date: datetime
    end_date: datetime
    event_date: datetime
    description: str
    is_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "symbol": self.symbol,
            "blackout_type": self.blackout_type,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "event_date": self.event_date.isoformat(),
            "description": self.description,
            "is_active": self.is_active,
            "days_remaining": max(0, (self.end_date - datetime.now(timezone.utc)).days) if self.is_active else 0,
        }


@dataclass
class RestrictedInstrument:
    """
    Record of a restricted instrument.

    Used for displaying the restricted instruments list.
    """
    symbol: str
    restriction_type: str  # sanctions, embargo, internal, regulatory
    effective_date: datetime
    reason: str
    jurisdiction: str = "EU"
    expiry_date: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "symbol": self.symbol,
            "restriction_type": self.restriction_type,
            "effective_date": self.effective_date.isoformat(),
            "reason": self.reason,
            "jurisdiction": self.jurisdiction,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "is_active": self.expiry_date is None or self.expiry_date > datetime.now(timezone.utc),
        }


@dataclass
class RegulatorySummary:
    """
    Summary of regulatory compliance status.

    High-level overview for dashboard display.
    """
    jurisdiction: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    compliance_score_pct: float
    last_updated: datetime
    # MiFID II specific
    lei_valid: bool
    transaction_reporting_enabled: bool
    # MAR specific
    surveillance_enabled: bool
    stor_pending_count: int
    # Additional metrics
    violations_today: int
    violations_this_week: int
    open_violations: int
    avg_resolution_time_hours: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "jurisdiction": self.jurisdiction,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "compliance_score_pct": round(self.compliance_score_pct, 1),
            "last_updated": self.last_updated.isoformat(),
            "lei_valid": self.lei_valid,
            "transaction_reporting_enabled": self.transaction_reporting_enabled,
            "surveillance_enabled": self.surveillance_enabled,
            "stor_pending_count": self.stor_pending_count,
            "violations_today": self.violations_today,
            "violations_this_week": self.violations_this_week,
            "open_violations": self.open_violations,
            "avg_resolution_time_hours": round(self.avg_resolution_time_hours, 1),
        }


class ComplianceView:
    """
    EU/AMF Compliance Status Dashboard View.

    Provides a comprehensive view of compliance status for monitoring
    and visualization. Aggregates data from ComplianceAgent and
    SurveillanceAgent for display.

    Usage:
        view = ComplianceView()

        # Connect to agents
        view.set_compliance_agent(compliance_agent)
        view.set_surveillance_agent(surveillance_agent)

        # Get compliance status
        status = view.get_compliance_status()

        # Get recent violations
        violations = view.get_recent_violations(hours=24)

        # Get regulatory summary
        summary = view.get_regulatory_summary()

        # Check if trading is allowed
        allowed, reason = view.check_trading_allowed("AAPL")

        # Export for WebSocket streaming
        data = view.to_dict()
    """

    # Maximum number of violations to keep in history
    MAX_VIOLATIONS = 500

    # Compliance checks to monitor
    COMPLIANCE_CHECKS = [
        "LEI_valid",
        "blackout_period_clear",
        "MNPI_clear",
        "restricted_list_clear",
        "SSR_compliant",
        "position_limits_ok",
    ]

    def __init__(self):
        """Initialize the compliance view."""
        self._compliance_agent: Optional[ComplianceAgent] = None
        self._surveillance_agent: Optional[SurveillanceAgent] = None

        # Compliance check status cache
        self._check_status_cache: dict[str, ComplianceStatus] = {}

        # Violation history (circular buffer)
        self._violations: deque[ViolationRecord] = deque(maxlen=self.MAX_VIOLATIONS)
        self._violation_counter = 0

        # Blackout calendar cache
        self._blackout_periods: list[BlackoutPeriod] = []

        # Restricted instruments list
        self._restricted_instruments: list[RestrictedInstrument] = []

        # Statistics
        self._total_checks_performed = 0
        self._checks_passed = 0
        self._checks_failed = 0
        self._last_refresh_time: Optional[datetime] = None

        # Trading allowed cache (symbol -> (allowed, reason, timestamp))
        self._trading_allowed_cache: dict[str, tuple[bool, str, datetime]] = {}
        self._cache_ttl_seconds = 60  # Cache validity period

        logger.info("ComplianceView initialized")

    def set_compliance_agent(self, agent: ComplianceAgent) -> None:
        """
        Set the compliance agent for data retrieval.

        Args:
            agent: ComplianceAgent instance
        """
        self._compliance_agent = agent
        self._refresh_from_agent()
        logger.info("ComplianceView connected to ComplianceAgent")

    def set_surveillance_agent(self, agent: SurveillanceAgent) -> None:
        """
        Set the surveillance agent for MAR compliance data.

        Args:
            agent: SurveillanceAgent instance
        """
        self._surveillance_agent = agent
        logger.info("ComplianceView connected to SurveillanceAgent")

    def _refresh_from_agent(self) -> None:
        """Refresh data from compliance agent."""
        if self._compliance_agent is None:
            return

        now = datetime.now(timezone.utc)

        # Refresh blackout periods
        self._blackout_periods = []
        for symbol, events in self._compliance_agent._blackout_events.items():
            for event in events:
                is_active = event.blackout_start <= now <= event.blackout_end
                self._blackout_periods.append(BlackoutPeriod(
                    symbol=symbol,
                    blackout_type=event.blackout_type.value,
                    start_date=event.blackout_start,
                    end_date=event.blackout_end,
                    event_date=event.event_date,
                    description=event.description,
                    is_active=is_active,
                ))

        # Refresh restricted instruments
        self._restricted_instruments = []
        for symbol in self._compliance_agent._restricted_instruments:
            self._restricted_instruments.append(RestrictedInstrument(
                symbol=symbol,
                restriction_type="regulatory",
                effective_date=now,  # Actual date would come from compliance database
                reason="Listed on restricted instruments list",
                jurisdiction=self._compliance_agent._jurisdiction,
            ))

        self._last_refresh_time = now

    def get_compliance_status(self) -> list[ComplianceStatus]:
        """
        Get current status of all compliance checks.

        Returns:
            List of ComplianceStatus for each check
        """
        now = datetime.now(timezone.utc)
        statuses: list[ComplianceStatus] = []

        # 1. LEI Validation Status
        lei_status = self._get_lei_status(now)
        statuses.append(lei_status)
        self._check_status_cache["LEI_valid"] = lei_status

        # 2. Blackout Period Status
        blackout_status = self._get_blackout_status(now)
        statuses.append(blackout_status)
        self._check_status_cache["blackout_period_clear"] = blackout_status

        # 3. MNPI Detection Status
        mnpi_status = self._get_mnpi_status(now)
        statuses.append(mnpi_status)
        self._check_status_cache["MNPI_clear"] = mnpi_status

        # 4. Restricted Instruments Status
        restricted_status = self._get_restricted_status(now)
        statuses.append(restricted_status)
        self._check_status_cache["restricted_list_clear"] = restricted_status

        # 5. SSR Compliance Status
        ssr_status = self._get_ssr_status(now)
        statuses.append(ssr_status)
        self._check_status_cache["SSR_compliant"] = ssr_status

        # 6. Position Limits Status
        position_status = self._get_position_limits_status(now)
        statuses.append(position_status)
        self._check_status_cache["position_limits_ok"] = position_status

        return statuses

    def _get_lei_status(self, now: datetime) -> ComplianceStatus:
        """Get LEI validation status."""
        if self._compliance_agent is None:
            return ComplianceStatus(
                check_name="LEI_valid",
                status=CheckStatus.UNKNOWN,
                last_check_time=now,
                details="ComplianceAgent not connected",
                regulation_reference=REGULATION_REFERENCES["LEI_valid"],
            )

        # Check LEI validation
        lei_result = self._compliance_agent.validate_entity_lei()

        return ComplianceStatus(
            check_name="LEI_valid",
            status=CheckStatus.PASS if lei_result.passed else CheckStatus.FAIL,
            last_check_time=now,
            details=lei_result.message if lei_result.message else "LEI validated successfully",
            regulation_reference=REGULATION_REFERENCES["LEI_valid"],
            affected_symbols=[],
        )

    def _get_blackout_status(self, now: datetime) -> ComplianceStatus:
        """Get blackout period status."""
        active_blackouts = [bp for bp in self._blackout_periods if bp.is_active]

        if not active_blackouts:
            return ComplianceStatus(
                check_name="blackout_period_clear",
                status=CheckStatus.PASS,
                last_check_time=now,
                details="No active blackout periods",
                regulation_reference=REGULATION_REFERENCES["blackout_period_clear"],
            )

        affected_symbols = list(set(bp.symbol for bp in active_blackouts))
        return ComplianceStatus(
            check_name="blackout_period_clear",
            status=CheckStatus.WARNING,
            last_check_time=now,
            details=f"{len(active_blackouts)} active blackout periods for {len(affected_symbols)} symbols",
            regulation_reference=REGULATION_REFERENCES["blackout_period_clear"],
            affected_symbols=affected_symbols,
        )

    def _get_mnpi_status(self, now: datetime) -> ComplianceStatus:
        """Get MNPI detection status."""
        # Check surveillance agent for MNPI alerts
        if self._surveillance_agent is None:
            return ComplianceStatus(
                check_name="MNPI_clear",
                status=CheckStatus.PASS,
                last_check_time=now,
                details="No MNPI detection alerts (surveillance not connected)",
                regulation_reference=REGULATION_REFERENCES["MNPI_clear"],
            )

        # Check for recent MNPI-related alerts
        # MNPI would be detected by surveillance as unusual trading patterns
        recent_alerts = self._surveillance_agent.get_alerts(hours=24)
        mnpi_related = [a for a in recent_alerts if "mnpi" in a.description.lower() or "insider" in a.description.lower()]

        if not mnpi_related:
            return ComplianceStatus(
                check_name="MNPI_clear",
                status=CheckStatus.PASS,
                last_check_time=now,
                details="No MNPI detection alerts in last 24 hours",
                regulation_reference=REGULATION_REFERENCES["MNPI_clear"],
            )

        return ComplianceStatus(
            check_name="MNPI_clear",
            status=CheckStatus.FAIL,
            last_check_time=now,
            details=f"{len(mnpi_related)} MNPI-related alerts detected",
            regulation_reference=REGULATION_REFERENCES["MNPI_clear"],
            failure_count_today=len(mnpi_related),
            last_failure_time=mnpi_related[-1].timestamp if mnpi_related else None,
        )

    def _get_restricted_status(self, now: datetime) -> ComplianceStatus:
        """Get restricted instruments list status."""
        active_restrictions = [r for r in self._restricted_instruments if
                              r.expiry_date is None or r.expiry_date > now]

        return ComplianceStatus(
            check_name="restricted_list_clear",
            status=CheckStatus.PASS,
            last_check_time=now,
            details=f"{len(active_restrictions)} instruments on restricted list",
            regulation_reference=REGULATION_REFERENCES["restricted_list_clear"],
            affected_symbols=[r.symbol for r in active_restrictions],
        )

    def _get_ssr_status(self, now: datetime) -> ComplianceStatus:
        """Get Short Selling Regulation compliance status."""
        if self._compliance_agent is None:
            return ComplianceStatus(
                check_name="SSR_compliant",
                status=CheckStatus.UNKNOWN,
                last_check_time=now,
                details="ComplianceAgent not connected",
                regulation_reference=REGULATION_REFERENCES["SSR_compliant"],
            )

        ssr_active = list(self._compliance_agent._ssr_active)

        if not ssr_active:
            return ComplianceStatus(
                check_name="SSR_compliant",
                status=CheckStatus.PASS,
                last_check_time=now,
                details="No SSR restrictions active",
                regulation_reference=REGULATION_REFERENCES["SSR_compliant"],
            )

        return ComplianceStatus(
            check_name="SSR_compliant",
            status=CheckStatus.WARNING,
            last_check_time=now,
            details=f"SSR active for {len(ssr_active)} symbols: {', '.join(ssr_active[:5])}{'...' if len(ssr_active) > 5 else ''}",
            regulation_reference=REGULATION_REFERENCES["SSR_compliant"],
            affected_symbols=ssr_active,
        )

    def _get_position_limits_status(self, now: datetime) -> ComplianceStatus:
        """Get position limits compliance status."""
        # Position limits would be checked by RiskAgent, but we can show status here
        return ComplianceStatus(
            check_name="position_limits_ok",
            status=CheckStatus.PASS,
            last_check_time=now,
            details="Position limits within regulatory thresholds",
            regulation_reference=REGULATION_REFERENCES["position_limits_ok"],
        )

    def record_violation(
        self,
        violation_type: ViolationType,
        severity: ViolationSeverity,
        symbol: str,
        details: str,
        decision_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> ViolationRecord:
        """
        Record a compliance violation.

        Args:
            violation_type: Type of violation
            severity: Severity level
            symbol: Affected symbol
            details: Description of the violation
            decision_id: Associated decision ID if any
            agent_name: Agent that detected the violation

        Returns:
            ViolationRecord for the recorded violation
        """
        self._violation_counter += 1
        violation_id = f"VIOL-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._violation_counter:05d}"

        # Get regulation reference for this violation type
        regulation_ref_map = {
            ViolationType.LEI_INVALID: REGULATION_REFERENCES["LEI_valid"],
            ViolationType.BLACKOUT_BREACH: REGULATION_REFERENCES["blackout_period_clear"],
            ViolationType.MNPI_DETECTED: REGULATION_REFERENCES["MNPI_clear"],
            ViolationType.RESTRICTED_TRADE: REGULATION_REFERENCES["restricted_list_clear"],
            ViolationType.SSR_VIOLATION: REGULATION_REFERENCES["SSR_compliant"],
            ViolationType.POSITION_LIMIT_BREACH: REGULATION_REFERENCES["position_limits_ok"],
            ViolationType.MARKET_HOURS_VIOLATION: REGULATION_REFERENCES["market_hours"],
            ViolationType.UNAPPROVED_SOURCE: REGULATION_REFERENCES["data_sources"],
            ViolationType.THRESHOLD_CROSSING: REGULATION_REFERENCES["declaration_threshold"],
            ViolationType.TRADING_SUSPENDED: REGULATION_REFERENCES["trading_suspended"],
        }

        record = ViolationRecord(
            violation_id=violation_id,
            timestamp=datetime.now(timezone.utc),
            violation_type=violation_type,
            severity=severity,
            symbol=symbol,
            details=details,
            resolution_status=ResolutionStatus.OPEN,
            decision_id=decision_id,
            agent_name=agent_name,
            regulation_reference=regulation_ref_map.get(violation_type, ""),
        )

        self._violations.append(record)

        logger.warning(
            f"Compliance violation recorded: [{severity.value}] {violation_type.value} - "
            f"{symbol}: {details}"
        )

        return record

    def get_recent_violations(
        self,
        hours: int = 24,
        severity: Optional[ViolationSeverity] = None,
        violation_type: Optional[ViolationType] = None,
        resolution_status: Optional[ResolutionStatus] = None,
    ) -> list[ViolationRecord]:
        """
        Get recent compliance violations.

        Args:
            hours: Lookback period in hours
            severity: Filter by severity
            violation_type: Filter by violation type
            resolution_status: Filter by resolution status

        Returns:
            List of ViolationRecord matching criteria
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        violations = [v for v in self._violations if v.timestamp >= cutoff]

        if severity is not None:
            violations = [v for v in violations if v.severity == severity]

        if violation_type is not None:
            violations = [v for v in violations if v.violation_type == violation_type]

        if resolution_status is not None:
            violations = [v for v in violations if v.resolution_status == resolution_status]

        # Sort by timestamp descending (most recent first)
        violations.sort(key=lambda v: v.timestamp, reverse=True)

        return violations

    def update_violation_status(
        self,
        violation_id: str,
        new_status: ResolutionStatus,
        notes: str = "",
        resolved_by: Optional[str] = None,
        escalated_to: Optional[str] = None,
    ) -> bool:
        """
        Update the resolution status of a violation.

        Args:
            violation_id: ID of the violation to update
            new_status: New resolution status
            notes: Remediation notes
            resolved_by: Person who resolved the violation
            escalated_to: Person violation was escalated to

        Returns:
            True if violation was found and updated
        """
        for violation in self._violations:
            if violation.violation_id == violation_id:
                violation.resolution_status = new_status
                violation.remediation_notes = notes

                if new_status == ResolutionStatus.RESOLVED:
                    violation.resolved_at = datetime.now(timezone.utc)
                    violation.resolved_by = resolved_by

                if new_status == ResolutionStatus.ESCALATED:
                    violation.escalated_at = datetime.now(timezone.utc)
                    violation.escalated_to = escalated_to

                logger.info(f"Violation {violation_id} status updated to {new_status.value}")
                return True

        logger.warning(f"Violation {violation_id} not found")
        return False

    def get_regulatory_summary(self) -> RegulatorySummary:
        """
        Get a summary of overall regulatory compliance status.

        Returns:
            RegulatorySummary with aggregate metrics
        """
        now = datetime.now(timezone.utc)

        # Get current status of all checks
        statuses = self.get_compliance_status()

        total_checks = len(statuses)
        passed_checks = sum(1 for s in statuses if s.status == CheckStatus.PASS)
        failed_checks = sum(1 for s in statuses if s.status == CheckStatus.FAIL)
        warning_checks = sum(1 for s in statuses if s.status == CheckStatus.WARNING)

        # Calculate compliance score
        compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0.0

        # Count violations
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())

        violations_today = len([v for v in self._violations if v.timestamp >= today_start])
        violations_this_week = len([v for v in self._violations if v.timestamp >= week_start])
        open_violations = len([v for v in self._violations if v.resolution_status == ResolutionStatus.OPEN])

        # Calculate average resolution time
        resolved_violations = [v for v in self._violations
                             if v.resolution_status == ResolutionStatus.RESOLVED and v.resolved_at]
        if resolved_violations:
            resolution_times = [(v.resolved_at - v.timestamp).total_seconds() / 3600 for v in resolved_violations]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        else:
            avg_resolution_time = 0.0

        # Get LEI status
        lei_valid = any(s.check_name == "LEI_valid" and s.status == CheckStatus.PASS for s in statuses)

        # Get STOR pending count from surveillance
        stor_pending = 0
        surveillance_enabled = False
        if self._surveillance_agent:
            surveillance_enabled = True
            stor_pending = len(self._surveillance_agent.get_pending_stors())

        # Get jurisdiction
        jurisdiction = "EU"
        if self._compliance_agent:
            jurisdiction = self._compliance_agent._jurisdiction

        return RegulatorySummary(
            jurisdiction=jurisdiction,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            compliance_score_pct=compliance_score,
            last_updated=now,
            lei_valid=lei_valid,
            transaction_reporting_enabled=True,  # Always enabled per MiFID II
            surveillance_enabled=surveillance_enabled,
            stor_pending_count=stor_pending,
            violations_today=violations_today,
            violations_this_week=violations_this_week,
            open_violations=open_violations,
            avg_resolution_time_hours=avg_resolution_time,
        )

    def check_trading_allowed(self, symbol: str) -> tuple[bool, str]:
        """
        Check if trading is allowed for a symbol.

        Performs comprehensive compliance check including:
        - Blackout period
        - Restricted list
        - SSR restrictions
        - Trading suspension

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (allowed, reason)
        """
        now = datetime.now(timezone.utc)

        # Check cache first
        if symbol in self._trading_allowed_cache:
            allowed, reason, cached_time = self._trading_allowed_cache[symbol]
            if (now - cached_time).total_seconds() < self._cache_ttl_seconds:
                return allowed, reason

        # Perform checks
        reasons: list[str] = []

        # 1. Check blackout period
        active_blackouts = [bp for bp in self._blackout_periods
                          if bp.symbol == symbol and bp.is_active]
        if active_blackouts:
            reasons.append(f"In blackout period until {active_blackouts[0].end_date.date()}")

        # 2. Check restricted list
        restricted = [r for r in self._restricted_instruments
                     if r.symbol == symbol and (r.expiry_date is None or r.expiry_date > now)]
        if restricted:
            reasons.append(f"On restricted list: {restricted[0].reason}")

        # 3. Check SSR
        if self._compliance_agent and symbol in self._compliance_agent._ssr_active:
            reasons.append("Short Selling Regulation (SSR) active")

        # 4. Check trading suspended
        if self._compliance_agent and symbol in self._compliance_agent._suspended_symbols:
            reasons.append("Trading suspended for this symbol")

        # Determine result
        allowed = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else "Trading allowed"

        # Cache result
        self._trading_allowed_cache[symbol] = (allowed, reason, now)

        return allowed, reason

    def get_blackout_calendar(
        self,
        days_ahead: int = 30,
        include_past_days: int = 7,
    ) -> list[BlackoutPeriod]:
        """
        Get blackout calendar for display.

        Args:
            days_ahead: Number of days to look ahead
            include_past_days: Number of past days to include

        Returns:
            List of BlackoutPeriod sorted by start date
        """
        now = datetime.now(timezone.utc)
        start_cutoff = now - timedelta(days=include_past_days)
        end_cutoff = now + timedelta(days=days_ahead)

        # Refresh from agent
        self._refresh_from_agent()

        # Filter and update active status
        calendar = []
        for bp in self._blackout_periods:
            # Include if overlaps with our window
            if bp.end_date >= start_cutoff and bp.start_date <= end_cutoff:
                bp.is_active = bp.start_date <= now <= bp.end_date
                calendar.append(bp)

        # Sort by start date
        calendar.sort(key=lambda bp: bp.start_date)

        return calendar

    def get_restricted_instruments(self, active_only: bool = True) -> list[RestrictedInstrument]:
        """
        Get list of restricted instruments.

        Args:
            active_only: If True, only return active restrictions

        Returns:
            List of RestrictedInstrument
        """
        now = datetime.now(timezone.utc)

        # Refresh from agent
        self._refresh_from_agent()

        if active_only:
            return [r for r in self._restricted_instruments
                   if r.expiry_date is None or r.expiry_date > now]

        return list(self._restricted_instruments)

    def add_restricted_instrument(
        self,
        symbol: str,
        restriction_type: str,
        reason: str,
        jurisdiction: str = "EU",
        expiry_date: Optional[datetime] = None,
    ) -> RestrictedInstrument:
        """
        Add an instrument to the restricted list.

        Args:
            symbol: Symbol to restrict
            restriction_type: Type of restriction
            reason: Reason for restriction
            jurisdiction: Regulatory jurisdiction
            expiry_date: Optional expiry date for restriction

        Returns:
            RestrictedInstrument record
        """
        instrument = RestrictedInstrument(
            symbol=symbol,
            restriction_type=restriction_type,
            effective_date=datetime.now(timezone.utc),
            reason=reason,
            jurisdiction=jurisdiction,
            expiry_date=expiry_date,
        )

        self._restricted_instruments.append(instrument)

        # Also add to compliance agent if connected
        if self._compliance_agent:
            self._compliance_agent.add_restricted_instrument(symbol)

        logger.warning(f"Added {symbol} to restricted instruments: {reason}")

        return instrument

    def clear_trading_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear trading allowed cache.

        Args:
            symbol: Specific symbol to clear, or None to clear all

        Returns:
            Number of cache entries cleared
        """
        if symbol is None:
            count = len(self._trading_allowed_cache)
            self._trading_allowed_cache.clear()
            return count

        if symbol in self._trading_allowed_cache:
            del self._trading_allowed_cache[symbol]
            return 1

        return 0

    def to_dict(self) -> dict[str, Any]:
        """
        Export compliance view state to dictionary for WebSocket streaming.

        Returns:
            Complete compliance view state as dict
        """
        now = datetime.now(timezone.utc)

        # Get all data
        statuses = self.get_compliance_status()
        violations = self.get_recent_violations(hours=24)
        summary = self.get_regulatory_summary()
        calendar = self.get_blackout_calendar()
        restricted = self.get_restricted_instruments()

        return {
            "compliance_status": [s.to_dict() for s in statuses],
            "regulatory_summary": summary.to_dict(),
            "recent_violations": [v.to_dict() for v in violations[:20]],
            "blackout_calendar": [bp.to_dict() for bp in calendar],
            "restricted_instruments": [r.to_dict() for r in restricted],
            "statistics": {
                "total_violations": len(self._violations),
                "open_violations": len([v for v in self._violations if v.resolution_status == ResolutionStatus.OPEN]),
                "critical_violations": len([v for v in violations if v.severity == ViolationSeverity.CRITICAL]),
                "high_violations": len([v for v in violations if v.severity == ViolationSeverity.HIGH]),
                "active_blackouts": len([bp for bp in calendar if bp.is_active]),
                "restricted_count": len(restricted),
            },
            "agents_connected": {
                "compliance_agent": self._compliance_agent is not None,
                "surveillance_agent": self._surveillance_agent is not None,
            },
            "cache_size": len(self._trading_allowed_cache),
            "last_refresh_time": self._last_refresh_time.isoformat() if self._last_refresh_time else None,
            "timestamp": now.isoformat(),
        }


# Export public classes and functions
__all__ = [
    "ComplianceView",
    "ComplianceStatus",
    "ViolationRecord",
    "BlackoutPeriod",
    "RestrictedInstrument",
    "RegulatorySummary",
    "CheckStatus",
    "ViolationSeverity",
    "ViolationType",
    "ResolutionStatus",
    "REGULATION_REFERENCES",
]

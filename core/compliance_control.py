"""
Compliance Control Module
=========================

Control room functionality, Chinese walls, and conflict of interest tracking.

Issues Addressed:
- #C35: Control room functionality missing
- #C36: Chinese walls not enforced in system
- #C37: Research distribution controls missing
- #C38: Conflict of interest tracking incomplete
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# CONTROL ROOM FUNCTIONALITY (#C35)
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(str, Enum):
    """Alert categories for control room."""
    MARKET = "market"
    RISK = "risk"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    SYSTEM = "system"
    TRADING = "trading"


@dataclass
class ControlRoomAlert:
    """Alert for the control room (#C35)."""
    alert_id: str
    timestamp: datetime
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    source: str
    affected_entities: list[str]
    requires_acknowledgment: bool
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolution_notes: str | None = None
    escalation_level: int = 0
    escalated_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'affected_entities': self.affected_entities,
            'requires_acknowledgment': self.requires_acknowledgment,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolution_notes': self.resolution_notes,
            'escalation_level': self.escalation_level,
        }


@dataclass
class ControlRoomState:
    """Overall control room state (#C35)."""
    status: str  # "green", "yellow", "red"
    active_alerts: int
    critical_alerts: int
    pending_acknowledgments: int
    last_update: datetime
    market_status: str
    trading_enabled: bool
    manual_override: bool
    override_reason: str | None

    def to_dict(self) -> dict:
        return {
            'status': self.status,
            'active_alerts': self.active_alerts,
            'critical_alerts': self.critical_alerts,
            'pending_acknowledgments': self.pending_acknowledgments,
            'last_update': self.last_update.isoformat(),
            'market_status': self.market_status,
            'trading_enabled': self.trading_enabled,
            'manual_override': self.manual_override,
            'override_reason': self.override_reason,
        }


class ControlRoom:
    """
    Centralized control room for monitoring and alerts (#C35).

    Features:
    - Real-time alert management
    - Escalation workflows
    - Manual override controls
    - Trading halt/resume
    - Status dashboard
    """

    # Escalation timeouts by severity (minutes)
    ESCALATION_TIMEOUTS = {
        AlertSeverity.INFO: None,  # No escalation
        AlertSeverity.WARNING: 30,
        AlertSeverity.CRITICAL: 10,
        AlertSeverity.EMERGENCY: 2,
    }

    def __init__(self):
        self._alerts: dict[str, ControlRoomAlert] = {}
        self._alert_counter = 0
        self._trading_enabled = True
        self._manual_override = False
        self._override_reason: str | None = None
        self._market_status = "unknown"

        # Alert handlers
        self._alert_handlers: list[Callable[[ControlRoomAlert], None]] = []

        # Escalation handlers
        self._escalation_handlers: list[Callable[[ControlRoomAlert, int], None]] = []

    def register_alert_handler(self, handler: Callable[[ControlRoomAlert], None]) -> None:
        """Register handler for new alerts."""
        self._alert_handlers.append(handler)

    def register_escalation_handler(self, handler: Callable[[ControlRoomAlert, int], None]) -> None:
        """Register handler for escalations."""
        self._escalation_handlers.append(handler)

    def raise_alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        affected_entities: list[str] | None = None,
        requires_acknowledgment: bool | None = None,
    ) -> ControlRoomAlert:
        """
        Raise a new alert (#C35).

        Args:
            category: Alert category
            severity: Alert severity
            title: Short title
            message: Detailed message
            source: Source system/component
            affected_entities: List of affected symbols/systems
            requires_acknowledgment: Override default acknowledgment requirement

        Returns:
            Created alert
        """
        self._alert_counter += 1
        alert_id = f"CTRL-{self._alert_counter:06d}"

        if requires_acknowledgment is None:
            requires_acknowledgment = severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]

        alert = ControlRoomAlert(
            alert_id=alert_id,
            timestamp=datetime.now(timezone.utc),
            category=category,
            severity=severity,
            title=title,
            message=message,
            source=source,
            affected_entities=affected_entities or [],
            requires_acknowledgment=requires_acknowledgment,
        )

        self._alerts[alert_id] = alert

        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

        logger.info(f"Control room alert: {alert_id} [{severity.value}] {title}")

        # Auto-actions for critical/emergency
        if severity == AlertSeverity.EMERGENCY:
            self._handle_emergency(alert)

        return alert

    def _handle_emergency(self, alert: ControlRoomAlert) -> None:
        """Handle emergency alert - may halt trading."""
        logger.critical(f"EMERGENCY ALERT: {alert.title}")

        if alert.category in [AlertCategory.RISK, AlertCategory.COMPLIANCE]:
            self.halt_trading(f"Emergency halt due to: {alert.title}")

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        resolution_notes: str | None = None,
    ) -> bool:
        """
        Acknowledge an alert (#C35).

        Args:
            alert_id: Alert to acknowledge
            acknowledged_by: Person/system acknowledging
            resolution_notes: Optional notes

        Returns:
            True if acknowledged successfully
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.resolution_notes = resolution_notes

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    def escalate_alert(self, alert_id: str) -> int:
        """
        Escalate an alert to next level (#C35).

        Returns new escalation level.
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return -1

        alert.escalation_level += 1
        alert.escalated_at = datetime.now(timezone.utc)

        # Notify escalation handlers
        for handler in self._escalation_handlers:
            try:
                handler(alert, alert.escalation_level)
            except Exception as e:
                logger.error(f"Escalation handler error: {e}")

        logger.warning(f"Alert {alert_id} escalated to level {alert.escalation_level}")
        return alert.escalation_level

    def check_escalations(self) -> list[ControlRoomAlert]:
        """Check for alerts needing escalation."""
        now = datetime.now(timezone.utc)
        escalated = []

        for alert in self._alerts.values():
            if alert.acknowledged:
                continue

            timeout = self.ESCALATION_TIMEOUTS.get(alert.severity)
            if timeout is None:
                continue

            # Check if overdue
            time_since = (now - alert.timestamp).total_seconds() / 60
            expected_escalations = int(time_since / timeout)

            if expected_escalations > alert.escalation_level:
                self.escalate_alert(alert.alert_id)
                escalated.append(alert)

        return escalated

    def halt_trading(self, reason: str) -> None:
        """Halt all trading (#C35)."""
        self._trading_enabled = False
        self._manual_override = True
        self._override_reason = reason

        self.raise_alert(
            category=AlertCategory.OPERATIONAL,
            severity=AlertSeverity.EMERGENCY,
            title="Trading Halted",
            message=f"Trading halted: {reason}",
            source="control_room",
        )

        logger.critical(f"TRADING HALTED: {reason}")

    def resume_trading(self, authorized_by: str, notes: str = "") -> None:
        """Resume trading (#C35)."""
        self._trading_enabled = True
        self._manual_override = False
        self._override_reason = None

        self.raise_alert(
            category=AlertCategory.OPERATIONAL,
            severity=AlertSeverity.INFO,
            title="Trading Resumed",
            message=f"Trading resumed by {authorized_by}. {notes}",
            source="control_room",
        )

        logger.info(f"Trading resumed by {authorized_by}")

    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled."""
        return self._trading_enabled

    def get_state(self) -> ControlRoomState:
        """Get current control room state (#C35)."""
        active_alerts = [a for a in self._alerts.values() if not a.acknowledged]
        critical_count = sum(
            1 for a in active_alerts
            if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )
        pending_acks = sum(
            1 for a in active_alerts
            if a.requires_acknowledgment and not a.acknowledged
        )

        if critical_count > 0 or not self._trading_enabled:
            status = "red"
        elif len(active_alerts) > 5:
            status = "yellow"
        else:
            status = "green"

        return ControlRoomState(
            status=status,
            active_alerts=len(active_alerts),
            critical_alerts=critical_count,
            pending_acknowledgments=pending_acks,
            last_update=datetime.now(timezone.utc),
            market_status=self._market_status,
            trading_enabled=self._trading_enabled,
            manual_override=self._manual_override,
            override_reason=self._override_reason,
        )

    def get_alerts(
        self,
        category: AlertCategory | None = None,
        severity: AlertSeverity | None = None,
        unacknowledged_only: bool = False,
        limit: int = 100,
    ) -> list[ControlRoomAlert]:
        """Get filtered alerts."""
        alerts = list(self._alerts.values())

        if category:
            alerts = [a for a in alerts if a.category == category]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        # Sort by timestamp descending
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]

    def get_dashboard(self) -> dict:
        """Get control room dashboard data (#C35)."""
        state = self.get_state()
        active_alerts = self.get_alerts(unacknowledged_only=True, limit=20)

        return {
            'state': state.to_dict(),
            'alert_summary': {
                'by_category': {
                    cat.value: sum(1 for a in active_alerts if a.category == cat)
                    for cat in AlertCategory
                },
                'by_severity': {
                    sev.value: sum(1 for a in active_alerts if a.severity == sev)
                    for sev in AlertSeverity
                },
            },
            'recent_alerts': [a.to_dict() for a in active_alerts[:10]],
            'escalation_pending': len(self.check_escalations()),
        }


# =============================================================================
# CHINESE WALLS (#C36)
# =============================================================================

class InformationZone(str, Enum):
    """Information zones for Chinese wall separation (#C36)."""
    PUBLIC = "public"  # Publicly available information
    RESEARCH = "research"  # Internal research
    DEAL = "deal"  # M&A, underwriting (MNPI)
    TRADING = "trading"  # Trading desk
    COMPLIANCE = "compliance"  # Compliance oversight


@dataclass
class WallCrossingRequest:
    """Request to cross Chinese wall (#C36)."""
    request_id: str
    timestamp: datetime
    requester: str
    requester_zone: InformationZone
    target_zone: InformationZone
    reason: str
    symbols_involved: list[str]
    approved: bool | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    expiry: datetime | None = None
    conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'requester': self.requester,
            'requester_zone': self.requester_zone.value,
            'target_zone': self.target_zone.value,
            'reason': self.reason,
            'symbols_involved': self.symbols_involved,
            'approved': self.approved,
            'approved_by': self.approved_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'conditions': self.conditions,
        }


@dataclass
class AccessAttempt:
    """Record of access attempt across zones (#C36)."""
    timestamp: datetime
    user: str
    user_zone: InformationZone
    target_zone: InformationZone
    resource: str
    allowed: bool
    reason: str


class ChineseWallManager:
    """
    Chinese wall enforcement (#C36).

    Implements information barriers between:
    - Public side (research, sales)
    - Private side (deal teams with MNPI)
    - Trading desk

    Features:
    - Zone-based access control
    - Wall crossing requests and approvals
    - Restricted list management
    - Access audit trail
    """

    # Allowed zone access without wall crossing
    ALLOWED_ACCESS = {
        InformationZone.PUBLIC: {InformationZone.PUBLIC},
        InformationZone.RESEARCH: {InformationZone.PUBLIC, InformationZone.RESEARCH},
        InformationZone.TRADING: {InformationZone.PUBLIC, InformationZone.RESEARCH, InformationZone.TRADING},
        InformationZone.DEAL: {InformationZone.DEAL},  # Isolated
        InformationZone.COMPLIANCE: {
            InformationZone.PUBLIC, InformationZone.RESEARCH,
            InformationZone.TRADING, InformationZone.DEAL, InformationZone.COMPLIANCE
        },  # Compliance can see all
    }

    def __init__(self):
        # User zone assignments
        self._user_zones: dict[str, InformationZone] = {}

        # Active wall crossing approvals
        self._active_crossings: dict[str, WallCrossingRequest] = {}

        # Access audit log
        self._access_log: list[AccessAttempt] = []

        # Restricted list (symbols with MNPI)
        self._restricted_list: dict[str, dict] = {}  # symbol -> restriction details

        # Request counter
        self._request_counter = 0

    def assign_zone(self, user: str, zone: InformationZone) -> None:
        """Assign user to information zone (#C36)."""
        self._user_zones[user] = zone
        logger.info(f"User {user} assigned to zone {zone.value}")

    def get_user_zone(self, user: str) -> InformationZone:
        """Get user's assigned zone."""
        return self._user_zones.get(user, InformationZone.PUBLIC)

    def add_to_restricted_list(
        self,
        symbol: str,
        reason: str,
        added_by: str,
        expiry: datetime | None = None,
    ) -> None:
        """Add symbol to restricted list (#C36)."""
        self._restricted_list[symbol] = {
            'symbol': symbol,
            'reason': reason,
            'added_by': added_by,
            'added_at': datetime.now(timezone.utc),
            'expiry': expiry,
        }
        logger.warning(f"Symbol {symbol} added to restricted list: {reason}")

    def remove_from_restricted_list(self, symbol: str, removed_by: str) -> None:
        """Remove symbol from restricted list."""
        if symbol in self._restricted_list:
            del self._restricted_list[symbol]
            logger.info(f"Symbol {symbol} removed from restricted list by {removed_by}")

    def is_restricted(self, symbol: str) -> bool:
        """Check if symbol is on restricted list."""
        if symbol not in self._restricted_list:
            return False

        restriction = self._restricted_list[symbol]
        if restriction.get('expiry') and datetime.now(timezone.utc) > restriction['expiry']:
            # Expired
            del self._restricted_list[symbol]
            return False

        return True

    def check_access(
        self,
        user: str,
        target_zone: InformationZone,
        resource: str,
    ) -> tuple[bool, str]:
        """
        Check if user can access resource in target zone (#C36).

        Args:
            user: User requesting access
            target_zone: Zone of the resource
            resource: Resource being accessed

        Returns:
            (allowed, reason)
        """
        user_zone = self.get_user_zone(user)

        # Check base zone access
        allowed_zones = self.ALLOWED_ACCESS.get(user_zone, set())
        base_allowed = target_zone in allowed_zones

        # Check for active wall crossing
        crossing_allowed = False
        crossing_key = f"{user}:{target_zone.value}"
        if crossing_key in self._active_crossings:
            crossing = self._active_crossings[crossing_key]
            if crossing.approved and (crossing.expiry is None or datetime.now(timezone.utc) < crossing.expiry):
                crossing_allowed = True

        allowed = base_allowed or crossing_allowed
        reason = "zone_access" if base_allowed else ("wall_crossing_approved" if crossing_allowed else "access_denied")

        # Log attempt
        self._access_log.append(AccessAttempt(
            timestamp=datetime.now(timezone.utc),
            user=user,
            user_zone=user_zone,
            target_zone=target_zone,
            resource=resource,
            allowed=allowed,
            reason=reason,
        ))

        # Trim log
        self._access_log = self._access_log[-10000:]

        return allowed, reason

    def request_wall_crossing(
        self,
        requester: str,
        target_zone: InformationZone,
        reason: str,
        symbols_involved: list[str] | None = None,
    ) -> WallCrossingRequest:
        """
        Request wall crossing approval (#C36).

        Args:
            requester: User requesting crossing
            target_zone: Zone to access
            reason: Business justification
            symbols_involved: Symbols that may be discussed

        Returns:
            Wall crossing request
        """
        self._request_counter += 1
        request_id = f"WCR-{self._request_counter:06d}"

        request = WallCrossingRequest(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            requester=requester,
            requester_zone=self.get_user_zone(requester),
            target_zone=target_zone,
            reason=reason,
            symbols_involved=symbols_involved or [],
        )

        logger.info(f"Wall crossing request {request_id}: {requester} -> {target_zone.value}")
        return request

    def approve_wall_crossing(
        self,
        request: WallCrossingRequest,
        approver: str,
        duration_hours: int = 24,
        conditions: list[str] | None = None,
    ) -> WallCrossingRequest:
        """
        Approve wall crossing request (#C36).

        Only compliance can approve.
        """
        approver_zone = self.get_user_zone(approver)
        if approver_zone != InformationZone.COMPLIANCE:
            raise PermissionError("Only compliance can approve wall crossings")

        request.approved = True
        request.approved_by = approver
        request.approved_at = datetime.now(timezone.utc)
        request.expiry = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
        request.conditions = conditions or []

        # Store active crossing
        crossing_key = f"{request.requester}:{request.target_zone.value}"
        self._active_crossings[crossing_key] = request

        logger.info(f"Wall crossing {request.request_id} approved by {approver}")
        return request

    def deny_wall_crossing(
        self,
        request: WallCrossingRequest,
        approver: str,
        reason: str,
    ) -> WallCrossingRequest:
        """Deny wall crossing request (#C36)."""
        request.approved = False
        request.approved_by = approver
        request.approved_at = datetime.now(timezone.utc)
        request.conditions = [f"DENIED: {reason}"]

        logger.info(f"Wall crossing {request.request_id} denied: {reason}")
        return request

    def get_restricted_list(self) -> list[dict]:
        """Get current restricted list."""
        now = datetime.now(timezone.utc)
        active = []
        for symbol, details in list(self._restricted_list.items()):
            if details.get('expiry') and now > details['expiry']:
                del self._restricted_list[symbol]
            else:
                active.append(details)
        return active

    def get_access_audit(
        self,
        user: str | None = None,
        denied_only: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """Get access audit log."""
        log = self._access_log

        if user:
            log = [a for a in log if a.user == user]
        if denied_only:
            log = [a for a in log if not a.allowed]

        return [
            {
                'timestamp': a.timestamp.isoformat(),
                'user': a.user,
                'user_zone': a.user_zone.value,
                'target_zone': a.target_zone.value,
                'resource': a.resource,
                'allowed': a.allowed,
                'reason': a.reason,
            }
            for a in log[-limit:]
        ]


# =============================================================================
# RESEARCH DISTRIBUTION CONTROLS (#C37)
# =============================================================================

class ResearchType(str, Enum):
    """Types of research for distribution control (#C37)."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    MACRO = "macro"
    QUANT = "quant"
    INTERNAL_ONLY = "internal_only"


class DistributionRestriction(str, Enum):
    """Distribution restriction levels (#C37)."""
    PUBLIC = "public"  # Can be distributed externally
    CLIENT_ONLY = "client_only"  # Clients only
    INTERNAL = "internal"  # Internal use only
    RESTRICTED = "restricted"  # Named recipients only


@dataclass
class ResearchDocument:
    """Research document with distribution controls (#C37)."""
    document_id: str
    title: str
    research_type: ResearchType
    restriction: DistributionRestriction
    author: str
    created_at: datetime
    symbols_covered: list[str]
    rating_changes: list[dict]  # {symbol, old_rating, new_rating}
    price_targets: list[dict]  # {symbol, target, current}
    allowed_recipients: list[str] | None  # For RESTRICTED
    embargo_until: datetime | None
    distribution_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'document_id': self.document_id,
            'title': self.title,
            'research_type': self.research_type.value,
            'restriction': self.restriction.value,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'symbols_covered': self.symbols_covered,
            'rating_changes': self.rating_changes,
            'price_targets': self.price_targets,
            'allowed_recipients': self.allowed_recipients,
            'embargo_until': self.embargo_until.isoformat() if self.embargo_until else None,
        }


class ResearchDistributionControl:
    """
    Research distribution controls (#C37).

    Features:
    - Embargo management
    - Distribution restrictions
    - Fair disclosure compliance
    - Distribution audit trail
    """

    def __init__(self):
        self._documents: dict[str, ResearchDocument] = {}
        self._document_counter = 0

        # Pre-release access tracking
        self._pre_release_access: dict[str, list[str]] = {}  # doc_id -> users

    def register_document(
        self,
        title: str,
        research_type: ResearchType,
        restriction: DistributionRestriction,
        author: str,
        symbols_covered: list[str],
        rating_changes: list[dict] | None = None,
        price_targets: list[dict] | None = None,
        allowed_recipients: list[str] | None = None,
        embargo_hours: int = 0,
    ) -> ResearchDocument:
        """
        Register new research document (#C37).

        Returns document with distribution controls set.
        """
        self._document_counter += 1
        doc_id = f"RES-{self._document_counter:06d}"

        embargo_until = None
        if embargo_hours > 0:
            embargo_until = datetime.now(timezone.utc) + timedelta(hours=embargo_hours)

        doc = ResearchDocument(
            document_id=doc_id,
            title=title,
            research_type=research_type,
            restriction=restriction,
            author=author,
            created_at=datetime.now(timezone.utc),
            symbols_covered=symbols_covered,
            rating_changes=rating_changes or [],
            price_targets=price_targets or [],
            allowed_recipients=allowed_recipients,
            embargo_until=embargo_until,
        )

        self._documents[doc_id] = doc
        logger.info(f"Research document registered: {doc_id} - {title}")

        return doc

    def can_distribute(
        self,
        document_id: str,
        recipient: str,
        is_client: bool = False,
        is_internal: bool = False,
    ) -> tuple[bool, str]:
        """
        Check if document can be distributed to recipient (#C37).

        Args:
            document_id: Document to distribute
            recipient: Intended recipient
            is_client: Whether recipient is a client
            is_internal: Whether recipient is internal

        Returns:
            (can_distribute, reason)
        """
        doc = self._documents.get(document_id)
        if not doc:
            return False, "document_not_found"

        # Check embargo
        if doc.embargo_until and datetime.now(timezone.utc) < doc.embargo_until:
            return False, f"under_embargo_until_{doc.embargo_until.isoformat()}"

        # Check restriction level
        if doc.restriction == DistributionRestriction.RESTRICTED:
            if doc.allowed_recipients and recipient not in doc.allowed_recipients:
                return False, "not_in_allowed_recipients"
        elif doc.restriction == DistributionRestriction.INTERNAL:
            if not is_internal:
                return False, "internal_only"
        elif doc.restriction == DistributionRestriction.CLIENT_ONLY:
            if not (is_client or is_internal):
                return False, "client_only"

        return True, "allowed"

    def record_distribution(
        self,
        document_id: str,
        recipient: str,
        channel: str,
    ) -> bool:
        """Record document distribution (#C37)."""
        doc = self._documents.get(document_id)
        if not doc:
            return False

        doc.distribution_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'recipient': recipient,
            'channel': channel,
        })

        logger.info(f"Document {document_id} distributed to {recipient} via {channel}")
        return True

    def record_pre_release_access(
        self,
        document_id: str,
        user: str,
        reason: str,
    ) -> None:
        """Record pre-release access for fair disclosure tracking (#C37)."""
        if document_id not in self._pre_release_access:
            self._pre_release_access[document_id] = []

        self._pre_release_access[document_id].append({
            'user': user,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'reason': reason,
        })

        logger.warning(f"Pre-release access: {user} accessed {document_id}")

    def get_distribution_report(self, document_id: str) -> dict:
        """Get distribution report for document (#C37)."""
        doc = self._documents.get(document_id)
        if not doc:
            return {'error': 'document_not_found'}

        return {
            'document': doc.to_dict(),
            'distribution_count': len(doc.distribution_log),
            'distribution_log': doc.distribution_log,
            'pre_release_access': self._pre_release_access.get(document_id, []),
        }

    def get_documents_by_symbol(self, symbol: str) -> list[ResearchDocument]:
        """Get all research documents covering a symbol."""
        return [
            doc for doc in self._documents.values()
            if symbol in doc.symbols_covered
        ]


# =============================================================================
# CONFLICT OF INTEREST TRACKING (#C38)
# =============================================================================

class ConflictType(str, Enum):
    """Types of conflicts of interest (#C38)."""
    PERSONAL_HOLDING = "personal_holding"
    FAMILY_RELATIONSHIP = "family_relationship"
    OUTSIDE_INTEREST = "outside_interest"
    BOARD_POSITION = "board_position"
    PRIOR_EMPLOYMENT = "prior_employment"
    INVESTMENT_BANKING = "investment_banking"
    RESEARCH_RELATIONSHIP = "research_relationship"
    MATERIAL_RELATIONSHIP = "material_relationship"


class ConflictStatus(str, Enum):
    """Conflict status (#C38)."""
    ACTIVE = "active"
    MANAGED = "managed"
    CLEARED = "cleared"
    EXPIRED = "expired"


@dataclass
class ConflictOfInterest:
    """Conflict of interest record (#C38)."""
    conflict_id: str
    conflict_type: ConflictType
    status: ConflictStatus
    person: str
    description: str
    symbols_affected: list[str]
    reported_at: datetime
    reported_by: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    mitigation_measures: list[str] = field(default_factory=list)
    expiry: datetime | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            'conflict_id': self.conflict_id,
            'conflict_type': self.conflict_type.value,
            'status': self.status.value,
            'person': self.person,
            'description': self.description,
            'symbols_affected': self.symbols_affected,
            'reported_at': self.reported_at.isoformat(),
            'reported_by': self.reported_by,
            'reviewed_by': self.reviewed_by,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'mitigation_measures': self.mitigation_measures,
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'notes': self.notes,
        }


class ConflictOfInterestTracker:
    """
    Conflict of interest tracking system (#C38).

    Features:
    - Conflict registration and tracking
    - Mitigation measure management
    - Trading restriction enforcement
    - Periodic attestation tracking
    - Reporting and audit
    """

    def __init__(self):
        self._conflicts: dict[str, ConflictOfInterest] = {}
        self._conflict_counter = 0

        # Personal holdings by person
        self._personal_holdings: dict[str, dict[str, dict]] = {}  # person -> symbol -> details

        # Attestation tracking
        self._attestations: list[dict] = []

    def register_conflict(
        self,
        conflict_type: ConflictType,
        person: str,
        description: str,
        symbols_affected: list[str],
        reported_by: str,
        expiry: datetime | None = None,
    ) -> ConflictOfInterest:
        """
        Register new conflict of interest (#C38).

        Args:
            conflict_type: Type of conflict
            person: Person with conflict
            description: Conflict description
            symbols_affected: Affected symbols
            reported_by: Person reporting
            expiry: When conflict expires

        Returns:
            Registered conflict
        """
        self._conflict_counter += 1
        conflict_id = f"COI-{self._conflict_counter:06d}"

        conflict = ConflictOfInterest(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            status=ConflictStatus.ACTIVE,
            person=person,
            description=description,
            symbols_affected=symbols_affected,
            reported_at=datetime.now(timezone.utc),
            reported_by=reported_by,
            expiry=expiry,
        )

        self._conflicts[conflict_id] = conflict
        logger.warning(f"Conflict of interest registered: {conflict_id} - {person} - {conflict_type.value}")

        return conflict

    def review_conflict(
        self,
        conflict_id: str,
        reviewer: str,
        mitigation_measures: list[str],
        new_status: ConflictStatus,
        notes: str = "",
    ) -> ConflictOfInterest | None:
        """
        Review and manage conflict (#C38).

        Args:
            conflict_id: Conflict to review
            reviewer: Compliance reviewer
            mitigation_measures: Measures to manage conflict
            new_status: Updated status
            notes: Review notes

        Returns:
            Updated conflict or None
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            return None

        conflict.reviewed_by = reviewer
        conflict.reviewed_at = datetime.now(timezone.utc)
        conflict.mitigation_measures = mitigation_measures
        conflict.status = new_status
        conflict.notes = notes

        logger.info(f"Conflict {conflict_id} reviewed: {new_status.value}")
        return conflict

    def register_personal_holding(
        self,
        person: str,
        symbol: str,
        shares: int,
        acquisition_date: datetime,
        acquisition_method: str,
    ) -> None:
        """Register personal holding for conflict tracking (#C38)."""
        if person not in self._personal_holdings:
            self._personal_holdings[person] = {}

        self._personal_holdings[person][symbol] = {
            'shares': shares,
            'acquisition_date': acquisition_date.isoformat(),
            'acquisition_method': acquisition_method,
            'reported_at': datetime.now(timezone.utc).isoformat(),
        }

        # Auto-register as conflict if material
        if shares > 0:
            self.register_conflict(
                conflict_type=ConflictType.PERSONAL_HOLDING,
                person=person,
                description=f"Holds {shares} shares of {symbol}",
                symbols_affected=[symbol],
                reported_by=person,
            )

    def check_can_trade(
        self,
        person: str,
        symbol: str,
    ) -> tuple[bool, str, list[str]]:
        """
        Check if person can trade a symbol (#C38).

        Args:
            person: Person wanting to trade
            symbol: Symbol to trade

        Returns:
            (can_trade, reason, required_disclosures)
        """
        conflicts = self.get_conflicts_for_person(person)
        active_conflicts = [c for c in conflicts if c.status == ConflictStatus.ACTIVE]

        # Check if symbol is affected
        affected_conflicts = [c for c in active_conflicts if symbol in c.symbols_affected]

        if affected_conflicts:
            # Check if mitigated
            managed = [c for c in affected_conflicts if c.status == ConflictStatus.MANAGED]
            if len(managed) == len(affected_conflicts):
                return True, "managed_conflict", [c.conflict_id for c in managed]
            else:
                return False, "unmanaged_conflict", [c.conflict_id for c in affected_conflicts]

        return True, "no_conflict", []

    def get_conflicts_for_person(self, person: str) -> list[ConflictOfInterest]:
        """Get all conflicts for a person."""
        return [c for c in self._conflicts.values() if c.person == person]

    def get_conflicts_for_symbol(self, symbol: str) -> list[ConflictOfInterest]:
        """Get all conflicts affecting a symbol."""
        return [c for c in self._conflicts.values() if symbol in c.symbols_affected]

    def record_attestation(
        self,
        person: str,
        attestation_type: str,
        content: str,
    ) -> dict:
        """Record compliance attestation (#C38)."""
        attestation = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'person': person,
            'type': attestation_type,
            'content': content,
            'attestation_hash': hashlib.sha256(
                f"{person}:{attestation_type}:{content}".encode()
            ).hexdigest()[:16],
        }

        self._attestations.append(attestation)
        logger.info(f"Attestation recorded: {person} - {attestation_type}")

        return attestation

    def get_conflicts_report(self) -> dict:
        """Generate conflicts report (#C38)."""
        all_conflicts = list(self._conflicts.values())

        return {
            'report_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_conflicts': len(all_conflicts),
            'by_status': {
                status.value: sum(1 for c in all_conflicts if c.status == status)
                for status in ConflictStatus
            },
            'by_type': {
                ctype.value: sum(1 for c in all_conflicts if c.conflict_type == ctype)
                for ctype in ConflictType
            },
            'active_conflicts': [
                c.to_dict() for c in all_conflicts
                if c.status in [ConflictStatus.ACTIVE, ConflictStatus.MANAGED]
            ],
            'people_with_conflicts': list(set(c.person for c in all_conflicts)),
            'symbols_affected': list(set(
                s for c in all_conflicts for s in c.symbols_affected
            )),
        }

    def check_expired_conflicts(self) -> list[ConflictOfInterest]:
        """Check and update expired conflicts."""
        now = datetime.now(timezone.utc)
        expired = []

        for conflict in self._conflicts.values():
            if conflict.expiry and now > conflict.expiry and conflict.status != ConflictStatus.EXPIRED:
                conflict.status = ConflictStatus.EXPIRED
                expired.append(conflict)
                logger.info(f"Conflict {conflict.conflict_id} expired")

        return expired

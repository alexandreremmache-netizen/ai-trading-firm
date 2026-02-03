"""
Alerting System
===============

Comprehensive alerting system for trading operations.
Addresses P3 (LOW) monitoring issues:
- Alert deduplication
- Escalation rules
- Alert acknowledgment

Features:
- Deduplication by key/fingerprint
- Multi-level escalation policies
- Acknowledgment workflow
- Alert grouping and correlation
- Notification channel routing
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Any, Optional
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(str, Enum):
    """Alert lifecycle states."""
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    EXPIRED = "expired"


class EscalationAction(str, Enum):
    """Actions to take during escalation."""
    NOTIFY = "notify"
    PAGE = "page"
    ESCALATE = "escalate"
    AUTO_RESOLVE = "auto_resolve"


@dataclass
class AlertFingerprint:
    """
    Unique fingerprint for alert deduplication.

    Alerts with the same fingerprint are considered duplicates.
    """
    source: str
    category: str
    metric_name: str
    labels: frozenset = field(default_factory=frozenset)

    def __hash__(self) -> int:
        return hash((self.source, self.category, self.metric_name, self.labels))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlertFingerprint):
            return False
        return (
            self.source == other.source
            and self.category == other.category
            and self.metric_name == other.metric_name
            and self.labels == other.labels
        )

    def to_string(self) -> str:
        """Generate string representation for hashing."""
        parts = [self.source, self.category, self.metric_name]
        if self.labels:
            parts.extend(sorted(f"{k}={v}" for k, v in self.labels))
        return "|".join(parts)

    def compute_hash(self) -> str:
        """Compute SHA256 hash of fingerprint."""
        return hashlib.sha256(self.to_string().encode()).hexdigest()[:16]


@dataclass
class Alert:
    """
    Alert with full lifecycle support.
    """
    alert_id: str
    fingerprint: AlertFingerprint
    severity: AlertSeverity
    title: str
    message: str

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # State
    state: AlertState = AlertState.FIRING
    occurrence_count: int = 1

    # Values
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None

    # Acknowledgment
    acknowledged_by: Optional[str] = None
    acknowledgment_comment: Optional[str] = None

    # Escalation tracking
    escalation_level: int = 0
    last_escalation_at: Optional[datetime] = None
    notified_channels: list[str] = field(default_factory=list)

    # Metadata
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    def acknowledge(self, user: str, comment: Optional[str] = None) -> None:
        """Acknowledge the alert."""
        self.state = AlertState.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = user
        self.acknowledgment_comment = comment
        self.updated_at = datetime.now(timezone.utc)
        logger.info(f"Alert {self.alert_id} acknowledged by {user}")

    def resolve(self) -> None:
        """Resolve the alert."""
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        logger.info(f"Alert {self.alert_id} resolved")

    def silence(self, duration: timedelta) -> None:
        """Silence the alert for a duration."""
        self.state = AlertState.SILENCED
        self.updated_at = datetime.now(timezone.utc)
        # Note: silence expiry would be tracked separately
        logger.info(f"Alert {self.alert_id} silenced for {duration}")

    def increment_occurrence(self) -> None:
        """Increment occurrence count for duplicate alerts."""
        self.occurrence_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "fingerprint": self.fingerprint.compute_hash(),
            "severity": self.severity.value,
            "state": self.state.value,
            "title": self.title,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "occurrence_count": self.occurrence_count,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "acknowledged_by": self.acknowledged_by,
            "acknowledgment_comment": self.acknowledgment_comment,
            "escalation_level": self.escalation_level,
            "labels": self.labels,
            "annotations": self.annotations,
        }


@dataclass
class EscalationRule:
    """
    Rule for alert escalation.

    Defines when and how to escalate alerts.
    """
    name: str
    severity_filter: list[AlertSeverity]
    delay_minutes: int  # Wait time before escalation
    action: EscalationAction
    notification_channels: list[str] = field(default_factory=list)
    escalate_to_level: int = 0
    max_escalations: int = 3

    # Optional conditions
    category_filter: Optional[list[str]] = None
    source_filter: Optional[list[str]] = None

    def matches(self, alert: Alert) -> bool:
        """Check if rule matches an alert."""
        if alert.severity not in self.severity_filter:
            return False

        if self.category_filter:
            if alert.fingerprint.category not in self.category_filter:
                return False

        if self.source_filter:
            if alert.fingerprint.source not in self.source_filter:
                return False

        return True


@dataclass
class SilenceRule:
    """
    Rule for silencing alerts.
    """
    rule_id: str
    created_by: str
    created_at: datetime
    expires_at: datetime

    # Matchers
    fingerprint_match: Optional[AlertFingerprint] = None
    severity_match: Optional[list[AlertSeverity]] = None
    source_match: Optional[str] = None
    category_match: Optional[str] = None

    comment: str = ""

    def matches(self, alert: Alert) -> bool:
        """Check if alert should be silenced."""
        now = datetime.now(timezone.utc)
        if now > self.expires_at:
            return False

        if self.fingerprint_match and alert.fingerprint != self.fingerprint_match:
            return False

        if self.severity_match and alert.severity not in self.severity_match:
            return False

        if self.source_match and alert.fingerprint.source != self.source_match:
            return False

        if self.category_match and alert.fingerprint.category != self.category_match:
            return False

        return True


class AlertDeduplicator:
    """
    Handles alert deduplication.

    Groups alerts by fingerprint to avoid alert fatigue.
    """

    def __init__(self, dedup_window_minutes: int = 60):
        self._dedup_window = timedelta(minutes=dedup_window_minutes)
        self._active_alerts: dict[str, Alert] = {}  # fingerprint hash -> alert
        self._fingerprint_to_id: dict[str, str] = {}  # fingerprint hash -> alert_id

    def check_duplicate(self, fingerprint: AlertFingerprint) -> Optional[Alert]:
        """
        Check if an alert with this fingerprint is already active.

        Returns the existing alert if duplicate, None otherwise.
        """
        fp_hash = fingerprint.compute_hash()

        if fp_hash in self._active_alerts:
            existing = self._active_alerts[fp_hash]

            # Check if still within dedup window
            age = datetime.now(timezone.utc) - existing.created_at
            if age < self._dedup_window and existing.state == AlertState.FIRING:
                return existing

            # Alert has expired or resolved, allow new one
            if existing.state in (AlertState.RESOLVED, AlertState.EXPIRED):
                del self._active_alerts[fp_hash]
                if fp_hash in self._fingerprint_to_id:
                    del self._fingerprint_to_id[fp_hash]
                return None

        return None

    def register_alert(self, alert: Alert) -> None:
        """Register an alert for deduplication."""
        fp_hash = alert.fingerprint.compute_hash()
        self._active_alerts[fp_hash] = alert
        self._fingerprint_to_id[fp_hash] = alert.alert_id

    def remove_alert(self, fingerprint: AlertFingerprint) -> None:
        """Remove an alert from deduplication tracking."""
        fp_hash = fingerprint.compute_hash()
        if fp_hash in self._active_alerts:
            del self._active_alerts[fp_hash]
        if fp_hash in self._fingerprint_to_id:
            del self._fingerprint_to_id[fp_hash]

    def get_active_count(self) -> int:
        """Get count of active deduplicated alerts."""
        return len(self._active_alerts)

    def cleanup_expired(self) -> int:
        """Remove expired alerts from tracking."""
        now = datetime.now(timezone.utc)
        expired = []

        for fp_hash, alert in self._active_alerts.items():
            age = now - alert.created_at
            if age > self._dedup_window or alert.state in (AlertState.RESOLVED, AlertState.EXPIRED):
                expired.append(fp_hash)

        for fp_hash in expired:
            del self._active_alerts[fp_hash]
            if fp_hash in self._fingerprint_to_id:
                del self._fingerprint_to_id[fp_hash]

        return len(expired)


class AlertEscalator:
    """
    Handles alert escalation based on rules.
    """

    def __init__(self):
        self._rules: list[EscalationRule] = []
        self._pending_escalations: dict[str, datetime] = {}  # alert_id -> next_escalation_time
        self._notification_handlers: dict[str, Callable[[Alert], None]] = {}

    def add_rule(self, rule: EscalationRule) -> None:
        """Add an escalation rule."""
        self._rules.append(rule)
        logger.info(f"Escalation rule added: {rule.name}")

    def add_notification_handler(self, channel: str, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler for a channel."""
        self._notification_handlers[channel] = handler

    def schedule_escalation(self, alert: Alert) -> None:
        """Schedule escalation check for an alert."""
        for rule in self._rules:
            if rule.matches(alert):
                next_check = alert.created_at + timedelta(minutes=rule.delay_minutes)
                self._pending_escalations[alert.alert_id] = next_check
                break

    def check_escalations(self, alerts: dict[str, Alert]) -> list[Alert]:
        """
        Check all pending escalations and execute if due.

        Returns list of escalated alerts.
        """
        now = datetime.now(timezone.utc)
        escalated = []

        for alert_id, next_time in list(self._pending_escalations.items()):
            if now >= next_time:
                if alert_id in alerts:
                    alert = alerts[alert_id]

                    # Skip if already acknowledged or resolved
                    if alert.state in (AlertState.ACKNOWLEDGED, AlertState.RESOLVED):
                        del self._pending_escalations[alert_id]
                        continue

                    # Find matching rule
                    for rule in self._rules:
                        if rule.matches(alert):
                            self._execute_escalation(alert, rule)
                            escalated.append(alert)

                            # Schedule next escalation if under max
                            if alert.escalation_level < rule.max_escalations:
                                self._pending_escalations[alert_id] = now + timedelta(minutes=rule.delay_minutes)
                            else:
                                del self._pending_escalations[alert_id]
                            break
                else:
                    # Alert no longer exists
                    del self._pending_escalations[alert_id]

        return escalated

    def _execute_escalation(self, alert: Alert, rule: EscalationRule) -> None:
        """Execute an escalation action."""
        alert.escalation_level += 1
        alert.last_escalation_at = datetime.now(timezone.utc)

        logger.warning(
            f"Escalating alert {alert.alert_id} to level {alert.escalation_level}: "
            f"{alert.title}"
        )

        # Execute action
        if rule.action == EscalationAction.NOTIFY:
            self._send_notifications(alert, rule.notification_channels)
        elif rule.action == EscalationAction.PAGE:
            self._send_page(alert, rule.notification_channels)
        elif rule.action == EscalationAction.AUTO_RESOLVE:
            alert.resolve()

    def _send_notifications(self, alert: Alert, channels: list[str]) -> None:
        """Send notifications to specified channels."""
        for channel in channels:
            if channel in self._notification_handlers:
                try:
                    self._notification_handlers[channel](alert)
                    alert.notified_channels.append(channel)
                except Exception as e:
                    logger.error(f"Failed to notify channel {channel}: {e}")
            else:
                logger.warning(f"No handler for notification channel: {channel}")

    def _send_page(self, alert: Alert, channels: list[str]) -> None:
        """Send page (urgent notification) to specified channels."""
        # Pages are urgent notifications - would integrate with PagerDuty, OpsGenie, etc.
        logger.critical(f"PAGING for alert {alert.alert_id}: {alert.title}")
        self._send_notifications(alert, channels)


class AlertingSystem:
    """
    Complete alerting system with deduplication, escalation, and acknowledgment.
    """

    def __init__(
        self,
        dedup_window_minutes: int = 60,
        auto_resolve_after_minutes: int = 1440,  # 24 hours
    ):
        self._deduplicator = AlertDeduplicator(dedup_window_minutes)
        self._escalator = AlertEscalator()

        self._alerts: dict[str, Alert] = {}  # alert_id -> Alert
        self._silence_rules: list[SilenceRule] = []

        self._auto_resolve_after = timedelta(minutes=auto_resolve_after_minutes)
        self._alert_counter = 0

        # Statistics
        self._stats = {
            "total_alerts": 0,
            "deduplicated_alerts": 0,
            "escalated_alerts": 0,
            "acknowledged_alerts": 0,
            "resolved_alerts": 0,
            "silenced_alerts": 0,
        }

        # Default escalation rules
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default escalation rules."""
        # Critical alerts: escalate after 5 minutes
        self._escalator.add_rule(EscalationRule(
            name="critical_5min",
            severity_filter=[AlertSeverity.CRITICAL],
            delay_minutes=5,
            action=EscalationAction.NOTIFY,
            notification_channels=["slack", "email"],
            max_escalations=3,
        ))

        # Emergency alerts: page immediately
        self._escalator.add_rule(EscalationRule(
            name="emergency_immediate",
            severity_filter=[AlertSeverity.EMERGENCY],
            delay_minutes=1,
            action=EscalationAction.PAGE,
            notification_channels=["pagerduty", "sms"],
            max_escalations=5,
        ))

        # Warning alerts: escalate after 15 minutes
        self._escalator.add_rule(EscalationRule(
            name="warning_15min",
            severity_filter=[AlertSeverity.WARNING],
            delay_minutes=15,
            action=EscalationAction.NOTIFY,
            notification_channels=["slack"],
            max_escalations=2,
        ))

    def create_alert(
        self,
        severity: AlertSeverity,
        source: str,
        category: str,
        title: str,
        message: str,
        metric_name: str = "",
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> Optional[Alert]:
        """
        Create an alert with deduplication.

        Returns the alert if new, or updates existing if duplicate.
        """
        # Create fingerprint
        fingerprint = AlertFingerprint(
            source=source,
            category=category,
            metric_name=metric_name,
            labels=frozenset(labels.items()) if labels else frozenset(),
        )

        # Check for silence
        for rule in self._silence_rules:
            if rule.matches(Alert(
                alert_id="",
                fingerprint=fingerprint,
                severity=severity,
                title=title,
                message=message,
            )):
                self._stats["silenced_alerts"] += 1
                logger.debug(f"Alert silenced by rule {rule.rule_id}: {title}")
                return None

        # Check for duplicate
        existing = self._deduplicator.check_duplicate(fingerprint)
        if existing:
            existing.increment_occurrence()
            self._stats["deduplicated_alerts"] += 1
            logger.debug(
                f"Alert deduplicated (count={existing.occurrence_count}): {title}"
            )
            return existing

        # Create new alert
        self._alert_counter += 1
        alert_id = f"ALERT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"

        alert = Alert(
            alert_id=alert_id,
            fingerprint=fingerprint,
            severity=severity,
            title=title,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            labels=labels or {},
        )

        # Register for deduplication
        self._deduplicator.register_alert(alert)

        # Store alert
        self._alerts[alert_id] = alert

        # Schedule escalation
        self._escalator.schedule_escalation(alert)

        self._stats["total_alerts"] += 1

        # Log based on severity
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(severity, logger.info)

        log_func(f"ALERT [{severity.value}] {source}/{category}: {title}")

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        user: str,
        comment: Optional[str] = None,
    ) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self._alerts:
            logger.warning(f"Alert not found for acknowledgment: {alert_id}")
            return False

        alert = self._alerts[alert_id]
        alert.acknowledge(user, comment)
        self._stats["acknowledged_alerts"] += 1

        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self._alerts:
            logger.warning(f"Alert not found for resolution: {alert_id}")
            return False

        alert = self._alerts[alert_id]
        alert.resolve()
        self._deduplicator.remove_alert(alert.fingerprint)
        self._stats["resolved_alerts"] += 1

        return True

    def add_silence_rule(
        self,
        created_by: str,
        duration: timedelta,
        comment: str = "",
        fingerprint: Optional[AlertFingerprint] = None,
        severity: Optional[list[AlertSeverity]] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
    ) -> SilenceRule:
        """Add a silence rule."""
        now = datetime.now(timezone.utc)
        rule = SilenceRule(
            rule_id=f"SILENCE-{now.strftime('%Y%m%d%H%M%S')}-{len(self._silence_rules):03d}",
            created_by=created_by,
            created_at=now,
            expires_at=now + duration,
            fingerprint_match=fingerprint,
            severity_match=severity,
            source_match=source,
            category_match=category,
            comment=comment,
        )

        self._silence_rules.append(rule)
        logger.info(f"Silence rule added: {rule.rule_id} by {created_by}")

        return rule

    def remove_silence_rule(self, rule_id: str) -> bool:
        """Remove a silence rule."""
        for i, rule in enumerate(self._silence_rules):
            if rule.rule_id == rule_id:
                del self._silence_rules[i]
                logger.info(f"Silence rule removed: {rule_id}")
                return True
        return False

    def add_escalation_rule(self, rule: EscalationRule) -> None:
        """Add a custom escalation rule."""
        self._escalator.add_rule(rule)

    def add_notification_handler(self, channel: str, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler."""
        self._escalator.add_notification_handler(channel, handler)

    def process_escalations(self) -> list[Alert]:
        """Process pending escalations."""
        escalated = self._escalator.check_escalations(self._alerts)
        self._stats["escalated_alerts"] += len(escalated)
        return escalated

    def cleanup(self) -> dict[str, int]:
        """
        Cleanup old alerts and expired silence rules.

        Returns counts of cleaned up items.
        """
        now = datetime.now(timezone.utc)
        cleaned = {
            "expired_alerts": 0,
            "auto_resolved": 0,
            "expired_silences": 0,
        }

        # Cleanup deduplicator
        cleaned["expired_alerts"] = self._deduplicator.cleanup_expired()

        # Auto-resolve old alerts
        for alert_id, alert in list(self._alerts.items()):
            age = now - alert.created_at
            if age > self._auto_resolve_after and alert.state == AlertState.FIRING:
                alert.state = AlertState.EXPIRED
                alert.updated_at = now
                cleaned["auto_resolved"] += 1

        # Cleanup expired silence rules
        self._silence_rules = [
            rule for rule in self._silence_rules
            if rule.expires_at > now
        ]
        # Note: we'd need original count to track this properly

        return cleaned

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[Alert]:
        """Get active (non-resolved) alerts with optional filters."""
        alerts = [
            a for a in self._alerts.values()
            if a.state not in (AlertState.RESOLVED, AlertState.EXPIRED)
        ]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if source:
            alerts = [a for a in alerts if a.fingerprint.source == source]

        if category:
            alerts = [a for a in alerts if a.fingerprint.category == category]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_counts(self) -> dict[str, int]:
        """Get alert counts by state and severity."""
        counts = {
            "total": len(self._alerts),
            "firing": 0,
            "acknowledged": 0,
            "resolved": 0,
            "silenced": 0,
            "expired": 0,
            "by_severity": {s.value: 0 for s in AlertSeverity},
        }

        for alert in self._alerts.values():
            counts[alert.state.value] += 1
            if alert.state == AlertState.FIRING:
                counts["by_severity"][alert.severity.value] += 1

        return counts

    def get_statistics(self) -> dict[str, Any]:
        """Get alerting statistics."""
        return {
            **self._stats,
            "deduplication_rate": (
                self._stats["deduplicated_alerts"] / self._stats["total_alerts"]
                if self._stats["total_alerts"] > 0 else 0
            ),
            "acknowledgment_rate": (
                self._stats["acknowledged_alerts"] / self._stats["total_alerts"]
                if self._stats["total_alerts"] > 0 else 0
            ),
            "active_silences": len(self._silence_rules),
            "current_counts": self.get_alert_counts(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export alerting state as dictionary."""
        return {
            "alerts": [a.to_dict() for a in self._alerts.values()],
            "statistics": self.get_statistics(),
            "silence_rules": [
                {
                    "rule_id": r.rule_id,
                    "created_by": r.created_by,
                    "expires_at": r.expires_at.isoformat(),
                    "comment": r.comment,
                }
                for r in self._silence_rules
            ],
        }


# Default notification handlers (can be replaced with actual implementations)
def slack_handler(alert: Alert) -> None:
    """Send alert to Slack (placeholder)."""
    logger.info(f"[SLACK] {alert.severity.value.upper()}: {alert.title} - {alert.message}")


def email_handler(alert: Alert) -> None:
    """Send alert via email (placeholder)."""
    logger.info(f"[EMAIL] {alert.severity.value.upper()}: {alert.title}")


def pagerduty_handler(alert: Alert) -> None:
    """Send alert to PagerDuty (placeholder)."""
    logger.info(f"[PAGERDUTY] {alert.severity.value.upper()}: {alert.title}")


def sms_handler(alert: Alert) -> None:
    """Send alert via SMS (placeholder)."""
    logger.info(f"[SMS] {alert.severity.value.upper()}: {alert.title}")


def create_alerting_system(
    dedup_window_minutes: int = 60,
    auto_resolve_after_minutes: int = 1440,
) -> AlertingSystem:
    """
    Create a configured alerting system with default handlers.
    """
    system = AlertingSystem(
        dedup_window_minutes=dedup_window_minutes,
        auto_resolve_after_minutes=auto_resolve_after_minutes,
    )

    # Register default handlers
    system.add_notification_handler("slack", slack_handler)
    system.add_notification_handler("email", email_handler)
    system.add_notification_handler("pagerduty", pagerduty_handler)
    system.add_notification_handler("sms", sms_handler)

    return system

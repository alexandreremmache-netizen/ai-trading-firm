"""
Notification System Module
==========================

Notification system for compliance and risk alerts (Issues #C33, #R27).

Features:
- Multi-channel notifications (email, webhook, file)
- Alert severity levels
- Notification throttling
- Audit trail of all notifications
- Compliance officer escalation
"""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"  # Requires immediate action


class AlertCategory(str, Enum):
    """Alert categories."""
    RISK = "risk"
    COMPLIANCE = "compliance"
    EXECUTION = "execution"
    SYSTEM = "system"
    MARKET = "market"


@dataclass
class Alert:
    """Alert notification."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    source: str  # Agent or component that generated alert
    details: dict = field(default_factory=dict)
    requires_acknowledgment: bool = False
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'details': self.details,
            'requires_acknowledgment': self.requires_acknowledgment,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send notification. Returns True if successful."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if channel is available."""
        pass


class FileNotificationChannel(NotificationChannel):
    """
    File-based notification channel.

    Writes alerts to a file for log aggregation systems.
    """

    def __init__(self, filepath: str = "logs/alerts.jsonl"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def send(self, alert: Alert) -> bool:
        """Write alert to file."""
        try:
            with self._lock:
                with open(self.filepath, 'a') as f:
                    f.write(json.dumps(alert.to_dict()) + '\n')
            return True
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
            return False

    def is_available(self) -> bool:
        """File channel is always available."""
        return True


class WebhookNotificationChannel(NotificationChannel):
    """
    Webhook notification channel.

    Sends alerts to a webhook URL (e.g., Slack, Teams, PagerDuty).
    """

    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: float = 10.0,
        headers: dict | None = None,
    ):
        self.webhook_url = webhook_url
        self.timeout = timeout_seconds
        self.headers = headers or {'Content-Type': 'application/json'}

    def send(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        try:
            payload = self._format_payload(alert)
            data = json.dumps(payload).encode('utf-8')

            request = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=self.headers,
                method='POST',
            )

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status == 200

        except urllib.error.URLError as e:
            logger.error(f"Webhook notification failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Webhook notification error: {e}")
            return False

    def _format_payload(self, alert: Alert) -> dict:
        """Format alert for webhook (Slack-compatible)."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔴",
        }

        return {
            'text': f"{severity_emoji.get(alert.severity, '')} *{alert.title}*",
            'attachments': [{
                'color': self._get_color(alert.severity),
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                    {'title': 'Category', 'value': alert.category.value, 'short': True},
                    {'title': 'Source', 'value': alert.source, 'short': True},
                    {'title': 'Time', 'value': alert.timestamp.isoformat(), 'short': True},
                    {'title': 'Message', 'value': alert.message, 'short': False},
                ],
            }],
        }

    def _get_color(self, severity: AlertSeverity) -> str:
        """Get color for severity."""
        colors = {
            AlertSeverity.INFO: '#36a64f',
            AlertSeverity.WARNING: '#ffa500',
            AlertSeverity.CRITICAL: '#ff0000',
            AlertSeverity.EMERGENCY: '#8b0000',
        }
        return colors.get(severity, '#808080')

    def is_available(self) -> bool:
        """Check if webhook is reachable."""
        return bool(self.webhook_url)


class EmailNotificationChannel(NotificationChannel):
    """
    Email notification channel.

    Sends alerts via SMTP email.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
        from_address: str = "alerts@trading.local",
        to_addresses: list[str] | None = None,
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses or []
        self.use_tls = use_tls

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.to_addresses:
            logger.warning("No email recipients configured")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)

            # Plain text version
            text_content = f"""
Alert: {alert.title}
Severity: {alert.severity.value}
Category: {alert.category.value}
Time: {alert.timestamp.isoformat()}
Source: {alert.source}

{alert.message}

Details:
{json.dumps(alert.details, indent=2)}
"""
            msg.attach(MIMEText(text_content, 'plain'))

            # HTML version
            html_content = self._format_html(alert)
            msg.attach(MIMEText(html_content, 'html'))

            # Send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_address, self.to_addresses, msg.as_string())

            return True

        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

    def _format_html(self, alert: Alert) -> str:
        """Format alert as HTML email."""
        color = {
            AlertSeverity.INFO: '#36a64f',
            AlertSeverity.WARNING: '#ffa500',
            AlertSeverity.CRITICAL: '#ff0000',
            AlertSeverity.EMERGENCY: '#8b0000',
        }.get(alert.severity, '#808080')

        return f"""
<html>
<body style="font-family: Arial, sans-serif;">
<div style="border-left: 4px solid {color}; padding-left: 15px; margin: 10px 0;">
<h2 style="color: {color};">{alert.title}</h2>
<p><strong>Severity:</strong> {alert.severity.value}</p>
<p><strong>Category:</strong> {alert.category.value}</p>
<p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
<p><strong>Source:</strong> {alert.source}</p>
<hr>
<p>{alert.message}</p>
</div>
</body>
</html>
"""

    def is_available(self) -> bool:
        """Check if SMTP is configured."""
        return bool(self.smtp_host and self.to_addresses)


class NotificationManager:
    """
    Central notification management (#C33, #R27).

    Handles alert routing, throttling, and escalation.
    """

    def __init__(
        self,
        channels: list[NotificationChannel] | None = None,
        throttle_minutes: float = 5.0,
        escalation_delay_minutes: float = 15.0,
    ):
        self.channels = channels or [FileNotificationChannel()]
        self.throttle_period = timedelta(minutes=throttle_minutes)
        self.escalation_delay = timedelta(minutes=escalation_delay_minutes)

        # Alert tracking
        self._alerts: dict[str, Alert] = {}
        self._alert_counter = 0
        self._last_alert_times: dict[str, datetime] = {}  # key -> last alert time
        self._pending_acknowledgments: dict[str, Alert] = {}

        # Escalation tracking
        self._escalation_callbacks: list[Callable[[Alert], None]] = []

        # Lock for thread safety
        self._lock = threading.Lock()

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels.append(channel)

    def register_escalation_callback(
        self,
        callback: Callable[[Alert], None],
    ) -> None:
        """Register callback for escalation."""
        self._escalation_callbacks.append(callback)

    def send_alert(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        message: str,
        source: str,
        details: dict | None = None,
        throttle_key: str | None = None,
        requires_acknowledgment: bool = False,
    ) -> Alert | None:
        """
        Send an alert through configured channels.

        Args:
            severity: Alert severity
            category: Alert category
            title: Alert title
            message: Alert message
            source: Source agent/component
            details: Additional details
            throttle_key: Key for throttling similar alerts
            requires_acknowledgment: Whether alert needs to be acknowledged

        Returns:
            Alert object if sent, None if throttled
        """
        with self._lock:
            # Check throttling
            if throttle_key:
                if not self._should_send(throttle_key, severity):
                    logger.debug(f"Alert throttled: {throttle_key}")
                    return None

            # Create alert
            self._alert_counter += 1
            now = datetime.now(timezone.utc)

            alert = Alert(
                alert_id=f"ALERT_{now.strftime('%Y%m%d_%H%M%S')}_{self._alert_counter}",
                timestamp=now,
                severity=severity,
                category=category,
                title=title,
                message=message,
                source=source,
                details=details or {},
                requires_acknowledgment=requires_acknowledgment,
            )

            # Store alert
            self._alerts[alert.alert_id] = alert

            if throttle_key:
                self._last_alert_times[throttle_key] = now

            if requires_acknowledgment:
                self._pending_acknowledgments[alert.alert_id] = alert

        # Send through channels
        success_count = 0
        for channel in self.channels:
            if channel.is_available():
                if self._should_send_to_channel(channel, alert):
                    if channel.send(alert):
                        success_count += 1

        logger.log(
            self._get_log_level(severity),
            f"Alert [{severity.value}] {title}: {message} "
            f"(sent to {success_count}/{len(self.channels)} channels)"
        )

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert to acknowledge
            acknowledged_by: Person/system acknowledging

        Returns:
            True if acknowledged successfully
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False

            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)

            if alert_id in self._pending_acknowledgments:
                del self._pending_acknowledgments[alert_id]

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

    def get_pending_acknowledgments(self) -> list[Alert]:
        """Get alerts pending acknowledgment."""
        with self._lock:
            return list(self._pending_acknowledgments.values())

    def check_escalations(self) -> list[Alert]:
        """
        Check for alerts requiring escalation.

        Escalates alerts that have been pending too long.
        """
        escalated = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for alert in self._pending_acknowledgments.values():
                if now - alert.timestamp > self.escalation_delay:
                    escalated.append(alert)

        # Trigger escalation callbacks
        for alert in escalated:
            for callback in self._escalation_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Escalation callback failed: {e}")

        return escalated

    def _should_send(self, throttle_key: str, severity: AlertSeverity) -> bool:
        """Check if alert should be sent based on throttling."""
        # Emergency alerts are never throttled
        if severity == AlertSeverity.EMERGENCY:
            return True

        last_time = self._last_alert_times.get(throttle_key)
        if last_time is None:
            return True

        now = datetime.now(timezone.utc)
        return now - last_time >= self.throttle_period

    def _should_send_to_channel(
        self,
        channel: NotificationChannel,
        alert: Alert,
    ) -> bool:
        """Check if alert should be sent to specific channel."""
        # File channel receives all alerts
        if isinstance(channel, FileNotificationChannel):
            return True

        # Webhook receives WARNING and above
        if isinstance(channel, WebhookNotificationChannel):
            return alert.severity in (
                AlertSeverity.WARNING,
                AlertSeverity.CRITICAL,
                AlertSeverity.EMERGENCY,
            )

        # Email receives CRITICAL and above
        if isinstance(channel, EmailNotificationChannel):
            return alert.severity in (
                AlertSeverity.CRITICAL,
                AlertSeverity.EMERGENCY,
            )

        return True

    def _get_log_level(self, severity: AlertSeverity) -> int:
        """Get logging level for severity."""
        levels = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL,
        }
        return levels.get(severity, logging.INFO)

    def get_statistics(self) -> dict:
        """Get notification statistics."""
        with self._lock:
            all_alerts = list(self._alerts.values())

        by_severity = defaultdict(int)
        by_category = defaultdict(int)
        acknowledged = 0

        for alert in all_alerts:
            by_severity[alert.severity.value] += 1
            by_category[alert.category.value] += 1
            if alert.acknowledged:
                acknowledged += 1

        return {
            'total_alerts': len(all_alerts),
            'pending_acknowledgment': len(self._pending_acknowledgments),
            'acknowledged': acknowledged,
            'by_severity': dict(by_severity),
            'by_category': dict(by_category),
            'channels': len(self.channels),
        }


class ComplianceOfficerNotifier:
    """
    Specialized notifier for compliance officer alerts (#C33).

    Handles compliance-specific notification requirements.
    """

    def __init__(
        self,
        notification_manager: NotificationManager,
        compliance_officer_email: str | None = None,
        compliance_webhook: str | None = None,
    ):
        self.manager = notification_manager
        self.compliance_email = compliance_officer_email
        self.compliance_webhook = compliance_webhook

    def notify_violation(
        self,
        violation_type: str,
        description: str,
        symbol: str | None = None,
        trade_id: str | None = None,
        details: dict | None = None,
    ) -> Alert | None:
        """Notify compliance officer of a violation."""
        full_details = {
            'violation_type': violation_type,
            'symbol': symbol,
            'trade_id': trade_id,
            **(details or {}),
        }

        return self.manager.send_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.COMPLIANCE,
            title=f"Compliance Violation: {violation_type}",
            message=description,
            source="ComplianceAgent",
            details=full_details,
            throttle_key=f"compliance_{violation_type}",
            requires_acknowledgment=True,
        )

    def notify_suspicious_activity(
        self,
        activity_type: str,
        description: str,
        details: dict | None = None,
    ) -> Alert | None:
        """Notify of suspicious trading activity (MAR requirement)."""
        return self.manager.send_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.COMPLIANCE,
            title=f"Suspicious Activity: {activity_type}",
            message=description,
            source="SurveillanceAgent",
            details=details or {},
            throttle_key=f"suspicious_{activity_type}",
            requires_acknowledgment=True,
        )

    def notify_regulatory_deadline(
        self,
        deadline_type: str,
        deadline_date: datetime,
        days_remaining: int,
    ) -> Alert | None:
        """Notify of upcoming regulatory deadline."""
        severity = AlertSeverity.CRITICAL if days_remaining <= 1 else (
            AlertSeverity.WARNING if days_remaining <= 7 else AlertSeverity.INFO
        )

        return self.manager.send_alert(
            severity=severity,
            category=AlertCategory.COMPLIANCE,
            title=f"Regulatory Deadline: {deadline_type}",
            message=f"Due in {days_remaining} days ({deadline_date.date()})",
            source="ComplianceCalendar",
            details={
                'deadline_type': deadline_type,
                'deadline_date': deadline_date.isoformat(),
                'days_remaining': days_remaining,
            },
            throttle_key=f"deadline_{deadline_type}_{deadline_date.date()}",
        )


class RiskLimitBreachNotifier:
    """
    Specialized notifier for risk limit breaches (#R27).

    Handles risk-specific notification requirements.
    """

    def __init__(self, notification_manager: NotificationManager):
        self.manager = notification_manager

    def notify_limit_breach(
        self,
        limit_name: str,
        current_value: float,
        limit_value: float,
        symbol: str | None = None,
    ) -> Alert | None:
        """Notify of a risk limit breach."""
        utilization = (current_value / limit_value * 100) if limit_value > 0 else 100

        return self.manager.send_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            title=f"Risk Limit Breach: {limit_name}",
            message=f"{limit_name} at {utilization:.1f}% (current: {current_value:.2f}, limit: {limit_value:.2f})",
            source="RiskAgent",
            details={
                'limit_name': limit_name,
                'current_value': current_value,
                'limit_value': limit_value,
                'utilization_pct': utilization,
                'symbol': symbol,
            },
            throttle_key=f"risk_breach_{limit_name}",
            requires_acknowledgment=True,
        )

    def notify_limit_warning(
        self,
        limit_name: str,
        current_value: float,
        limit_value: float,
        warning_threshold: float = 0.8,
    ) -> Alert | None:
        """Notify of approaching limit (warning level)."""
        utilization = (current_value / limit_value * 100) if limit_value > 0 else 100

        return self.manager.send_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            title=f"Risk Limit Warning: {limit_name}",
            message=f"{limit_name} at {utilization:.1f}% of limit",
            source="RiskAgent",
            details={
                'limit_name': limit_name,
                'current_value': current_value,
                'limit_value': limit_value,
                'utilization_pct': utilization,
            },
            throttle_key=f"risk_warning_{limit_name}",
        )

    def notify_var_breach(
        self,
        var_type: str,
        var_value: float,
        var_limit: float,
    ) -> Alert | None:
        """Notify of VaR limit breach."""
        return self.manager.send_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            title=f"VaR Breach: {var_type}",
            message=f"{var_type} = {var_value*100:.2f}% exceeds limit of {var_limit*100:.2f}%",
            source="RiskAgent",
            details={
                'var_type': var_type,
                'var_value': var_value,
                'var_limit': var_limit,
            },
            throttle_key=f"var_breach_{var_type}",
            requires_acknowledgment=True,
        )

# notifications

**Path**: `C:\Users\Alexa\ai-trading-firm\core\notifications.py`

## Overview

Notification System Module
==========================

Notification system for compliance and risk alerts (Issues #C33, #R27).

Features:
- Multi-channel notifications (email, webhook, file)
- Alert severity levels
- Notification throttling
- Audit trail of all notifications
- Compliance officer escalation

## Classes

### AlertSeverity

**Inherits from**: str, Enum

Alert severity levels.

### AlertCategory

**Inherits from**: str, Enum

Alert categories.

### Alert

Alert notification.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

### NotificationChannel

**Inherits from**: ABC

Abstract base class for notification channels.

#### Methods

##### `def send(self, alert: Alert) -> bool`

Send notification. Returns True if successful.

##### `def is_available(self) -> bool`

Check if channel is available.

### FileNotificationChannel

**Inherits from**: NotificationChannel

File-based notification channel.

Writes alerts to a file for log aggregation systems.

#### Methods

##### `def __init__(self, filepath: str)`

##### `def send(self, alert: Alert) -> bool`

Write alert to file.

##### `def is_available(self) -> bool`

File channel is always available.

### WebhookNotificationChannel

**Inherits from**: NotificationChannel

Webhook notification channel.

Sends alerts to a webhook URL (e.g., Slack, Teams, PagerDuty).

#### Methods

##### `def __init__(self, webhook_url: str, timeout_seconds: float, headers: )`

##### `def send(self, alert: Alert) -> bool`

Send alert to webhook.

##### `def is_available(self) -> bool`

Check if webhook is reachable.

### EmailNotificationChannel

**Inherits from**: NotificationChannel

Email notification channel.

Sends alerts via SMTP email.

#### Methods

##### `def __init__(self, smtp_host: str, smtp_port: int, username: , password: , from_address: str, to_addresses: , use_tls: bool)`

##### `def send(self, alert: Alert) -> bool`

Send alert via email.

##### `def is_available(self) -> bool`

Check if SMTP is configured.

### NotificationManager

Central notification management (#C33, #R27).

Handles alert routing, throttling, and escalation.

#### Methods

##### `def __init__(self, channels: , throttle_minutes: float, escalation_delay_minutes: float)`

##### `def add_channel(self, channel: NotificationChannel) -> None`

Add a notification channel.

##### `def register_escalation_callback(self, callback: Callable[, None]) -> None`

Register callback for escalation.

##### `def send_alert(self, severity: AlertSeverity, category: AlertCategory, title: str, message: str, source: str, details: , throttle_key: , requires_acknowledgment: bool)`

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

##### `def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool`

Acknowledge an alert.

Args:
    alert_id: Alert to acknowledge
    acknowledged_by: Person/system acknowledging

Returns:
    True if acknowledged successfully

##### `def get_pending_acknowledgments(self) -> list[Alert]`

Get alerts pending acknowledgment.

##### `def check_escalations(self) -> list[Alert]`

Check for alerts requiring escalation.

Escalates alerts that have been pending too long.

##### `def get_statistics(self) -> dict`

Get notification statistics.

### ComplianceOfficerNotifier

Specialized notifier for compliance officer alerts (#C33).

Handles compliance-specific notification requirements.

#### Methods

##### `def __init__(self, notification_manager: NotificationManager, compliance_officer_email: , compliance_webhook: )`

##### `def notify_violation(self, violation_type: str, description: str, symbol: , trade_id: , details: )`

Notify compliance officer of a violation.

##### `def notify_suspicious_activity(self, activity_type: str, description: str, details: )`

Notify of suspicious trading activity (MAR requirement).

##### `def notify_regulatory_deadline(self, deadline_type: str, deadline_date: datetime, days_remaining: int)`

Notify of upcoming regulatory deadline.

### RiskLimitBreachNotifier

Specialized notifier for risk limit breaches (#R27).

Handles risk-specific notification requirements.

#### Methods

##### `def __init__(self, notification_manager: NotificationManager)`

##### `def notify_limit_breach(self, limit_name: str, current_value: float, limit_value: float, symbol: )`

Notify of a risk limit breach.

##### `def notify_limit_warning(self, limit_name: str, current_value: float, limit_value: float, warning_threshold: float)`

Notify of approaching limit (warning level).

##### `def notify_var_breach(self, var_type: str, var_value: float, var_limit: float)`

Notify of VaR limit breach.

"""
Real-Time Alerts and Notifications System
==========================================

Dashboard component for managing and displaying real-time alerts.

Features:
- AlertManager class for centralized alert management
- Alert dataclass with full lifecycle tracking
- AlertRule dataclass for configurable alert triggers
- Built-in rules for common trading system conditions
- WebSocket notification support
- Alert aggregation to prevent spam
- Thread-safe with asyncio locks
- WebSocket-ready export to dict

Built-in Rules:
- drawdown_warning: Alerts when drawdown approaches limit
- position_limit: Alerts when position size approaches limit
- leverage_warning: Alerts when leverage approaches limit
- compliance_violation: Alerts on compliance rule violations
- execution_error: Alerts on order execution failures
- agent_timeout: Alerts when agents stop responding

Usage:
    manager = AlertManager()

    # Create an alert
    alert = await manager.create_alert(
        severity=AlertSeverity.WARNING,
        category=AlertCategory.RISK,
        title="Position Limit Warning",
        message="AAPL position at 80% of limit",
        source_agent="RiskAgent",
    )

    # Acknowledge an alert
    await manager.acknowledge_alert(alert.alert_id, "admin")

    # Get active alerts for dashboard
    active = await manager.get_active_alerts()

    # Export for WebSocket streaming
    data = await manager.to_dict()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.event_bus import EventBus


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class AlertSeverity(str, Enum):
    """Alert severity levels matching core/notifications.py."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(str, Enum):
    """Alert categories for classification."""
    RISK = "risk"
    COMPLIANCE = "compliance"
    EXECUTION = "execution"
    SYSTEM = "system"
    MARKET = "market"
    PERFORMANCE = "performance"


class AlertState(str, Enum):
    """Alert lifecycle states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    SUPPRESSED = "suppressed"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Alert:
    """
    Alert notification with full lifecycle tracking.

    Captures all information needed for compliance audit trails
    and dashboard display.
    """
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    source_agent: str

    # State tracking
    state: AlertState = AlertState.ACTIVE
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    # Metadata
    details: dict[str, Any] = field(default_factory=dict)
    related_symbols: list[str] = field(default_factory=list)
    rule_id: str | None = None

    # Aggregation tracking
    occurrence_count: int = 1
    first_occurrence: datetime | None = None
    last_occurrence: datetime | None = None
    aggregation_key: str | None = None

    # Resolution
    resolved_at: datetime | None = None
    resolution_message: str | None = None

    def __post_init__(self):
        """Initialize derived fields."""
        if self.first_occurrence is None:
            self.first_occurrence = self.timestamp
        if self.last_occurrence is None:
            self.last_occurrence = self.timestamp

    def acknowledge(self, acknowledged_by: str) -> None:
        """Mark the alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now(timezone.utc)
        self.state = AlertState.ACKNOWLEDGED
        logger.info(f"Alert {self.alert_id} acknowledged by {acknowledged_by}")

    def resolve(self, resolution_message: str | None = None) -> None:
        """Mark the alert as resolved."""
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)
        self.resolution_message = resolution_message
        logger.info(f"Alert {self.alert_id} resolved: {resolution_message or 'No message'}")

    def increment_occurrence(self) -> None:
        """Increment occurrence count for aggregated alerts."""
        self.occurrence_count += 1
        self.last_occurrence = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "source_agent": self.source_agent,
            "state": self.state.value,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "details": self.details,
            "related_symbols": self.related_symbols,
            "rule_id": self.rule_id,
            "occurrence_count": self.occurrence_count,
            "first_occurrence": self.first_occurrence.isoformat() if self.first_occurrence else None,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None,
            "aggregation_key": self.aggregation_key,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_message": self.resolution_message,
        }


@dataclass
class AlertRule:
    """
    Configurable alert rule for automatic alert generation.

    Defines conditions that trigger alerts when monitored metrics
    cross specified thresholds.
    """
    rule_id: str
    name: str
    condition: str  # Description of the condition (e.g., "drawdown > threshold")
    threshold: float
    severity: AlertSeverity
    category: AlertCategory
    message_template: str  # Template with {placeholders} for values
    source_agent: str = "AlertManager"

    # Rule configuration
    enabled: bool = True
    cooldown_minutes: float = 5.0  # Minimum time between alerts from same rule
    requires_acknowledgment: bool = False

    # Optional filters
    symbols: list[str] | None = None  # Only applies to these symbols
    time_start: str | None = None  # Only active from this time (HH:MM)
    time_end: str | None = None  # Only active until this time (HH:MM)

    # Tracking
    last_triggered: datetime | None = None
    trigger_count: int = 0

    def can_trigger(self, now: datetime | None = None) -> bool:
        """
        Check if the rule can trigger based on cooldown and time restrictions.

        Args:
            now: Current time (defaults to now)

        Returns:
            True if rule can trigger, False if on cooldown or outside time window
        """
        if not self.enabled:
            return False

        if now is None:
            now = datetime.now(timezone.utc)

        # Check cooldown
        if self.last_triggered is not None:
            cooldown = timedelta(minutes=self.cooldown_minutes)
            if now - self.last_triggered < cooldown:
                return False

        # Check time window
        if self.time_start and self.time_end:
            current_time = now.strftime("%H:%M")
            if not (self.time_start <= current_time <= self.time_end):
                return False

        return True

    def format_message(self, **kwargs: Any) -> str:
        """
        Format the message template with provided values.

        Args:
            **kwargs: Values to substitute in the template

        Returns:
            Formatted message string
        """
        try:
            return self.message_template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template key in alert rule {self.rule_id}: {e}")
            return self.message_template

    def mark_triggered(self) -> None:
        """Mark the rule as triggered."""
        self.last_triggered = datetime.now(timezone.utc)
        self.trigger_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for configuration export."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "category": self.category.value,
            "message_template": self.message_template,
            "source_agent": self.source_agent,
            "enabled": self.enabled,
            "cooldown_minutes": self.cooldown_minutes,
            "requires_acknowledgment": self.requires_acknowledgment,
            "symbols": self.symbols,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
        }


# =============================================================================
# BUILT-IN RULES
# =============================================================================


def create_builtin_rules() -> list[AlertRule]:
    """
    Create the standard built-in alert rules.

    Returns:
        List of pre-configured AlertRule instances
    """
    return [
        # Drawdown warning
        AlertRule(
            rule_id="drawdown_warning",
            name="Drawdown Warning",
            condition="drawdown_pct >= threshold",
            threshold=7.0,  # 7% drawdown (max is 10%)
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            message_template="Portfolio drawdown at {current_value:.1f}% (limit: {limit_value:.1f}%)",
            source_agent="RiskAgent",
            cooldown_minutes=15.0,
            requires_acknowledgment=False,
        ),
        # Drawdown critical
        AlertRule(
            rule_id="drawdown_critical",
            name="Drawdown Critical",
            condition="drawdown_pct >= threshold",
            threshold=9.0,  # 9% drawdown (max is 10%)
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            message_template="CRITICAL: Portfolio drawdown at {current_value:.1f}% - approaching kill-switch threshold",
            source_agent="RiskAgent",
            cooldown_minutes=5.0,
            requires_acknowledgment=True,
        ),
        # Position limit warning
        AlertRule(
            rule_id="position_limit_warning",
            name="Position Limit Warning",
            condition="position_pct >= threshold",
            threshold=80.0,  # 80% of position limit
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            message_template="{symbol} position at {current_value:.1f}% of limit ({limit_value:.1f}%)",
            source_agent="RiskAgent",
            cooldown_minutes=10.0,
            requires_acknowledgment=False,
        ),
        # Position limit breach
        AlertRule(
            rule_id="position_limit_breach",
            name="Position Limit Breach",
            condition="position_pct >= threshold",
            threshold=100.0,  # 100% of position limit
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            message_template="{symbol} position EXCEEDS limit at {current_value:.1f}%",
            source_agent="RiskAgent",
            cooldown_minutes=1.0,
            requires_acknowledgment=True,
        ),
        # Leverage warning
        AlertRule(
            rule_id="leverage_warning",
            name="Leverage Warning",
            condition="leverage >= threshold",
            threshold=1.6,  # 1.6x leverage (max is 2.0x)
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            message_template="Portfolio leverage at {current_value:.2f}x (limit: {limit_value:.1f}x)",
            source_agent="RiskAgent",
            cooldown_minutes=10.0,
            requires_acknowledgment=False,
        ),
        # Leverage critical
        AlertRule(
            rule_id="leverage_critical",
            name="Leverage Critical",
            condition="leverage >= threshold",
            threshold=1.9,  # 1.9x leverage (max is 2.0x)
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            message_template="CRITICAL: Leverage at {current_value:.2f}x - approaching limit",
            source_agent="RiskAgent",
            cooldown_minutes=5.0,
            requires_acknowledgment=True,
        ),
        # Compliance violation
        AlertRule(
            rule_id="compliance_violation",
            name="Compliance Violation",
            condition="compliance_check_failed",
            threshold=1.0,  # Binary: any violation
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.COMPLIANCE,
            message_template="Compliance violation: {violation_type} - {description}",
            source_agent="ComplianceAgent",
            cooldown_minutes=0.0,  # No cooldown for compliance
            requires_acknowledgment=True,
        ),
        # Execution error
        AlertRule(
            rule_id="execution_error",
            name="Execution Error",
            condition="order_execution_failed",
            threshold=1.0,  # Binary: any error
            severity=AlertSeverity.WARNING,
            category=AlertCategory.EXECUTION,
            message_template="Order execution failed for {symbol}: {error_message}",
            source_agent="ExecutionAgent",
            cooldown_minutes=1.0,
            requires_acknowledgment=False,
        ),
        # Multiple execution errors
        AlertRule(
            rule_id="execution_errors_critical",
            name="Multiple Execution Errors",
            condition="execution_error_count >= threshold",
            threshold=3.0,  # 3 errors in window
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.EXECUTION,
            message_template="Multiple execution errors: {error_count} failures in last {window_minutes} minutes",
            source_agent="ExecutionAgent",
            cooldown_minutes=5.0,
            requires_acknowledgment=True,
        ),
        # Agent timeout
        AlertRule(
            rule_id="agent_timeout",
            name="Agent Timeout",
            condition="agent_last_response_seconds >= threshold",
            threshold=60.0,  # 60 seconds without response
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            message_template="Agent {agent_name} has not responded for {seconds:.0f} seconds",
            source_agent="SystemMonitor",
            cooldown_minutes=5.0,
            requires_acknowledgment=False,
        ),
        # Agent error
        AlertRule(
            rule_id="agent_error",
            name="Agent Error",
            condition="agent_error_occurred",
            threshold=1.0,  # Binary: any error
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            message_template="Agent {agent_name} error: {error_message}",
            source_agent="SystemMonitor",
            cooldown_minutes=2.0,
            requires_acknowledgment=False,
        ),
        # VaR breach
        AlertRule(
            rule_id="var_breach",
            name="VaR Limit Breach",
            condition="var_pct >= threshold",
            threshold=95.0,  # 95% of VaR limit
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            message_template="VaR at {current_value:.2f}% exceeds {var_type} limit of {limit_value:.2f}%",
            source_agent="RiskAgent",
            cooldown_minutes=10.0,
            requires_acknowledgment=True,
        ),
        # Daily loss warning
        AlertRule(
            rule_id="daily_loss_warning",
            name="Daily Loss Warning",
            condition="daily_loss_pct >= threshold",
            threshold=2.0,  # 2% daily loss (max is 3%)
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            message_template="Daily P&L at -{current_value:.2f}% (limit: -{limit_value:.1f}%)",
            source_agent="RiskAgent",
            cooldown_minutes=15.0,
            requires_acknowledgment=False,
        ),
        # Kill switch activated
        AlertRule(
            rule_id="kill_switch_activated",
            name="Kill Switch Activated",
            condition="kill_switch_triggered",
            threshold=1.0,  # Binary
            severity=AlertSeverity.EMERGENCY,
            category=AlertCategory.RISK,
            message_template="KILL SWITCH ACTIVATED: {reason}",
            source_agent="RiskAgent",
            cooldown_minutes=0.0,  # No cooldown for kill switch
            requires_acknowledgment=True,
        ),
        # Market data stale
        AlertRule(
            rule_id="market_data_stale",
            name="Market Data Stale",
            condition="market_data_age_seconds >= threshold",
            threshold=30.0,  # 30 seconds stale
            severity=AlertSeverity.WARNING,
            category=AlertCategory.MARKET,
            message_template="Market data for {symbol} is stale ({age_seconds:.0f}s old)",
            source_agent="MarketDataManager",
            cooldown_minutes=5.0,
            requires_acknowledgment=False,
        ),
        # Broker disconnected
        AlertRule(
            rule_id="broker_disconnected",
            name="Broker Disconnected",
            condition="broker_connection_lost",
            threshold=1.0,  # Binary
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            message_template="Broker connection lost: {reason}",
            source_agent="BrokerManager",
            cooldown_minutes=0.0,  # No cooldown for connection issues
            requires_acknowledgment=True,
        ),
    ]


# =============================================================================
# ALERT MANAGER
# =============================================================================


class AlertManager:
    """
    Central alert management system for the trading dashboard.

    Provides comprehensive alert lifecycle management including:
    - Alert creation with automatic severity-based handling
    - Alert aggregation to prevent notification spam
    - WebSocket notification support for real-time updates
    - Alert history with configurable retention
    - Built-in rules for common trading system conditions

    Thread Safety:
        All public methods acquire an asyncio lock before modifying state,
        making the manager safe for concurrent use.

    Usage:
        manager = AlertManager()

        # Create alerts programmatically
        alert = await manager.create_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            title="Position Limit Warning",
            message="AAPL position at 80% of limit",
            source_agent="RiskAgent",
        )

        # Check rules against current metrics
        triggered = await manager.check_rules(metrics={
            "drawdown_pct": 7.5,
            "leverage": 1.7,
        })

        # Get alerts for dashboard
        active = await manager.get_active_alerts()

        # Export for WebSocket
        data = await manager.to_dict()
    """

    # Aggregation window in seconds
    AGGREGATION_WINDOW_SECONDS = 60.0

    # Default alert retention in hours
    ALERT_RETENTION_HOURS = 168  # 7 days

    # Maximum alerts to keep in history
    MAX_HISTORY_SIZE = 10000

    def __init__(
        self,
        aggregation_window_seconds: float = AGGREGATION_WINDOW_SECONDS,
        retention_hours: float = ALERT_RETENTION_HOURS,
        max_history_size: int = MAX_HISTORY_SIZE,
        enable_builtin_rules: bool = True,
        websocket_callback: Callable[[Alert], None] | None = None,
        audit_log_path: str | None = None,
    ):
        """
        Initialize the AlertManager.

        Args:
            aggregation_window_seconds: Window for aggregating similar alerts
            retention_hours: Hours to retain alerts in history
            max_history_size: Maximum number of alerts to keep in history
            enable_builtin_rules: Whether to enable built-in alert rules
            websocket_callback: Callback for WebSocket notifications
            audit_log_path: Path for audit log file (optional)
        """
        self._aggregation_window = timedelta(seconds=aggregation_window_seconds)
        self._retention_period = timedelta(hours=retention_hours)
        self._max_history_size = max_history_size
        self._websocket_callback = websocket_callback

        # Alert storage
        self._alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._alert_counter = 0

        # Rules storage
        self._rules: dict[str, AlertRule] = {}

        # Aggregation tracking
        self._aggregation_map: dict[str, str] = {}  # aggregation_key -> alert_id

        # Statistics
        self._stats = {
            "total_alerts": 0,
            "alerts_by_severity": defaultdict(int),
            "alerts_by_category": defaultdict(int),
            "aggregated_count": 0,
            "acknowledged_count": 0,
            "resolved_count": 0,
            "rule_triggers": defaultdict(int),
        }

        # Audit logging
        self._audit_log_path = Path(audit_log_path) if audit_log_path else None
        if self._audit_log_path:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = asyncio.Lock()

        # Initialize built-in rules
        if enable_builtin_rules:
            for rule in create_builtin_rules():
                self._rules[rule.rule_id] = rule

        logger.info(
            f"AlertManager initialized with {len(self._rules)} rules, "
            f"aggregation_window={aggregation_window_seconds}s"
        )

    def _generate_alert_id(self) -> str:
        """Generate a unique alert ID."""
        self._alert_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"ALERT_{timestamp}_{self._alert_counter:06d}"

    def _generate_aggregation_key(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        source_agent: str,
        rule_id: str | None = None,
        symbols: list[str] | None = None,
    ) -> str:
        """
        Generate a key for alert aggregation.

        Alerts with the same aggregation key within the aggregation window
        will be grouped together.
        """
        key_parts = [
            severity.value,
            category.value,
            source_agent,
            rule_id or "manual",
        ]
        if symbols:
            key_parts.extend(sorted(symbols))

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _should_aggregate(self, aggregation_key: str) -> tuple[bool, Alert | None]:
        """
        Check if a new alert should be aggregated with an existing one.

        Args:
            aggregation_key: The aggregation key for the new alert

        Returns:
            Tuple of (should_aggregate, existing_alert)
        """
        if aggregation_key not in self._aggregation_map:
            return False, None

        alert_id = self._aggregation_map[aggregation_key]
        if alert_id not in self._alerts:
            # Alert was removed, clean up mapping
            del self._aggregation_map[aggregation_key]
            return False, None

        existing = self._alerts[alert_id]

        # Check if still within aggregation window
        if existing.last_occurrence is None:
            return False, None

        now = datetime.now(timezone.utc)
        if now - existing.last_occurrence > self._aggregation_window:
            # Outside aggregation window, allow new alert
            del self._aggregation_map[aggregation_key]
            return False, None

        # Only aggregate if still active
        if existing.state != AlertState.ACTIVE:
            del self._aggregation_map[aggregation_key]
            return False, None

        return True, existing

    def _write_audit_log(self, alert: Alert, action: str) -> None:
        """Write alert action to audit log."""
        if self._audit_log_path is None:
            return

        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "category": alert.category.value,
                "title": alert.title,
                "source_agent": alert.source_agent,
            }

            with open(self._audit_log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.exception(f"Failed to write audit log: {e}")

    async def create_alert(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        message: str,
        source_agent: str,
        details: dict[str, Any] | None = None,
        related_symbols: list[str] | None = None,
        rule_id: str | None = None,
    ) -> Alert:
        """
        Create a new alert.

        If a similar alert exists within the aggregation window,
        it will be aggregated instead of creating a new one.

        Args:
            severity: Alert severity level
            category: Alert category
            title: Alert title
            message: Alert message
            source_agent: Name of the agent creating the alert
            details: Additional details (optional)
            related_symbols: Related trading symbols (optional)
            rule_id: ID of the rule that triggered this alert (optional)

        Returns:
            The created or aggregated Alert
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            # Generate aggregation key
            aggregation_key = self._generate_aggregation_key(
                severity=severity,
                category=category,
                source_agent=source_agent,
                rule_id=rule_id,
                symbols=related_symbols,
            )

            # Check for aggregation
            should_aggregate, existing = self._should_aggregate(aggregation_key)
            if should_aggregate and existing:
                existing.increment_occurrence()
                self._stats["aggregated_count"] += 1
                logger.debug(
                    f"Alert aggregated (count={existing.occurrence_count}): {title}"
                )
                return existing

            # Create new alert
            alert = Alert(
                alert_id=self._generate_alert_id(),
                timestamp=now,
                severity=severity,
                category=category,
                title=title,
                message=message,
                source_agent=source_agent,
                details=details or {},
                related_symbols=related_symbols or [],
                rule_id=rule_id,
                aggregation_key=aggregation_key,
            )

            # Store alert
            self._alerts[alert.alert_id] = alert
            self._aggregation_map[aggregation_key] = alert.alert_id

            # Update statistics
            self._stats["total_alerts"] += 1
            self._stats["alerts_by_severity"][severity.value] += 1
            self._stats["alerts_by_category"][category.value] += 1

            # Write audit log
            self._write_audit_log(alert, "created")

            # Trigger WebSocket notification
            if self._websocket_callback:
                try:
                    self._websocket_callback(alert)
                except Exception as e:
                    logger.exception(f"WebSocket callback failed: {e}")

            # Log based on severity
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.ERROR,
                AlertSeverity.EMERGENCY: logging.CRITICAL,
            }.get(severity, logging.INFO)

            logger.log(
                log_level,
                f"ALERT [{severity.value.upper()}] {source_agent}/{category.value}: {title}"
            )

            return alert

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Name of the person/system acknowledging

        Returns:
            True if acknowledged successfully, False if not found
        """
        async with self._lock:
            if alert_id not in self._alerts:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False

            alert = self._alerts[alert_id]

            if alert.state != AlertState.ACTIVE:
                logger.warning(
                    f"Cannot acknowledge alert {alert_id} in state {alert.state.value}"
                )
                return False

            alert.acknowledge(acknowledged_by)
            self._stats["acknowledged_count"] += 1
            self._write_audit_log(alert, "acknowledged")

            return True

    async def resolve_alert(
        self,
        alert_id: str,
        resolution_message: str | None = None,
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: ID of the alert to resolve
            resolution_message: Optional resolution message

        Returns:
            True if resolved successfully, False if not found
        """
        async with self._lock:
            if alert_id not in self._alerts:
                logger.warning(f"Alert not found for resolution: {alert_id}")
                return False

            alert = self._alerts[alert_id]

            if alert.state == AlertState.RESOLVED:
                logger.warning(f"Alert {alert_id} is already resolved")
                return False

            alert.resolve(resolution_message)
            self._stats["resolved_count"] += 1
            self._write_audit_log(alert, "resolved")

            # Clean up aggregation mapping
            if alert.aggregation_key and alert.aggregation_key in self._aggregation_map:
                del self._aggregation_map[alert.aggregation_key]

            # Move to history
            self._alert_history.append(alert)
            del self._alerts[alert_id]

            # Trim history if needed
            while len(self._alert_history) > self._max_history_size:
                self._alert_history.pop(0)

            return True

    async def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        category: AlertCategory | None = None,
        source_agent: str | None = None,
        limit: int | None = None,
    ) -> list[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            severity: Filter by severity (optional)
            category: Filter by category (optional)
            source_agent: Filter by source agent (optional)
            limit: Maximum number of alerts to return (optional)

        Returns:
            List of active alerts, sorted by timestamp (newest first)
        """
        async with self._lock:
            alerts = [
                a for a in self._alerts.values()
                if a.state == AlertState.ACTIVE
            ]

            # Apply filters
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if category:
                alerts = [a for a in alerts if a.category == category]
            if source_agent:
                alerts = [a for a in alerts if a.source_agent == source_agent]

            # Sort by timestamp (newest first)
            alerts.sort(key=lambda a: a.timestamp, reverse=True)

            # Apply limit
            if limit:
                alerts = alerts[:limit]

            return alerts

    async def get_alert_history(
        self,
        hours: float | None = None,
        severity: AlertSeverity | None = None,
        category: AlertCategory | None = None,
        limit: int | None = None,
    ) -> list[Alert]:
        """
        Get alert history with optional filtering.

        Args:
            hours: Only return alerts from the last N hours (optional)
            severity: Filter by severity (optional)
            category: Filter by category (optional)
            limit: Maximum number of alerts to return (optional)

        Returns:
            List of historical alerts, sorted by timestamp (newest first)
        """
        async with self._lock:
            alerts = list(self._alert_history)

            # Add resolved alerts from current storage
            alerts.extend([
                a for a in self._alerts.values()
                if a.state in (AlertState.RESOLVED, AlertState.EXPIRED)
            ])

            # Apply time filter
            if hours:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
                alerts = [a for a in alerts if a.timestamp >= cutoff]

            # Apply other filters
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if category:
                alerts = [a for a in alerts if a.category == category]

            # Sort by timestamp (newest first)
            alerts.sort(key=lambda a: a.timestamp, reverse=True)

            # Apply limit
            if limit:
                alerts = alerts[:limit]

            return alerts

    async def get_alert_by_id(self, alert_id: str) -> Alert | None:
        """
        Get a specific alert by ID.

        Args:
            alert_id: ID of the alert

        Returns:
            Alert if found, None otherwise
        """
        async with self._lock:
            # Check active alerts
            if alert_id in self._alerts:
                return self._alerts[alert_id]

            # Check history
            for alert in self._alert_history:
                if alert.alert_id == alert_id:
                    return alert

            return None

    async def add_rule(self, rule: AlertRule) -> None:
        """
        Add or update an alert rule.

        Args:
            rule: AlertRule to add or update
        """
        async with self._lock:
            self._rules[rule.rule_id] = rule
            logger.info(f"Alert rule added/updated: {rule.rule_id} ({rule.name})")

    async def remove_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule.

        Args:
            rule_id: ID of the rule to remove

        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info(f"Alert rule removed: {rule_id}")
                return True
            return False

    async def get_rule(self, rule_id: str) -> AlertRule | None:
        """
        Get a specific alert rule.

        Args:
            rule_id: ID of the rule

        Returns:
            AlertRule if found, None otherwise
        """
        async with self._lock:
            return self._rules.get(rule_id)

    async def get_all_rules(self) -> list[AlertRule]:
        """
        Get all alert rules.

        Returns:
            List of all alert rules
        """
        async with self._lock:
            return list(self._rules.values())

    async def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        async with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = True
                return True
            return False

    async def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        async with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = False
                return True
            return False

    async def check_rules(
        self,
        metrics: dict[str, Any],
        source_agent: str = "RuleChecker",
    ) -> list[Alert]:
        """
        Check all rules against provided metrics and generate alerts.

        This method evaluates each rule's condition against the metrics
        and creates alerts for any triggered rules.

        Args:
            metrics: Dictionary of metric names to values
            source_agent: Agent to attribute alerts to

        Returns:
            List of alerts created from triggered rules
        """
        triggered_alerts: list[Alert] = []

        async with self._lock:
            now = datetime.now(timezone.utc)

            for rule in self._rules.values():
                if not rule.can_trigger(now):
                    continue

                # Check if rule condition is met based on metrics
                triggered, alert_params = self._evaluate_rule(rule, metrics)

                if not triggered:
                    continue

                # Rule triggered - create alert
                rule.mark_triggered()
                self._stats["rule_triggers"][rule.rule_id] += 1

        # Create alerts outside the lock to avoid deadlock
        for rule in self._rules.values():
            if self._stats["rule_triggers"].get(rule.rule_id, 0) > 0:
                triggered, alert_params = self._evaluate_rule(rule, metrics)
                if triggered:
                    message = rule.format_message(**alert_params)
                    alert = await self.create_alert(
                        severity=rule.severity,
                        category=rule.category,
                        title=rule.name,
                        message=message,
                        source_agent=rule.source_agent or source_agent,
                        details=alert_params,
                        related_symbols=rule.symbols,
                        rule_id=rule.rule_id,
                    )
                    triggered_alerts.append(alert)

        return triggered_alerts

    def _evaluate_rule(
        self,
        rule: AlertRule,
        metrics: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Evaluate a rule condition against metrics.

        Args:
            rule: Rule to evaluate
            metrics: Metrics to check against

        Returns:
            Tuple of (triggered, parameters_for_message)
        """
        # Map rule IDs to their metric keys and evaluation logic
        rule_evaluators = {
            "drawdown_warning": self._check_threshold("drawdown_pct", metrics, rule.threshold, ">="),
            "drawdown_critical": self._check_threshold("drawdown_pct", metrics, rule.threshold, ">="),
            "position_limit_warning": self._check_threshold("position_pct", metrics, rule.threshold, ">="),
            "position_limit_breach": self._check_threshold("position_pct", metrics, rule.threshold, ">="),
            "leverage_warning": self._check_threshold("leverage", metrics, rule.threshold, ">="),
            "leverage_critical": self._check_threshold("leverage", metrics, rule.threshold, ">="),
            "daily_loss_warning": self._check_threshold("daily_loss_pct", metrics, rule.threshold, ">="),
            "var_breach": self._check_threshold("var_pct", metrics, rule.threshold, ">="),
            "agent_timeout": self._check_threshold("agent_last_response_seconds", metrics, rule.threshold, ">="),
            "market_data_stale": self._check_threshold("market_data_age_seconds", metrics, rule.threshold, ">="),
            # Binary checks
            "compliance_violation": self._check_binary("compliance_check_failed", metrics),
            "execution_error": self._check_binary("order_execution_failed", metrics),
            "execution_errors_critical": self._check_threshold("execution_error_count", metrics, rule.threshold, ">="),
            "agent_error": self._check_binary("agent_error_occurred", metrics),
            "kill_switch_activated": self._check_binary("kill_switch_triggered", metrics),
            "broker_disconnected": self._check_binary("broker_connection_lost", metrics),
        }

        if rule.rule_id in rule_evaluators:
            return rule_evaluators[rule.rule_id]

        # Generic threshold check for custom rules
        return self._check_generic_threshold(rule, metrics)

    def _check_threshold(
        self,
        metric_key: str,
        metrics: dict[str, Any],
        threshold: float,
        operator: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Check a threshold-based condition."""
        if metric_key not in metrics:
            return False, {}

        value = metrics[metric_key]
        if not isinstance(value, (int, float)):
            return False, {}

        triggered = False
        if operator == ">=":
            triggered = value >= threshold
        elif operator == ">":
            triggered = value > threshold
        elif operator == "<=":
            triggered = value <= threshold
        elif operator == "<":
            triggered = value < threshold
        elif operator == "==":
            triggered = value == threshold

        params = {
            "current_value": value,
            "limit_value": threshold,
            **{k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool))},
        }

        return triggered, params

    def _check_binary(
        self,
        metric_key: str,
        metrics: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Check a binary (true/false) condition."""
        if metric_key not in metrics:
            return False, {}

        value = metrics[metric_key]
        triggered = bool(value)

        params = {k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool))}
        return triggered, params

    def _check_generic_threshold(
        self,
        rule: AlertRule,
        metrics: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Generic threshold check for custom rules."""
        # Try to find a matching metric key
        for key, value in metrics.items():
            if key in rule.condition and isinstance(value, (int, float)):
                triggered = value >= rule.threshold
                params = {
                    "current_value": value,
                    "limit_value": rule.threshold,
                    **{k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool))},
                }
                return triggered, params

        return False, {}

    async def cleanup_expired(self) -> int:
        """
        Clean up expired alerts and old history.

        Returns:
            Number of alerts cleaned up
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - self._retention_period
            cleaned = 0

            # Expire old active alerts
            for alert_id, alert in list(self._alerts.items()):
                if alert.timestamp < cutoff and alert.state == AlertState.ACTIVE:
                    alert.state = AlertState.EXPIRED
                    self._alert_history.append(alert)
                    del self._alerts[alert_id]
                    cleaned += 1

                    if alert.aggregation_key in self._aggregation_map:
                        del self._aggregation_map[alert.aggregation_key]

            # Clean old history
            original_history_len = len(self._alert_history)
            self._alert_history = [
                a for a in self._alert_history
                if a.timestamp >= cutoff
            ]
            cleaned += original_history_len - len(self._alert_history)

            # Trim history if still too large
            while len(self._alert_history) > self._max_history_size:
                self._alert_history.pop(0)
                cleaned += 1

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired alerts")

            return cleaned

    def set_websocket_callback(
        self,
        callback: Callable[[Alert], None] | None,
    ) -> None:
        """
        Set the WebSocket notification callback.

        Args:
            callback: Function to call when new alerts are created
        """
        self._websocket_callback = callback

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            Dictionary of alert statistics
        """
        async with self._lock:
            active_alerts = [
                a for a in self._alerts.values()
                if a.state == AlertState.ACTIVE
            ]

            return {
                "total_alerts": self._stats["total_alerts"],
                "active_alerts": len(active_alerts),
                "acknowledged_alerts": len([
                    a for a in self._alerts.values()
                    if a.state == AlertState.ACKNOWLEDGED
                ]),
                "history_size": len(self._alert_history),
                "aggregated_count": self._stats["aggregated_count"],
                "acknowledged_count": self._stats["acknowledged_count"],
                "resolved_count": self._stats["resolved_count"],
                "by_severity": dict(self._stats["alerts_by_severity"]),
                "by_category": dict(self._stats["alerts_by_category"]),
                "rule_triggers": dict(self._stats["rule_triggers"]),
                "active_by_severity": {
                    s.value: len([a for a in active_alerts if a.severity == s])
                    for s in AlertSeverity
                },
                "rules_count": len(self._rules),
                "enabled_rules_count": len([r for r in self._rules.values() if r.enabled]),
            }

    def to_dict(self) -> dict[str, Any]:
        """
        Export manager state to dictionary for WebSocket streaming.

        Note: This is synchronous for compatibility with simple JSON serialization.
        For async contexts, use to_dict_async().

        Returns:
            Complete manager state as dict
        """
        now = datetime.now(timezone.utc)

        # Get active alerts
        active_alerts = [
            a for a in self._alerts.values()
            if a.state == AlertState.ACTIVE
        ]
        active_alerts.sort(key=lambda a: a.timestamp, reverse=True)

        # Get acknowledged alerts
        acknowledged_alerts = [
            a for a in self._alerts.values()
            if a.state == AlertState.ACKNOWLEDGED
        ]
        acknowledged_alerts.sort(key=lambda a: a.timestamp, reverse=True)

        # Get recent history (last 24 hours)
        cutoff = now - timedelta(hours=24)
        recent_history = [
            a for a in self._alert_history
            if a.timestamp >= cutoff
        ]
        recent_history.sort(key=lambda a: a.timestamp, reverse=True)

        # Count by severity for active alerts
        severity_counts = {
            s.value: len([a for a in active_alerts if a.severity == s])
            for s in AlertSeverity
        }

        # Count by category for active alerts
        category_counts = {
            c.value: len([a for a in active_alerts if a.category == c])
            for c in AlertCategory
        }

        return {
            "active_alerts": [a.to_dict() for a in active_alerts],
            "acknowledged_alerts": [a.to_dict() for a in acknowledged_alerts],
            "recent_history": [a.to_dict() for a in recent_history[:100]],
            "summary": {
                "total_active": len(active_alerts),
                "total_acknowledged": len(acknowledged_alerts),
                "by_severity": severity_counts,
                "by_category": category_counts,
                "has_critical": severity_counts.get("critical", 0) > 0,
                "has_emergency": severity_counts.get("emergency", 0) > 0,
            },
            "rules": {
                rule_id: rule.to_dict()
                for rule_id, rule in self._rules.items()
            },
            "statistics": {
                "total_alerts": self._stats["total_alerts"],
                "aggregated_count": self._stats["aggregated_count"],
                "acknowledged_count": self._stats["acknowledged_count"],
                "resolved_count": self._stats["resolved_count"],
            },
            "timestamp": now.isoformat(),
        }

    async def to_dict_async(self) -> dict[str, Any]:
        """
        Async version of to_dict with proper locking.

        Returns:
            Complete manager state as dict
        """
        async with self._lock:
            return self.to_dict()

    @property
    def active_alert_count(self) -> int:
        """Get count of active alerts."""
        return len([
            a for a in self._alerts.values()
            if a.state == AlertState.ACTIVE
        ])

    @property
    def critical_alert_count(self) -> int:
        """Get count of critical active alerts."""
        return len([
            a for a in self._alerts.values()
            if a.state == AlertState.ACTIVE and a.severity == AlertSeverity.CRITICAL
        ])

    @property
    def emergency_alert_count(self) -> int:
        """Get count of emergency active alerts."""
        return len([
            a for a in self._alerts.values()
            if a.state == AlertState.ACTIVE and a.severity == AlertSeverity.EMERGENCY
        ])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_alert_manager(
    enable_builtin_rules: bool = True,
    websocket_callback: Callable[[Alert], None] | None = None,
    audit_log_path: str = "logs/alerts_audit.jsonl",
) -> AlertManager:
    """
    Create a configured AlertManager instance.

    Args:
        enable_builtin_rules: Whether to enable built-in alert rules
        websocket_callback: Callback for WebSocket notifications
        audit_log_path: Path for audit log file

    Returns:
        Configured AlertManager instance
    """
    return AlertManager(
        enable_builtin_rules=enable_builtin_rules,
        websocket_callback=websocket_callback,
        audit_log_path=audit_log_path,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "AlertSeverity",
    "AlertCategory",
    "AlertState",
    # Data classes
    "Alert",
    "AlertRule",
    # Main class
    "AlertManager",
    # Factory function
    "create_alert_manager",
    "create_builtin_rules",
]

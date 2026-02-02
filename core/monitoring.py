"""
Monitoring & Observability
==========================

Centralized monitoring for the trading system.
Collects metrics, logs, alerts per agent.

Per CLAUDE.md: "Le système doit être testable, observable et auditable"
"""

from __future__ import annotations

import asyncio
import logging
import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Optional, Callable, Any
from enum import Enum
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from core.event_bus import EventBus

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    labels: dict[str, str] = field(default_factory=dict)
    agent: str = ""


@dataclass
class Alert:
    """An alert from the monitoring system."""
    alert_id: str
    severity: AlertSeverity
    source: str
    title: str
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    agent_name: str

    # Performance metrics
    events_processed: int = 0
    events_per_second: float = 0.0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0

    # Error metrics
    errors_total: int = 0
    errors_last_hour: int = 0

    # Agent-specific
    custom_metrics: dict[str, float] = field(default_factory=dict)

    # Status
    is_healthy: bool = True
    last_event_time: Optional[datetime] = None
    uptime_seconds: float = 0.0


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: datetime

    # Trading metrics
    total_decisions: int = 0
    decisions_approved: int = 0
    decisions_rejected: int = 0
    rejection_rate: float = 0.0

    total_orders: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    fill_rate: float = 0.0

    # P&L metrics
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Risk metrics
    current_drawdown_pct: float = 0.0
    var_95: float = 0.0
    leverage: float = 0.0

    # Latency metrics
    avg_decision_latency_ms: float = 0.0
    avg_execution_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0

    # System health
    active_agents: int = 0
    unhealthy_agents: list[str] = field(default_factory=list)
    kill_switch_active: bool = False


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    is_anomaly: bool
    current_value: float
    expected_value: float
    z_score: float
    threshold: float
    timestamp: datetime


class MetricsCollector:
    """
    Collects and aggregates metrics from all agents.
    """

    def __init__(self, retention_hours: int = 24):
        self._metrics: dict[str, list[Metric]] = {}
        self._retention_hours = retention_hours
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._processing_times: dict[str, list[float]] = {}

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[dict[str, str]] = None,
        agent: str = "",
    ) -> None:
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            metric_type=metric_type,
            labels=labels or {},
            agent=agent,
        )

        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append(metric)

        # Cleanup old metrics
        self._cleanup_old_metrics(name)

    def record_processing_time(self, agent: str, time_ms: float) -> None:
        """Record processing time for an agent."""
        if agent not in self._processing_times:
            self._processing_times[agent] = []

        self._processing_times[agent].append(time_ms)

        # Keep last 1000 samples
        if len(self._processing_times[agent]) > 1000:
            self._processing_times[agent] = self._processing_times[agent][-1000:]

    def increment_counter(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
        agent: str = "",
    ) -> None:
        """Increment a counter metric."""
        current = self.get_latest_value(name, labels)
        new_value = (current or 0) + 1
        self.record_metric(name, new_value, MetricType.COUNTER, labels, agent)

    def get_latest_value(
        self, name: str, labels: Optional[dict[str, str]] = None
    ) -> Optional[float]:
        """Get the latest value for a metric."""
        if name not in self._metrics:
            return None

        metrics = self._metrics[name]
        if labels:
            metrics = [m for m in metrics if m.labels == labels]

        if not metrics:
            return None

        return metrics[-1].value

    def get_metric_history(
        self,
        name: str,
        duration_minutes: int = 60,
        labels: Optional[dict[str, str]] = None,
    ) -> list[Metric]:
        """Get metric history for a duration."""
        if name not in self._metrics:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        metrics = [m for m in self._metrics[name] if m.timestamp >= cutoff]

        if labels:
            metrics = [m for m in metrics if m.labels == labels]

        return metrics

    def get_agent_metrics(self, agent: str) -> AgentMetrics:
        """Get aggregated metrics for an agent."""
        if agent not in self._agent_metrics:
            self._agent_metrics[agent] = AgentMetrics(agent_name=agent)

        metrics = self._agent_metrics[agent]

        # Update processing time stats
        if agent in self._processing_times and self._processing_times[agent]:
            times = self._processing_times[agent]
            metrics.avg_processing_time_ms = statistics.mean(times)
            metrics.max_processing_time_ms = max(times)
            metrics.p99_processing_time_ms = (
                np.percentile(times, 99) if len(times) >= 100 else max(times)
            )

        return metrics

    def update_agent_metrics(self, agent: str, **kwargs) -> None:
        """Update agent metrics."""
        if agent not in self._agent_metrics:
            self._agent_metrics[agent] = AgentMetrics(agent_name=agent)

        for key, value in kwargs.items():
            if hasattr(self._agent_metrics[agent], key):
                setattr(self._agent_metrics[agent], key, value)

    def _cleanup_old_metrics(self, name: str) -> None:
        """Remove metrics older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._retention_hours)
        self._metrics[name] = [m for m in self._metrics[name] if m.timestamp >= cutoff]


class AlertManager:
    """
    Manages alerts and notifications.
    """

    def __init__(self, alert_handlers: Optional[list[Callable[[Alert], None]]] = None):
        self._alerts: list[Alert] = []
        self._alert_handlers: list[Callable[[Alert], None]] = alert_handlers or []
        self._alert_rules: list[dict] = []
        self._alert_counter = 0

        # Alert thresholds
        self._thresholds: dict[str, tuple[float, AlertSeverity]] = {
            "daily_pnl_pct": (-0.02, AlertSeverity.WARNING),
            "daily_pnl_pct_critical": (-0.03, AlertSeverity.CRITICAL),
            "drawdown_pct": (0.05, AlertSeverity.WARNING),
            "drawdown_pct_critical": (0.10, AlertSeverity.EMERGENCY),
            "avg_latency_ms": (500, AlertSeverity.WARNING),
            "error_rate": (0.05, AlertSeverity.WARNING),
            "var_95": (0.02, AlertSeverity.WARNING),
        }

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        self._alert_handlers.append(handler)

    def create_alert(
        self,
        severity: AlertSeverity,
        source: str,
        title: str,
        message: str,
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
    ) -> Alert:
        """Create and dispatch an alert."""
        self._alert_counter += 1

        alert = Alert(
            alert_id=f"ALERT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}",
            severity=severity,
            source=source,
            title=title,
            message=message,
            timestamp=datetime.now(timezone.utc),
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
        )

        self._alerts.append(alert)
        self._dispatch_alert(alert)

        # Log based on severity
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(severity, logger.info)

        log_func(f"ALERT [{severity.value}] {source}: {title} - {message}")

        return alert

    def check_metric(self, metric_name: str, value: float) -> Optional[Alert]:
        """
        Check a single metric against its threshold and create an alert if needed.

        Args:
            metric_name: Name of the metric (e.g., "daily_pnl_pct", "drawdown_pct")
            value: Current value of the metric

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        # Check for critical threshold first
        critical_key = f"{metric_name}_critical"
        if critical_key in self._thresholds:
            threshold, severity = self._thresholds[critical_key]
            # For negative thresholds (like daily_pnl), trigger when value is below
            if threshold < 0 and value < threshold:
                return self.create_alert(
                    severity,
                    "monitoring",
                    f"{metric_name} Critical",
                    f"{metric_name} at {value:.4f} (threshold: {threshold})",
                    metric_name,
                    value,
                    threshold,
                )
            # For positive thresholds (like drawdown), trigger when value is above
            elif threshold > 0 and value > threshold:
                return self.create_alert(
                    severity,
                    "monitoring",
                    f"{metric_name} Critical",
                    f"{metric_name} at {value:.4f} (threshold: {threshold})",
                    metric_name,
                    value,
                    threshold,
                )

        # Check warning threshold
        if metric_name in self._thresholds:
            threshold, severity = self._thresholds[metric_name]
            if threshold < 0 and value < threshold:
                return self.create_alert(
                    severity,
                    "monitoring",
                    f"{metric_name} Warning",
                    f"{metric_name} at {value:.4f} (threshold: {threshold})",
                    metric_name,
                    value,
                    threshold,
                )
            elif threshold > 0 and value > threshold:
                return self.create_alert(
                    severity,
                    "monitoring",
                    f"{metric_name} Warning",
                    f"{metric_name} at {value:.4f} (threshold: {threshold})",
                    metric_name,
                    value,
                    threshold,
                )

        return None

    def check_thresholds(self, metrics: SystemMetrics) -> list[Alert]:
        """Check metrics against thresholds and create alerts."""
        alerts = []

        # Daily P&L check
        if metrics.daily_pnl_pct < self._thresholds["daily_pnl_pct_critical"][0]:
            alerts.append(self.create_alert(
                AlertSeverity.CRITICAL,
                "monitoring",
                "Daily Loss Critical",
                f"Daily P&L at {metrics.daily_pnl_pct*100:.2f}%",
                "daily_pnl_pct",
                metrics.daily_pnl_pct,
                self._thresholds["daily_pnl_pct_critical"][0],
            ))
        elif metrics.daily_pnl_pct < self._thresholds["daily_pnl_pct"][0]:
            alerts.append(self.create_alert(
                AlertSeverity.WARNING,
                "monitoring",
                "Daily Loss Warning",
                f"Daily P&L at {metrics.daily_pnl_pct*100:.2f}%",
                "daily_pnl_pct",
                metrics.daily_pnl_pct,
                self._thresholds["daily_pnl_pct"][0],
            ))

        # Drawdown check
        if metrics.current_drawdown_pct > self._thresholds["drawdown_pct_critical"][0]:
            alerts.append(self.create_alert(
                AlertSeverity.EMERGENCY,
                "monitoring",
                "Max Drawdown Exceeded",
                f"Drawdown at {metrics.current_drawdown_pct*100:.2f}%",
                "drawdown_pct",
                metrics.current_drawdown_pct,
                self._thresholds["drawdown_pct_critical"][0],
            ))
        elif metrics.current_drawdown_pct > self._thresholds["drawdown_pct"][0]:
            alerts.append(self.create_alert(
                AlertSeverity.WARNING,
                "monitoring",
                "Drawdown Warning",
                f"Drawdown at {metrics.current_drawdown_pct*100:.2f}%",
                "drawdown_pct",
                metrics.current_drawdown_pct,
                self._thresholds["drawdown_pct"][0],
            ))

        # Latency check
        if metrics.avg_total_latency_ms > self._thresholds["avg_latency_ms"][0]:
            alerts.append(self.create_alert(
                AlertSeverity.WARNING,
                "monitoring",
                "High Latency",
                f"Average latency at {metrics.avg_total_latency_ms:.0f}ms",
                "avg_latency_ms",
                metrics.avg_total_latency_ms,
                self._thresholds["avg_latency_ms"][0],
            ))

        # VaR check
        if metrics.var_95 > self._thresholds["var_95"][0]:
            alerts.append(self.create_alert(
                AlertSeverity.WARNING,
                "monitoring",
                "VaR Limit Warning",
                f"VaR at {metrics.var_95*100:.2f}%",
                "var_95",
                metrics.var_95,
                self._thresholds["var_95"][0],
            ))

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False

    def get_active_alerts(self) -> list[Alert]:
        """Get all unresolved alerts."""
        return [a for a in self._alerts if not a.resolved]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts by severity."""
        return [a for a in self._alerts if a.severity == severity and not a.resolved]

    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert to all handlers."""
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")


class AnomalyDetector:
    """
    Detects anomalies in metrics using statistical methods.
    """

    def __init__(self, z_score_threshold: float = 3.0, min_samples: int = 30):
        self._z_score_threshold = z_score_threshold
        self._min_samples = min_samples
        self._metric_history: dict[str, list[float]] = {}
        self._max_history = 1000

    def add_sample(self, metric_name: str, value: float) -> Optional[AnomalyDetection]:
        """Add a sample and check for anomaly."""
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []

        history = self._metric_history[metric_name]
        history.append(value)

        # Maintain max history size
        if len(history) > self._max_history:
            self._metric_history[metric_name] = history[-self._max_history:]
            history = self._metric_history[metric_name]

        # Need minimum samples for detection
        if len(history) < self._min_samples:
            return None

        # Calculate z-score
        mean = statistics.mean(history[:-1])  # Exclude current value
        std = statistics.stdev(history[:-1]) if len(history) > 2 else 1.0

        if std == 0:
            std = 1.0

        z_score = (value - mean) / std
        is_anomaly = abs(z_score) > self._z_score_threshold

        return AnomalyDetection(
            metric_name=metric_name,
            is_anomaly=is_anomaly,
            current_value=value,
            expected_value=mean,
            z_score=z_score,
            threshold=self._z_score_threshold,
            timestamp=datetime.now(timezone.utc),
        )

    def get_baseline(self, metric_name: str) -> Optional[tuple[float, float]]:
        """Get baseline (mean, std) for a metric."""
        if metric_name not in self._metric_history:
            return None

        history = self._metric_history[metric_name]
        if len(history) < self._min_samples:
            return None

        return statistics.mean(history), statistics.stdev(history)


class MonitoringSystem:
    """
    Central monitoring system for the trading platform.

    Integrates:
    - Metrics collection
    - Alert management
    - Anomaly detection
    - Per-agent logging
    """

    def __init__(
        self,
        log_dir: str = "logs",
        metrics_retention_hours: int = 24,
        alert_handlers: Optional[list[Callable[[Alert], None]]] = None,
    ):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.metrics = MetricsCollector(retention_hours=metrics_retention_hours)
        self.alerts = AlertManager(alert_handlers=alert_handlers)
        self.anomaly_detector = AnomalyDetector()

        # Agent loggers
        self._agent_loggers: dict[str, logging.Logger] = {}

        # System state
        self._start_time = datetime.now(timezone.utc)
        self._registered_agents: set[str] = set()

        # Background task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    def get_agent_logger(self, agent_name: str) -> logging.Logger:
        """Get or create a logger for an agent."""
        if agent_name not in self._agent_loggers:
            agent_logger = logging.getLogger(f"agent.{agent_name}")

            # File handler for this agent
            log_file = self._log_dir / f"{agent_name}.log"
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            ))
            agent_logger.addHandler(handler)
            agent_logger.setLevel(logging.DEBUG)

            self._agent_loggers[agent_name] = agent_logger

        return self._agent_loggers[agent_name]

    def register_agent(self, agent_name: str) -> None:
        """Register an agent for monitoring."""
        self._registered_agents.add(agent_name)
        self.metrics.update_agent_metrics(
            agent_name,
            is_healthy=True,
            uptime_seconds=0,
        )
        logger.info(f"Agent registered for monitoring: {agent_name}")

    def record_event(
        self,
        agent: str,
        event_type: str,
        processing_time_ms: float,
        success: bool = True,
        details: Optional[dict] = None,
    ) -> None:
        """Record an event processed by an agent."""
        # Update counters
        self.metrics.increment_counter(f"{agent}_events_total", agent=agent)
        if success:
            self.metrics.increment_counter(f"{agent}_events_success", agent=agent)
        else:
            self.metrics.increment_counter(f"{agent}_events_error", agent=agent)

        # Record processing time
        self.metrics.record_processing_time(agent, processing_time_ms)

        # Check for latency anomaly
        anomaly = self.anomaly_detector.add_sample(
            f"{agent}_processing_time", processing_time_ms
        )
        if anomaly and anomaly.is_anomaly:
            self.alerts.create_alert(
                AlertSeverity.WARNING,
                agent,
                "Processing Time Anomaly",
                f"Processing time {processing_time_ms:.0f}ms (z-score: {anomaly.z_score:.2f})",
                f"{agent}_processing_time",
                processing_time_ms,
                anomaly.expected_value,
            )

        # Log to agent-specific log
        agent_logger = self.get_agent_logger(agent)
        agent_logger.info(
            f"event_type={event_type} | "
            f"processing_time_ms={processing_time_ms:.2f} | "
            f"success={success} | "
            f"details={json.dumps(details) if details else '{}'}"
        )

    def record_decision(
        self,
        decision_id: str,
        symbol: str,
        action: str,
        quantity: int,
        approved: bool,
        rejection_reason: Optional[str] = None,
        latency_ms: float = 0,
    ) -> None:
        """Record a trading decision."""
        self.metrics.increment_counter("decisions_total")
        if approved:
            self.metrics.increment_counter("decisions_approved")
        else:
            self.metrics.increment_counter("decisions_rejected")
            self.metrics.increment_counter(
                "rejections_by_reason",
                labels={"reason": rejection_reason or "unknown"}
            )

        # Record latency
        self.metrics.record_metric(
            "decision_latency_ms",
            latency_ms,
            MetricType.HISTOGRAM,
        )

    def record_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        status: str,
        fill_price: Optional[float] = None,
        latency_ms: float = 0,
    ) -> None:
        """Record an order."""
        self.metrics.increment_counter("orders_total")
        self.metrics.increment_counter(f"orders_{status}")

        if status == "filled" and fill_price:
            self.metrics.record_metric(
                "execution_latency_ms",
                latency_ms,
                MetricType.HISTOGRAM,
            )

    def record_pnl(
        self,
        daily_pnl: float,
        daily_pnl_pct: float,
        unrealized_pnl: float,
        realized_pnl: float,
    ) -> None:
        """Record P&L metrics."""
        self.metrics.record_metric("daily_pnl", daily_pnl, MetricType.GAUGE)
        self.metrics.record_metric("daily_pnl_pct", daily_pnl_pct, MetricType.GAUGE)
        self.metrics.record_metric("unrealized_pnl", unrealized_pnl, MetricType.GAUGE)
        self.metrics.record_metric("realized_pnl", realized_pnl, MetricType.GAUGE)

        # Check for P&L anomaly
        anomaly = self.anomaly_detector.add_sample("daily_pnl_pct", daily_pnl_pct)
        if anomaly and anomaly.is_anomaly and daily_pnl_pct < 0:
            self.alerts.create_alert(
                AlertSeverity.WARNING,
                "monitoring",
                "P&L Anomaly Detected",
                f"Daily P&L at {daily_pnl_pct*100:.2f}% (z-score: {anomaly.z_score:.2f})",
                "daily_pnl_pct",
                daily_pnl_pct,
                anomaly.expected_value,
            )

    def record_risk_metrics(
        self,
        drawdown_pct: float,
        var_95: float,
        leverage: float,
    ) -> None:
        """Record risk metrics."""
        self.metrics.record_metric("drawdown_pct", drawdown_pct, MetricType.GAUGE)
        self.metrics.record_metric("var_95", var_95, MetricType.GAUGE)
        self.metrics.record_metric("leverage", leverage, MetricType.GAUGE)

    def get_system_metrics(self) -> SystemMetrics:
        """Get aggregated system metrics."""
        now = datetime.now(timezone.utc)

        # Aggregate from collectors
        decisions_total = int(self.metrics.get_latest_value("decisions_total") or 0)
        decisions_approved = int(self.metrics.get_latest_value("decisions_approved") or 0)
        decisions_rejected = int(self.metrics.get_latest_value("decisions_rejected") or 0)

        orders_total = int(self.metrics.get_latest_value("orders_total") or 0)
        orders_filled = int(self.metrics.get_latest_value("orders_filled") or 0)

        # Calculate rates
        rejection_rate = (
            decisions_rejected / decisions_total if decisions_total > 0 else 0
        )
        fill_rate = orders_filled / orders_total if orders_total > 0 else 0

        # Get latencies
        decision_latencies = self.metrics.get_metric_history("decision_latency_ms", 60)
        execution_latencies = self.metrics.get_metric_history("execution_latency_ms", 60)

        avg_decision_latency = (
            statistics.mean([m.value for m in decision_latencies])
            if decision_latencies else 0
        )
        avg_execution_latency = (
            statistics.mean([m.value for m in execution_latencies])
            if execution_latencies else 0
        )

        # Check agent health
        unhealthy = []
        for agent in self._registered_agents:
            agent_metrics = self.metrics.get_agent_metrics(agent)
            if not agent_metrics.is_healthy:
                unhealthy.append(agent)

        return SystemMetrics(
            timestamp=now,
            total_decisions=decisions_total,
            decisions_approved=decisions_approved,
            decisions_rejected=decisions_rejected,
            rejection_rate=rejection_rate,
            total_orders=orders_total,
            orders_filled=orders_filled,
            fill_rate=fill_rate,
            daily_pnl=self.metrics.get_latest_value("daily_pnl") or 0,
            daily_pnl_pct=self.metrics.get_latest_value("daily_pnl_pct") or 0,
            unrealized_pnl=self.metrics.get_latest_value("unrealized_pnl") or 0,
            realized_pnl=self.metrics.get_latest_value("realized_pnl") or 0,
            current_drawdown_pct=self.metrics.get_latest_value("drawdown_pct") or 0,
            var_95=self.metrics.get_latest_value("var_95") or 0,
            leverage=self.metrics.get_latest_value("leverage") or 0,
            avg_decision_latency_ms=avg_decision_latency,
            avg_execution_latency_ms=avg_execution_latency,
            avg_total_latency_ms=avg_decision_latency + avg_execution_latency,
            active_agents=len(self._registered_agents),
            unhealthy_agents=unhealthy,
            kill_switch_active=False,  # Would get from risk agent
        )

    async def start(self, check_interval_seconds: int = 60) -> None:
        """Start background monitoring."""
        self._running = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval_seconds)
        )
        logger.info("Monitoring system started")

    async def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring system stopped")

    async def _monitoring_loop(self, interval: int) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Get current metrics
                system_metrics = self.get_system_metrics()

                # Check thresholds and create alerts
                self.alerts.check_thresholds(system_metrics)

                # Log summary
                logger.info(
                    f"MONITORING | "
                    f"decisions={system_metrics.total_decisions} | "
                    f"rejection_rate={system_metrics.rejection_rate*100:.1f}% | "
                    f"pnl={system_metrics.daily_pnl_pct*100:.2f}% | "
                    f"dd={system_metrics.current_drawdown_pct*100:.2f}% | "
                    f"latency={system_metrics.avg_total_latency_ms:.0f}ms"
                )

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)

    def export_metrics(self, filepath: str) -> None:
        """Export current metrics to JSON file."""
        system_metrics = self.get_system_metrics()
        agent_metrics = {
            agent: asdict(self.metrics.get_agent_metrics(agent))
            for agent in self._registered_agents
        }
        active_alerts = [asdict(a) for a in self.alerts.get_active_alerts()]

        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_metrics": asdict(system_metrics),
            "agent_metrics": agent_metrics,
            "active_alerts": active_alerts,
        }

        # Handle datetime serialization
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=datetime_handler)

        logger.info(f"Metrics exported to {filepath}")

    def get_status_summary(self) -> dict:
        """Get a status summary for display."""
        metrics = self.get_system_metrics()
        active_alerts = self.alerts.get_active_alerts()

        return {
            "uptime_hours": (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds() / 3600,
            "active_agents": metrics.active_agents,
            "unhealthy_agents": metrics.unhealthy_agents,
            "decisions_today": metrics.total_decisions,
            "rejection_rate_pct": metrics.rejection_rate * 100,
            "daily_pnl_pct": metrics.daily_pnl_pct * 100,
            "drawdown_pct": metrics.current_drawdown_pct * 100,
            "avg_latency_ms": metrics.avg_total_latency_ms,
            "active_alerts_count": len(active_alerts),
            "critical_alerts": len([
                a for a in active_alerts
                if a.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)
            ]),
        }

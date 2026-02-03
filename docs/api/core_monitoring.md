# monitoring

**Path**: `C:\Users\Alexa\ai-trading-firm\core\monitoring.py`

## Overview

Monitoring & Observability
==========================

Centralized monitoring for the trading system.
Collects metrics, logs, alerts per agent.

Per CLAUDE.md: "Le système doit être testable, observable et auditable"

## Classes

### AlertSeverity

**Inherits from**: Enum

Alert severity levels.

### MetricType

**Inherits from**: Enum

Types of metrics.

### Metric

A single metric data point.

### Alert

An alert from the monitoring system.

### AgentMetrics

Metrics for a single agent.

### SystemMetrics

System-wide metrics.

### AnomalyDetection

Anomaly detection result.

### MetricsCollector

Collects and aggregates metrics from all agents.

#### Methods

##### `def __init__(self, retention_hours: int)`

##### `def record_metric(self, name: str, value: float, metric_type: MetricType, labels: Optional[dict[str, str]], agent: str) -> None`

Record a metric value.

##### `def record_processing_time(self, agent: str, time_ms: float) -> None`

Record processing time for an agent.

##### `def increment_counter(self, name: str, labels: Optional[dict[str, str]], agent: str) -> None`

Increment a counter metric.

##### `def get_latest_value(self, name: str, labels: Optional[dict[str, str]]) -> Optional[float]`

Get the latest value for a metric.

##### `def get_metric_history(self, name: str, duration_minutes: int, labels: Optional[dict[str, str]]) -> list[Metric]`

Get metric history for a duration.

##### `def get_agent_metrics(self, agent: str) -> AgentMetrics`

Get aggregated metrics for an agent.

##### `def update_agent_metrics(self, agent: str) -> None`

Update agent metrics.

### AlertManager

Manages alerts and notifications.

#### Methods

##### `def __init__(self, alert_handlers: Optional[list[Callable[, None]]])`

##### `def add_handler(self, handler: Callable[, None]) -> None`

Add an alert handler.

##### `def create_alert(self, severity: AlertSeverity, source: str, title: str, message: str, metric_name: Optional[str], current_value: Optional[float], threshold_value: Optional[float]) -> Alert`

Create and dispatch an alert.

##### `def check_metric(self, metric_name: str, value: float) -> Optional[Alert]`

Check a single metric against its threshold and create an alert if needed.

Args:
    metric_name: Name of the metric (e.g., "daily_pnl_pct", "drawdown_pct")
    value: Current value of the metric

Returns:
    Alert if threshold exceeded, None otherwise

##### `def check_thresholds(self, metrics: SystemMetrics) -> list[Alert]`

Check metrics against thresholds and create alerts.

##### `def acknowledge_alert(self, alert_id: str) -> bool`

Acknowledge an alert.

##### `def resolve_alert(self, alert_id: str) -> bool`

Resolve an alert.

##### `def get_active_alerts(self) -> list[Alert]`

Get all unresolved alerts.

##### `def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]`

Get alerts by severity.

### AnomalyDetector

Detects anomalies in metrics using statistical methods.

#### Methods

##### `def __init__(self, z_score_threshold: float, min_samples: int)`

##### `def add_sample(self, metric_name: str, value: float) -> Optional[AnomalyDetection]`

Add a sample and check for anomaly.

##### `def get_baseline(self, metric_name: str) -> Optional[tuple[float, float]]`

Get baseline (mean, std) for a metric.

### MonitoringSystem

Central monitoring system for the trading platform.

Integrates:
- Metrics collection
- Alert management
- Anomaly detection
- Per-agent logging

#### Methods

##### `def __init__(self, log_dir: str, metrics_retention_hours: int, alert_handlers: Optional[list[Callable[, None]]])`

##### `def get_agent_logger(self, agent_name: str) -> logging.Logger`

Get or create a logger for an agent.

##### `def register_agent(self, agent_name: str) -> None`

Register an agent for monitoring.

##### `def record_event(self, agent: str, event_type: str, processing_time_ms: float, success: bool, details: Optional[dict]) -> None`

Record an event processed by an agent.

##### `def record_decision(self, decision_id: str, symbol: str, action: str, quantity: int, approved: bool, rejection_reason: Optional[str], latency_ms: float) -> None`

Record a trading decision.

##### `def record_order(self, order_id: str, symbol: str, side: str, quantity: int, status: str, fill_price: Optional[float], latency_ms: float) -> None`

Record an order.

##### `def record_pnl(self, daily_pnl: float, daily_pnl_pct: float, unrealized_pnl: float, realized_pnl: float) -> None`

Record P&L metrics.

##### `def record_risk_metrics(self, drawdown_pct: float, var_95: float, leverage: float) -> None`

Record risk metrics.

##### `def get_system_metrics(self) -> SystemMetrics`

Get aggregated system metrics.

##### `async def start(self, check_interval_seconds: int) -> None`

Start background monitoring.

##### `async def stop(self) -> None`

Stop background monitoring.

##### `def export_metrics(self, filepath: str) -> None`

Export current metrics to JSON file.

##### `def get_status_summary(self) -> dict`

Get a status summary for display.

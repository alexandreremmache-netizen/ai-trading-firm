"""
Infrastructure Operations Module

Addresses MEDIUM priority issues:
- #I15: Log aggregation framework
- #I16: Application Performance Monitoring (APM)
- #I17: Alert management and escalation
- #I18: Chaos testing framework
- #I19: Connection pooling
- #I20: Cache warming strategies
- #I21: Request tracing
- #I22: Blue-green deployment support
- #I23: Feature flags system

Provides production-grade infrastructure components for trading systems.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from queue import PriorityQueue, Queue
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class DeploymentColor(Enum):
    """Blue-green deployment color."""
    BLUE = "blue"
    GREEN = "green"


class FeatureFlagStatus(Enum):
    """Feature flag status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PERCENTAGE_ROLLOUT = "percentage_rollout"
    USER_SEGMENT = "user_segment"


class ChaosExperimentType(Enum):
    """Types of chaos experiments."""
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DEPENDENCY_FAILURE = "dependency_failure"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    service: str
    component: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None


@dataclass
class PerformanceMetric:
    """APM performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation: str = "gauge"  # gauge, counter, histogram


@dataclass
class Alert:
    """Alert definition."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.OPEN
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    status: str = "ok"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    status: FeatureFlagStatus
    default_value: bool
    description: str = ""
    percentage: float = 0.0  # For percentage rollout
    user_segments: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosExperiment:
    """Chaos experiment definition."""
    id: str
    experiment_type: ChaosExperimentType
    target_service: str
    parameters: Dict[str, Any]
    duration_seconds: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    is_active: bool = False
    results: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Log Aggregation Framework (#I15)
# =============================================================================

class LogAggregator:
    """
    Centralized log aggregation system.

    Features:
    - Structured logging
    - Log buffering and batching
    - Multiple output destinations
    - Log correlation with trace IDs
    - Log level filtering
    - Log retention policies
    """

    def __init__(
        self,
        service_name: str,
        buffer_size: int = 1000,
        flush_interval_seconds: int = 5
    ):
        """Initialize log aggregator."""
        self.service_name = service_name
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        self._buffer: List[LogEntry] = []
        self._lock = threading.Lock()
        self._outputs: List[Callable[[List[LogEntry]], None]] = []
        self._filters: List[Callable[[LogEntry], bool]] = []
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._current_trace_id: Optional[str] = None

    def start(self) -> None:
        """Start the log aggregator background flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info(f"Log aggregator started for service: {self.service_name}")

    def stop(self) -> None:
        """Stop the log aggregator and flush remaining logs."""
        self._running = False
        self.flush()
        if self._flush_thread:
            self._flush_thread.join(timeout=5)

    def add_output(self, output_handler: Callable[[List[LogEntry]], None]) -> None:
        """Add an output destination for logs."""
        self._outputs.append(output_handler)

    def add_filter(self, filter_func: Callable[[LogEntry], bool]) -> None:
        """Add a log filter (returns True to keep log)."""
        self._filters.append(filter_func)

    def set_trace_context(self, trace_id: str) -> None:
        """Set current trace context for log correlation."""
        self._current_trace_id = trace_id

    def clear_trace_context(self) -> None:
        """Clear current trace context."""
        self._current_trace_id = None

    def log(
        self,
        level: LogLevel,
        message: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
        exception: Optional[str] = None
    ) -> None:
        """
        Log a message with structured data.

        Args:
            level: Log level
            message: Log message
            component: Component name
            metadata: Additional metadata
            exception: Exception string if any
        """
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            service=self.service_name,
            component=component,
            trace_id=self._current_trace_id,
            metadata=metadata or {},
            exception=exception
        )

        # Apply filters
        for filter_func in self._filters:
            if not filter_func(entry):
                return

        with self._lock:
            self._buffer.append(entry)

            if len(self._buffer) >= self.buffer_size:
                self._flush_internal()

    def debug(self, message: str, component: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, component, kwargs)

    def info(self, message: str, component: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, component, kwargs)

    def warning(self, message: str, component: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, component, kwargs)

    def error(self, message: str, component: str, exception: Optional[str] = None, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, component, kwargs, exception)

    def critical(self, message: str, component: str, exception: Optional[str] = None, **kwargs) -> None:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, component, kwargs, exception)

    def flush(self) -> None:
        """Flush buffered logs to outputs."""
        with self._lock:
            self._flush_internal()

    def _flush_internal(self) -> None:
        """Internal flush (must hold lock)."""
        if not self._buffer:
            return

        logs_to_send = self._buffer.copy()
        self._buffer.clear()

        for output in self._outputs:
            try:
                output(logs_to_send)
            except Exception as e:
                logger.error(f"Failed to send logs to output: {e}")

    def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            time.sleep(self.flush_interval)
            self.flush()

    def search_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        level: Optional[LogLevel] = None,
        component: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Search logs (in-memory buffer only for this implementation).

        In production, this would query a log storage backend.
        """
        results = []
        with self._lock:
            for entry in self._buffer:
                if entry.timestamp < start_time or entry.timestamp > end_time:
                    continue
                if level and entry.level.value < level.value:
                    continue
                if component and entry.component != component:
                    continue
                if trace_id and entry.trace_id != trace_id:
                    continue
                results.append(entry)
        return results


def console_log_output(entries: List[LogEntry]) -> None:
    """Console output handler for logs."""
    for entry in entries:
        print(f"[{entry.timestamp.isoformat()}] [{entry.level.name}] "
              f"[{entry.service}/{entry.component}] {entry.message}")


def json_log_output(entries: List[LogEntry]) -> List[str]:
    """JSON output handler for logs."""
    json_logs = []
    for entry in entries:
        log_dict = {
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level.name,
            "message": entry.message,
            "service": entry.service,
            "component": entry.component,
            "trace_id": entry.trace_id,
            "metadata": entry.metadata
        }
        if entry.exception:
            log_dict["exception"] = entry.exception
        json_logs.append(json.dumps(log_dict))
    return json_logs


# =============================================================================
# Application Performance Monitoring (#I16)
# =============================================================================

class APMCollector:
    """
    Application Performance Monitoring collector.

    Features:
    - Metric collection (counters, gauges, histograms)
    - Transaction tracing
    - Resource utilization tracking
    - SLA monitoring
    - Custom dashboards data
    """

    def __init__(self, service_name: str, flush_interval_seconds: int = 10):
        """Initialize APM collector."""
        self.service_name = service_name
        self.flush_interval = flush_interval_seconds
        self._metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._transaction_times: Dict[str, List[float]] = defaultdict(list)
        self._sla_thresholds: Dict[str, float] = {}

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        aggregation: str = "gauge"
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {},
            aggregation=aggregation
        )

        with self._lock:
            self._metrics[name].append(metric)

            if aggregation == "counter":
                self._counters[name] += value
            elif aggregation == "gauge":
                self._gauges[name] = value
            elif aggregation == "histogram":
                self._histograms[name].append(value)

    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, tags=tags, aggregation="counter")

    def set_gauge(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, unit=unit, tags=tags, aggregation="gauge")

    def record_histogram(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        self.record_metric(name, value, unit=unit, tags=tags, aggregation="histogram")

    def start_transaction(self, transaction_name: str) -> "TransactionContext":
        """Start a transaction for timing."""
        return TransactionContext(self, transaction_name)

    def record_transaction_time(self, transaction_name: str, duration_ms: float) -> None:
        """Record a transaction duration."""
        with self._lock:
            self._transaction_times[transaction_name].append(duration_ms)

        # Check SLA
        if transaction_name in self._sla_thresholds:
            threshold = self._sla_thresholds[transaction_name]
            if duration_ms > threshold:
                logger.warning(f"SLA breach: {transaction_name} took {duration_ms:.2f}ms (threshold: {threshold}ms)")

    def set_sla_threshold(self, transaction_name: str, threshold_ms: float) -> None:
        """Set SLA threshold for a transaction."""
        self._sla_thresholds[transaction_name] = threshold_ms

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = self._histograms.get(name, [])

        if not values:
            return {"count": 0}

        import statistics
        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
            "stddev": statistics.stdev(values) if n > 1 else 0
        }

    def get_transaction_stats(self, transaction_name: str) -> Dict[str, Any]:
        """Get transaction timing statistics."""
        with self._lock:
            times = self._transaction_times.get(transaction_name, [])

        if not times:
            return {"count": 0}

        import statistics
        sorted_times = sorted(times)
        n = len(sorted_times)

        stats = {
            "count": n,
            "min_ms": min(times),
            "max_ms": max(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "p95_ms": sorted_times[int(n * 0.95)] if n > 0 else 0,
            "p99_ms": sorted_times[int(n * 0.99)] if n > 0 else 0
        }

        # SLA compliance
        if transaction_name in self._sla_thresholds:
            threshold = self._sla_thresholds[transaction_name]
            violations = sum(1 for t in times if t > threshold)
            stats["sla_threshold_ms"] = threshold
            stats["sla_violations"] = violations
            stats["sla_compliance_pct"] = (n - violations) / n * 100 if n > 0 else 100

        return stats

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.get_histogram_stats(k) for k in self._histograms.keys()},
                "transactions": {k: self.get_transaction_stats(k) for k in self._transaction_times.keys()}
            }


class TransactionContext:
    """Context manager for transaction timing."""

    def __init__(self, apm: APMCollector, transaction_name: str):
        self.apm = apm
        self.transaction_name = transaction_name
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TransactionContext":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.apm.record_transaction_time(self.transaction_name, duration_ms)


def apm_timed(apm: APMCollector, transaction_name: str):
    """Decorator for automatic transaction timing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with apm.start_transaction(transaction_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Alert Management and Escalation (#I17)
# =============================================================================

@dataclass
class EscalationPolicy:
    """Escalation policy definition."""
    name: str
    levels: List[Dict[str, Any]]  # List of escalation level configs
    repeat_interval_minutes: int = 30


class AlertManager:
    """
    Alert management and escalation system.

    Features:
    - Alert creation and routing
    - Escalation policies
    - Alert grouping and deduplication
    - On-call integration
    - Alert suppression rules
    """

    def __init__(self, service_name: str):
        """Initialize alert manager."""
        self.service_name = service_name
        self._alerts: Dict[str, Alert] = {}
        self._escalation_policies: Dict[str, EscalationPolicy] = {}
        self._notification_handlers: Dict[AlertSeverity, List[Callable[[Alert], None]]] = defaultdict(list)
        self._suppression_rules: List[Callable[[Alert], bool]] = []
        self._alert_groups: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self._escalation_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the alert manager escalation thread."""
        self._running = True
        self._escalation_thread = threading.Thread(target=self._escalation_loop, daemon=True)
        self._escalation_thread.start()
        logger.info("Alert manager started")

    def stop(self) -> None:
        """Stop the alert manager."""
        self._running = False
        if self._escalation_thread:
            self._escalation_thread.join(timeout=5)

    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        group_key: Optional[str] = None
    ) -> Alert:
        """
        Create a new alert.

        Args:
            severity: Alert severity
            title: Alert title
            description: Detailed description
            source: Source component
            metadata: Additional metadata
            group_key: Key for alert grouping

        Returns:
            Created alert
        """
        # Check suppression rules
        test_alert = Alert(
            id="test",
            severity=severity,
            title=title,
            description=description,
            source=source,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )

        for rule in self._suppression_rules:
            if rule(test_alert):
                logger.debug(f"Alert suppressed: {title}")
                test_alert.status = AlertStatus.SUPPRESSED
                return test_alert

        # Generate alert ID
        alert_id = hashlib.md5(
            f"{title}{source}{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]

        # Check for duplicate in group
        if group_key:
            with self._lock:
                if group_key in self._alert_groups:
                    for existing_id in self._alert_groups[group_key]:
                        if existing_id in self._alerts:
                            existing = self._alerts[existing_id]
                            if existing.status == AlertStatus.OPEN:
                                logger.debug(f"Alert deduplicated: {title}")
                                return existing

        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            source=source,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )

        with self._lock:
            self._alerts[alert_id] = alert
            if group_key:
                self._alert_groups[group_key].add(alert_id)

        # Send notifications
        self._notify(alert)

        logger.info(f"Alert created: [{severity.value}] {title}")
        return alert

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return False

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return False

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)

        logger.info(f"Alert resolved: {alert_id}")
        return True

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy."""
        self._escalation_policies[policy.name] = policy

    def add_notification_handler(
        self,
        severity: AlertSeverity,
        handler: Callable[[Alert], None]
    ) -> None:
        """Add a notification handler for a severity level."""
        self._notification_handlers[severity].append(handler)

    def add_suppression_rule(self, rule: Callable[[Alert], bool]) -> None:
        """Add an alert suppression rule (returns True to suppress)."""
        self._suppression_rules.append(rule)

    def get_open_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get all open alerts, optionally filtered by severity."""
        with self._lock:
            alerts = [a for a in self._alerts.values() if a.status == AlertStatus.OPEN]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            alerts = list(self._alerts.values())

        stats = {
            "total": len(alerts),
            "by_status": defaultdict(int),
            "by_severity": defaultdict(int),
            "mttr_minutes": []  # Mean time to resolve
        }

        for alert in alerts:
            stats["by_status"][alert.status.value] += 1
            stats["by_severity"][alert.severity.value] += 1

            if alert.resolved_at:
                resolve_time = (alert.resolved_at - alert.timestamp).total_seconds() / 60
                stats["mttr_minutes"].append(resolve_time)

        if stats["mttr_minutes"]:
            import statistics
            stats["avg_mttr_minutes"] = statistics.mean(stats["mttr_minutes"])
        else:
            stats["avg_mttr_minutes"] = 0

        return stats

    def _notify(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        handlers = self._notification_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    def _escalation_loop(self) -> None:
        """Background escalation loop."""
        while self._running:
            time.sleep(60)  # Check every minute

            with self._lock:
                for alert in self._alerts.values():
                    if alert.status != AlertStatus.OPEN:
                        continue

                    # Check if escalation is needed
                    age_minutes = (datetime.now(timezone.utc) - alert.timestamp).total_seconds() / 60

                    # Simple escalation: escalate after 15, 30, 60 minutes
                    if age_minutes > 60 and alert.escalation_level < 3:
                        alert.escalation_level = 3
                        logger.warning(f"Alert escalated to level 3: {alert.title}")
                    elif age_minutes > 30 and alert.escalation_level < 2:
                        alert.escalation_level = 2
                        logger.warning(f"Alert escalated to level 2: {alert.title}")
                    elif age_minutes > 15 and alert.escalation_level < 1:
                        alert.escalation_level = 1
                        logger.warning(f"Alert escalated to level 1: {alert.title}")


# =============================================================================
# Chaos Testing Framework (#I18)
# =============================================================================

class ChaosEngine:
    """
    Chaos engineering framework for resilience testing.

    Features:
    - Latency injection
    - Error injection
    - Resource exhaustion simulation
    - Network partition simulation
    - Dependency failure simulation
    """

    def __init__(self):
        """Initialize chaos engine."""
        self._experiments: Dict[str, ChaosExperiment] = {}
        self._active_experiments: Set[str] = set()
        self._lock = threading.Lock()
        self._interceptors: Dict[str, Callable] = {}

    def create_experiment(
        self,
        experiment_type: ChaosExperimentType,
        target_service: str,
        parameters: Dict[str, Any],
        duration_seconds: int
    ) -> ChaosExperiment:
        """
        Create a chaos experiment.

        Args:
            experiment_type: Type of chaos to inject
            target_service: Target service name
            parameters: Experiment-specific parameters
            duration_seconds: How long to run the experiment

        Returns:
            Created experiment
        """
        exp_id = hashlib.md5(
            f"{experiment_type.value}{target_service}{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]

        experiment = ChaosExperiment(
            id=exp_id,
            experiment_type=experiment_type,
            target_service=target_service,
            parameters=parameters,
            duration_seconds=duration_seconds
        )

        with self._lock:
            self._experiments[exp_id] = experiment

        logger.info(f"Chaos experiment created: {exp_id} ({experiment_type.value})")
        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """Start a chaos experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False

            experiment = self._experiments[experiment_id]
            experiment.is_active = True
            experiment.start_time = datetime.now(timezone.utc)
            self._active_experiments.add(experiment_id)

            # Set up interceptor based on experiment type
            self._setup_interceptor(experiment)

        logger.warning(f"Chaos experiment STARTED: {experiment_id}")

        # Schedule auto-stop
        threading.Timer(
            experiment.duration_seconds,
            lambda: self.stop_experiment(experiment_id)
        ).start()

        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a chaos experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False

            experiment = self._experiments[experiment_id]
            experiment.is_active = False
            experiment.end_time = datetime.now(timezone.utc)
            self._active_experiments.discard(experiment_id)

            # Remove interceptor
            self._teardown_interceptor(experiment)

        logger.info(f"Chaos experiment STOPPED: {experiment_id}")
        return True

    def _setup_interceptor(self, experiment: ChaosExperiment) -> None:
        """Set up chaos interceptor for experiment."""
        key = f"{experiment.target_service}_{experiment.experiment_type.value}"

        if experiment.experiment_type == ChaosExperimentType.LATENCY_INJECTION:
            latency_ms = experiment.parameters.get("latency_ms", 1000)
            self._interceptors[key] = lambda: time.sleep(latency_ms / 1000)

        elif experiment.experiment_type == ChaosExperimentType.ERROR_INJECTION:
            error_rate = experiment.parameters.get("error_rate", 0.5)
            error_message = experiment.parameters.get("error_message", "Chaos error")
            self._interceptors[key] = lambda: self._maybe_raise(error_rate, error_message)

    def _teardown_interceptor(self, experiment: ChaosExperiment) -> None:
        """Remove chaos interceptor."""
        key = f"{experiment.target_service}_{experiment.experiment_type.value}"
        self._interceptors.pop(key, None)

    def _maybe_raise(self, error_rate: float, message: str) -> None:
        """Maybe raise an error based on error rate."""
        if random.random() < error_rate:
            raise RuntimeError(f"Chaos: {message}")

    def inject_chaos(self, service_name: str, experiment_type: ChaosExperimentType) -> None:
        """
        Inject chaos if there's an active experiment for this service/type.

        Call this from your code at points where you want chaos injection.
        """
        key = f"{service_name}_{experiment_type.value}"
        interceptor = self._interceptors.get(key)
        if interceptor:
            interceptor()

    def get_active_experiments(self) -> List[ChaosExperiment]:
        """Get all active experiments."""
        with self._lock:
            return [self._experiments[eid] for eid in self._active_experiments]

    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return None
            return self._experiments[experiment_id].results


def chaos_injection_point(engine: ChaosEngine, service_name: str, experiment_type: ChaosExperimentType):
    """Decorator to add chaos injection point to a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            engine.inject_chaos(service_name, experiment_type)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Connection Pooling (#I19)
# =============================================================================

T = TypeVar('T')


@dataclass
class PooledConnection:
    """Wrapper for pooled connections."""
    connection: Any
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_healthy: bool = True


class ConnectionPool:
    """
    Generic connection pool implementation.

    Features:
    - Min/max pool size
    - Connection health checking
    - Automatic connection recycling
    - Connection timeout handling
    - Pool statistics
    """

    def __init__(
        self,
        factory: Callable[[], T],
        min_size: int = 2,
        max_size: int = 10,
        max_idle_seconds: int = 300,
        health_check: Optional[Callable[[T], bool]] = None,
        cleanup: Optional[Callable[[T], None]] = None
    ):
        """
        Initialize connection pool.

        Args:
            factory: Function to create new connections
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_seconds: Max idle time before recycling
            health_check: Function to check connection health
            cleanup: Function to cleanup connections
        """
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_seconds = max_idle_seconds
        self.health_check = health_check
        self.cleanup = cleanup

        self._pool: Queue[PooledConnection] = Queue()
        self._in_use: Set[int] = set()
        self._total_created = 0
        self._lock = threading.Lock()
        self._stats = {
            "checkouts": 0,
            "checkins": 0,
            "created": 0,
            "recycled": 0,
            "health_check_failures": 0
        }

        # Initialize minimum connections
        for _ in range(min_size):
            self._create_connection()

    def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        conn = self.factory()
        pooled = PooledConnection(
            connection=conn,
            created_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc)
        )
        self._pool.put(pooled)
        self._total_created += 1
        self._stats["created"] += 1
        return pooled

    def acquire(self, timeout: float = 30.0) -> T:
        """
        Acquire a connection from the pool.

        Args:
            timeout: Timeout in seconds

        Returns:
            Connection object

        Raises:
            TimeoutError if no connection available
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                pooled = self._pool.get(timeout=1.0)

                # Check if connection is still healthy
                if not self._is_connection_healthy(pooled):
                    self._recycle_connection(pooled)
                    continue

                # Check if connection has been idle too long
                idle_time = (datetime.now(timezone.utc) - pooled.last_used).total_seconds()
                if idle_time > self.max_idle_seconds:
                    self._recycle_connection(pooled)
                    continue

                pooled.last_used = datetime.now(timezone.utc)
                pooled.use_count += 1

                with self._lock:
                    self._in_use.add(id(pooled.connection))
                    self._stats["checkouts"] += 1

                return pooled.connection

            except queue.Empty:
                # Pool empty, try to create new connection
                logger.debug("Connection pool empty, attempting to create new connection")
                with self._lock:
                    if self._total_created < self.max_size:
                        self._create_connection()
            except Exception as e:
                logger.error(f"Failed to acquire connection from pool: {e}", exc_info=True)
                raise

        raise TimeoutError("Could not acquire connection from pool")

    def release(self, connection: T) -> None:
        """Release a connection back to the pool."""
        with self._lock:
            self._in_use.discard(id(connection))
            self._stats["checkins"] += 1

        pooled = PooledConnection(
            connection=connection,
            created_at=datetime.now(timezone.utc),  # Reset for simplicity
            last_used=datetime.now(timezone.utc)
        )
        self._pool.put(pooled)

    def _is_connection_healthy(self, pooled: PooledConnection) -> bool:
        """Check if a connection is healthy."""
        if not self.health_check:
            return True

        try:
            is_healthy = self.health_check(pooled.connection)
            if not is_healthy:
                self._stats["health_check_failures"] += 1
            return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for connection: {e}")
            self._stats["health_check_failures"] += 1
            return False

    def _recycle_connection(self, pooled: PooledConnection) -> None:
        """Recycle an unhealthy or stale connection."""
        if self.cleanup:
            try:
                self.cleanup(pooled.connection)
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}")

        self._stats["recycled"] += 1

        # Create replacement if below min size
        # Note: Don't decrement _total_created during recycle - only on actual removal
        # The _create_connection will increment it when creating replacement
        with self._lock:
            # Only decrement if we're actually removing without replacement
            current_pool_size = self._pool.qsize() + len(self._in_use)
            if current_pool_size < self.min_size:
                # Creating replacement, so net change to _total_created is 0
                self._create_connection()
            else:
                # Actually removing, decrement the count
                self._total_created -= 1

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self._stats,
                "pool_size": self._pool.qsize(),
                "in_use": len(self._in_use),
                "total_created": self._total_created
            }

    def close(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                if self.cleanup:
                    self.cleanup(pooled.connection)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error closing connection in pool: {e}")
                break


class PooledConnectionContext:
    """Context manager for automatic connection release."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self.connection: Optional[Any] = None

    def __enter__(self) -> Any:
        self.connection = self.pool.acquire()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.connection:
            self.pool.release(self.connection)


# =============================================================================
# Cache Warming Strategies (#I20)
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class CacheWarmer:
    """
    Cache warming and management system.

    Features:
    - Proactive cache warming
    - TTL-based expiration
    - LRU eviction
    - Warm-up strategies
    - Cache statistics
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl_seconds: int = 3600
    ):
        """Initialize cache warmer."""
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._warming_tasks: List[Callable[[], Dict[str, Any]]] = []
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "warm_ups": 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check TTL
            age = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
            if age > entry.ttl_seconds:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set value in cache."""
        ttl = ttl_seconds or self.default_ttl

        with self._lock:
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(timezone.utc),
                ttl_seconds=ttl
            )

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at
        )

        del self._cache[lru_key]
        self._stats["evictions"] += 1

    def add_warming_task(self, task: Callable[[], Dict[str, Any]]) -> None:
        """
        Add a cache warming task.

        Task should return a dict of {key: value} to cache.
        """
        self._warming_tasks.append(task)

    def warm_cache(self, ttl_seconds: Optional[int] = None) -> int:
        """
        Execute all warming tasks and populate cache.

        Returns number of entries added.
        """
        entries_added = 0

        for task in self._warming_tasks:
            try:
                data = task()
                for key, value in data.items():
                    self.set(key, value, ttl_seconds)
                    entries_added += 1
            except Exception as e:
                logger.error(f"Cache warming task failed: {e}")

        self._stats["warm_ups"] += 1
        logger.info(f"Cache warmed with {entries_added} entries")
        return entries_added

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl_seconds: Optional[int] = None
    ) -> Any:
        """Get from cache or compute and set."""
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl_seconds)
        return value

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                **self._stats,
                "size": len(self._cache),
                "hit_rate": hit_rate,
                "max_size": self.max_size
            }

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()


def cached(cache: CacheWarmer, ttl_seconds: Optional[int] = None):
    """Decorator for automatic caching."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            return cache.get_or_set(key, lambda: func(*args, **kwargs), ttl_seconds)
        return wrapper
    return decorator


# =============================================================================
# Request Tracing (#I21)
# =============================================================================

class RequestTracer:
    """
    Distributed request tracing system.

    Features:
    - Trace context propagation
    - Span creation and management
    - Cross-service tracing
    - Trace sampling
    - Trace export
    """

    def __init__(
        self,
        service_name: str,
        sample_rate: float = 1.0
    ):
        """Initialize request tracer."""
        self.service_name = service_name
        self.sample_rate = sample_rate
        self._traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self._current_trace: Optional[str] = None
        self._current_span: Optional[str] = None
        self._lock = threading.Lock()
        self._exporters: List[Callable[[List[TraceSpan]], None]] = []

    def start_trace(self, operation_name: str) -> TraceSpan:
        """Start a new trace."""
        # Sampling decision
        if random.random() > self.sample_rate:
            return self._create_noop_span(operation_name)

        trace_id = self._generate_id()
        span_id = self._generate_id()

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(timezone.utc)
        )

        with self._lock:
            self._traces[trace_id].append(span)
            self._current_trace = trace_id
            self._current_span = span_id

        return span

    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[TraceSpan] = None
    ) -> TraceSpan:
        """Start a new span within current trace."""
        with self._lock:
            trace_id = parent_span.trace_id if parent_span else self._current_trace
            parent_id = parent_span.span_id if parent_span else self._current_span

        if not trace_id:
            # No active trace, start a new one
            return self.start_trace(operation_name)

        span_id = self._generate_id()

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(timezone.utc)
        )

        with self._lock:
            self._traces[trace_id].append(span)
            self._current_span = span_id

        return span

    def end_span(self, span: TraceSpan, status: str = "ok") -> None:
        """End a span."""
        span.end_time = datetime.now(timezone.utc)
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status

        # If this was the root span, export the trace
        if span.parent_span_id is None:
            self._export_trace(span.trace_id)

    def add_span_tag(self, span: TraceSpan, key: str, value: str) -> None:
        """Add a tag to a span."""
        span.tags[key] = value

    def add_span_log(self, span: TraceSpan, message: str, fields: Optional[Dict[str, Any]] = None) -> None:
        """Add a log entry to a span."""
        span.logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "fields": fields or {}
        })

    def add_exporter(self, exporter: Callable[[List[TraceSpan]], None]) -> None:
        """Add a trace exporter."""
        self._exporters.append(exporter)

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            return self._traces.get(trace_id, [])

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return hashlib.md5(
            f"{random.random()}{time.time()}".encode()
        ).hexdigest()[:16]

    def _create_noop_span(self, operation_name: str) -> TraceSpan:
        """Create a no-op span for sampled-out traces."""
        return TraceSpan(
            trace_id="",
            span_id="",
            parent_span_id=None,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(timezone.utc)
        )

    def _export_trace(self, trace_id: str) -> None:
        """Export a completed trace."""
        with self._lock:
            spans = self._traces.get(trace_id, [])

        for exporter in self._exporters:
            try:
                exporter(spans)
            except Exception as e:
                logger.error(f"Trace export failed: {e}")

    def inject_context(self, span: TraceSpan) -> Dict[str, str]:
        """Inject trace context for propagation to other services."""
        return {
            "trace-id": span.trace_id,
            "span-id": span.span_id,
            "sampled": "1" if span.trace_id else "0"
        }

    def extract_context(self, headers: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """Extract trace context from propagated headers."""
        trace_id = headers.get("trace-id")
        span_id = headers.get("span-id")

        if trace_id and span_id:
            return trace_id, span_id
        return None


class SpanContext:
    """Context manager for automatic span management."""

    def __init__(self, tracer: RequestTracer, operation_name: str, parent: Optional[TraceSpan] = None):
        self.tracer = tracer
        self.operation_name = operation_name
        self.parent = parent
        self.span: Optional[TraceSpan] = None

    def __enter__(self) -> TraceSpan:
        self.span = self.tracer.start_span(self.operation_name, self.parent)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.span:
            status = "error" if exc_type else "ok"
            if exc_type:
                self.tracer.add_span_log(self.span, f"Exception: {exc_val}")
            self.tracer.end_span(self.span, status)


def traced(tracer: RequestTracer, operation_name: Optional[str] = None):
    """Decorator for automatic span creation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with SpanContext(tracer, op_name) as span:
                tracer.add_span_tag(span, "function", func.__name__)
                return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Blue-Green Deployment Support (#I22)
# =============================================================================

@dataclass
class DeploymentInstance:
    """Deployment instance information."""
    color: DeploymentColor
    version: str
    health_endpoint: str
    weight: float = 0.0  # Traffic weight (0-1)
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None


class BlueGreenDeployer:
    """
    Blue-green deployment manager.

    Features:
    - Zero-downtime deployments
    - Traffic shifting
    - Health checking
    - Rollback support
    - Canary deployments
    """

    def __init__(self, service_name: str):
        """Initialize blue-green deployer."""
        self.service_name = service_name
        self._deployments: Dict[DeploymentColor, DeploymentInstance] = {}
        self._active_color: Optional[DeploymentColor] = None
        self._lock = threading.Lock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False

    def register_deployment(
        self,
        color: DeploymentColor,
        version: str,
        health_endpoint: str
    ) -> None:
        """Register a deployment instance."""
        instance = DeploymentInstance(
            color=color,
            version=version,
            health_endpoint=health_endpoint
        )

        with self._lock:
            self._deployments[color] = instance

        logger.info(f"Registered deployment: {color.value} (v{version})")

    def start_health_checks(self, interval_seconds: int = 10) -> None:
        """Start background health checking."""
        self._running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._health_check_thread.start()

    def stop_health_checks(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)

    def _health_check_loop(self, interval: int) -> None:
        """Background health check loop."""
        while self._running:
            self._perform_health_checks()
            time.sleep(interval)

    def _perform_health_checks(self) -> None:
        """Check health of all deployments."""
        with self._lock:
            for color, instance in self._deployments.items():
                # In production, this would make HTTP call to health endpoint
                # For now, simulate health check
                instance.is_healthy = True  # Assume healthy
                instance.last_health_check = datetime.now(timezone.utc)

    def set_active(self, color: DeploymentColor) -> bool:
        """Set the active deployment (100% traffic)."""
        with self._lock:
            if color not in self._deployments:
                return False

            instance = self._deployments[color]
            if not instance.is_healthy:
                logger.warning(f"Cannot activate unhealthy deployment: {color.value}")
                return False

            # Set traffic weights
            for c, inst in self._deployments.items():
                inst.weight = 1.0 if c == color else 0.0

            self._active_color = color

        logger.info(f"Active deployment set to: {color.value}")
        return True

    def shift_traffic(
        self,
        target_color: DeploymentColor,
        percentage: float
    ) -> bool:
        """
        Shift a percentage of traffic to target deployment.

        Args:
            target_color: Deployment to shift traffic to
            percentage: Percentage of traffic (0-100)

        Returns:
            True if successful
        """
        if percentage < 0 or percentage > 100:
            return False

        with self._lock:
            if target_color not in self._deployments:
                return False

            target_weight = percentage / 100
            other_color = (
                DeploymentColor.GREEN
                if target_color == DeploymentColor.BLUE
                else DeploymentColor.BLUE
            )

            self._deployments[target_color].weight = target_weight
            if other_color in self._deployments:
                self._deployments[other_color].weight = 1.0 - target_weight

        logger.info(f"Traffic shifted: {target_color.value} = {percentage}%")
        return True

    def get_deployment_for_request(self) -> Optional[DeploymentInstance]:
        """
        Get deployment to route a request to based on traffic weights.

        Returns:
            Selected deployment instance
        """
        with self._lock:
            healthy = [
                (color, inst)
                for color, inst in self._deployments.items()
                if inst.is_healthy and inst.weight > 0
            ]

        if not healthy:
            return None

        # Weighted random selection
        total_weight = sum(inst.weight for _, inst in healthy)
        rand = random.random() * total_weight

        cumulative = 0.0
        for color, inst in healthy:
            cumulative += inst.weight
            if rand <= cumulative:
                return inst

        return healthy[-1][1]

    def rollback(self) -> bool:
        """Rollback to the inactive deployment."""
        with self._lock:
            if not self._active_color:
                return False

            other_color = (
                DeploymentColor.GREEN
                if self._active_color == DeploymentColor.BLUE
                else DeploymentColor.BLUE
            )

            if other_color not in self._deployments:
                return False

        logger.warning(f"Rolling back from {self._active_color.value} to {other_color.value}")
        return self.set_active(other_color)

    def get_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        with self._lock:
            return {
                "active": self._active_color.value if self._active_color else None,
                "deployments": {
                    color.value: {
                        "version": inst.version,
                        "weight": inst.weight,
                        "is_healthy": inst.is_healthy,
                        "last_health_check": inst.last_health_check.isoformat() if inst.last_health_check else None
                    }
                    for color, inst in self._deployments.items()
                }
            }


# =============================================================================
# Feature Flags System (#I23)
# =============================================================================

class FeatureFlagManager:
    """
    Feature flag management system.

    Features:
    - Boolean flags
    - Percentage rollouts
    - User segment targeting
    - A/B testing support
    - Flag analytics
    """

    def __init__(self):
        """Initialize feature flag manager."""
        self._flags: Dict[str, FeatureFlag] = {}
        self._user_overrides: Dict[str, Dict[str, bool]] = defaultdict(dict)  # user_id -> flag -> value
        self._lock = threading.Lock()
        self._evaluation_log: List[Dict[str, Any]] = []

    def create_flag(
        self,
        name: str,
        default_value: bool = False,
        description: str = "",
        status: FeatureFlagStatus = FeatureFlagStatus.DISABLED
    ) -> FeatureFlag:
        """Create a new feature flag."""
        flag = FeatureFlag(
            name=name,
            status=status,
            default_value=default_value,
            description=description
        )

        with self._lock:
            self._flags[name] = flag

        logger.info(f"Feature flag created: {name}")
        return flag

    def set_flag_status(self, name: str, status: FeatureFlagStatus) -> bool:
        """Set flag status."""
        with self._lock:
            if name not in self._flags:
                return False

            self._flags[name].status = status
            self._flags[name].updated_at = datetime.now(timezone.utc)

        logger.info(f"Feature flag {name} status set to: {status.value}")
        return True

    def set_percentage_rollout(self, name: str, percentage: float) -> bool:
        """Set percentage rollout for a flag."""
        if percentage < 0 or percentage > 100:
            return False

        with self._lock:
            if name not in self._flags:
                return False

            self._flags[name].status = FeatureFlagStatus.PERCENTAGE_ROLLOUT
            self._flags[name].percentage = percentage
            self._flags[name].updated_at = datetime.now(timezone.utc)

        logger.info(f"Feature flag {name} set to {percentage}% rollout")
        return True

    def set_user_segments(self, name: str, segments: List[str]) -> bool:
        """Set user segments for a flag."""
        with self._lock:
            if name not in self._flags:
                return False

            self._flags[name].status = FeatureFlagStatus.USER_SEGMENT
            self._flags[name].user_segments = segments
            self._flags[name].updated_at = datetime.now(timezone.utc)

        logger.info(f"Feature flag {name} set for segments: {segments}")
        return True

    def set_user_override(self, user_id: str, flag_name: str, value: bool) -> None:
        """Set a user-specific override for a flag."""
        with self._lock:
            self._user_overrides[user_id][flag_name] = value

    def is_enabled(
        self,
        name: str,
        user_id: Optional[str] = None,
        user_segments: Optional[List[str]] = None,
        default: bool = False
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            name: Flag name
            user_id: Optional user ID for percentage/override checks
            user_segments: Optional user segments for targeting
            default: Default value if flag not found

        Returns:
            True if flag is enabled
        """
        with self._lock:
            # Check user override first
            if user_id and user_id in self._user_overrides:
                if name in self._user_overrides[user_id]:
                    result = self._user_overrides[user_id][name]
                    self._log_evaluation(name, user_id, result, "user_override")
                    return result

            if name not in self._flags:
                return default

            flag = self._flags[name]

        # Evaluate based on status
        if flag.status == FeatureFlagStatus.ENABLED:
            result = True
            reason = "enabled"
        elif flag.status == FeatureFlagStatus.DISABLED:
            result = False
            reason = "disabled"
        elif flag.status == FeatureFlagStatus.PERCENTAGE_ROLLOUT:
            # Use user_id for consistent bucketing
            if user_id:
                bucket = int(hashlib.md5(f"{name}{user_id}".encode()).hexdigest(), 16) % 100
            else:
                bucket = random.randint(0, 99)
            result = bucket < flag.percentage
            reason = f"percentage_rollout_{flag.percentage}"
        elif flag.status == FeatureFlagStatus.USER_SEGMENT:
            result = bool(
                user_segments and
                any(seg in flag.user_segments for seg in user_segments)
            )
            reason = "user_segment"
        else:
            result = flag.default_value
            reason = "default"

        self._log_evaluation(name, user_id, result, reason)
        return result

    def _log_evaluation(
        self,
        flag_name: str,
        user_id: Optional[str],
        result: bool,
        reason: str
    ) -> None:
        """Log flag evaluation for analytics."""
        self._evaluation_log.append({
            "flag": flag_name,
            "user_id": user_id,
            "result": result,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Keep log bounded
        if len(self._evaluation_log) > 10000:
            self._evaluation_log = self._evaluation_log[-5000:]

    def get_flag_analytics(self, name: str) -> Dict[str, Any]:
        """Get analytics for a flag."""
        evaluations = [e for e in self._evaluation_log if e["flag"] == name]

        if not evaluations:
            return {"total_evaluations": 0}

        enabled_count = sum(1 for e in evaluations if e["result"])

        return {
            "total_evaluations": len(evaluations),
            "enabled_count": enabled_count,
            "disabled_count": len(evaluations) - enabled_count,
            "enabled_rate": enabled_count / len(evaluations),
            "by_reason": self._group_by_key(evaluations, "reason")
        }

    def _group_by_key(self, items: List[Dict], key: str) -> Dict[str, int]:
        """Group items by key and count."""
        counts: Dict[str, int] = defaultdict(int)
        for item in items:
            counts[item[key]] += 1
        return dict(counts)

    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all flags and their status."""
        with self._lock:
            return {
                name: {
                    "status": flag.status.value,
                    "default_value": flag.default_value,
                    "percentage": flag.percentage,
                    "user_segments": flag.user_segments,
                    "description": flag.description,
                    "updated_at": flag.updated_at.isoformat()
                }
                for name, flag in self._flags.items()
            }


def feature_flag(
    manager: FeatureFlagManager,
    flag_name: str,
    default: bool = False
):
    """Decorator for feature-flagged functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if manager.is_enabled(flag_name, default=default):
                return func(*args, **kwargs)
            else:
                return None
        return wrapper
    return decorator


# =============================================================================
# Automatic Backup System for Audit Logs
# =============================================================================

class AuditLogBackupManager:
    """
    Automatic backup system for critical audit logs.

    Backs up:
    - logs/audit.jsonl
    - logs/trades.jsonl
    - logs/decisions.jsonl

    To: logs/backup/<timestamp>/
    """

    def __init__(
        self,
        log_dir: str = "logs",
        backup_dir: str = "logs/backup",
        retention_days: int = 90,
    ):
        """
        Initialize audit log backup manager.

        Args:
            log_dir: Directory containing audit logs
            backup_dir: Directory for backups
            retention_days: Days to retain backups (0 = forever)
        """
        from pathlib import Path

        self.log_dir = Path(log_dir)
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days

        # Critical audit log files to backup
        self.critical_logs = [
            "audit.jsonl",
            "trades.jsonl",
            "decisions.jsonl",
        ]

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup tracking
        self._last_backup_time: Optional[datetime] = None
        self._backup_count: int = 0
        self._lock = threading.Lock()

        logger.info(f"AuditLogBackupManager initialized: {backup_dir}")

    def backup_now(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Perform immediate backup of all critical audit logs.

        Args:
            reason: Reason for backup (e.g., "daily", "manual", "shutdown")

        Returns:
            Backup result with files backed up and paths
        """
        import shutil
        from pathlib import Path

        with self._lock:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

            # Create timestamped backup directory
            backup_path = self.backup_dir / timestamp_str
            backup_path.mkdir(parents=True, exist_ok=True)

            results = {
                "timestamp": timestamp.isoformat(),
                "reason": reason,
                "backup_path": str(backup_path),
                "files_backed_up": [],
                "files_missing": [],
                "bytes_total": 0,
                "success": True,
            }

            for log_file in self.critical_logs:
                source_path = self.log_dir / log_file

                if source_path.exists():
                    dest_path = backup_path / log_file
                    try:
                        shutil.copy2(source_path, dest_path)
                        file_size = dest_path.stat().st_size
                        results["files_backed_up"].append({
                            "file": log_file,
                            "size_bytes": file_size,
                            "path": str(dest_path),
                        })
                        results["bytes_total"] += file_size
                        logger.info(f"Backed up {log_file} ({file_size} bytes)")
                    except Exception as e:
                        logger.error(f"Failed to backup {log_file}: {e}")
                        results["files_missing"].append({
                            "file": log_file,
                            "error": str(e),
                        })
                        results["success"] = False
                else:
                    logger.warning(f"Audit log not found for backup: {source_path}")
                    results["files_missing"].append({
                        "file": log_file,
                        "error": "File not found",
                    })

            # Write backup manifest
            manifest_path = backup_path / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(results, f, indent=2)

            self._last_backup_time = timestamp
            self._backup_count += 1

            logger.info(
                f"Backup complete: {len(results['files_backed_up'])} files, "
                f"{results['bytes_total']} bytes to {backup_path}"
            )

            return results

    def cleanup_old_backups(self) -> int:
        """
        Remove backups older than retention period.

        Returns:
            Number of backup directories removed
        """
        if self.retention_days <= 0:
            return 0

        import shutil
        from pathlib import Path

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0

        for backup_dir in self.backup_dir.iterdir():
            if not backup_dir.is_dir():
                continue

            try:
                # Parse timestamp from directory name
                dir_timestamp = datetime.strptime(backup_dir.name, "%Y%m%d_%H%M%S")

                if dir_timestamp < cutoff:
                    shutil.rmtree(backup_dir)
                    removed_count += 1
                    logger.info(f"Removed old backup: {backup_dir}")
            except ValueError:
                # Not a backup directory (wrong name format)
                continue
            except Exception as e:
                logger.error(f"Error removing old backup {backup_dir}: {e}")

        return removed_count

    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup system status."""
        from pathlib import Path

        backups = []
        total_size = 0

        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                try:
                    # Try to parse as backup directory
                    datetime.strptime(backup_dir.name, "%Y%m%d_%H%M%S")

                    dir_size = sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())
                    backups.append({
                        "name": backup_dir.name,
                        "size_bytes": dir_size,
                    })
                    total_size += dir_size
                except ValueError:
                    continue

        return {
            "backup_count": len(backups),
            "total_backup_count": self._backup_count,
            "last_backup_time": self._last_backup_time.isoformat() if self._last_backup_time else None,
            "total_size_bytes": total_size,
            "retention_days": self.retention_days,
            "backup_dir": str(self.backup_dir),
            "recent_backups": sorted(backups, key=lambda x: x["name"], reverse=True)[:5],
        }

    def schedule_daily_backup(self, hour: int = 0, minute: int = 0) -> None:
        """
        Schedule daily backup at specified time.

        Args:
            hour: Hour of day (0-23)
            minute: Minute of hour (0-59)
        """
        def backup_loop():
            while True:
                now = datetime.now()

                # Calculate next backup time
                next_backup = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_backup <= now:
                    next_backup += timedelta(days=1)

                # Sleep until backup time
                sleep_seconds = (next_backup - now).total_seconds()
                logger.info(f"Next scheduled backup in {sleep_seconds/3600:.1f} hours")
                time.sleep(sleep_seconds)

                # Perform backup
                try:
                    self.backup_now(reason="daily")
                    self.cleanup_old_backups()
                except Exception as e:
                    logger.error(f"Scheduled backup failed: {e}")

        backup_thread = threading.Thread(target=backup_loop, daemon=True, name="audit-backup")
        backup_thread.start()
        logger.info(f"Scheduled daily backup at {hour:02d}:{minute:02d}")


# =============================================================================
# P3: Health Check Endpoint Customization
# =============================================================================

class HealthCheckStatus(Enum):
    """Health check result status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthCheckStatus
    message: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HealthCheckConfig:
    """Configuration for a health check."""
    name: str
    check_func: Callable[[], bool]
    timeout_seconds: float = 5.0
    critical: bool = True  # If False, failure won't mark system unhealthy
    interval_seconds: int = 30
    description: str = ""


class HealthCheckManager:
    """
    Customizable health check endpoint manager.

    P3 Enhancement: Provides:
    - Custom health checks registration
    - Configurable timeouts
    - Critical vs non-critical checks
    - Detailed health reports
    - Health history tracking
    """

    def __init__(self, service_name: str):
        """Initialize health check manager."""
        self.service_name = service_name
        self._checks: Dict[str, HealthCheckConfig] = {}
        self._results_history: List[Dict[str, HealthCheckResult]] = []
        self._lock = threading.Lock()
        self._max_history = 100

    def register_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        interval_seconds: int = 30,
        description: str = "",
    ) -> None:
        """
        Register a custom health check.

        Args:
            name: Unique name for the check
            check_func: Function that returns True if healthy
            timeout_seconds: Timeout for the check
            critical: Whether failure marks system unhealthy
            interval_seconds: Suggested check interval
            description: Human-readable description
        """
        config = HealthCheckConfig(
            name=name,
            check_func=check_func,
            timeout_seconds=timeout_seconds,
            critical=critical,
            interval_seconds=interval_seconds,
            description=description,
        )
        with self._lock:
            self._checks[name] = config
        logger.info(f"Registered health check: {name} (critical={critical})")

    def unregister_check(self, name: str) -> bool:
        """Remove a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                return True
            return False

    def run_check(self, name: str) -> HealthCheckResult:
        """
        Run a single health check.

        Args:
            name: Name of the check to run

        Returns:
            HealthCheckResult with status and details
        """
        with self._lock:
            if name not in self._checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthCheckStatus.UNHEALTHY,
                    message=f"Unknown health check: {name}",
                )
            config = self._checks[name]

        start_time = time.time()
        try:
            # Run check with timeout
            result = self._run_with_timeout(config.check_func, config.timeout_seconds)
            duration_ms = (time.time() - start_time) * 1000

            if result:
                return HealthCheckResult(
                    name=name,
                    status=HealthCheckStatus.HEALTHY,
                    message="Check passed",
                    duration_ms=duration_ms,
                    metadata={"critical": config.critical},
                )
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthCheckStatus.UNHEALTHY,
                    message="Check failed",
                    duration_ms=duration_ms,
                    metadata={"critical": config.critical},
                )

        except TimeoutError:
            return HealthCheckResult(
                name=name,
                status=HealthCheckStatus.UNHEALTHY,
                message=f"Check timed out after {config.timeout_seconds}s",
                duration_ms=config.timeout_seconds * 1000,
                metadata={"critical": config.critical, "timeout": True},
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthCheckStatus.UNHEALTHY,
                message=f"Check error: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"critical": config.critical, "error": str(e)},
            )

    def _run_with_timeout(self, func: Callable[[], bool], timeout: float) -> bool:
        """Run a function with timeout."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Check timed out after {timeout}s")

    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.

        Returns:
            Complete health report
        """
        results: Dict[str, HealthCheckResult] = {}
        overall_status = HealthCheckStatus.HEALTHY

        with self._lock:
            check_names = list(self._checks.keys())

        for name in check_names:
            result = self.run_check(name)
            results[name] = result

            # Determine overall status
            if result.status == HealthCheckStatus.UNHEALTHY:
                config = self._checks.get(name)
                if config and config.critical:
                    overall_status = HealthCheckStatus.UNHEALTHY
                elif overall_status == HealthCheckStatus.HEALTHY:
                    overall_status = HealthCheckStatus.DEGRADED

        # Store in history
        with self._lock:
            self._results_history.append(results)
            if len(self._results_history) > self._max_history:
                self._results_history = self._results_history[-self._max_history:]

        return {
            "service": self.service_name,
            "status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                name: {
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "critical": r.metadata.get("critical", True),
                }
                for name, r in results.items()
            },
        }

    def get_health_endpoint_response(self) -> Tuple[int, Dict[str, Any]]:
        """
        Get health check response suitable for HTTP endpoint.

        Returns:
            Tuple of (HTTP status code, response body)
        """
        report = self.run_all_checks()
        status = report["status"]

        if status == "healthy":
            return (200, report)
        elif status == "degraded":
            return (200, report)  # Still OK but with warnings
        else:
            return (503, report)  # Service unavailable

    def get_liveness_response(self) -> Tuple[int, Dict[str, Any]]:
        """
        Simple liveness check (is the service running?).

        Returns:
            Tuple of (HTTP status code, response body)
        """
        return (200, {
            "service": self.service_name,
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_readiness_response(self) -> Tuple[int, Dict[str, Any]]:
        """
        Readiness check (is the service ready to receive traffic?).

        Returns:
            Tuple of (HTTP status code, response body)
        """
        report = self.run_all_checks()
        status = report["status"]

        ready = status in ("healthy", "degraded")
        return (
            200 if ready else 503,
            {
                "service": self.service_name,
                "ready": ready,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


# =============================================================================
# P3: Metrics Export Formats (Prometheus, StatsD)
# =============================================================================

class MetricsExportFormat(Enum):
    """Supported metrics export formats."""
    PROMETHEUS = "prometheus"
    STATSD = "statsd"
    JSON = "json"
    INFLUXDB = "influxdb"


class MetricsExporter:
    """
    Export metrics in various formats.

    P3 Enhancement: Supports multiple formats:
    - Prometheus text format
    - StatsD format
    - JSON format
    - InfluxDB line protocol
    """

    def __init__(self, service_name: str, apm: Optional[APMCollector] = None):
        """
        Initialize metrics exporter.

        Args:
            service_name: Name of the service
            apm: APMCollector to export metrics from
        """
        self.service_name = service_name
        self.apm = apm
        self._custom_metrics: Dict[str, Tuple[str, float, Dict[str, str]]] = {}
        self._lock = threading.Lock()

    def add_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a custom metric for export."""
        with self._lock:
            self._custom_metrics[name] = (metric_type, value, labels or {})

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        lines.append(f"# Metrics for {self.service_name}")
        lines.append(f"# Generated at {datetime.now(timezone.utc).isoformat()}")
        lines.append("")

        # Export APM metrics if available
        if self.apm:
            metrics = self.apm.get_all_metrics()

            # Counters
            for name, value in metrics.get("counters", {}).items():
                metric_name = self._sanitize_name(name)
                lines.append(f"# TYPE {metric_name}_total counter")
                lines.append(f'{metric_name}_total{{service="{self.service_name}"}} {value}')

            # Gauges
            for name, value in metrics.get("gauges", {}).items():
                metric_name = self._sanitize_name(name)
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f'{metric_name}{{service="{self.service_name}"}} {value}')

            # Histograms (export summary stats)
            for name, stats in metrics.get("histograms", {}).items():
                metric_name = self._sanitize_name(name)
                if stats.get("count", 0) > 0:
                    lines.append(f"# TYPE {metric_name} summary")
                    lines.append(f'{metric_name}_count{{service="{self.service_name}"}} {stats["count"]}')
                    lines.append(f'{metric_name}_sum{{service="{self.service_name}"}} {stats.get("mean", 0) * stats["count"]}')
                    lines.append(f'{metric_name}{{service="{self.service_name}",quantile="0.5"}} {stats.get("median", 0)}')
                    lines.append(f'{metric_name}{{service="{self.service_name}",quantile="0.95"}} {stats.get("p95", 0)}')
                    lines.append(f'{metric_name}{{service="{self.service_name}",quantile="0.99"}} {stats.get("p99", 0)}')

            # Transaction timings
            for name, stats in metrics.get("transactions", {}).items():
                metric_name = self._sanitize_name(f"transaction_{name}_duration_ms")
                if stats.get("count", 0) > 0:
                    lines.append(f"# TYPE {metric_name} summary")
                    lines.append(f'{metric_name}_count{{service="{self.service_name}"}} {stats["count"]}')
                    lines.append(f'{metric_name}{{service="{self.service_name}",quantile="0.5"}} {stats.get("median_ms", 0)}')
                    lines.append(f'{metric_name}{{service="{self.service_name}",quantile="0.95"}} {stats.get("p95_ms", 0)}')
                    lines.append(f'{metric_name}{{service="{self.service_name}",quantile="0.99"}} {stats.get("p99_ms", 0)}')

        # Export custom metrics
        with self._lock:
            for name, (metric_type, value, labels) in self._custom_metrics.items():
                metric_name = self._sanitize_name(name)
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                if label_str:
                    label_str = f'service="{self.service_name}",' + label_str
                else:
                    label_str = f'service="{self.service_name}"'

                lines.append(f"# TYPE {metric_name} {metric_type}")
                lines.append(f"{metric_name}{{{label_str}}} {value}")

        return "\n".join(lines)

    def export_statsd(self) -> List[str]:
        """
        Export metrics in StatsD format.

        Returns:
            List of StatsD-formatted metric strings
        """
        messages = []
        prefix = self._sanitize_name(self.service_name)

        if self.apm:
            metrics = self.apm.get_all_metrics()

            # Counters
            for name, value in metrics.get("counters", {}).items():
                metric_name = self._sanitize_name(name)
                messages.append(f"{prefix}.{metric_name}:{value}|c")

            # Gauges
            for name, value in metrics.get("gauges", {}).items():
                metric_name = self._sanitize_name(name)
                messages.append(f"{prefix}.{metric_name}:{value}|g")

            # Histograms as timings
            for name, stats in metrics.get("histograms", {}).items():
                metric_name = self._sanitize_name(name)
                if stats.get("count", 0) > 0:
                    messages.append(f"{prefix}.{metric_name}.mean:{stats.get('mean', 0)}|ms")
                    messages.append(f"{prefix}.{metric_name}.p95:{stats.get('p95', 0)}|ms")
                    messages.append(f"{prefix}.{metric_name}.count:{stats['count']}|c")

            # Transactions
            for name, stats in metrics.get("transactions", {}).items():
                metric_name = self._sanitize_name(f"transaction_{name}")
                if stats.get("count", 0) > 0:
                    messages.append(f"{prefix}.{metric_name}.mean:{stats.get('mean_ms', 0)}|ms")
                    messages.append(f"{prefix}.{metric_name}.p95:{stats.get('p95_ms', 0)}|ms")
                    messages.append(f"{prefix}.{metric_name}.count:{stats['count']}|c")

        # Custom metrics
        with self._lock:
            for name, (metric_type, value, _) in self._custom_metrics.items():
                metric_name = self._sanitize_name(name)
                type_char = {"counter": "c", "gauge": "g", "timing": "ms"}.get(metric_type, "g")
                messages.append(f"{prefix}.{metric_name}:{value}|{type_char}")

        return messages

    def export_json(self) -> Dict[str, Any]:
        """
        Export metrics in JSON format.

        Returns:
            Dictionary with all metrics
        """
        result = {
            "service": self.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
        }

        if self.apm:
            result["metrics"] = self.apm.get_all_metrics()

        with self._lock:
            result["custom_metrics"] = {
                name: {"type": t, "value": v, "labels": l}
                for name, (t, v, l) in self._custom_metrics.items()
            }

        return result

    def export_influxdb(self) -> List[str]:
        """
        Export metrics in InfluxDB line protocol.

        Returns:
            List of InfluxDB line protocol strings
        """
        lines = []
        timestamp_ns = int(time.time() * 1e9)

        if self.apm:
            metrics = self.apm.get_all_metrics()

            # Counters
            for name, value in metrics.get("counters", {}).items():
                lines.append(f"counters,service={self.service_name},name={name} value={value} {timestamp_ns}")

            # Gauges
            for name, value in metrics.get("gauges", {}).items():
                lines.append(f"gauges,service={self.service_name},name={name} value={value} {timestamp_ns}")

            # Histograms
            for name, stats in metrics.get("histograms", {}).items():
                if stats.get("count", 0) > 0:
                    lines.append(
                        f"histograms,service={self.service_name},name={name} "
                        f"count={stats['count']},mean={stats.get('mean', 0)},p95={stats.get('p95', 0)} {timestamp_ns}"
                    )

            # Transactions
            for name, stats in metrics.get("transactions", {}).items():
                if stats.get("count", 0) > 0:
                    lines.append(
                        f"transactions,service={self.service_name},name={name} "
                        f"count={stats['count']},mean_ms={stats.get('mean_ms', 0)},p95_ms={stats.get('p95_ms', 0)} {timestamp_ns}"
                    )

        return lines

    def export(self, format: MetricsExportFormat) -> Union[str, List[str], Dict[str, Any]]:
        """
        Export metrics in specified format.

        Args:
            format: Export format

        Returns:
            Formatted metrics
        """
        if format == MetricsExportFormat.PROMETHEUS:
            return self.export_prometheus()
        elif format == MetricsExportFormat.STATSD:
            return self.export_statsd()
        elif format == MetricsExportFormat.JSON:
            return self.export_json()
        elif format == MetricsExportFormat.INFLUXDB:
            return self.export_influxdb()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize metric name for export."""
        # Replace non-alphanumeric chars with underscore
        return "".join(c if c.isalnum() else "_" for c in name).lower()


# =============================================================================
# P3: Log Aggregation Configuration
# =============================================================================

@dataclass
class LogDestinationConfig:
    """Configuration for a log destination."""
    name: str
    destination_type: str  # "file", "stdout", "http", "syslog"
    enabled: bool = True
    level_filter: Optional[LogLevel] = None  # Min level to send
    format: str = "json"  # "json", "text", "structured"
    # Type-specific config
    file_path: Optional[str] = None
    http_endpoint: Optional[str] = None
    syslog_host: Optional[str] = None
    syslog_port: int = 514
    batch_size: int = 100
    flush_interval_seconds: int = 5


class LogAggregationConfig:
    """
    Configurable log aggregation system.

    P3 Enhancement: Provides:
    - Multiple destination support
    - Configurable log formats
    - Level-based routing
    - Batch configuration
    """

    def __init__(self, service_name: str):
        """Initialize log aggregation config."""
        self.service_name = service_name
        self._destinations: Dict[str, LogDestinationConfig] = {}
        self._formatters: Dict[str, Callable[[LogEntry], str]] = {}
        self._register_default_formatters()

    def _register_default_formatters(self) -> None:
        """Register default log formatters."""
        self._formatters["json"] = self._format_json
        self._formatters["text"] = self._format_text
        self._formatters["structured"] = self._format_structured

    def _format_json(self, entry: LogEntry) -> str:
        """Format log entry as JSON."""
        return json.dumps({
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level.name,
            "service": entry.service,
            "component": entry.component,
            "message": entry.message,
            "trace_id": entry.trace_id,
            "metadata": entry.metadata,
            "exception": entry.exception,
        })

    def _format_text(self, entry: LogEntry) -> str:
        """Format log entry as plain text."""
        return (
            f"[{entry.timestamp.isoformat()}] [{entry.level.name}] "
            f"[{entry.service}/{entry.component}] {entry.message}"
        )

    def _format_structured(self, entry: LogEntry) -> str:
        """Format log entry as structured key-value pairs."""
        parts = [
            f"ts={entry.timestamp.isoformat()}",
            f"level={entry.level.name}",
            f"service={entry.service}",
            f"component={entry.component}",
            f'msg="{entry.message}"',
        ]
        if entry.trace_id:
            parts.append(f"trace_id={entry.trace_id}")
        for k, v in entry.metadata.items():
            parts.append(f"{k}={v}")
        return " ".join(parts)

    def add_destination(self, config: LogDestinationConfig) -> None:
        """Add a log destination."""
        self._destinations[config.name] = config
        logger.info(f"Added log destination: {config.name} ({config.destination_type})")

    def remove_destination(self, name: str) -> bool:
        """Remove a log destination."""
        if name in self._destinations:
            del self._destinations[name]
            return True
        return False

    def get_destinations(self) -> Dict[str, LogDestinationConfig]:
        """Get all configured destinations."""
        return self._destinations.copy()

    def create_aggregator_from_config(self) -> LogAggregator:
        """
        Create a LogAggregator instance from current configuration.

        Returns:
            Configured LogAggregator
        """
        aggregator = LogAggregator(self.service_name)

        for name, config in self._destinations.items():
            if not config.enabled:
                continue

            # Create output handler based on destination type
            handler = self._create_handler(config)
            if handler:
                # Wrap with level filter if configured
                if config.level_filter:
                    original_handler = handler
                    min_level = config.level_filter.value

                    def filtered_handler(entries: List[LogEntry], h=original_handler, ml=min_level):
                        filtered = [e for e in entries if e.level.value >= ml]
                        if filtered:
                            h(filtered)

                    aggregator.add_output(filtered_handler)
                else:
                    aggregator.add_output(handler)

        return aggregator

    def _create_handler(self, config: LogDestinationConfig) -> Optional[Callable[[List[LogEntry]], None]]:
        """Create a log handler based on configuration."""
        formatter = self._formatters.get(config.format, self._format_json)

        if config.destination_type == "stdout":
            def stdout_handler(entries: List[LogEntry]):
                for entry in entries:
                    print(formatter(entry))
            return stdout_handler

        elif config.destination_type == "file" and config.file_path:
            def file_handler(entries: List[LogEntry]):
                try:
                    with open(config.file_path, "a") as f:
                        for entry in entries:
                            f.write(formatter(entry) + "\n")
                except Exception as e:
                    logger.error(f"Failed to write to log file: {e}")
            return file_handler

        elif config.destination_type == "http" and config.http_endpoint:
            def http_handler(entries: List[LogEntry]):
                try:
                    import urllib.request
                    data = json.dumps([json.loads(formatter(e)) for e in entries]).encode()
                    req = urllib.request.Request(
                        config.http_endpoint,
                        data=data,
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(req, timeout=5)
                except Exception as e:
                    logger.error(f"Failed to send logs to HTTP endpoint: {e}")
            return http_handler

        return None

    def register_formatter(self, name: str, formatter: Callable[[LogEntry], str]) -> None:
        """Register a custom log formatter."""
        self._formatters[name] = formatter

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "service_name": self.service_name,
            "destinations": {
                name: {
                    "destination_type": cfg.destination_type,
                    "enabled": cfg.enabled,
                    "level_filter": cfg.level_filter.name if cfg.level_filter else None,
                    "format": cfg.format,
                    "file_path": cfg.file_path,
                    "http_endpoint": cfg.http_endpoint,
                    "batch_size": cfg.batch_size,
                    "flush_interval_seconds": cfg.flush_interval_seconds,
                }
                for name, cfg in self._destinations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogAggregationConfig":
        """Create configuration from dictionary."""
        config = cls(data.get("service_name", "unknown"))

        for name, dest_data in data.get("destinations", {}).items():
            level_filter = None
            if dest_data.get("level_filter"):
                level_filter = LogLevel[dest_data["level_filter"]]

            dest_config = LogDestinationConfig(
                name=name,
                destination_type=dest_data.get("destination_type", "stdout"),
                enabled=dest_data.get("enabled", True),
                level_filter=level_filter,
                format=dest_data.get("format", "json"),
                file_path=dest_data.get("file_path"),
                http_endpoint=dest_data.get("http_endpoint"),
                batch_size=dest_data.get("batch_size", 100),
                flush_interval_seconds=dest_data.get("flush_interval_seconds", 5),
            )
            config.add_destination(dest_config)

        return config


# =============================================================================
# Module Integration
# =============================================================================

def create_infrastructure_suite(service_name: str) -> Dict[str, Any]:
    """
    Create a complete infrastructure suite.

    Returns:
        Dictionary containing all infrastructure components
    """
    apm = APMCollector(service_name)
    return {
        "log_aggregator": LogAggregator(service_name),
        "log_config": LogAggregationConfig(service_name),
        "apm": apm,
        "metrics_exporter": MetricsExporter(service_name, apm),
        "health_check_manager": HealthCheckManager(service_name),
        "alert_manager": AlertManager(service_name),
        "chaos_engine": ChaosEngine(),
        "cache": CacheWarmer(),
        "tracer": RequestTracer(service_name),
        "deployer": BlueGreenDeployer(service_name),
        "feature_flags": FeatureFlagManager()
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create infrastructure suite
    infra = create_infrastructure_suite("trading-service")

    # Log aggregation
    log_agg = infra["log_aggregator"]
    log_agg.add_output(console_log_output)
    log_agg.info("System started", "main", environment="production")

    # APM
    apm = infra["apm"]
    apm.set_sla_threshold("order_execution", 100)  # 100ms SLA
    with apm.start_transaction("order_execution"):
        time.sleep(0.05)  # Simulate work
    print(f"APM stats: {apm.get_transaction_stats('order_execution')}")

    # Alert management
    alerts = infra["alert_manager"]
    alert = alerts.create_alert(
        AlertSeverity.HIGH,
        "High latency detected",
        "Order execution latency exceeded threshold",
        "execution-engine"
    )
    print(f"Alert created: {alert.id}")

    # Feature flags
    ff = infra["feature_flags"]
    ff.create_flag("new_algo", default_value=False)
    ff.set_percentage_rollout("new_algo", 25)
    print(f"New algo enabled: {ff.is_enabled('new_algo', user_id='user123')}")

    # Cache warming
    cache = infra["cache"]
    cache.add_warming_task(lambda: {"key1": "value1", "key2": "value2"})
    cache.warm_cache()
    print(f"Cache stats: {cache.get_stats()}")

    # Request tracing
    tracer = infra["tracer"]
    span = tracer.start_trace("handle_order")
    tracer.add_span_tag(span, "order_id", "12345")
    child_span = tracer.start_span("validate_order", span)
    tracer.end_span(child_span)
    tracer.end_span(span)
    print(f"Trace recorded: {span.trace_id}")

    # Blue-green deployment
    deployer = infra["deployer"]
    deployer.register_deployment(DeploymentColor.BLUE, "1.0.0", "/health")
    deployer.register_deployment(DeploymentColor.GREEN, "1.1.0", "/health")
    deployer.set_active(DeploymentColor.BLUE)
    deployer.shift_traffic(DeploymentColor.GREEN, 10)  # 10% canary
    print(f"Deployment status: {deployer.get_status()}")

"""
Health Check Server
===================

HTTP health check endpoints for monitoring and orchestration systems.
Addresses issue #S5: No health check endpoint for monitoring.

Features:
- Kubernetes-style liveness and readiness probes
- Detailed health status with component checks
- Prometheus-compatible metrics endpoint
- Configurable health check criteria
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Callable, Any
import threading
from functools import partial

# System metrics - optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.monitoring import MonitoringSystem


logger = logging.getLogger(__name__)


# ============================================================================
# P3: Rate Limiting for API Endpoints
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API endpoints (P3: Add rate limiting).

    Limits requests per IP address to prevent abuse of health check endpoints.
    Uses thread-safe implementation for use with the HTTP server.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval_seconds: int = 300,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum sustained requests per minute per IP
            burst_size: Maximum burst size (initial token count)
            cleanup_interval_seconds: How often to clean up old entries
        """
        self._requests_per_minute = requests_per_minute
        self._burst_size = burst_size
        self._cleanup_interval = cleanup_interval_seconds

        # Token buckets per IP: {ip: (tokens, last_update_time)}
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

        # Refill rate: tokens per second
        self._refill_rate = requests_per_minute / 60.0

    def is_allowed(self, client_ip: str) -> tuple[bool, dict[str, Any]]:
        """
        Check if request from client IP is allowed.

        Args:
            client_ip: Client IP address

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        with self._lock:
            current_time = time.time()

            # Periodic cleanup
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup_stale_entries(current_time)
                self._last_cleanup = current_time

            # Get or create bucket
            if client_ip in self._buckets:
                tokens, last_update = self._buckets[client_ip]

                # Refill tokens based on elapsed time
                elapsed = current_time - last_update
                tokens = min(self._burst_size, tokens + elapsed * self._refill_rate)
            else:
                tokens = self._burst_size

            # Rate limit info for headers
            rate_info = {
                "limit": self._requests_per_minute,
                "remaining": max(0, int(tokens) - 1),
                "reset": int(current_time + (self._burst_size - tokens) / self._refill_rate),
            }

            # Check if request is allowed
            if tokens >= 1:
                self._buckets[client_ip] = (tokens - 1, current_time)
                return True, rate_info
            else:
                self._buckets[client_ip] = (tokens, current_time)
                return False, rate_info

    def _cleanup_stale_entries(self, current_time: float) -> None:
        """Remove entries that haven't been used in a while."""
        stale_threshold = current_time - (self._cleanup_interval * 2)
        stale_ips = [
            ip for ip, (_, last_update) in self._buckets.items()
            if last_update < stale_threshold
        ]
        for ip in stale_ips:
            del self._buckets[ip]

        if stale_ips:
            logger.debug(f"Rate limiter cleaned up {len(stale_ips)} stale entries")


# ============================================================================
# P3: Response Caching for API Endpoints
# ============================================================================

class ResponseCache:
    """
    Simple response cache for health check endpoints (P3: Add response caching).

    Caches responses for a short TTL to reduce load on expensive health checks.
    Thread-safe for use with the HTTP server.
    """

    def __init__(self, default_ttl_seconds: float = 1.0, max_entries: int = 100):
        """
        Initialize response cache.

        Args:
            default_ttl_seconds: Default cache TTL (short for health checks)
            max_entries: Maximum cache entries to prevent memory leaks
        """
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries

        # Cache: {key: (response_data, status_code, content_type, expiry_time)}
        self._cache: dict[str, tuple[str, int, str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[str, int, str] | None:
        """
        Get cached response if valid.

        Args:
            key: Cache key (typically endpoint path)

        Returns:
            Tuple of (response_data, status_code, content_type) or None if not cached
        """
        with self._lock:
            if key not in self._cache:
                return None

            response_data, status_code, content_type, expiry_time = self._cache[key]

            if time.time() > expiry_time:
                # Expired
                del self._cache[key]
                return None

            return response_data, status_code, content_type

    def set(
        self,
        key: str,
        response_data: str,
        status_code: int,
        content_type: str,
        ttl_seconds: float | None = None,
    ) -> None:
        """
        Cache a response.

        Args:
            key: Cache key
            response_data: Response body
            status_code: HTTP status code
            content_type: Content-Type header value
            ttl_seconds: Optional TTL override
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expiry_time = time.time() + ttl

        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_entries:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][3])
                del self._cache[oldest_key]

            self._cache[key] = (response_data, status_code, content_type, expiry_time)

    def invalidate(self, key: str | None = None) -> None:
        """
        Invalidate cache entries.

        Args:
            key: Specific key to invalidate, or None to clear all
        """
        with self._lock:
            if key is None:
                self._cache.clear()
            elif key in self._cache:
                del self._cache[key]


# ============================================================================
# P3: Request Validation
# ============================================================================

class RequestValidator:
    """
    Request validator for health check endpoints (P3: Add request validation).

    Validates incoming requests and provides security checks.
    """

    # Valid endpoint paths
    VALID_PATHS = {"/", "/health", "/live", "/livez", "/ready", "/readyz", "/metrics"}

    # Maximum URL length to prevent DoS
    MAX_URL_LENGTH = 2048

    # Allowed HTTP methods
    ALLOWED_METHODS = {"GET", "HEAD"}

    # Path pattern (alphanumeric, slashes, hyphens, underscores)
    PATH_PATTERN = re.compile(r"^[a-zA-Z0-9/_-]*$")

    @classmethod
    def validate_request(
        cls,
        method: str,
        path: str,
        client_ip: str,
    ) -> tuple[bool, str | None]:
        """
        Validate an incoming request.

        Args:
            method: HTTP method
            path: Request path
            client_ip: Client IP address

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check method
        if method not in cls.ALLOWED_METHODS:
            return False, f"Method not allowed: {method}"

        # Check URL length
        if len(path) > cls.MAX_URL_LENGTH:
            return False, "URL too long"

        # Extract path without query string
        clean_path = path.split("?")[0]

        # Check path pattern (prevent path traversal, etc.)
        if not cls.PATH_PATTERN.match(clean_path):
            return False, "Invalid characters in path"

        # Check for path traversal attempts
        if ".." in path or "//" in path:
            logger.warning(f"Potential path traversal attempt from {client_ip}: {path}")
            return False, "Invalid path"

        # Check if path is a known endpoint
        if clean_path not in cls.VALID_PATHS:
            # Not an error, just 404
            return True, None

        return True, None

    @classmethod
    def sanitize_path(cls, path: str) -> str:
        """
        Sanitize request path for logging.

        Args:
            path: Raw request path

        Returns:
            Sanitized path safe for logging
        """
        # Truncate long paths
        if len(path) > 100:
            path = path[:100] + "..."

        # Remove control characters
        path = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", path)

        return path


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Overall health check result."""
    status: HealthStatus
    timestamp: datetime
    components: dict[str, ComponentHealth]
    uptime_seconds: float = 0.0
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "version": self.version,
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "last_check": comp.last_check.isoformat(),
                    "details": comp.details,
                }
                for name, comp in self.components.items()
            }
        }


@dataclass
class AlertThreshold:
    """Threshold configuration for alerting."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = ">"  # ">" or "<"
    description: str = ""


@dataclass
class HealthCheckConfig:
    """Configuration for health check server."""
    host: str = "0.0.0.0"
    port: int = 8080
    # Thresholds for health determination
    max_event_queue_pct: float = 90.0  # Queue above this = unhealthy
    max_latency_ms: float = 1000.0  # Latency above this = degraded
    max_error_rate_pct: float = 10.0  # Error rate above this = degraded
    min_active_agents: int = 3  # Below this = degraded
    broker_required: bool = False  # If True, disconnected broker = unhealthy
    # Prometheus labels for multi-instance deployments (MON-004)
    instance_id: str = field(default_factory=lambda: socket.gethostname())
    environment: str = "paper"  # paper, staging, production
    version: str = "1.0.0"
    # Security: authentication requirement for non-localhost bindings (SEC-001)
    require_auth_for_remote: bool = True

    # Alerting thresholds (P3: Add alerting thresholds)
    alert_thresholds: list[AlertThreshold] = field(default_factory=lambda: [
        AlertThreshold("drawdown_pct", 0.05, 0.10, ">", "Portfolio drawdown"),
        AlertThreshold("var_95_pct", 0.02, 0.03, ">", "Value at Risk 95%"),
        AlertThreshold("leverage_ratio", 2.0, 3.0, ">", "Leverage ratio"),
        AlertThreshold("fill_rate", 0.90, 0.80, "<", "Order fill rate"),
        AlertThreshold("daily_pnl_pct", -0.02, -0.03, "<", "Daily P&L"),
        AlertThreshold("event_queue_pct", 70.0, 90.0, ">", "Event queue utilization"),
        AlertThreshold("latency_p99_ms", 500.0, 1000.0, ">", "P99 latency"),
    ])
    # P2: Disk space monitoring thresholds
    disk_warning_pct: float = 80.0  # Disk usage above this = degraded
    disk_critical_pct: float = 95.0  # Disk usage above this = unhealthy
    disk_paths: list[str] = field(default_factory=lambda: ["."])  # Paths to monitor
    # P2: Memory usage alert thresholds
    memory_warning_pct: float = 80.0  # Memory usage above this = degraded
    memory_critical_pct: float = 95.0  # Memory usage above this = unhealthy
    # P2: Network latency thresholds (ms)
    network_latency_warning_ms: float = 100.0  # Latency above this = degraded
    network_latency_critical_ms: float = 500.0  # Latency above this = unhealthy
    network_check_hosts: list[str] = field(default_factory=lambda: [])  # Hosts to ping


@dataclass
class MetricAggregation:
    """
    Aggregated metric statistics (P3: Add metric aggregation).
    """
    metric_name: str
    current_value: float
    min_value: float
    max_value: float
    avg_value: float
    count: int
    window_seconds: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "current": self.current_value,
            "min": self.min_value,
            "max": self.max_value,
            "avg": self.avg_value,
            "count": self.count,
            "window_seconds": self.window_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CustomBusinessMetric:
    """
    Custom business metric definition (P3: Add custom business metrics).
    """
    name: str
    value: float
    unit: str
    category: str  # trading, risk, execution, compliance
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


class MetricAggregator:
    """
    Aggregates metrics over time windows (P3: Add metric aggregation).
    """

    def __init__(self, default_window_seconds: int = 300):
        self._default_window = default_window_seconds
        self._metric_history: dict[str, list[tuple[datetime, float]]] = {}
        self._max_history_size = 1000

    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []

        now = datetime.now(timezone.utc)
        self._metric_history[metric_name].append((now, value))

        # Trim to max size
        if len(self._metric_history[metric_name]) > self._max_history_size:
            self._metric_history[metric_name] = self._metric_history[metric_name][-self._max_history_size:]

    def get_aggregation(
        self, metric_name: str, window_seconds: int | None = None
    ) -> MetricAggregation | None:
        """Get aggregated stats for a metric."""
        if metric_name not in self._metric_history:
            return None

        window = window_seconds or self._default_window
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window)

        values = [v for ts, v in self._metric_history[metric_name] if ts >= cutoff]

        if not values:
            return None

        return MetricAggregation(
            metric_name=metric_name,
            current_value=values[-1],
            min_value=min(values),
            max_value=max(values),
            avg_value=sum(values) / len(values),
            count=len(values),
            window_seconds=window,
        )

    def get_all_aggregations(self, window_seconds: int | None = None) -> list[MetricAggregation]:
        """Get aggregations for all tracked metrics."""
        aggregations = []
        for metric_name in self._metric_history:
            agg = self.get_aggregation(metric_name, window_seconds)
            if agg:
                aggregations.append(agg)
        return aggregations

    def cleanup_old(self, max_age_seconds: int = 3600) -> int:
        """Remove data older than max_age_seconds."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        removed = 0

        for metric_name in self._metric_history:
            original_len = len(self._metric_history[metric_name])
            self._metric_history[metric_name] = [
                (ts, v) for ts, v in self._metric_history[metric_name]
                if ts >= cutoff
            ]
            removed += original_len - len(self._metric_history[metric_name])

        return removed


class HealthChecker:
    """
    Health checking logic for the trading system.

    Performs checks on:
    - Event bus health
    - Broker connectivity
    - Agent status
    - Resource utilization
    """

    def __init__(
        self,
        config: HealthCheckConfig | None = None,
        get_status_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        self._config = config or HealthCheckConfig()
        self._get_status_fn = get_status_fn
        self._start_time = datetime.now(timezone.utc)
        self._event_bus: "EventBus | None" = None
        self._monitoring: "MonitoringSystem | None" = None
        self._is_ready = False
        self._custom_checks: list[Callable[[], ComponentHealth]] = []

        # P3: Add metric aggregation support
        self._aggregator = MetricAggregator()
        self._custom_metrics: list[CustomBusinessMetric] = []
        self._threshold_violations: list[dict[str, Any]] = []

    def set_event_bus(self, event_bus: "EventBus") -> None:
        """Set the event bus reference for health checks."""
        self._event_bus = event_bus

    def set_monitoring(self, monitoring: "MonitoringSystem") -> None:
        """Set the monitoring system reference."""
        self._monitoring = monitoring

    def set_ready(self, ready: bool = True) -> None:
        """Mark the system as ready (or not ready)."""
        self._is_ready = ready
        logger.info(f"System readiness set to: {ready}")

    def add_custom_check(self, check_fn: Callable[[], ComponentHealth]) -> None:
        """Add a custom health check function."""
        self._custom_checks.append(check_fn)

    def record_custom_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        category: str = "trading",
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record a custom business metric (P3: Add custom business metrics).

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement (e.g., "USD", "bps", "%")
            category: Category (trading, risk, execution, compliance)
            labels: Optional labels for the metric
        """
        metric = CustomBusinessMetric(
            name=name,
            value=value,
            unit=unit,
            category=category,
            labels=labels or {},
        )
        self._custom_metrics.append(metric)

        # Also record in aggregator
        self._aggregator.record(name, value)

        # Keep last 100 custom metrics
        if len(self._custom_metrics) > 100:
            self._custom_metrics = self._custom_metrics[-100:]

        # Check thresholds
        self._check_threshold(name, value)

    def _check_threshold(self, metric_name: str, value: float) -> None:
        """Check if metric violates any threshold (P3: Add alerting thresholds)."""
        for threshold in self._config.alert_thresholds:
            if threshold.metric_name != metric_name:
                continue

            violation = None
            if threshold.comparison == ">":
                if value > threshold.critical_threshold:
                    violation = {"level": "critical", "threshold": threshold.critical_threshold}
                elif value > threshold.warning_threshold:
                    violation = {"level": "warning", "threshold": threshold.warning_threshold}
            else:  # "<"
                if value < threshold.critical_threshold:
                    violation = {"level": "critical", "threshold": threshold.critical_threshold}
                elif value < threshold.warning_threshold:
                    violation = {"level": "warning", "threshold": threshold.warning_threshold}

            if violation:
                self._threshold_violations.append({
                    "metric_name": metric_name,
                    "current_value": value,
                    "level": violation["level"],
                    "threshold": violation["threshold"],
                    "comparison": threshold.comparison,
                    "description": threshold.description,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                # Keep last 50 violations
                if len(self._threshold_violations) > 50:
                    self._threshold_violations = self._threshold_violations[-50:]

                logger.warning(
                    f"Threshold violation [{violation['level']}]: {metric_name}={value} "
                    f"({threshold.comparison} {violation['threshold']})"
                )

    def get_metric_aggregations(self, window_seconds: int = 300) -> list[dict[str, Any]]:
        """Get metric aggregations (P3: Add metric aggregation)."""
        return [agg.to_dict() for agg in self._aggregator.get_all_aggregations(window_seconds)]

    def get_custom_metrics(self, category: str | None = None) -> list[dict[str, Any]]:
        """Get custom business metrics (P3: Add custom business metrics)."""
        metrics = self._custom_metrics
        if category:
            metrics = [m for m in metrics if m.category == category]
        return [m.to_dict() for m in metrics]

    def get_threshold_violations(self, level: str | None = None) -> list[dict[str, Any]]:
        """Get recent threshold violations (P3: Add alerting thresholds)."""
        violations = self._threshold_violations
        if level:
            violations = [v for v in violations if v["level"] == level]
        return violations

    def check_liveness(self) -> tuple[bool, str]:
        """
        Liveness probe - is the process alive and not deadlocked?

        Returns:
            Tuple of (is_live, message)
        """
        # Basic liveness: can we execute code and report time?
        try:
            _ = datetime.now(timezone.utc)
            return True, "Process is alive"
        except Exception as e:
            return False, f"Liveness check failed: {e}"

    def check_readiness(self) -> tuple[bool, str]:
        """
        Readiness probe - is the system ready to accept work?

        Returns:
            Tuple of (is_ready, message)
        """
        if not self._is_ready:
            return False, "System not initialized"

        # Check event bus
        if self._event_bus and not self._event_bus.is_running:
            return False, "Event bus not running"

        return True, "System ready"

    def _check_event_bus(self) -> ComponentHealth:
        """Check event bus health."""
        if not self._event_bus:
            return ComponentHealth(
                name="event_bus",
                status=HealthStatus.UNHEALTHY,
                message="Event bus not configured",
            )

        status = self._event_bus.get_status()
        queue_pct = status.get("queue_utilization_pct", 0)
        backpressure = status.get("backpressure_level", "normal")

        if not status.get("running", False):
            return ComponentHealth(
                name="event_bus",
                status=HealthStatus.UNHEALTHY,
                message="Event bus not running",
                details=status,
            )

        if queue_pct > self._config.max_event_queue_pct:
            return ComponentHealth(
                name="event_bus",
                status=HealthStatus.UNHEALTHY,
                message=f"Queue utilization critical: {queue_pct:.1f}%",
                details=status,
            )

        if backpressure in ("high", "critical"):
            return ComponentHealth(
                name="event_bus",
                status=HealthStatus.DEGRADED,
                message=f"Backpressure level: {backpressure}",
                details=status,
            )

        return ComponentHealth(
            name="event_bus",
            status=HealthStatus.HEALTHY,
            message=f"Queue at {queue_pct:.1f}%",
            details={
                "queue_size": status.get("queue_size", 0),
                "backpressure_level": backpressure,
            },
        )

    def _check_broker(self) -> ComponentHealth:
        """Check broker connectivity."""
        if not self._get_status_fn:
            return ComponentHealth(
                name="broker",
                status=HealthStatus.DEGRADED,
                message="Status function not configured",
            )

        status = self._get_status_fn()
        broker_info = status.get("broker", {})
        connected = broker_info.get("connected", False)

        if not connected:
            if self._config.broker_required:
                return ComponentHealth(
                    name="broker",
                    status=HealthStatus.UNHEALTHY,
                    message="Broker disconnected (required)",
                    details=broker_info,
                )
            return ComponentHealth(
                name="broker",
                status=HealthStatus.DEGRADED,
                message="Broker disconnected (simulated mode)",
                details=broker_info,
            )

        return ComponentHealth(
            name="broker",
            status=HealthStatus.HEALTHY,
            message=f"Connected to account {broker_info.get('account_id', 'N/A')}",
            details=broker_info,
        )

    def _check_agents(self) -> ComponentHealth:
        """Check agent health."""
        if not self._get_status_fn:
            return ComponentHealth(
                name="agents",
                status=HealthStatus.DEGRADED,
                message="Status function not configured",
            )

        status = self._get_status_fn()
        agents = status.get("agents", {})

        # Count running agents
        running_count = 0
        unhealthy_agents = []

        # Check signal agents
        signal_agents = agents.get("signal", [])
        for agent_status in signal_agents:
            if agent_status.get("running", False):
                running_count += 1
            elif agent_status.get("name"):
                unhealthy_agents.append(agent_status.get("name"))

        # Check core agents
        for agent_type in ["cio", "risk", "compliance", "execution"]:
            agent_status = agents.get(agent_type)
            if agent_status:
                if agent_status.get("running", False):
                    running_count += 1
                elif agent_status.get("name"):
                    unhealthy_agents.append(agent_type)

        if len(unhealthy_agents) > 0 and "cio" in unhealthy_agents:
            return ComponentHealth(
                name="agents",
                status=HealthStatus.UNHEALTHY,
                message="CIO agent not running",
                details={
                    "running_count": running_count,
                    "unhealthy": unhealthy_agents,
                },
            )

        if running_count < self._config.min_active_agents:
            return ComponentHealth(
                name="agents",
                status=HealthStatus.DEGRADED,
                message=f"Only {running_count} agents running (min: {self._config.min_active_agents})",
                details={
                    "running_count": running_count,
                    "unhealthy": unhealthy_agents,
                },
            )

        return ComponentHealth(
            name="agents",
            status=HealthStatus.HEALTHY,
            message=f"{running_count} agents running",
            details={"running_count": running_count},
        )

    def _check_monitoring(self) -> ComponentHealth:
        """Check monitoring system health."""
        if not self._monitoring:
            return ComponentHealth(
                name="monitoring",
                status=HealthStatus.DEGRADED,
                message="Monitoring system not configured",
            )

        try:
            summary = self._monitoring.get_status_summary()
            critical_alerts = summary.get("critical_alerts", 0)

            if critical_alerts > 0:
                return ComponentHealth(
                    name="monitoring",
                    status=HealthStatus.DEGRADED,
                    message=f"{critical_alerts} critical alerts active",
                    details=summary,
                )

            return ComponentHealth(
                name="monitoring",
                status=HealthStatus.HEALTHY,
                message=f"Uptime: {summary.get('uptime_hours', 0):.1f}h",
                details=summary,
            )

        except AttributeError as e:
            logger.warning(f"Monitoring interface error: {e}")
            return ComponentHealth(
                name="monitoring",
                status=HealthStatus.DEGRADED,
                message="Monitoring interface incomplete",
            )
        except Exception as e:
            logger.exception("Unexpected error checking monitoring health")
            return ComponentHealth(
                name="monitoring",
                status=HealthStatus.DEGRADED,
                message=f"Error checking monitoring: {type(e).__name__}",
            )

    def _check_disk_space(self) -> ComponentHealth:
        """
        P2: Check disk space health.

        Monitors disk usage on configured paths and alerts on thresholds.
        """
        if not HAS_PSUTIL:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.DEGRADED,
                message="psutil not installed - disk monitoring unavailable",
            )

        try:
            disk_info = {}
            worst_status = HealthStatus.HEALTHY
            worst_usage = 0.0
            worst_path = ""

            for path in self._config.disk_paths:
                try:
                    usage = psutil.disk_usage(path)
                    usage_pct = usage.percent

                    disk_info[path] = {
                        "total_gb": usage.total / (1024 ** 3),
                        "used_gb": usage.used / (1024 ** 3),
                        "free_gb": usage.free / (1024 ** 3),
                        "percent": usage_pct,
                    }

                    # Track worst case
                    if usage_pct > worst_usage:
                        worst_usage = usage_pct
                        worst_path = path

                    # Determine status for this path
                    if usage_pct >= self._config.disk_critical_pct:
                        worst_status = HealthStatus.UNHEALTHY
                    elif usage_pct >= self._config.disk_warning_pct and worst_status != HealthStatus.UNHEALTHY:
                        worst_status = HealthStatus.DEGRADED

                except Exception as path_error:
                    disk_info[path] = {"error": str(path_error)}
                    if worst_status == HealthStatus.HEALTHY:
                        worst_status = HealthStatus.DEGRADED

            # Generate message based on status
            if worst_status == HealthStatus.UNHEALTHY:
                message = f"CRITICAL: Disk {worst_path} at {worst_usage:.1f}% (>= {self._config.disk_critical_pct}%)"
                logger.critical(f"Disk space alert: {message}")
            elif worst_status == HealthStatus.DEGRADED:
                message = f"WARNING: Disk {worst_path} at {worst_usage:.1f}% (>= {self._config.disk_warning_pct}%)"
                logger.warning(f"Disk space alert: {message}")
            else:
                message = f"Disk usage OK, highest: {worst_usage:.1f}%"

            return ComponentHealth(
                name="disk",
                status=worst_status,
                message=message,
                details=disk_info,
            )

        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return ComponentHealth(
                name="disk",
                status=HealthStatus.DEGRADED,
                message=f"Error checking disk: {e}",
            )

    def _check_memory_alerts(self) -> ComponentHealth:
        """
        P2: Check memory usage with alert thresholds.

        Monitors system memory usage and alerts when thresholds are exceeded.
        """
        if not HAS_PSUTIL:
            return ComponentHealth(
                name="memory_alerts",
                status=HealthStatus.DEGRADED,
                message="psutil not installed - memory alerting unavailable",
            )

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            memory_pct = mem.percent
            swap_pct = swap.percent

            memory_info = {
                "memory_percent": memory_pct,
                "memory_total_gb": mem.total / (1024 ** 3),
                "memory_available_gb": mem.available / (1024 ** 3),
                "memory_used_gb": mem.used / (1024 ** 3),
                "swap_percent": swap_pct,
                "swap_total_gb": swap.total / (1024 ** 3),
                "swap_used_gb": swap.used / (1024 ** 3),
            }

            # Determine status based on memory usage
            status = HealthStatus.HEALTHY
            message = f"Memory at {memory_pct:.1f}%, swap at {swap_pct:.1f}%"

            if memory_pct >= self._config.memory_critical_pct:
                status = HealthStatus.UNHEALTHY
                message = f"CRITICAL: Memory at {memory_pct:.1f}% (>= {self._config.memory_critical_pct}%)"
                logger.critical(f"Memory alert: {message}")
            elif memory_pct >= self._config.memory_warning_pct:
                status = HealthStatus.DEGRADED
                message = f"WARNING: Memory at {memory_pct:.1f}% (>= {self._config.memory_warning_pct}%)"
                logger.warning(f"Memory alert: {message}")

            # Also check swap as secondary indicator
            if swap_pct > 80 and status != HealthStatus.UNHEALTHY:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                message += f" - High swap usage: {swap_pct:.1f}%"
                logger.warning(f"Swap usage high: {swap_pct:.1f}%")

            return ComponentHealth(
                name="memory_alerts",
                status=status,
                message=message,
                details=memory_info,
            )

        except Exception as e:
            logger.error(f"Error checking memory alerts: {e}")
            return ComponentHealth(
                name="memory_alerts",
                status=HealthStatus.DEGRADED,
                message=f"Error checking memory: {e}",
            )

    def _check_network_latency(self) -> ComponentHealth:
        """
        P2: Check network latency to configured hosts.

        Uses TCP connect time as a proxy for network latency.
        """
        if not self._config.network_check_hosts:
            return ComponentHealth(
                name="network",
                status=HealthStatus.HEALTHY,
                message="No network hosts configured for monitoring",
                details={"configured": False},
            )

        import time

        latencies = {}
        worst_latency = 0.0
        worst_host = ""
        failed_hosts = []

        for host in self._config.network_check_hosts:
            try:
                # Parse host:port format (default port 443)
                if ":" in host:
                    parts = host.split(":")
                    hostname = parts[0]
                    port = int(parts[1])
                else:
                    hostname = host
                    port = 443

                # Measure TCP connect time
                start = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)  # 5 second timeout
                sock.connect((hostname, port))
                sock.close()
                latency_ms = (time.time() - start) * 1000

                latencies[host] = {
                    "latency_ms": round(latency_ms, 2),
                    "status": "ok",
                }

                if latency_ms > worst_latency:
                    worst_latency = latency_ms
                    worst_host = host

            except socket.timeout:
                latencies[host] = {"latency_ms": None, "status": "timeout"}
                failed_hosts.append(host)
            except socket.gaierror as e:
                latencies[host] = {"latency_ms": None, "status": f"DNS error: {e}"}
                failed_hosts.append(host)
            except Exception as e:
                latencies[host] = {"latency_ms": None, "status": f"error: {e}"}
                failed_hosts.append(host)

        # Determine status
        status = HealthStatus.HEALTHY
        if failed_hosts:
            # Some hosts unreachable
            if len(failed_hosts) == len(self._config.network_check_hosts):
                status = HealthStatus.UNHEALTHY
                message = f"All network hosts unreachable: {failed_hosts}"
                logger.critical(f"Network alert: {message}")
            else:
                status = HealthStatus.DEGRADED
                message = f"Some network hosts unreachable: {failed_hosts}"
                logger.warning(f"Network alert: {message}")
        elif worst_latency >= self._config.network_latency_critical_ms:
            status = HealthStatus.UNHEALTHY
            message = f"CRITICAL: Network latency to {worst_host} is {worst_latency:.0f}ms (>= {self._config.network_latency_critical_ms}ms)"
            logger.critical(f"Network latency alert: {message}")
        elif worst_latency >= self._config.network_latency_warning_ms:
            status = HealthStatus.DEGRADED
            message = f"WARNING: Network latency to {worst_host} is {worst_latency:.0f}ms (>= {self._config.network_latency_warning_ms}ms)"
            logger.warning(f"Network latency alert: {message}")
        else:
            message = f"Network OK, max latency: {worst_latency:.0f}ms to {worst_host}"

        return ComponentHealth(
            name="network",
            status=status,
            message=message,
            details={
                "hosts": latencies,
                "max_latency_ms": worst_latency,
                "failed_hosts": failed_hosts,
            },
        )

    def perform_health_check(self) -> HealthCheckResult:
        """
        Perform comprehensive health check.

        Returns:
            HealthCheckResult with all component statuses
        """
        components: dict[str, ComponentHealth] = {}

        # Run standard checks
        components["event_bus"] = self._check_event_bus()
        components["broker"] = self._check_broker()
        components["agents"] = self._check_agents()
        components["monitoring"] = self._check_monitoring()

        # P2: Run infrastructure checks
        components["disk"] = self._check_disk_space()
        components["memory_alerts"] = self._check_memory_alerts()
        components["network"] = self._check_network_latency()

        # Run custom checks
        for check_fn in self._custom_checks:
            try:
                result = check_fn()
                components[result.name] = result
            except Exception as e:
                components[f"custom_check_{len(components)}"] = ComponentHealth(
                    name="custom_check",
                    status=HealthStatus.DEGRADED,
                    message=f"Check failed: {e}",
                )

        # Determine overall status
        statuses = [comp.status for comp in components.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return HealthCheckResult(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            components=components,
            uptime_seconds=uptime,
        )

    def get_business_metrics(self) -> dict[str, Any]:
        """
        Get business metrics (P&L, drawdown, fill rate).

        MON-011: Expose business-critical metrics for trading system monitoring.

        Returns:
            Dictionary with business metrics
        """
        metrics = {
            "pnl_daily": 0.0,
            "pnl_total": 0.0,
            "drawdown_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "fill_rate": 0.0,
            "orders_total": 0,
            "orders_filled": 0,
            "positions_count": 0,
        }

        if not self._get_status_fn:
            return metrics

        try:
            status = self._get_status_fn()

            # Extract P&L metrics from portfolio/broker status
            portfolio = status.get("portfolio", {})
            metrics["pnl_daily"] = portfolio.get("daily_pnl", 0.0)
            metrics["pnl_total"] = portfolio.get("total_pnl", 0.0)
            metrics["drawdown_pct"] = portfolio.get("current_drawdown_pct", 0.0)
            metrics["max_drawdown_pct"] = portfolio.get("max_drawdown_pct", 0.0)
            metrics["positions_count"] = portfolio.get("position_count", 0)

            # Extract execution metrics
            execution = status.get("execution", {})
            orders_total = execution.get("orders_total", 0)
            orders_filled = execution.get("orders_filled", 0)
            metrics["orders_total"] = orders_total
            metrics["orders_filled"] = orders_filled

            # Calculate fill rate
            if orders_total > 0:
                metrics["fill_rate"] = orders_filled / orders_total
            else:
                metrics["fill_rate"] = 1.0  # No orders = 100% fill rate (no failures)

        except Exception as e:
            logger.warning(f"Error collecting business metrics: {e}")

        return metrics

    def get_system_metrics(self) -> dict[str, Any]:
        """
        Get system metrics (CPU, memory, event bus latency).

        Returns:
            Dictionary with system metrics
        """
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu": {},
            "memory": {},
            "event_bus": {},
            "process": {},
        }

        # System CPU and Memory (requires psutil)
        if HAS_PSUTIL:
            try:
                # CPU metrics
                metrics["cpu"]["percent"] = psutil.cpu_percent(interval=0.1)
                metrics["cpu"]["count"] = psutil.cpu_count()
                metrics["cpu"]["count_logical"] = psutil.cpu_count(logical=True)

                # Memory metrics
                mem = psutil.virtual_memory()
                metrics["memory"]["total_bytes"] = mem.total
                metrics["memory"]["available_bytes"] = mem.available
                metrics["memory"]["used_bytes"] = mem.used
                metrics["memory"]["percent"] = mem.percent

                # Process-specific metrics
                process = psutil.Process(os.getpid())
                metrics["process"]["cpu_percent"] = process.cpu_percent(interval=0.1)
                metrics["process"]["memory_bytes"] = process.memory_info().rss
                metrics["process"]["memory_percent"] = process.memory_percent()
                metrics["process"]["threads"] = process.num_threads()
                metrics["process"]["open_files"] = len(process.open_files())
            except Exception as e:
                logger.warning(f"Error collecting psutil metrics: {e}")
                metrics["error"] = str(e)
        else:
            metrics["cpu"]["error"] = "psutil not installed"
            metrics["memory"]["error"] = "psutil not installed"

        # Event bus metrics
        if self._event_bus:
            try:
                eb_status = self._event_bus.get_status()
                metrics["event_bus"]["queue_size"] = eb_status.get("queue_size", 0)
                metrics["event_bus"]["queue_max"] = eb_status.get("max_queue_size", 0)
                metrics["event_bus"]["queue_utilization_pct"] = eb_status.get("queue_utilization_pct", 0)
                metrics["event_bus"]["backpressure_level"] = eb_status.get("backpressure_level", "unknown")
                metrics["event_bus"]["events_processed"] = eb_status.get("events_processed", 0)

                # Latency metrics if available
                latency_stats = eb_status.get("latency_stats", {})
                if latency_stats:
                    metrics["event_bus"]["latency_avg_ms"] = latency_stats.get("avg_ms", 0)
                    metrics["event_bus"]["latency_p95_ms"] = latency_stats.get("p95_ms", 0)
                    metrics["event_bus"]["latency_p99_ms"] = latency_stats.get("p99_ms", 0)
                    metrics["event_bus"]["latency_max_ms"] = latency_stats.get("max_ms", 0)
            except Exception as e:
                logger.warning(f"Error collecting event bus metrics: {e}")
                metrics["event_bus"]["error"] = str(e)
        else:
            metrics["event_bus"]["error"] = "Event bus not configured"

        # P2: Disk metrics
        metrics["disk"] = {}
        if HAS_PSUTIL:
            try:
                for path in self._config.disk_paths:
                    try:
                        usage = psutil.disk_usage(path)
                        metrics["disk"][path] = {
                            "total_bytes": usage.total,
                            "used_bytes": usage.used,
                            "free_bytes": usage.free,
                            "percent": usage.percent,
                        }
                    except Exception as path_error:
                        metrics["disk"][path] = {"error": str(path_error)}
            except Exception as e:
                metrics["disk"]["error"] = str(e)
        else:
            metrics["disk"]["error"] = "psutil not installed"

        # P2: Network latency metrics
        metrics["network"] = {}
        if self._config.network_check_hosts:
            import time as time_module
            for host in self._config.network_check_hosts:
                try:
                    if ":" in host:
                        parts = host.split(":")
                        hostname = parts[0]
                        port = int(parts[1])
                    else:
                        hostname = host
                        port = 443

                    start = time_module.time()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5.0)
                    sock.connect((hostname, port))
                    sock.close()
                    latency_ms = (time_module.time() - start) * 1000
                    metrics["network"][host] = {"latency_ms": round(latency_ms, 2)}
                except Exception as e:
                    metrics["network"][host] = {"error": str(e)}

        return metrics


class HealthCheckHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for health check endpoints.

    P3 Improvements:
    - Rate limiting per IP address
    - Request validation and sanitization
    - Response caching for expensive endpoints
    """

    # Class-level rate limiter and cache (shared across all handler instances)
    _rate_limiter: RateLimiter | None = None
    _response_cache: ResponseCache | None = None
    _rate_limiting_enabled: bool = True
    _caching_enabled: bool = True

    # Cache TTLs per endpoint (seconds)
    CACHE_TTLS = {
        "/health": 1.0,      # Short TTL for health (frequent updates)
        "/live": 0.5,        # Very short for liveness
        "/ready": 0.5,       # Very short for readiness
        "/metrics": 2.0,     # Slightly longer for metrics (expensive)
    }

    @classmethod
    def configure(
        cls,
        rate_limiting_enabled: bool = True,
        requests_per_minute: int = 60,
        caching_enabled: bool = True,
        cache_ttl_seconds: float = 1.0,
    ) -> None:
        """
        Configure handler class settings.

        Args:
            rate_limiting_enabled: Enable rate limiting
            requests_per_minute: Rate limit (requests per minute per IP)
            caching_enabled: Enable response caching
            cache_ttl_seconds: Default cache TTL
        """
        cls._rate_limiting_enabled = rate_limiting_enabled
        cls._caching_enabled = caching_enabled

        if rate_limiting_enabled:
            cls._rate_limiter = RateLimiter(
                requests_per_minute=requests_per_minute,
                burst_size=min(requests_per_minute // 2, 20),
            )

        if caching_enabled:
            cls._response_cache = ResponseCache(
                default_ttl_seconds=cache_ttl_seconds,
            )

    def __init__(self, checker: HealthChecker, *args, **kwargs):
        self.checker = checker

        # Initialize class-level resources if not already done
        if HealthCheckHandler._rate_limiter is None and HealthCheckHandler._rate_limiting_enabled:
            HealthCheckHandler._rate_limiter = RateLimiter()
        if HealthCheckHandler._response_cache is None and HealthCheckHandler._caching_enabled:
            HealthCheckHandler._response_cache = ResponseCache()

        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging, use our logger instead."""
        logger.debug(f"Health check request: {args[0]}")

    def _get_client_ip(self) -> str:
        """Get client IP address, considering X-Forwarded-For header."""
        # Check for proxy headers (when behind load balancer/proxy)
        forwarded_for = self.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()
        return self.client_address[0]

    def _check_rate_limit(self) -> bool:
        """
        Check rate limit and send 429 response if exceeded.

        Returns:
            True if request is allowed, False if rate limited
        """
        if not self._rate_limiting_enabled or self._rate_limiter is None:
            return True

        client_ip = self._get_client_ip()
        is_allowed, rate_info = self._rate_limiter.is_allowed(client_ip)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-RateLimit-Limit", str(rate_info["limit"]))
            self.send_header("X-RateLimit-Remaining", str(rate_info["remaining"]))
            self.send_header("X-RateLimit-Reset", str(rate_info["reset"]))
            self.send_header("Retry-After", "60")
            self.end_headers()
            error_response = json.dumps({
                "error": "Rate limit exceeded",
                "retry_after_seconds": 60,
            })
            self.wfile.write(error_response.encode())
            return False

        return True

    def _get_cached_response(self, path: str) -> tuple[str, int, str] | None:
        """Get cached response if available."""
        if not self._caching_enabled or self._response_cache is None:
            return None
        return self._response_cache.get(path)

    def _cache_response(
        self, path: str, response_data: str, status_code: int, content_type: str
    ) -> None:
        """Cache a response."""
        if not self._caching_enabled or self._response_cache is None:
            return

        ttl = self.CACHE_TTLS.get(path, 1.0)
        self._response_cache.set(path, response_data, status_code, content_type, ttl)

    def do_GET(self) -> None:
        """Handle GET requests with rate limiting, validation, and caching."""
        client_ip = self._get_client_ip()
        path = self.path.split("?")[0]  # Remove query string

        # P3: Validate request
        is_valid, error_msg = RequestValidator.validate_request("GET", self.path, client_ip)
        if not is_valid:
            logger.warning(f"Invalid request from {client_ip}: {error_msg}")
            self._send_response(400, {"error": error_msg or "Bad request"})
            return

        # P3: Check rate limit
        if not self._check_rate_limit():
            return  # Already sent 429 response

        # P3: Check cache first (for expensive endpoints)
        cached = self._get_cached_response(path)
        if cached:
            response_data, status_code, content_type = cached
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response_data)))
            self.send_header("X-Cache", "HIT")
            self.end_headers()
            self.wfile.write(response_data.encode())
            return

        # Route to appropriate handler
        if path == "/health" or path == "/":
            self._handle_health()
        elif path == "/live" or path == "/livez":
            self._handle_liveness()
        elif path == "/ready" or path == "/readyz":
            self._handle_readiness()
        elif path == "/metrics":
            self._handle_metrics()
        else:
            self._send_response(404, {"error": "Not found"})

    def _handle_health(self) -> None:
        """Handle /health endpoint - comprehensive health check."""
        result = self.checker.perform_health_check()
        status_code = {
            HealthStatus.HEALTHY: 200,
            HealthStatus.DEGRADED: 200,  # Still operational
            HealthStatus.UNHEALTHY: 503,
        }.get(result.status, 503)

        self._send_response(status_code, result.to_dict())

    def _handle_liveness(self) -> None:
        """Handle /live endpoint - Kubernetes liveness probe."""
        is_live, message = self.checker.check_liveness()
        status_code = 200 if is_live else 503

        self._send_response(status_code, {
            "status": "live" if is_live else "dead",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _handle_readiness(self) -> None:
        """Handle /ready endpoint - Kubernetes readiness probe."""
        is_ready, message = self.checker.check_readiness()
        status_code = 200 if is_ready else 503

        self._send_response(status_code, {
            "status": "ready" if is_ready else "not_ready",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _handle_metrics(self) -> None:
        """Handle /metrics endpoint - Prometheus-compatible metrics."""
        result = self.checker.perform_health_check()
        system_metrics = self.checker.get_system_metrics()
        business_metrics = self.checker.get_business_metrics()

        # MON-004: Get labels from config for multi-instance distinction
        config = self.checker._config
        labels = f'instance="{config.instance_id}",environment="{config.environment}",version="{config.version}"'

        # Format as Prometheus metrics
        lines = [
            "# HELP trading_system_up System health status (1=healthy, 0=unhealthy)",
            "# TYPE trading_system_up gauge",
            f"trading_system_up{{{labels}}} {1 if result.status != HealthStatus.UNHEALTHY else 0}",
            "",
            "# HELP trading_system_uptime_seconds System uptime in seconds",
            "# TYPE trading_system_uptime_seconds counter",
            f"trading_system_uptime_seconds{{{labels}}} {result.uptime_seconds:.2f}",
            "",
            "# HELP trading_system_component_health Component health status",
            "# TYPE trading_system_component_health gauge",
        ]

        for name, comp in result.components.items():
            value = {
                HealthStatus.HEALTHY: 1,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0,
            }.get(comp.status, 0)
            lines.append(f'trading_system_component_health{{{labels},component="{name}"}} {value}')

        # MON-011: Business metrics (P&L, drawdown, fill rate)
        lines.extend([
            "",
            "# HELP trading_pnl_daily Daily P&L in base currency",
            "# TYPE trading_pnl_daily gauge",
            f"trading_pnl_daily{{{labels}}} {business_metrics.get('pnl_daily', 0):.2f}",
            "",
            "# HELP trading_pnl_total Total cumulative P&L in base currency",
            "# TYPE trading_pnl_total gauge",
            f"trading_pnl_total{{{labels}}} {business_metrics.get('pnl_total', 0):.2f}",
            "",
            "# HELP trading_drawdown_pct Current drawdown percentage from peak",
            "# TYPE trading_drawdown_pct gauge",
            f"trading_drawdown_pct{{{labels}}} {business_metrics.get('drawdown_pct', 0):.4f}",
            "",
            "# HELP trading_drawdown_max_pct Maximum drawdown percentage",
            "# TYPE trading_drawdown_max_pct gauge",
            f"trading_drawdown_max_pct{{{labels}}} {business_metrics.get('max_drawdown_pct', 0):.4f}",
            "",
            "# HELP trading_fill_rate Order fill rate (filled/total orders)",
            "# TYPE trading_fill_rate gauge",
            f"trading_fill_rate{{{labels}}} {business_metrics.get('fill_rate', 0):.4f}",
            "",
            "# HELP trading_orders_total Total number of orders submitted",
            "# TYPE trading_orders_total counter",
            f"trading_orders_total{{{labels}}} {business_metrics.get('orders_total', 0)}",
            "",
            "# HELP trading_orders_filled Number of orders filled",
            "# TYPE trading_orders_filled counter",
            f"trading_orders_filled{{{labels}}} {business_metrics.get('orders_filled', 0)}",
            "",
            "# HELP trading_positions_count Number of open positions",
            "# TYPE trading_positions_count gauge",
            f"trading_positions_count{{{labels}}} {business_metrics.get('positions_count', 0)}",
        ])

        # CPU metrics
        cpu = system_metrics.get("cpu", {})
        if "percent" in cpu:
            lines.extend([
                "",
                "# HELP trading_system_cpu_percent System CPU utilization percentage",
                "# TYPE trading_system_cpu_percent gauge",
                f"trading_system_cpu_percent{{{labels}}} {cpu['percent']:.2f}",
            ])

        # Memory metrics
        memory = system_metrics.get("memory", {})
        if "percent" in memory:
            lines.extend([
                "",
                "# HELP trading_system_memory_percent System memory utilization percentage",
                "# TYPE trading_system_memory_percent gauge",
                f"trading_system_memory_percent{{{labels}}} {memory['percent']:.2f}",
                "",
                "# HELP trading_system_memory_used_bytes System memory used in bytes",
                "# TYPE trading_system_memory_used_bytes gauge",
                f"trading_system_memory_used_bytes{{{labels}}} {memory.get('used_bytes', 0)}",
            ])

        # Process metrics
        process = system_metrics.get("process", {})
        if "memory_bytes" in process:
            lines.extend([
                "",
                "# HELP trading_process_memory_bytes Process memory (RSS) in bytes",
                "# TYPE trading_process_memory_bytes gauge",
                f"trading_process_memory_bytes{{{labels}}} {process['memory_bytes']}",
                "",
                "# HELP trading_process_cpu_percent Process CPU utilization percentage",
                "# TYPE trading_process_cpu_percent gauge",
                f"trading_process_cpu_percent{{{labels}}} {process.get('cpu_percent', 0):.2f}",
                "",
                "# HELP trading_process_threads Number of threads",
                "# TYPE trading_process_threads gauge",
                f"trading_process_threads{{{labels}}} {process.get('threads', 0)}",
            ])

        # Event bus metrics
        event_bus = system_metrics.get("event_bus", {})
        if "queue_size" in event_bus:
            lines.extend([
                "",
                "# HELP trading_event_bus_queue_size Current event queue size",
                "# TYPE trading_event_bus_queue_size gauge",
                f"trading_event_bus_queue_size{{{labels}}} {event_bus['queue_size']}",
                "",
                "# HELP trading_event_bus_queue_utilization_pct Event queue utilization percentage",
                "# TYPE trading_event_bus_queue_utilization_pct gauge",
                f"trading_event_bus_queue_utilization_pct{{{labels}}} {event_bus.get('queue_utilization_pct', 0):.2f}",
                "",
                "# HELP trading_event_bus_events_processed Total events processed",
                "# TYPE trading_event_bus_events_processed counter",
                f"trading_event_bus_events_processed{{{labels}}} {event_bus.get('events_processed', 0)}",
            ])

            # Event bus latency metrics (if available)
            if "latency_avg_ms" in event_bus:
                lines.extend([
                    "",
                    "# HELP trading_event_bus_latency_avg_ms Average event processing latency in milliseconds",
                    "# TYPE trading_event_bus_latency_avg_ms gauge",
                    f"trading_event_bus_latency_avg_ms{{{labels}}} {event_bus['latency_avg_ms']:.2f}",
                    "",
                    "# HELP trading_event_bus_latency_p95_ms 95th percentile event processing latency in milliseconds",
                    "# TYPE trading_event_bus_latency_p95_ms gauge",
                    f"trading_event_bus_latency_p95_ms{{{labels}}} {event_bus.get('latency_p95_ms', 0):.2f}",
                    "",
                    "# HELP trading_event_bus_latency_p99_ms 99th percentile event processing latency in milliseconds",
                    "# TYPE trading_event_bus_latency_p99_ms gauge",
                    f"trading_event_bus_latency_p99_ms{{{labels}}} {event_bus.get('latency_p99_ms', 0):.2f}",
                    "",
                    "# HELP trading_event_bus_latency_max_ms Maximum event processing latency in milliseconds",
                    "# TYPE trading_event_bus_latency_max_ms gauge",
                    f"trading_event_bus_latency_max_ms{{{labels}}} {event_bus.get('latency_max_ms', 0):.2f}",
                ])

        content = "\n".join(lines)
        self._send_text_response(200, content, "text/plain; charset=utf-8")

    def _send_response(self, status_code: int, data: dict, cache: bool = True) -> None:
        """Send JSON response with optional caching."""
        content = json.dumps(data, indent=2)
        self._send_text_response(status_code, content, "application/json", cache=cache)

    def _send_text_response(
        self, status_code: int, content: str, content_type: str, cache: bool = True
    ) -> None:
        """Send text response with optional caching."""
        # Cache the response for future requests
        if cache:
            path = self.path.split("?")[0]
            self._cache_response(path, content, status_code, content_type)

        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Cache", "MISS")
        self.end_headers()
        self.wfile.write(content.encode())


class HealthCheckServer:
    """
    HTTP server for health check endpoints.

    Runs in a separate thread to avoid blocking the event loop.
    """

    def __init__(
        self,
        checker: HealthChecker,
        config: HealthCheckConfig | None = None,
    ):
        self._checker = checker
        self._config = config or HealthCheckConfig()
        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the health check server."""
        if self._running:
            logger.warning("Health check server already running")
            return

        # SEC-001: Security warning for non-localhost bindings without auth
        if self._config.host != "127.0.0.1" and self._config.host != "localhost":
            if self._config.require_auth_for_remote:
                logger.warning(
                    f"SECURITY WARNING: Health check server binding to {self._config.host}:{self._config.port} "
                    "which is accessible from network. Consider binding to 127.0.0.1 for local-only access, "
                    "or ensure proper network-level authentication/firewall is configured. "
                    "Set require_auth_for_remote=False to suppress this warning."
                )

        # Create handler with checker reference
        handler_class = partial(HealthCheckHandler, self._checker)

        try:
            self._server = HTTPServer(
                (self._config.host, self._config.port),
                handler_class,
            )
        except OSError as e:
            logger.error(f"Failed to start health check server: {e}")
            return

        self._running = True

        # Run in separate thread
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="health-check-server",
        )
        self._server_thread.start()

        logger.info(
            f"Health check server started at http://{self._config.host}:{self._config.port}"
        )
        logger.info(f"  Endpoints: /health, /live, /ready, /metrics")

    def _run_server(self) -> None:
        """Server loop running in thread."""
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.error(f"Health check server error: {e}")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the health check server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.shutdown()
            self._server = None

        if self._server_thread:
            self._server_thread.join(timeout=5.0)
            self._server_thread = None

        logger.info("Health check server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


# Convenience function to create and configure health checking
def create_health_check_server(
    get_status_fn: Callable[[], dict[str, Any]],
    event_bus: "EventBus | None" = None,
    monitoring: "MonitoringSystem | None" = None,
    config: HealthCheckConfig | None = None,
) -> tuple[HealthChecker, HealthCheckServer]:
    """
    Create a configured health check server.

    Args:
        get_status_fn: Function that returns system status dict
        event_bus: Optional event bus reference
        monitoring: Optional monitoring system reference
        config: Optional configuration

    Returns:
        Tuple of (HealthChecker, HealthCheckServer)
    """
    config = config or HealthCheckConfig()

    checker = HealthChecker(config=config, get_status_fn=get_status_fn)

    if event_bus:
        checker.set_event_bus(event_bus)
    if monitoring:
        checker.set_monitoring(monitoring)

    server = HealthCheckServer(checker, config)

    return checker, server

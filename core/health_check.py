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
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Callable, Any
import threading
from functools import partial

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.monitoring import MonitoringSystem


logger = logging.getLogger(__name__)


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

        except Exception as e:
            return ComponentHealth(
                name="monitoring",
                status=HealthStatus.DEGRADED,
                message=f"Error checking monitoring: {e}",
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


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints."""

    def __init__(self, checker: HealthChecker, *args, **kwargs):
        self.checker = checker
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging, use our logger instead."""
        logger.debug(f"Health check request: {args[0]}")

    def do_GET(self) -> None:
        """Handle GET requests."""
        path = self.path.split("?")[0]  # Remove query string

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

        # Format as Prometheus metrics
        lines = [
            "# HELP trading_system_up System health status (1=healthy, 0=unhealthy)",
            "# TYPE trading_system_up gauge",
            f"trading_system_up {1 if result.status != HealthStatus.UNHEALTHY else 0}",
            "",
            "# HELP trading_system_uptime_seconds System uptime in seconds",
            "# TYPE trading_system_uptime_seconds counter",
            f"trading_system_uptime_seconds {result.uptime_seconds:.2f}",
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
            lines.append(f'trading_system_component_health{{component="{name}"}} {value}')

        content = "\n".join(lines)
        self._send_text_response(200, content, "text/plain; charset=utf-8")

    def _send_response(self, status_code: int, data: dict) -> None:
        """Send JSON response."""
        content = json.dumps(data, indent=2)
        self._send_text_response(status_code, content, "application/json")

    def _send_text_response(
        self, status_code: int, content: str, content_type: str
    ) -> None:
        """Send text response."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
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

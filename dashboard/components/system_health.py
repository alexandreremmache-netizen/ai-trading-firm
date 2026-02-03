"""
System Health Monitor
=====================

Infrastructure and system health monitoring for the trading system dashboard.

Tracks health status of all system components including:
- EventBus
- Broker connection
- Database (if applicable)
- WebSocket server
- All registered agents

Features:
- Component health tracking with heartbeats
- System metrics collection (CPU, memory, disk, network)
- Health history for trend analysis
- Uptime reporting
- Alert generation for degraded components
- WebSocket-ready export to dict
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

# System metrics - optional dependency
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.broker import BrokerInterface
    from dashboard.components.agent_status import AgentStatusTracker


logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Health status levels for components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Severity levels for health alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """
    Health status for a single system component.

    Captures comprehensive health information for monitoring.
    """
    component_name: str
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_heartbeat: datetime | None = None
    uptime_seconds: float = 0.0
    error_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    # Additional tracking
    started_at: datetime | None = None
    last_status_change: datetime | None = None
    consecutive_failures: int = 0
    last_error: str | None = None
    last_error_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "error_count": self.error_count,
            "details": self.details,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_status_change": self.last_status_change.isoformat() if self.last_status_change else None,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
        }


@dataclass
class SystemMetrics:
    """
    System-level resource metrics.

    Captures CPU, memory, disk, and network metrics.
    """
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_latency_ms: float = 0.0
    event_queue_size: int = 0
    # Extended metrics
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0
    thread_count: int = 0
    open_file_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "disk_percent": round(self.disk_percent, 1),
            "network_latency_ms": round(self.network_latency_ms, 2),
            "event_queue_size": self.event_queue_size,
            "cpu_count": self.cpu_count,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "memory_available_gb": round(self.memory_available_gb, 2),
            "disk_total_gb": round(self.disk_total_gb, 2),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "process_memory_mb": round(self.process_memory_mb, 2),
            "process_cpu_percent": round(self.process_cpu_percent, 1),
            "thread_count": self.thread_count,
            "open_file_count": self.open_file_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthHistoryEntry:
    """Single entry in health history."""
    timestamp: datetime
    component_name: str
    status: ComponentStatus
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component_name": self.component_name,
            "status": self.status.value,
            "details": self.details,
        }


@dataclass
class HealthAlert:
    """Alert for degraded or unhealthy components."""
    alert_id: str
    timestamp: datetime
    component_name: str
    severity: AlertSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "component_name": self.component_name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
        }


@dataclass
class UptimeReport:
    """Uptime report for system and components."""
    report_time: datetime
    total_uptime_seconds: float
    system_start_time: datetime | None
    component_uptimes: dict[str, float] = field(default_factory=dict)
    component_availability_pct: dict[str, float] = field(default_factory=dict)
    downtime_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_time": self.report_time.isoformat(),
            "total_uptime_seconds": round(self.total_uptime_seconds, 2),
            "total_uptime_hours": round(self.total_uptime_seconds / 3600, 2),
            "system_start_time": self.system_start_time.isoformat() if self.system_start_time else None,
            "component_uptimes": {
                k: round(v, 2) for k, v in self.component_uptimes.items()
            },
            "component_availability_pct": {
                k: round(v, 2) for k, v in self.component_availability_pct.items()
            },
            "downtime_events": self.downtime_events,
        }


class SystemHealthMonitor:
    """
    Monitors infrastructure and system health across all components.

    Provides centralized health monitoring for:
    - EventBus health and queue metrics
    - Broker connection status and latency
    - Database connectivity (if applicable)
    - WebSocket server status
    - All registered agents
    - System resources (CPU, memory, disk)

    Usage:
        monitor = SystemHealthMonitor()

        # Set up component references
        monitor.set_event_bus(event_bus)
        monitor.set_broker(broker)
        monitor.set_agent_tracker(agent_tracker)

        # Start monitoring
        await monitor.start()

        # Check health
        health = await monitor.check_health()

        # Get all components
        components = monitor.get_all_components()

        # Get system metrics
        metrics = monitor.get_system_metrics()

        # Get health history
        history = monitor.get_health_history(limit=100)

        # Get uptime report
        report = monitor.get_uptime_report()

        # Export for dashboard
        data = monitor.to_dict()
    """

    # Component names
    COMPONENT_EVENT_BUS = "EventBus"
    COMPONENT_BROKER = "Broker"
    COMPONENT_DATABASE = "Database"
    COMPONENT_WEBSOCKET = "WebSocketServer"

    # Health thresholds
    HEARTBEAT_TIMEOUT_SECONDS = 60.0  # Consider unhealthy if no heartbeat
    DEGRADED_LATENCY_MS = 500.0  # Network latency threshold for degraded
    CRITICAL_LATENCY_MS = 2000.0  # Network latency threshold for unhealthy
    QUEUE_WARNING_PCT = 70.0  # Event queue utilization warning threshold
    QUEUE_CRITICAL_PCT = 90.0  # Event queue utilization critical threshold

    # History settings
    MAX_HISTORY_ENTRIES = 1000
    MAX_ALERTS = 100
    METRICS_HISTORY_SIZE = 500

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        network_check_host: str = "8.8.8.8",
        network_check_port: int = 53,
        enable_auto_alerts: bool = True,
    ):
        """
        Initialize the system health monitor.

        Args:
            check_interval_seconds: Interval between health checks
            network_check_host: Host to check for network latency
            network_check_port: Port to use for network check
            enable_auto_alerts: Whether to automatically generate alerts
        """
        self._check_interval = check_interval_seconds
        self._network_check_host = network_check_host
        self._network_check_port = network_check_port
        self._enable_auto_alerts = enable_auto_alerts

        # Component references
        self._event_bus: EventBus | None = None
        self._broker: BrokerInterface | None = None
        self._agent_tracker: AgentStatusTracker | None = None
        self._websocket_server: Any = None
        self._database: Any = None

        # Component health tracking
        self._components: dict[str, ComponentHealth] = {}
        self._custom_components: dict[str, Callable[[], ComponentHealth]] = {}

        # Health history
        self._health_history: deque[HealthHistoryEntry] = deque(maxlen=self.MAX_HISTORY_ENTRIES)
        self._metrics_history: deque[SystemMetrics] = deque(maxlen=self.METRICS_HISTORY_SIZE)

        # Alerts
        self._alerts: deque[HealthAlert] = deque(maxlen=self.MAX_ALERTS)
        self._alert_counter = 0
        self._alert_callbacks: list[Callable[[HealthAlert], None]] = []

        # State
        self._start_time: datetime | None = None
        self._running = False
        self._check_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Downtime tracking for uptime report
        self._downtime_events: deque[dict[str, Any]] = deque(maxlen=100)
        self._component_healthy_time: dict[str, float] = {}  # Total healthy seconds
        self._component_total_time: dict[str, float] = {}  # Total tracked seconds

        # Initialize core components
        self._initialize_core_components()

        logger.info(
            f"SystemHealthMonitor initialized with check_interval={check_interval_seconds}s"
        )

    def _initialize_core_components(self) -> None:
        """Initialize tracking for core components."""
        core_components = [
            self.COMPONENT_EVENT_BUS,
            self.COMPONENT_BROKER,
            self.COMPONENT_DATABASE,
            self.COMPONENT_WEBSOCKET,
        ]

        for name in core_components:
            self._components[name] = ComponentHealth(
                component_name=name,
                status=ComponentStatus.UNKNOWN,
            )

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set the event bus reference for monitoring."""
        self._event_bus = event_bus
        logger.debug("EventBus reference set for health monitoring")

    def set_broker(self, broker: BrokerInterface) -> None:
        """Set the broker reference for monitoring."""
        self._broker = broker
        logger.debug("Broker reference set for health monitoring")

    def set_agent_tracker(self, tracker: AgentStatusTracker) -> None:
        """Set the agent status tracker reference."""
        self._agent_tracker = tracker
        logger.debug("AgentStatusTracker reference set for health monitoring")

    def set_websocket_server(self, server: Any) -> None:
        """Set the WebSocket server reference for monitoring."""
        self._websocket_server = server
        logger.debug("WebSocket server reference set for health monitoring")

    def set_database(self, database: Any) -> None:
        """Set the database reference for monitoring."""
        self._database = database
        logger.debug("Database reference set for health monitoring")

    def register_custom_component(
        self,
        component_name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a custom component with a health check function.

        Args:
            component_name: Name of the component
            check_fn: Function that returns ComponentHealth
        """
        self._custom_components[component_name] = check_fn
        self._components[component_name] = ComponentHealth(
            component_name=component_name,
            status=ComponentStatus.UNKNOWN,
        )
        logger.info(f"Registered custom component for health monitoring: {component_name}")

    def on_alert(self, callback: Callable[[HealthAlert], None]) -> None:
        """Register callback for health alerts."""
        self._alert_callbacks.append(callback)

    async def start(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            logger.warning("SystemHealthMonitor already running")
            return

        self._running = True
        self._start_time = datetime.now(timezone.utc)

        # Initial health check
        await self.check_health()

        # Start periodic check task
        self._check_task = asyncio.create_task(self._run_check_loop())

        logger.info("SystemHealthMonitor started")

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False

        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("SystemHealthMonitor stopped")

    async def _run_check_loop(self) -> None:
        """Run periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)

                if not self._running:
                    break

                await self.check_health()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health check loop error: {e}")

    async def check_health(self) -> dict[str, ComponentHealth]:
        """
        Perform comprehensive health check on all components.

        Returns:
            Dictionary mapping component names to their health status
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            # Check each component
            await self._check_event_bus()
            await self._check_broker()
            await self._check_database()
            await self._check_websocket()
            await self._check_agents()
            await self._check_custom_components()

            # Collect system metrics
            metrics = self._collect_system_metrics()
            self._metrics_history.append(metrics)

            # Update history and tracking
            for name, health in self._components.items():
                # Record history
                self._health_history.append(HealthHistoryEntry(
                    timestamp=now,
                    component_name=name,
                    status=health.status,
                    details=dict(health.details),
                ))

                # Update uptime tracking
                if name not in self._component_total_time:
                    self._component_total_time[name] = 0.0
                    self._component_healthy_time[name] = 0.0

                self._component_total_time[name] += self._check_interval
                if health.status == ComponentStatus.HEALTHY:
                    self._component_healthy_time[name] += self._check_interval

                # Generate alerts if enabled
                if self._enable_auto_alerts:
                    self._check_and_generate_alert(health)

            return dict(self._components)

    async def _check_event_bus(self) -> None:
        """Check EventBus health."""
        health = self._components[self.COMPONENT_EVENT_BUS]
        now = datetime.now(timezone.utc)

        if not self._event_bus:
            health.status = ComponentStatus.UNKNOWN
            health.details = {"error": "EventBus not configured"}
            return

        try:
            status = self._event_bus.get_status()
            is_running = status.get("running", False)
            queue_size = status.get("queue_size", 0)
            max_queue = status.get("max_queue_size", 10000)
            queue_pct = (queue_size / max_queue * 100) if max_queue > 0 else 0
            backpressure = status.get("backpressure_level", "normal")

            # Update health
            health.last_heartbeat = now

            if not is_running:
                health.status = ComponentStatus.UNHEALTHY
                health.details = {
                    "running": False,
                    "error": "EventBus not running",
                }
            elif queue_pct >= self.QUEUE_CRITICAL_PCT:
                health.status = ComponentStatus.UNHEALTHY
                health.details = {
                    "running": True,
                    "queue_size": queue_size,
                    "queue_utilization_pct": round(queue_pct, 1),
                    "backpressure": backpressure,
                    "warning": "Queue at critical level",
                }
            elif queue_pct >= self.QUEUE_WARNING_PCT or backpressure in ("high", "critical"):
                health.status = ComponentStatus.DEGRADED
                health.details = {
                    "running": True,
                    "queue_size": queue_size,
                    "queue_utilization_pct": round(queue_pct, 1),
                    "backpressure": backpressure,
                    "warning": "Elevated queue depth or backpressure",
                }
            else:
                health.status = ComponentStatus.HEALTHY
                health.details = {
                    "running": True,
                    "queue_size": queue_size,
                    "queue_utilization_pct": round(queue_pct, 1),
                    "backpressure": backpressure,
                    "events_processed": status.get("metrics", {}).get("total_processed", 0),
                }

            # Update uptime
            eb_health = status.get("health", {})
            if health.started_at is None and is_running:
                health.started_at = now
            if health.started_at:
                health.uptime_seconds = (now - health.started_at).total_seconds()

            health.consecutive_failures = 0

        except Exception as e:
            logger.exception(f"Error checking EventBus health: {e}")
            health.status = ComponentStatus.UNHEALTHY
            health.consecutive_failures += 1
            health.error_count += 1
            health.last_error = str(e)
            health.last_error_time = now
            health.details = {"error": str(e)}

    async def _check_broker(self) -> None:
        """Check Broker connection health."""
        health = self._components[self.COMPONENT_BROKER]
        now = datetime.now(timezone.utc)

        if not self._broker:
            health.status = ComponentStatus.UNKNOWN
            health.details = {"error": "Broker not configured"}
            return

        try:
            # Check if broker has is_connected method
            is_connected = False
            latency_ms = 0.0
            broker_details = {}

            if hasattr(self._broker, "is_connected"):
                is_connected = self._broker.is_connected
            elif hasattr(self._broker, "connected"):
                is_connected = self._broker.connected

            # Get connection quality metrics if available
            if hasattr(self._broker, "get_connection_quality"):
                quality = self._broker.get_connection_quality()
                latency_ms = quality.get("avg_latency_ms", 0.0)
                broker_details["latency_ms"] = round(latency_ms, 2)
                broker_details["reconnect_count"] = quality.get("reconnect_count", 0)
                broker_details["uptime_percentage"] = quality.get("uptime_percentage", 0.0)

            # Get account info if available
            if hasattr(self._broker, "account_id"):
                broker_details["account_id"] = self._broker.account_id

            # Update health
            health.last_heartbeat = now

            if not is_connected:
                health.status = ComponentStatus.UNHEALTHY
                health.details = {
                    "connected": False,
                    "error": "Broker disconnected",
                    **broker_details,
                }
            elif latency_ms >= self.CRITICAL_LATENCY_MS:
                health.status = ComponentStatus.UNHEALTHY
                health.details = {
                    "connected": True,
                    "warning": f"Critical latency: {latency_ms:.0f}ms",
                    **broker_details,
                }
            elif latency_ms >= self.DEGRADED_LATENCY_MS:
                health.status = ComponentStatus.DEGRADED
                health.details = {
                    "connected": True,
                    "warning": f"High latency: {latency_ms:.0f}ms",
                    **broker_details,
                }
            else:
                health.status = ComponentStatus.HEALTHY
                health.details = {
                    "connected": True,
                    **broker_details,
                }

            # Update uptime
            if health.started_at is None and is_connected:
                health.started_at = now
            if health.started_at and is_connected:
                health.uptime_seconds = (now - health.started_at).total_seconds()

            health.consecutive_failures = 0

        except Exception as e:
            logger.exception(f"Error checking Broker health: {e}")
            health.status = ComponentStatus.UNHEALTHY
            health.consecutive_failures += 1
            health.error_count += 1
            health.last_error = str(e)
            health.last_error_time = now
            health.details = {"error": str(e)}

    async def _check_database(self) -> None:
        """Check Database health."""
        health = self._components[self.COMPONENT_DATABASE]
        now = datetime.now(timezone.utc)

        if not self._database:
            # Database not configured - mark as healthy (optional component)
            health.status = ComponentStatus.HEALTHY
            health.details = {"configured": False, "note": "Database not configured (optional)"}
            return

        try:
            # Check database connectivity
            is_connected = False

            if hasattr(self._database, "is_connected"):
                is_connected = self._database.is_connected
            elif hasattr(self._database, "connected"):
                is_connected = self._database.connected
            elif hasattr(self._database, "ping"):
                # Try a ping if available
                is_connected = await self._database.ping() if asyncio.iscoroutinefunction(self._database.ping) else self._database.ping()

            health.last_heartbeat = now

            if is_connected:
                health.status = ComponentStatus.HEALTHY
                health.details = {"connected": True}
                health.consecutive_failures = 0
            else:
                health.status = ComponentStatus.UNHEALTHY
                health.details = {"connected": False, "error": "Database disconnected"}

            if health.started_at is None and is_connected:
                health.started_at = now
            if health.started_at and is_connected:
                health.uptime_seconds = (now - health.started_at).total_seconds()

        except Exception as e:
            logger.exception(f"Error checking Database health: {e}")
            health.status = ComponentStatus.UNHEALTHY
            health.consecutive_failures += 1
            health.error_count += 1
            health.last_error = str(e)
            health.last_error_time = now
            health.details = {"error": str(e)}

    async def _check_websocket(self) -> None:
        """Check WebSocket server health."""
        health = self._components[self.COMPONENT_WEBSOCKET]
        now = datetime.now(timezone.utc)

        if not self._websocket_server:
            # WebSocket not configured - mark as healthy (optional component)
            health.status = ComponentStatus.HEALTHY
            health.details = {"configured": False, "note": "WebSocket server not configured (optional)"}
            return

        try:
            is_running = False
            client_count = 0

            if hasattr(self._websocket_server, "is_running"):
                is_running = self._websocket_server.is_running
            elif hasattr(self._websocket_server, "running"):
                is_running = self._websocket_server.running

            if hasattr(self._websocket_server, "client_count"):
                client_count = self._websocket_server.client_count
            elif hasattr(self._websocket_server, "clients"):
                client_count = len(self._websocket_server.clients)

            health.last_heartbeat = now

            if is_running:
                health.status = ComponentStatus.HEALTHY
                health.details = {
                    "running": True,
                    "client_count": client_count,
                }
                health.consecutive_failures = 0
            else:
                health.status = ComponentStatus.UNHEALTHY
                health.details = {
                    "running": False,
                    "error": "WebSocket server not running",
                }

            if health.started_at is None and is_running:
                health.started_at = now
            if health.started_at and is_running:
                health.uptime_seconds = (now - health.started_at).total_seconds()

        except Exception as e:
            logger.exception(f"Error checking WebSocket health: {e}")
            health.status = ComponentStatus.UNHEALTHY
            health.consecutive_failures += 1
            health.error_count += 1
            health.last_error = str(e)
            health.last_error_time = now
            health.details = {"error": str(e)}

    async def _check_agents(self) -> None:
        """Check health of all registered agents via AgentStatusTracker."""
        now = datetime.now(timezone.utc)

        if not self._agent_tracker:
            return

        try:
            # Get all agent statuses
            all_statuses = await self._agent_tracker.get_all_statuses()

            for agent_name, record in all_statuses.items():
                component_name = f"Agent:{agent_name}"

                # Create component health if not exists
                if component_name not in self._components:
                    self._components[component_name] = ComponentHealth(
                        component_name=component_name,
                    )

                health = self._components[component_name]
                health.last_heartbeat = now

                # Map agent status to component status
                agent_status = record.status.value if hasattr(record.status, "value") else str(record.status)

                if agent_status in ("active", "idle"):
                    health.status = ComponentStatus.HEALTHY
                elif agent_status == "error":
                    health.status = ComponentStatus.UNHEALTHY
                elif agent_status in ("stopped", "shutting_down"):
                    health.status = ComponentStatus.DEGRADED
                else:
                    health.status = ComponentStatus.UNKNOWN

                health.details = {
                    "agent_status": agent_status,
                    "events_processed": record.events_processed,
                    "error_count": record.error_count,
                    "health_score": round(record.health_score, 1),
                    "events_per_minute": round(record.events_per_minute, 2),
                }

                if record.started_at:
                    health.started_at = record.started_at
                    health.uptime_seconds = record.uptime_seconds

                health.error_count = record.error_count
                health.consecutive_failures = record.consecutive_errors
                health.last_error = record.last_error
                health.last_error_time = record.last_error_time

        except Exception as e:
            logger.exception(f"Error checking agents health: {e}")

    async def _check_custom_components(self) -> None:
        """Check health of custom registered components."""
        now = datetime.now(timezone.utc)

        for name, check_fn in self._custom_components.items():
            try:
                result = check_fn()
                self._components[name] = result
                result.last_heartbeat = now
            except Exception as e:
                logger.exception(f"Error checking custom component {name}: {e}")
                health = self._components.get(name) or ComponentHealth(component_name=name)
                health.status = ComponentStatus.UNHEALTHY
                health.last_error = str(e)
                health.last_error_time = now
                health.error_count += 1
                self._components[name] = health

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics."""
        metrics = SystemMetrics(timestamp=datetime.now(timezone.utc))

        # Get event queue size from EventBus
        if self._event_bus:
            try:
                status = self._event_bus.get_status()
                metrics.event_queue_size = status.get("queue_size", 0)
            except Exception as e:
                logger.debug(f"Could not get event queue size: {e}")

        # Get system metrics if psutil is available
        if HAS_PSUTIL:
            try:
                import os

                # CPU metrics
                metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                metrics.cpu_count = psutil.cpu_count()

                # Memory metrics
                mem = psutil.virtual_memory()
                metrics.memory_percent = mem.percent
                metrics.memory_total_gb = mem.total / (1024 ** 3)
                metrics.memory_available_gb = mem.available / (1024 ** 3)

                # Disk metrics
                disk = psutil.disk_usage(".")
                metrics.disk_percent = disk.percent
                metrics.disk_total_gb = disk.total / (1024 ** 3)
                metrics.disk_free_gb = disk.free / (1024 ** 3)

                # Process metrics
                process = psutil.Process(os.getpid())
                metrics.process_memory_mb = process.memory_info().rss / (1024 ** 2)
                metrics.process_cpu_percent = process.cpu_percent(interval=0.1)
                metrics.thread_count = process.num_threads()

                try:
                    metrics.open_file_count = len(process.open_files())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    metrics.open_file_count = 0

            except Exception as e:
                logger.debug(f"Error collecting system metrics: {e}")

        # Measure network latency
        metrics.network_latency_ms = self._measure_network_latency()

        return metrics

    def _measure_network_latency(self) -> float:
        """Measure network latency to configured host."""
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self._network_check_host, self._network_check_port))
            sock.close()
            latency_ms = (time.time() - start) * 1000
            return latency_ms
        except Exception as e:
            logger.debug(f"Network latency check failed: {e}")
            return -1.0  # Indicate failure

    def _check_and_generate_alert(self, health: ComponentHealth) -> None:
        """Check component health and generate alert if needed."""
        if health.status == ComponentStatus.HEALTHY:
            return

        # Determine severity
        if health.status == ComponentStatus.UNHEALTHY:
            severity = AlertSeverity.CRITICAL
        elif health.status == ComponentStatus.DEGRADED:
            severity = AlertSeverity.WARNING
        else:
            return  # Don't alert on unknown

        # Generate alert
        self._alert_counter += 1
        alert = HealthAlert(
            alert_id=f"alert_{self._alert_counter}_{int(time.time())}",
            timestamp=datetime.now(timezone.utc),
            component_name=health.component_name,
            severity=severity,
            message=f"{health.component_name} is {health.status.value}",
            details=dict(health.details),
        )

        self._alerts.append(alert)

        # Track downtime
        if health.status == ComponentStatus.UNHEALTHY:
            self._downtime_events.append({
                "component": health.component_name,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "error": health.last_error,
            })

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.exception(f"Alert callback error: {e}")

        logger.warning(
            f"Health alert [{severity.value}]: {health.component_name} - {alert.message}"
        )

    def get_all_components(self) -> dict[str, ComponentHealth]:
        """
        Get health status of all components.

        Returns:
            Dictionary mapping component names to their health status
        """
        return dict(self._components)

    def get_component(self, component_name: str) -> ComponentHealth | None:
        """
        Get health status of a specific component.

        Args:
            component_name: Name of the component

        Returns:
            ComponentHealth or None if not found
        """
        return self._components.get(component_name)

    def get_system_metrics(self) -> SystemMetrics:
        """
        Get current system metrics.

        Returns:
            Current SystemMetrics
        """
        return self._collect_system_metrics()

    def get_metrics_history(self, limit: int = 100) -> list[SystemMetrics]:
        """
        Get historical system metrics.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of SystemMetrics (most recent first)
        """
        history = list(self._metrics_history)
        history.reverse()
        return history[:limit]

    def get_health_history(
        self,
        component_name: str | None = None,
        status: ComponentStatus | None = None,
        limit: int = 100,
    ) -> list[HealthHistoryEntry]:
        """
        Get health history with optional filtering.

        Args:
            component_name: Filter by component name
            status: Filter by status
            limit: Maximum number of entries to return

        Returns:
            List of HealthHistoryEntry (most recent first)
        """
        history = list(self._health_history)

        if component_name:
            history = [h for h in history if h.component_name == component_name]

        if status:
            history = [h for h in history if h.status == status]

        history.reverse()
        return history[:limit]

    def get_uptime_report(self) -> UptimeReport:
        """
        Generate uptime report for system and all components.

        Returns:
            UptimeReport with uptime statistics
        """
        now = datetime.now(timezone.utc)

        # Calculate total uptime
        total_uptime = 0.0
        if self._start_time:
            total_uptime = (now - self._start_time).total_seconds()

        # Calculate component availability percentages
        availability = {}
        for name, total_time in self._component_total_time.items():
            if total_time > 0:
                healthy_time = self._component_healthy_time.get(name, 0.0)
                availability[name] = (healthy_time / total_time) * 100
            else:
                availability[name] = 100.0  # Assume 100% if no data

        # Get component uptimes
        uptimes = {}
        for name, health in self._components.items():
            uptimes[name] = health.uptime_seconds

        # Compile downtime events
        downtime = list(self._downtime_events)

        return UptimeReport(
            report_time=now,
            total_uptime_seconds=total_uptime,
            system_start_time=self._start_time,
            component_uptimes=uptimes,
            component_availability_pct=availability,
            downtime_events=downtime,
        )

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        acknowledged: bool | None = None,
        limit: int = 50,
    ) -> list[HealthAlert]:
        """
        Get health alerts with optional filtering.

        Args:
            severity: Filter by severity
            acknowledged: Filter by acknowledged status
            limit: Maximum number of alerts to return

        Returns:
            List of HealthAlert (most recent first)
        """
        alerts = list(self._alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        alerts.reverse()
        return alerts[:limit]

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = "system",
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Who acknowledged the alert

        Returns:
            True if alert was found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def record_heartbeat(self, component_name: str) -> bool:
        """
        Record a heartbeat for a component.

        Args:
            component_name: Name of the component

        Returns:
            True if component exists and heartbeat recorded
        """
        if component_name in self._components:
            self._components[component_name].last_heartbeat = datetime.now(timezone.utc)
            return True
        return False

    def get_overall_status(self) -> ComponentStatus:
        """
        Get overall system health status.

        Returns:
            ComponentStatus representing overall system health
        """
        if not self._components:
            return ComponentStatus.UNKNOWN

        statuses = [h.status for h in self._components.values()]

        if any(s == ComponentStatus.UNHEALTHY for s in statuses):
            return ComponentStatus.UNHEALTHY
        elif any(s == ComponentStatus.DEGRADED for s in statuses):
            return ComponentStatus.DEGRADED
        elif all(s == ComponentStatus.HEALTHY for s in statuses):
            return ComponentStatus.HEALTHY
        else:
            return ComponentStatus.DEGRADED

    def to_dict(self) -> dict[str, Any]:
        """
        Export monitor state to dictionary for WebSocket streaming.

        Returns:
            Complete monitor state as dict
        """
        now = datetime.now(timezone.utc)
        metrics = self.get_system_metrics()
        uptime_report = self.get_uptime_report()
        recent_alerts = self.get_alerts(limit=10)

        # Group components by category
        core_components = {}
        agent_components = {}
        custom_components = {}

        for name, health in self._components.items():
            if name.startswith("Agent:"):
                agent_components[name] = health.to_dict()
            elif name in self._custom_components:
                custom_components[name] = health.to_dict()
            else:
                core_components[name] = health.to_dict()

        return {
            "overall_status": self.get_overall_status().value,
            "system_metrics": metrics.to_dict(),
            "components": {
                "core": core_components,
                "agents": agent_components,
                "custom": custom_components,
            },
            "uptime": {
                "total_seconds": round(uptime_report.total_uptime_seconds, 2),
                "total_hours": round(uptime_report.total_uptime_seconds / 3600, 2),
                "start_time": uptime_report.system_start_time.isoformat() if uptime_report.system_start_time else None,
            },
            "alerts": {
                "total": len(self._alerts),
                "unacknowledged": sum(1 for a in self._alerts if not a.acknowledged),
                "recent": [a.to_dict() for a in recent_alerts],
            },
            "monitoring": {
                "running": self._running,
                "check_interval_seconds": self._check_interval,
                "history_entries": len(self._health_history),
                "metrics_history_entries": len(self._metrics_history),
            },
            "timestamp": now.isoformat(),
        }

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def component_count(self) -> int:
        """Get total number of tracked components."""
        return len(self._components)

    @property
    def healthy_component_count(self) -> int:
        """Get number of healthy components."""
        return sum(
            1 for h in self._components.values()
            if h.status == ComponentStatus.HEALTHY
        )

    @property
    def unhealthy_component_count(self) -> int:
        """Get number of unhealthy components."""
        return sum(
            1 for h in self._components.values()
            if h.status == ComponentStatus.UNHEALTHY
        )

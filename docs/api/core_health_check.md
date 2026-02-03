# health_check

**Path**: `C:\Users\Alexa\ai-trading-firm\core\health_check.py`

## Overview

Health Check Server
===================

HTTP health check endpoints for monitoring and orchestration systems.
Addresses issue #S5: No health check endpoint for monitoring.

Features:
- Kubernetes-style liveness and readiness probes
- Detailed health status with component checks
- Prometheus-compatible metrics endpoint
- Configurable health check criteria

## Classes

### HealthStatus

**Inherits from**: Enum

Health status levels.

### ComponentHealth

Health status for a single component.

### HealthCheckResult

Overall health check result.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary for JSON response.

### HealthCheckConfig

Configuration for health check server.

### HealthChecker

Health checking logic for the trading system.

Performs checks on:
- Event bus health
- Broker connectivity
- Agent status
- Resource utilization

#### Methods

##### `def __init__(self, config: , get_status_fn: )`

##### `def set_event_bus(self, event_bus: EventBus) -> None`

Set the event bus reference for health checks.

##### `def set_monitoring(self, monitoring: MonitoringSystem) -> None`

Set the monitoring system reference.

##### `def set_ready(self, ready: bool) -> None`

Mark the system as ready (or not ready).

##### `def add_custom_check(self, check_fn: Callable[, ComponentHealth]) -> None`

Add a custom health check function.

##### `def check_liveness(self) -> tuple[bool, str]`

Liveness probe - is the process alive and not deadlocked?

Returns:
    Tuple of (is_live, message)

##### `def check_readiness(self) -> tuple[bool, str]`

Readiness probe - is the system ready to accept work?

Returns:
    Tuple of (is_ready, message)

##### `def perform_health_check(self) -> HealthCheckResult`

Perform comprehensive health check.

Returns:
    HealthCheckResult with all component statuses

### HealthCheckHandler

**Inherits from**: BaseHTTPRequestHandler

HTTP request handler for health check endpoints.

#### Methods

##### `def __init__(self, checker: HealthChecker)`

##### `def log_message(self, format: str) -> None`

Suppress default logging, use our logger instead.

##### `def do_GET(self) -> None`

Handle GET requests.

### HealthCheckServer

HTTP server for health check endpoints.

Runs in a separate thread to avoid blocking the event loop.

#### Methods

##### `def __init__(self, checker: HealthChecker, config: )`

##### `def start(self) -> None`

Start the health check server.

##### `def stop(self) -> None`

Stop the health check server.

##### `def is_running(self) -> bool`

Check if server is running.

## Functions

### `def create_health_check_server(get_status_fn: Callable[, dict[str, Any]], event_bus: EventBus | None, monitoring: MonitoringSystem | None, config: ) -> tuple[HealthChecker, HealthCheckServer]`

Create a configured health check server.

Args:
    get_status_fn: Function that returns system status dict
    event_bus: Optional event bus reference
    monitoring: Optional monitoring system reference
    config: Optional configuration

Returns:
    Tuple of (HealthChecker, HealthCheckServer)

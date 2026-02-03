# infrastructure_ops

**Path**: `C:\Users\Alexa\ai-trading-firm\core\infrastructure_ops.py`

## Overview

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

## Classes

### LogLevel

**Inherits from**: Enum

Log level enumeration.

### AlertSeverity

**Inherits from**: Enum

Alert severity levels.

### AlertStatus

**Inherits from**: Enum

Alert status.

### DeploymentColor

**Inherits from**: Enum

Blue-green deployment color.

### FeatureFlagStatus

**Inherits from**: Enum

Feature flag status.

### ChaosExperimentType

**Inherits from**: Enum

Types of chaos experiments.

### LogEntry

Structured log entry.

### PerformanceMetric

APM performance metric.

### Alert

Alert definition.

### TraceSpan

Distributed trace span.

### FeatureFlag

Feature flag definition.

### ChaosExperiment

Chaos experiment definition.

### LogAggregator

Centralized log aggregation system.

Features:
- Structured logging
- Log buffering and batching
- Multiple output destinations
- Log correlation with trace IDs
- Log level filtering
- Log retention policies

#### Methods

##### `def __init__(self, service_name: str, buffer_size: int, flush_interval_seconds: int)`

Initialize log aggregator.

##### `def start(self) -> None`

Start the log aggregator background flush thread.

##### `def stop(self) -> None`

Stop the log aggregator and flush remaining logs.

##### `def add_output(self, output_handler: Callable[, None]) -> None`

Add an output destination for logs.

##### `def add_filter(self, filter_func: Callable[, bool]) -> None`

Add a log filter (returns True to keep log).

##### `def set_trace_context(self, trace_id: str) -> None`

Set current trace context for log correlation.

##### `def clear_trace_context(self) -> None`

Clear current trace context.

##### `def log(self, level: LogLevel, message: str, component: str, metadata: Optional[Dict[str, Any]], exception: Optional[str]) -> None`

Log a message with structured data.

Args:
    level: Log level
    message: Log message
    component: Component name
    metadata: Additional metadata
    exception: Exception string if any

##### `def debug(self, message: str, component: str) -> None`

Log debug message.

##### `def info(self, message: str, component: str) -> None`

Log info message.

##### `def warning(self, message: str, component: str) -> None`

Log warning message.

##### `def error(self, message: str, component: str, exception: Optional[str]) -> None`

Log error message.

##### `def critical(self, message: str, component: str, exception: Optional[str]) -> None`

Log critical message.

##### `def flush(self) -> None`

Flush buffered logs to outputs.

##### `def search_logs(self, start_time: datetime, end_time: datetime, level: Optional[LogLevel], component: Optional[str], trace_id: Optional[str]) -> List[LogEntry]`

Search logs (in-memory buffer only for this implementation).

In production, this would query a log storage backend.

### APMCollector

Application Performance Monitoring collector.

Features:
- Metric collection (counters, gauges, histograms)
- Transaction tracing
- Resource utilization tracking
- SLA monitoring
- Custom dashboards data

#### Methods

##### `def __init__(self, service_name: str, flush_interval_seconds: int)`

Initialize APM collector.

##### `def record_metric(self, name: str, value: float, unit: str, tags: Optional[Dict[str, str]], aggregation: str) -> None`

Record a performance metric.

##### `def increment_counter(self, name: str, value: float, tags: Optional[Dict[str, str]]) -> None`

Increment a counter metric.

##### `def set_gauge(self, name: str, value: float, unit: str, tags: Optional[Dict[str, str]]) -> None`

Set a gauge metric.

##### `def record_histogram(self, name: str, value: float, unit: str, tags: Optional[Dict[str, str]]) -> None`

Record a histogram value.

##### `def start_transaction(self, transaction_name: str) -> TransactionContext`

Start a transaction for timing.

##### `def record_transaction_time(self, transaction_name: str, duration_ms: float) -> None`

Record a transaction duration.

##### `def set_sla_threshold(self, transaction_name: str, threshold_ms: float) -> None`

Set SLA threshold for a transaction.

##### `def get_counter(self, name: str) -> float`

Get current counter value.

##### `def get_gauge(self, name: str) -> float`

Get current gauge value.

##### `def get_histogram_stats(self, name: str) -> Dict[str, float]`

Get histogram statistics.

##### `def get_transaction_stats(self, transaction_name: str) -> Dict[str, Any]`

Get transaction timing statistics.

##### `def get_all_metrics(self) -> Dict[str, Any]`

Get all collected metrics.

### TransactionContext

Context manager for transaction timing.

#### Methods

##### `def __init__(self, apm: APMCollector, transaction_name: str)`

##### `def __enter__(self) -> TransactionContext`

##### `def __exit__(self, exc_type, exc_val, exc_tb) -> None`

### EscalationPolicy

Escalation policy definition.

### AlertManager

Alert management and escalation system.

Features:
- Alert creation and routing
- Escalation policies
- Alert grouping and deduplication
- On-call integration
- Alert suppression rules

#### Methods

##### `def __init__(self, service_name: str)`

Initialize alert manager.

##### `def start(self) -> None`

Start the alert manager escalation thread.

##### `def stop(self) -> None`

Stop the alert manager.

##### `def create_alert(self, severity: AlertSeverity, title: str, description: str, source: str, metadata: Optional[Dict[str, Any]], group_key: Optional[str]) -> Alert`

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

##### `def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool`

Acknowledge an alert.

##### `def resolve_alert(self, alert_id: str) -> bool`

Resolve an alert.

##### `def add_escalation_policy(self, policy: EscalationPolicy) -> None`

Add an escalation policy.

##### `def add_notification_handler(self, severity: AlertSeverity, handler: Callable[, None]) -> None`

Add a notification handler for a severity level.

##### `def add_suppression_rule(self, rule: Callable[, bool]) -> None`

Add an alert suppression rule (returns True to suppress).

##### `def get_open_alerts(self, severity: Optional[AlertSeverity]) -> List[Alert]`

Get all open alerts, optionally filtered by severity.

##### `def get_alert_statistics(self) -> Dict[str, Any]`

Get alert statistics.

### ChaosEngine

Chaos engineering framework for resilience testing.

Features:
- Latency injection
- Error injection
- Resource exhaustion simulation
- Network partition simulation
- Dependency failure simulation

#### Methods

##### `def __init__(self)`

Initialize chaos engine.

##### `def create_experiment(self, experiment_type: ChaosExperimentType, target_service: str, parameters: Dict[str, Any], duration_seconds: int) -> ChaosExperiment`

Create a chaos experiment.

Args:
    experiment_type: Type of chaos to inject
    target_service: Target service name
    parameters: Experiment-specific parameters
    duration_seconds: How long to run the experiment

Returns:
    Created experiment

##### `def start_experiment(self, experiment_id: str) -> bool`

Start a chaos experiment.

##### `def stop_experiment(self, experiment_id: str) -> bool`

Stop a chaos experiment.

##### `def inject_chaos(self, service_name: str, experiment_type: ChaosExperimentType) -> None`

Inject chaos if there's an active experiment for this service/type.

Call this from your code at points where you want chaos injection.

##### `def get_active_experiments(self) -> List[ChaosExperiment]`

Get all active experiments.

##### `def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]`

Get results for an experiment.

### PooledConnection

Wrapper for pooled connections.

### ConnectionPool

Generic connection pool implementation.

Features:
- Min/max pool size
- Connection health checking
- Automatic connection recycling
- Connection timeout handling
- Pool statistics

#### Methods

##### `def __init__(self, factory: Callable[, T], min_size: int, max_size: int, max_idle_seconds: int, health_check: Optional[Callable[, bool]], cleanup: Optional[Callable[, None]])`

Initialize connection pool.

Args:
    factory: Function to create new connections
    min_size: Minimum pool size
    max_size: Maximum pool size
    max_idle_seconds: Max idle time before recycling
    health_check: Function to check connection health
    cleanup: Function to cleanup connections

##### `def acquire(self, timeout: float) -> T`

Acquire a connection from the pool.

Args:
    timeout: Timeout in seconds

Returns:
    Connection object

Raises:
    TimeoutError if no connection available

##### `def release(self, connection: T) -> None`

Release a connection back to the pool.

##### `def get_stats(self) -> Dict[str, Any]`

Get pool statistics.

##### `def close(self) -> None`

Close all connections in the pool.

### PooledConnectionContext

Context manager for automatic connection release.

#### Methods

##### `def __init__(self, pool: ConnectionPool)`

##### `def __enter__(self) -> Any`

##### `def __exit__(self, exc_type, exc_val, exc_tb) -> None`

### CacheEntry

Cache entry with metadata.

### CacheWarmer

Cache warming and management system.

Features:
- Proactive cache warming
- TTL-based expiration
- LRU eviction
- Warm-up strategies
- Cache statistics

#### Methods

##### `def __init__(self, max_size: int, default_ttl_seconds: int)`

Initialize cache warmer.

##### `def get(self, key: str) -> Optional[Any]`

Get value from cache.

##### `def set(self, key: str, value: Any, ttl_seconds: Optional[int]) -> None`

Set value in cache.

##### `def delete(self, key: str) -> bool`

Delete key from cache.

##### `def add_warming_task(self, task: Callable[, Dict[str, Any]]) -> None`

Add a cache warming task.

Task should return a dict of {key: value} to cache.

##### `def warm_cache(self, ttl_seconds: Optional[int]) -> int`

Execute all warming tasks and populate cache.

Returns number of entries added.

##### `def get_or_set(self, key: str, factory: Callable[, Any], ttl_seconds: Optional[int]) -> Any`

Get from cache or compute and set.

##### `def get_stats(self) -> Dict[str, Any]`

Get cache statistics.

##### `def clear(self) -> None`

Clear all cache entries.

### RequestTracer

Distributed request tracing system.

Features:
- Trace context propagation
- Span creation and management
- Cross-service tracing
- Trace sampling
- Trace export

#### Methods

##### `def __init__(self, service_name: str, sample_rate: float)`

Initialize request tracer.

##### `def start_trace(self, operation_name: str) -> TraceSpan`

Start a new trace.

##### `def start_span(self, operation_name: str, parent_span: Optional[TraceSpan]) -> TraceSpan`

Start a new span within current trace.

##### `def end_span(self, span: TraceSpan, status: str) -> None`

End a span.

##### `def add_span_tag(self, span: TraceSpan, key: str, value: str) -> None`

Add a tag to a span.

##### `def add_span_log(self, span: TraceSpan, message: str, fields: Optional[Dict[str, Any]]) -> None`

Add a log entry to a span.

##### `def add_exporter(self, exporter: Callable[, None]) -> None`

Add a trace exporter.

##### `def get_trace(self, trace_id: str) -> List[TraceSpan]`

Get all spans for a trace.

##### `def inject_context(self, span: TraceSpan) -> Dict[str, str]`

Inject trace context for propagation to other services.

##### `def extract_context(self, headers: Dict[str, str]) -> Optional[Tuple[str, str]]`

Extract trace context from propagated headers.

### SpanContext

Context manager for automatic span management.

#### Methods

##### `def __init__(self, tracer: RequestTracer, operation_name: str, parent: Optional[TraceSpan])`

##### `def __enter__(self) -> TraceSpan`

##### `def __exit__(self, exc_type, exc_val, exc_tb) -> None`

### DeploymentInstance

Deployment instance information.

### BlueGreenDeployer

Blue-green deployment manager.

Features:
- Zero-downtime deployments
- Traffic shifting
- Health checking
- Rollback support
- Canary deployments

#### Methods

##### `def __init__(self, service_name: str)`

Initialize blue-green deployer.

##### `def register_deployment(self, color: DeploymentColor, version: str, health_endpoint: str) -> None`

Register a deployment instance.

##### `def start_health_checks(self, interval_seconds: int) -> None`

Start background health checking.

##### `def stop_health_checks(self) -> None`

Stop health checking.

##### `def set_active(self, color: DeploymentColor) -> bool`

Set the active deployment (100% traffic).

##### `def shift_traffic(self, target_color: DeploymentColor, percentage: float) -> bool`

Shift a percentage of traffic to target deployment.

Args:
    target_color: Deployment to shift traffic to
    percentage: Percentage of traffic (0-100)

Returns:
    True if successful

##### `def get_deployment_for_request(self) -> Optional[DeploymentInstance]`

Get deployment to route a request to based on traffic weights.

Returns:
    Selected deployment instance

##### `def rollback(self) -> bool`

Rollback to the inactive deployment.

##### `def get_status(self) -> Dict[str, Any]`

Get deployment status.

### FeatureFlagManager

Feature flag management system.

Features:
- Boolean flags
- Percentage rollouts
- User segment targeting
- A/B testing support
- Flag analytics

#### Methods

##### `def __init__(self)`

Initialize feature flag manager.

##### `def create_flag(self, name: str, default_value: bool, description: str, status: FeatureFlagStatus) -> FeatureFlag`

Create a new feature flag.

##### `def set_flag_status(self, name: str, status: FeatureFlagStatus) -> bool`

Set flag status.

##### `def set_percentage_rollout(self, name: str, percentage: float) -> bool`

Set percentage rollout for a flag.

##### `def set_user_segments(self, name: str, segments: List[str]) -> bool`

Set user segments for a flag.

##### `def set_user_override(self, user_id: str, flag_name: str, value: bool) -> None`

Set a user-specific override for a flag.

##### `def is_enabled(self, name: str, user_id: Optional[str], user_segments: Optional[List[str]], default: bool) -> bool`

Check if a feature flag is enabled.

Args:
    name: Flag name
    user_id: Optional user ID for percentage/override checks
    user_segments: Optional user segments for targeting
    default: Default value if flag not found

Returns:
    True if flag is enabled

##### `def get_flag_analytics(self, name: str) -> Dict[str, Any]`

Get analytics for a flag.

##### `def get_all_flags(self) -> Dict[str, Dict[str, Any]]`

Get all flags and their status.

## Functions

### `def console_log_output(entries: List[LogEntry]) -> None`

Console output handler for logs.

### `def json_log_output(entries: List[LogEntry]) -> List[str]`

JSON output handler for logs.

### `def apm_timed(apm: APMCollector, transaction_name: str)`

Decorator for automatic transaction timing.

### `def chaos_injection_point(engine: ChaosEngine, service_name: str, experiment_type: ChaosExperimentType)`

Decorator to add chaos injection point to a function.

### `def cached(cache: CacheWarmer, ttl_seconds: Optional[int])`

Decorator for automatic caching.

### `def traced(tracer: RequestTracer, operation_name: Optional[str])`

Decorator for automatic span creation.

### `def feature_flag(manager: FeatureFlagManager, flag_name: str, default: bool)`

Decorator for feature-flagged functions.

### `def create_infrastructure_suite(service_name: str) -> Dict[str, Any]`

Create a complete infrastructure suite.

Returns:
    Dictionary containing all infrastructure components

## Constants

- `T`

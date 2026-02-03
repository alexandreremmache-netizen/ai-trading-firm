# system_infrastructure

**Path**: `C:\Users\Alexa\ai-trading-firm\core\system_infrastructure.py`

## Overview

System Infrastructure Module
============================

Hot reload, dependency injection, and distributed tracing support.

Issues Addressed:
- #S9: No hot reload of configuration
- #S10: Missing dependency injection framework
- #S12: No distributed tracing support

## Classes

### ConfigChangeType

**Inherits from**: str, Enum

Type of configuration change.

### ConfigChange

Record of configuration change (#S9).

#### Methods

##### `def to_dict(self) -> dict`

### HotReloadableConfig

Configuration with hot reload support (#S9).

Features:
- File watching for changes
- Automatic reload on change
- Change callbacks for dependent systems
- Validation before applying changes
- Rollback on failed validation

#### Methods

##### `def __init__(self, config_path: , auto_reload: bool, poll_interval_seconds: float)`

##### `def load_from_file(self, path: str) -> dict[str, Any]`

Load configuration from file.

##### `def load_from_dict(self, config: dict[str, Any], source: str) -> None`

Load configuration from dictionary.

##### `def get(self, key: str, default: Any) -> Any`

Get configuration value.

##### `def get_nested(self, path: str, default: Any, separator: str) -> Any`

Get nested configuration value using dot notation.

##### `def set(self, key: str, value: Any) -> None`

Set configuration value (triggers callbacks).

##### `def register_callback(self, key_pattern: str, callback: Callable[, None]) -> None`

Register callback for configuration changes (#S9).

Args:
    key_pattern: Key or pattern (e.g., "database.*")
    callback: Function called with (key, old_value, new_value)

##### `def register_validator(self, key_pattern: str, validator: Callable[, tuple[bool, str]]) -> None`

Register validator for configuration values (#S9).

Args:
    key_pattern: Key or pattern
    validator: Function returning (is_valid, error_message)

##### `def start_watching(self) -> None`

Start watching configuration file for changes (#S9).

##### `def stop_watching(self) -> None`

Stop watching configuration file.

##### `def get_change_history(self, limit: int) -> list[ConfigChange]`

Get configuration change history.

##### `def to_dict(self) -> dict[str, Any]`

Get full configuration as dictionary.

### Scope

**Inherits from**: str, Enum

Dependency injection scope (#S10).

### ServiceRegistration

Service registration for DI (#S10).

### DIContainer

Dependency injection container (#S10).

Features:
- Constructor injection
- Interface-to-implementation binding
- Singleton, transient, and scoped lifetimes
- Factory functions
- Circular dependency detection
- Child containers for scoped resolution

#### Methods

##### `def __init__(self, parent: DIContainer | None)`

##### `def register_singleton(self, service_type: Type[T], implementation: ) -> DIContainer`

Register singleton service (#S10).

Args:
    service_type: Interface/base type
    implementation: Implementation class or instance

Returns:
    Self for chaining

##### `def register_transient(self, service_type: Type[T], implementation: ) -> DIContainer`

Register transient service (#S10).

New instance created for each resolution.

##### `def register_scoped(self, service_type: Type[T], implementation: ) -> DIContainer`

Register scoped service (#S10).

One instance per scope (child container).

##### `def register_factory(self, service_type: Type[T], factory: Callable[, T], scope: Scope) -> DIContainer`

Register factory function (#S10).

Args:
    service_type: Type to register
    factory: Function taking container and returning instance
    scope: Lifetime scope

##### `def resolve(self, service_type: Type[T]) -> T`

Resolve service from container (#S10).

Args:
    service_type: Type to resolve

Returns:
    Service instance

Raises:
    ValueError: If service not registered
    RuntimeError: If circular dependency detected

##### `def create_scope(self) -> DIContainer`

Create child container for scoped resolution (#S10).

Returns:
    New container with this as parent

##### `def try_resolve(self, service_type: Type[T])`

Try to resolve service, returning None if not found.

##### `def is_registered(self, service_type: Type) -> bool`

Check if service is registered.

### SpanContext

Context for a trace span (#S12).

#### Methods

##### `def to_dict(self) -> dict`

### TraceContext

Trace context for distributed tracing (#S12).

Follows OpenTelemetry-like semantics.

#### Methods

##### `def __init__(self, trace_id: , parent_span_id: , baggage: )`

##### `def start_span(self, operation_name: str, service_name: str, tags: ) -> SpanContext`

Start a new span (#S12).

Args:
    operation_name: Name of operation
    service_name: Name of service
    tags: Initial tags

Returns:
    New span context

##### `def end_span(self, span: , status: str, error: ) -> None`

End a span (#S12).

Args:
    span: Span to end (current if None)
    status: Final status
    error: Error message if failed

##### `def add_tag(self, key: str, value: str, span: ) -> None`

Add tag to span.

##### `def add_log(self, message: str, level: str, span: ) -> None`

Add log entry to span.

##### `def set_baggage(self, key: str, value: str) -> None`

Set baggage item (propagates to child spans).

##### `def get_spans(self) -> list[SpanContext]`

Get all spans in trace.

##### `def current_span(self)`

Get current span.

##### `def to_dict(self) -> dict`

### DistributedTracer

Distributed tracing manager (#S12).

Features:
- Trace context propagation
- Span creation and management
- Trace collection and export
- Sampling

#### Methods

##### `def __init__(self, service_name: str, sampling_rate: float)`

##### `def register_exporter(self, exporter: Callable[, None]) -> None`

Register trace exporter.

##### `def start_trace(self, operation_name: str, parent_trace_id: , parent_span_id: , baggage: ) -> TraceContext`

Start a new trace (#S12).

Args:
    operation_name: Root operation name
    parent_trace_id: ID of parent trace (for distributed tracing)
    parent_span_id: ID of parent span
    baggage: Initial baggage

Returns:
    New trace context

##### `def end_trace(self, trace: ) -> None`

End a trace and export (#S12).

##### `def get_current_trace()`

Get current trace context.

##### `def inject_headers(self, trace: ) -> dict[str, str]`

Inject trace context into HTTP headers (#S12).

Returns headers for propagation.

##### `def extract_headers(self, headers: dict[str, str]) -> tuple[, , dict]`

Extract trace context from HTTP headers (#S12).

Returns (trace_id, span_id, baggage).

## Functions

### `def inject(service_type: Type[T]) -> Callable[, Callable]`

Decorator for dependency injection (#S10).

Usage:
    @inject(MyService)
    def my_function(service: MyService):
        ...

### `def traced(operation_name: )`

Decorator for tracing functions (#S12).

Supports both sync and async functions.

Usage:
    @traced("process_order")
    def process_order(order_id: str):
        ...

    @traced("async_process")
    async def async_process(order_id: str):
        ...

### `def log_trace_exporter(trace: TraceContext) -> None`

Export trace to logs.

## Constants

- `T`

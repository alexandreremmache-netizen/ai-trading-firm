"""
System Infrastructure Module
============================

Hot reload, dependency injection, and distributed tracing support.

Issues Addressed:
- #S9: No hot reload of configuration
- #S10: Missing dependency injection framework
- #S12: No distributed tracing support
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import json
import yaml
import hashlib
import threading
import time
import uuid
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar, Generic, Type
from functools import wraps
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


# =============================================================================
# HOT RELOAD OF CONFIGURATION (#S9)
# =============================================================================

class ConfigChangeType(str, Enum):
    """Type of configuration change."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class ConfigChange:
    """Record of configuration change (#S9)."""
    timestamp: datetime
    change_type: ConfigChangeType
    key: str
    old_value: Any
    new_value: Any
    source: str

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'change_type': self.change_type.value,
            'key': self.key,
            'old_value': str(self.old_value)[:100],
            'new_value': str(self.new_value)[:100],
            'source': self.source,
        }


class HotReloadableConfig:
    """
    Configuration with hot reload support (#S9).

    Features:
    - File watching for changes
    - Automatic reload on change
    - Change callbacks for dependent systems
    - Validation before applying changes
    - Rollback on failed validation
    """

    def __init__(
        self,
        config_path: str | None = None,
        auto_reload: bool = True,
        poll_interval_seconds: float = 5.0,
    ):
        self._config_path = config_path
        self._auto_reload = auto_reload
        self._poll_interval = poll_interval_seconds

        # Current configuration
        self._config: dict[str, Any] = {}
        self._config_hash: str = ""

        # Change callbacks: key pattern -> callback
        self._callbacks: dict[str, list[Callable[[str, Any, Any], None]]] = defaultdict(list)

        # Change history
        self._change_history: list[ConfigChange] = []

        # Validators: key pattern -> validator
        self._validators: dict[str, Callable[[str, Any], tuple[bool, str]]] = {}

        # Watch thread
        self._watch_thread: threading.Thread | None = None
        self._stop_watching = threading.Event()

        # Load initial config
        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, path: str) -> dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}")
            return {}

        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                new_config = yaml.safe_load(f) or {}
            else:
                new_config = json.load(f)

        self._apply_config(new_config, source=path)
        self._config_path = path

        return new_config

    def load_from_dict(self, config: dict[str, Any], source: str = "dict") -> None:
        """Load configuration from dictionary."""
        self._apply_config(config, source=source)

    def _apply_config(self, new_config: dict[str, Any], source: str) -> None:
        """Apply new configuration, triggering callbacks for changes."""
        old_config = self._config.copy()

        # Find changes
        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)

            if old_value != new_value:
                # Determine change type
                if old_value is None:
                    change_type = ConfigChangeType.CREATED
                elif new_value is None:
                    change_type = ConfigChangeType.DELETED
                else:
                    change_type = ConfigChangeType.MODIFIED

                # Validate
                valid, error = self._validate(key, new_value)
                if not valid:
                    logger.error(f"Config validation failed for {key}: {error}")
                    continue

                # Record change
                change = ConfigChange(
                    timestamp=datetime.now(timezone.utc),
                    change_type=change_type,
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    source=source,
                )
                self._change_history.append(change)

                # Apply change
                if new_value is None:
                    self._config.pop(key, None)
                else:
                    self._config[key] = new_value

                # Trigger callbacks
                self._trigger_callbacks(key, old_value, new_value)

        # Update hash
        self._config_hash = self._compute_hash(self._config)

        logger.info(f"Config reloaded from {source}: {len(new_config)} keys")

    def _validate(self, key: str, value: Any) -> tuple[bool, str]:
        """Validate configuration value."""
        # Check exact key match
        if key in self._validators:
            return self._validators[key](key, value)

        # Check pattern match
        for pattern, validator in self._validators.items():
            if key.startswith(pattern.replace('*', '')):
                return validator(key, value)

        return True, ""

    def _trigger_callbacks(self, key: str, old_value: Any, new_value: Any) -> None:
        """Trigger callbacks for configuration change."""
        # Exact match callbacks
        for callback in self._callbacks.get(key, []):
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Config callback error for {key}: {e}")

        # Wildcard callbacks
        for pattern, callbacks in self._callbacks.items():
            if '*' in pattern and key.startswith(pattern.replace('*', '')):
                for callback in callbacks:
                    try:
                        callback(key, old_value, new_value)
                    except Exception as e:
                        logger.error(f"Config callback error for {key}: {e}")

    def _compute_hash(self, config: dict) -> str:
        """Compute hash of configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def get_nested(self, path: str, default: Any = None, separator: str = '.') -> Any:
        """Get nested configuration value using dot notation."""
        keys = path.split(separator)
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value (triggers callbacks)."""
        self._apply_config({**self._config, key: value}, source="runtime")

    def register_callback(
        self,
        key_pattern: str,
        callback: Callable[[str, Any, Any], None],
    ) -> None:
        """
        Register callback for configuration changes (#S9).

        Args:
            key_pattern: Key or pattern (e.g., "database.*")
            callback: Function called with (key, old_value, new_value)
        """
        self._callbacks[key_pattern].append(callback)
        logger.debug(f"Registered config callback for {key_pattern}")

    def register_validator(
        self,
        key_pattern: str,
        validator: Callable[[str, Any], tuple[bool, str]],
    ) -> None:
        """
        Register validator for configuration values (#S9).

        Args:
            key_pattern: Key or pattern
            validator: Function returning (is_valid, error_message)
        """
        self._validators[key_pattern] = validator

    def start_watching(self) -> None:
        """Start watching configuration file for changes (#S9)."""
        if not self._config_path:
            logger.warning("No config path set, cannot watch")
            return

        if self._watch_thread and self._watch_thread.is_alive():
            return

        self._stop_watching.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        logger.info(f"Started config file watching: {self._config_path}")

    def stop_watching(self) -> None:
        """Stop watching configuration file."""
        self._stop_watching.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=2)
        logger.info("Stopped config file watching")

    def _watch_loop(self) -> None:
        """Background loop to watch for config changes."""
        last_mtime = 0.0

        while not self._stop_watching.is_set():
            try:
                if self._config_path and os.path.exists(self._config_path):
                    current_mtime = os.path.getmtime(self._config_path)

                    if current_mtime > last_mtime:
                        last_mtime = current_mtime
                        self.load_from_file(self._config_path)
                        logger.info(f"Config auto-reloaded: {self._config_path}")

            except Exception as e:
                logger.error(f"Config watch error: {e}")

            self._stop_watching.wait(self._poll_interval)

    def get_change_history(self, limit: int = 100) -> list[ConfigChange]:
        """Get configuration change history."""
        return self._change_history[-limit:]

    def to_dict(self) -> dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()


# =============================================================================
# DEPENDENCY INJECTION FRAMEWORK (#S10)
# =============================================================================

T = TypeVar('T')


class Scope(str, Enum):
    """Dependency injection scope (#S10)."""
    SINGLETON = "singleton"  # One instance per container
    TRANSIENT = "transient"  # New instance each time
    SCOPED = "scoped"  # One instance per scope


@dataclass
class ServiceRegistration:
    """Service registration for DI (#S10)."""
    service_type: Type
    implementation: Type | Callable | object
    scope: Scope
    factory: Callable | None = None
    instance: object | None = None


class DIContainer:
    """
    Dependency injection container (#S10).

    Features:
    - Constructor injection
    - Interface-to-implementation binding
    - Singleton, transient, and scoped lifetimes
    - Factory functions
    - Circular dependency detection
    - Child containers for scoped resolution
    """

    def __init__(self, parent: "DIContainer | None" = None):
        self._parent = parent
        self._registrations: dict[Type, ServiceRegistration] = {}
        self._scoped_instances: dict[Type, object] = {}
        self._resolving: set[Type] = set()  # For circular dependency detection

    def register_singleton(
        self,
        service_type: Type[T],
        implementation: Type[T] | T | None = None,
    ) -> "DIContainer":
        """
        Register singleton service (#S10).

        Args:
            service_type: Interface/base type
            implementation: Implementation class or instance

        Returns:
            Self for chaining
        """
        if implementation is None:
            implementation = service_type

        # If instance passed directly
        if not isinstance(implementation, type):
            self._registrations[service_type] = ServiceRegistration(
                service_type=service_type,
                implementation=type(implementation),
                scope=Scope.SINGLETON,
                instance=implementation,
            )
        else:
            self._registrations[service_type] = ServiceRegistration(
                service_type=service_type,
                implementation=implementation,
                scope=Scope.SINGLETON,
            )

        logger.debug(f"Registered singleton: {service_type.__name__}")
        return self

    def register_transient(
        self,
        service_type: Type[T],
        implementation: Type[T] | None = None,
    ) -> "DIContainer":
        """
        Register transient service (#S10).

        New instance created for each resolution.
        """
        if implementation is None:
            implementation = service_type

        self._registrations[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            scope=Scope.TRANSIENT,
        )

        logger.debug(f"Registered transient: {service_type.__name__}")
        return self

    def register_scoped(
        self,
        service_type: Type[T],
        implementation: Type[T] | None = None,
    ) -> "DIContainer":
        """
        Register scoped service (#S10).

        One instance per scope (child container).
        """
        if implementation is None:
            implementation = service_type

        self._registrations[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            scope=Scope.SCOPED,
        )

        logger.debug(f"Registered scoped: {service_type.__name__}")
        return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[["DIContainer"], T],
        scope: Scope = Scope.TRANSIENT,
    ) -> "DIContainer":
        """
        Register factory function (#S10).

        Args:
            service_type: Type to register
            factory: Function taking container and returning instance
            scope: Lifetime scope
        """
        self._registrations[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=service_type,
            scope=scope,
            factory=factory,
        )

        logger.debug(f"Registered factory: {service_type.__name__}")
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve service from container (#S10).

        Args:
            service_type: Type to resolve

        Returns:
            Service instance

        Raises:
            ValueError: If service not registered
            RuntimeError: If circular dependency detected
        """
        # Check for circular dependency
        if service_type in self._resolving:
            raise RuntimeError(f"Circular dependency detected for {service_type.__name__}")

        # Check local registration
        registration = self._registrations.get(service_type)

        # Check parent
        if registration is None and self._parent:
            return self._parent.resolve(service_type)

        if registration is None:
            raise ValueError(f"Service not registered: {service_type.__name__}")

        # Handle singleton
        if registration.scope == Scope.SINGLETON:
            if registration.instance is not None:
                return registration.instance

        # Handle scoped
        if registration.scope == Scope.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]

        # Create instance
        self._resolving.add(service_type)
        try:
            if registration.factory:
                instance = registration.factory(self)
            else:
                instance = self._create_instance(registration.implementation)
        finally:
            self._resolving.remove(service_type)

        # Store based on scope
        if registration.scope == Scope.SINGLETON:
            registration.instance = instance
        elif registration.scope == Scope.SCOPED:
            self._scoped_instances[service_type] = instance

        return instance

    def _create_instance(self, impl_type: Type) -> object:
        """Create instance with constructor injection."""
        import inspect

        # Get constructor parameters
        sig = inspect.signature(impl_type.__init__)
        params = list(sig.parameters.values())[1:]  # Skip 'self'

        args = []
        for param in params:
            if param.annotation != inspect.Parameter.empty:
                # Try to resolve dependency
                try:
                    args.append(self.resolve(param.annotation))
                except ValueError:
                    if param.default != inspect.Parameter.empty:
                        args.append(param.default)
                    else:
                        raise
            elif param.default != inspect.Parameter.empty:
                args.append(param.default)

        return impl_type(*args)

    def create_scope(self) -> "DIContainer":
        """
        Create child container for scoped resolution (#S10).

        Returns:
            New container with this as parent
        """
        return DIContainer(parent=self)

    def try_resolve(self, service_type: Type[T]) -> T | None:
        """Try to resolve service, returning None if not found."""
        try:
            return self.resolve(service_type)
        except (ValueError, RuntimeError):
            return None

    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered."""
        if service_type in self._registrations:
            return True
        if self._parent:
            return self._parent.is_registered(service_type)
        return False


def inject(service_type: Type[T]) -> Callable[[Callable], Callable]:
    """
    Decorator for dependency injection (#S10).

    Usage:
        @inject(MyService)
        def my_function(service: MyService):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, container: DIContainer = None, **kwargs):
            if container:
                instance = container.resolve(service_type)
                return func(instance, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# DISTRIBUTED TRACING SUPPORT (#S12)
# =============================================================================

# Context variable for current trace
_current_trace: contextvars.ContextVar["TraceContext | None"] = contextvars.ContextVar(
    'current_trace', default=None
)


@dataclass
class SpanContext:
    """Context for a trace span (#S12)."""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "ok"  # "ok", "error"
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[dict] = field(default_factory=list)
    baggage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'service_name': self.service_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': (
                (self.end_time - self.start_time).total_seconds() * 1000
                if self.end_time else None
            ),
            'status': self.status,
            'tags': self.tags,
            'logs': self.logs,
            'baggage': self.baggage,
        }


class TraceContext:
    """
    Trace context for distributed tracing (#S12).

    Follows OpenTelemetry-like semantics.
    """

    def __init__(
        self,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        baggage: dict[str, str] | None = None,
    ):
        self.trace_id = trace_id or self._generate_trace_id()
        self.parent_span_id = parent_span_id
        self.baggage = baggage or {}
        self._spans: list[SpanContext] = []
        self._current_span: SpanContext | None = None

    @staticmethod
    def _generate_trace_id() -> str:
        """Generate unique trace ID."""
        return uuid.uuid4().hex[:32]

    @staticmethod
    def _generate_span_id() -> str:
        """Generate unique span ID."""
        return uuid.uuid4().hex[:16]

    def start_span(
        self,
        operation_name: str,
        service_name: str = "trading-system",
        tags: dict[str, str] | None = None,
    ) -> SpanContext:
        """
        Start a new span (#S12).

        Args:
            operation_name: Name of operation
            service_name: Name of service
            tags: Initial tags

        Returns:
            New span context
        """
        parent_id = self._current_span.span_id if self._current_span else self.parent_span_id

        span = SpanContext(
            trace_id=self.trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_id,
            operation_name=operation_name,
            service_name=service_name,
            start_time=datetime.now(timezone.utc),
            tags=tags or {},
            baggage=dict(self.baggage),
        )

        self._spans.append(span)
        self._current_span = span

        return span

    def end_span(
        self,
        span: SpanContext | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        """
        End a span (#S12).

        Args:
            span: Span to end (current if None)
            status: Final status
            error: Error message if failed
        """
        target_span = span or self._current_span
        if not target_span:
            return

        target_span.end_time = datetime.now(timezone.utc)
        target_span.status = status

        if error:
            target_span.logs.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': 'error',
                'message': error,
            })

        # Update current span to parent
        if target_span == self._current_span:
            parent_spans = [
                s for s in self._spans
                if s.span_id == target_span.parent_span_id
            ]
            self._current_span = parent_spans[0] if parent_spans else None

    def add_tag(self, key: str, value: str, span: SpanContext | None = None) -> None:
        """Add tag to span."""
        target = span or self._current_span
        if target:
            target.tags[key] = value

    def add_log(
        self,
        message: str,
        level: str = "info",
        span: SpanContext | None = None,
    ) -> None:
        """Add log entry to span."""
        target = span or self._current_span
        if target:
            target.logs.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': level,
                'message': message,
            })

    def set_baggage(self, key: str, value: str) -> None:
        """Set baggage item (propagates to child spans)."""
        self.baggage[key] = value

    def get_spans(self) -> list[SpanContext]:
        """Get all spans in trace."""
        return self._spans

    @property
    def current_span(self) -> SpanContext | None:
        """Get current span."""
        return self._current_span

    def to_dict(self) -> dict:
        return {
            'trace_id': self.trace_id,
            'spans': [s.to_dict() for s in self._spans],
            'baggage': self.baggage,
        }


class DistributedTracer:
    """
    Distributed tracing manager (#S12).

    Features:
    - Trace context propagation
    - Span creation and management
    - Trace collection and export
    - Sampling
    """

    def __init__(
        self,
        service_name: str = "trading-system",
        sampling_rate: float = 1.0,
    ):
        self.service_name = service_name
        self.sampling_rate = sampling_rate

        # Trace storage
        self._traces: dict[str, TraceContext] = {}

        # Export handlers
        self._exporters: list[Callable[[TraceContext], None]] = []

    def register_exporter(self, exporter: Callable[[TraceContext], None]) -> None:
        """Register trace exporter."""
        self._exporters.append(exporter)

    def start_trace(
        self,
        operation_name: str,
        parent_trace_id: str | None = None,
        parent_span_id: str | None = None,
        baggage: dict[str, str] | None = None,
    ) -> TraceContext:
        """
        Start a new trace (#S12).

        Args:
            operation_name: Root operation name
            parent_trace_id: ID of parent trace (for distributed tracing)
            parent_span_id: ID of parent span
            baggage: Initial baggage

        Returns:
            New trace context
        """
        # Sampling
        import random
        if random.random() > self.sampling_rate:
            # Return a no-op trace
            return TraceContext()

        trace = TraceContext(
            trace_id=parent_trace_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
        )

        # Start root span
        trace.start_span(operation_name, self.service_name)

        # Store trace
        self._traces[trace.trace_id] = trace

        # Set as current
        _current_trace.set(trace)

        return trace

    def end_trace(self, trace: TraceContext | None = None) -> None:
        """
        End a trace and export (#S12).
        """
        target = trace or _current_trace.get()
        if not target:
            return

        # End any open spans
        for span in target.get_spans():
            if span.end_time is None:
                target.end_span(span)

        # Export
        for exporter in self._exporters:
            try:
                exporter(target)
            except Exception as e:
                logger.error(f"Trace export error: {e}")

        # Remove from storage
        self._traces.pop(target.trace_id, None)

        # Clear current
        if _current_trace.get() == target:
            _current_trace.set(None)

    @staticmethod
    def get_current_trace() -> TraceContext | None:
        """Get current trace context."""
        return _current_trace.get()

    def inject_headers(self, trace: TraceContext | None = None) -> dict[str, str]:
        """
        Inject trace context into HTTP headers (#S12).

        Returns headers for propagation.
        """
        target = trace or _current_trace.get()
        if not target:
            return {}

        headers = {
            'x-trace-id': target.trace_id,
        }

        if target.current_span:
            headers['x-span-id'] = target.current_span.span_id

        # Include baggage
        for key, value in target.baggage.items():
            headers[f'x-baggage-{key}'] = value

        return headers

    def extract_headers(self, headers: dict[str, str]) -> tuple[str | None, str | None, dict]:
        """
        Extract trace context from HTTP headers (#S12).

        Returns (trace_id, span_id, baggage).
        """
        trace_id = headers.get('x-trace-id')
        span_id = headers.get('x-span-id')

        baggage = {}
        for key, value in headers.items():
            if key.startswith('x-baggage-'):
                baggage_key = key[10:]  # Remove prefix
                baggage[baggage_key] = value

        return trace_id, span_id, baggage


def traced(operation_name: str | None = None):
    """
    Decorator for tracing functions (#S12).

    Supports both sync and async functions.

    Usage:
        @traced("process_order")
        def process_order(order_id: str):
            ...

        @traced("async_process")
        async def async_process(order_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        if inspect.iscoroutinefunction(func):
            # Async function wrapper
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                trace = _current_trace.get()

                if trace:
                    span = trace.start_span(op_name)
                    try:
                        result = await func(*args, **kwargs)
                        trace.end_span(span, status="ok")
                        return result
                    except Exception as e:
                        trace.end_span(span, status="error", error=str(e))
                        raise
                else:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            # Sync function wrapper
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                trace = _current_trace.get()

                if trace:
                    span = trace.start_span(op_name)
                    try:
                        result = func(*args, **kwargs)
                        trace.end_span(span, status="ok")
                        return result
                    except Exception as e:
                        trace.end_span(span, status="error", error=str(e))
                        raise
                else:
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


# Logging exporter for traces
def log_trace_exporter(trace: TraceContext) -> None:
    """Export trace to logs."""
    spans = trace.get_spans()
    logger.info(
        f"Trace completed: {trace.trace_id}, "
        f"spans={len(spans)}, "
        f"duration={(spans[-1].end_time - spans[0].start_time).total_seconds() * 1000:.1f}ms"
    )

    for span in spans:
        logger.debug(
            f"  Span: {span.operation_name}, "
            f"duration={(span.end_time - span.start_time).total_seconds() * 1000:.1f}ms, "
            f"status={span.status}"
        )

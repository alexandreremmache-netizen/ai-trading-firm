"""
Structured Logging Module
=========================

Implements JSON-structured logging for production environments (Issue #S11).

Features:
- JSON-formatted log entries for machine parsing
- Correlation ID tracking across distributed components
- Log context propagation
- Performance metrics in logs
- Log level filtering by component
- ELK/Splunk compatible output format
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Callable
import uuid


# Context variable for correlation ID (thread-safe)
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    'correlation_id', default=''
)

# Context variable for additional context
log_context_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    'log_context', default={}
)


class LogLevel(str, Enum):
    """Standardized log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # Special level for compliance


@dataclass
class StructuredLogEntry:
    """Structured log entry for JSON output."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    correlation_id: str = ""
    component: str = ""
    event_type: str = ""
    duration_ms: float | None = None
    context: dict = field(default_factory=dict)
    exception: dict | None = None
    metrics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {k: v for k, v in asdict(self).items() if v}
        return json.dumps(data, default=str, ensure_ascii=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v}


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_stack: bool = True):
        super().__init__()
        self.include_stack = include_stack

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base entry
        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            correlation_id=correlation_id_var.get(),
            component=getattr(record, 'component', ''),
            event_type=getattr(record, 'event_type', ''),
            duration_ms=getattr(record, 'duration_ms', None),
            context={**log_context_var.get(), **getattr(record, 'context', {})},
            metrics=getattr(record, 'metrics', {}),
        )

        # Add exception info if present
        if record.exc_info and self.include_stack:
            entry.exception = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'stack': ''.join(traceback.format_exception(*record.exc_info)),
            }

        return entry.to_json()


class StructuredLogger:
    """
    Enhanced logger with structured JSON output.

    Features:
    - JSON-formatted logs for ELK/Splunk ingestion
    - Correlation ID tracking
    - Performance timing
    - Context propagation
    - Async logging support
    """

    def __init__(
        self,
        name: str,
        component: str = "",
        log_dir: str = "logs",
        json_file: str = "structured.jsonl",
        console_json: bool = False,
        max_bytes: int = 50 * 1024 * 1024,
        backup_count: int = 30,
        async_logging: bool = True,
    ):
        self.name = name
        self.component = component
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create underlying Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if not self._logger.handlers:
            self._setup_handlers(
                json_file, console_json, max_bytes, backup_count, async_logging
            )

        # Performance tracking
        self._timers: dict[str, float] = {}
        self._lock = Lock()

    def _setup_handlers(
        self,
        json_file: str,
        console_json: bool,
        max_bytes: int,
        backup_count: int,
        async_logging: bool,
    ) -> None:
        """Configure log handlers."""
        json_formatter = JsonFormatter()
        text_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        # JSON file handler
        json_path = self.log_dir / json_file
        json_handler = RotatingFileHandler(
            json_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        json_handler.setFormatter(json_formatter)
        json_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if console_json:
            console_handler.setFormatter(json_formatter)
        else:
            console_handler.setFormatter(text_formatter)
        console_handler.setLevel(logging.INFO)

        if async_logging:
            # Use queue-based async logging
            log_queue: Queue = Queue(-1)
            queue_handler = QueueHandler(log_queue)

            # Listener processes queue in background thread
            listener = QueueListener(
                log_queue, json_handler, console_handler, respect_handler_level=True
            )
            listener.start()

            self._logger.addHandler(queue_handler)
            self._listener = listener
        else:
            self._logger.addHandler(json_handler)
            self._logger.addHandler(console_handler)

    def _log(
        self,
        level: int,
        message: str,
        event_type: str = "",
        context: dict | None = None,
        metrics: dict | None = None,
        duration_ms: float | None = None,
        exc_info: bool = False,
    ) -> None:
        """Internal log method with structured data."""
        extra = {
            'component': self.component,
            'event_type': event_type,
            'context': context or {},
            'metrics': metrics or {},
            'duration_ms': duration_ms,
        }
        self._logger.log(level, message, extra=extra, exc_info=exc_info)

    def debug(
        self,
        message: str,
        event_type: str = "",
        context: dict | None = None,
        **kwargs,
    ) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, event_type, context, **kwargs)

    def info(
        self,
        message: str,
        event_type: str = "",
        context: dict | None = None,
        metrics: dict | None = None,
        **kwargs,
    ) -> None:
        """Log info message."""
        self._log(logging.INFO, message, event_type, context, metrics, **kwargs)

    def warning(
        self,
        message: str,
        event_type: str = "",
        context: dict | None = None,
        **kwargs,
    ) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, event_type, context, **kwargs)

    def error(
        self,
        message: str,
        event_type: str = "",
        context: dict | None = None,
        exc_info: bool = True,
        **kwargs,
    ) -> None:
        """Log error message with optional exception info."""
        self._log(logging.ERROR, message, event_type, context, exc_info=exc_info, **kwargs)

    def critical(
        self,
        message: str,
        event_type: str = "",
        context: dict | None = None,
        exc_info: bool = True,
        **kwargs,
    ) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, event_type, context, exc_info=exc_info, **kwargs)

    def audit(
        self,
        message: str,
        event_type: str,
        context: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Log audit event (compliance requirement)."""
        # Audit logs always use INFO level but with AUDIT event type
        self._log(
            logging.INFO,
            message,
            event_type=f"AUDIT_{event_type}",
            context=context,
            metrics=metrics,
        )

    def start_timer(self, operation: str) -> str:
        """Start a performance timer."""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._timers[timer_id] = time.perf_counter()
        return timer_id

    def stop_timer(self, timer_id: str) -> float:
        """Stop timer and return duration in milliseconds."""
        with self._lock:
            start = self._timers.pop(timer_id, None)
        if start is None:
            return 0.0
        return (time.perf_counter() - start) * 1000

    def timed(
        self,
        message: str,
        event_type: str = "",
        context: dict | None = None,
    ):
        """Context manager for timed operations."""
        return TimedOperation(self, message, event_type, context)

    def with_context(self, **kwargs) -> 'LogContextManager':
        """Add context to all logs within the context manager."""
        return LogContextManager(kwargs)


class TimedOperation:
    """Context manager for timing operations."""

    def __init__(
        self,
        logger: StructuredLogger,
        message: str,
        event_type: str,
        context: dict | None,
    ):
        self.logger = logger
        self.message = message
        self.event_type = event_type
        self.context = context or {}
        self.start_time: float = 0

    def __enter__(self) -> 'TimedOperation':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            self.logger.error(
                f"{self.message} failed after {duration_ms:.2f}ms",
                event_type=self.event_type,
                context=self.context,
                duration_ms=duration_ms,
            )
        else:
            self.logger.info(
                f"{self.message} completed in {duration_ms:.2f}ms",
                event_type=self.event_type,
                context=self.context,
                duration_ms=duration_ms,
                metrics={'duration_ms': duration_ms},
            )


class LogContextManager:
    """Context manager for adding context to logs."""

    def __init__(self, context: dict):
        self.context = context
        self.token: contextvars.Token | None = None

    def __enter__(self) -> 'LogContextManager':
        current = log_context_var.get()
        new_context = {**current, **self.context}
        self.token = log_context_var.set(new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.token:
            log_context_var.reset(self.token)


def set_correlation_id(correlation_id: str | None = None) -> str:
    """
    Set or generate correlation ID for request tracing.

    Returns the correlation ID.
    """
    if correlation_id is None:
        correlation_id = uuid.uuid4().hex
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get()


def add_log_context(**kwargs) -> contextvars.Token:
    """Add context to current log context."""
    current = log_context_var.get()
    new_context = {**current, **kwargs}
    return log_context_var.set(new_context)


def clear_log_context() -> None:
    """Clear log context."""
    log_context_var.set({})


# Singleton loggers per component
_loggers: dict[str, StructuredLogger] = {}
_logger_lock = Lock()


def get_logger(name: str, component: str = "") -> StructuredLogger:
    """Get or create a structured logger."""
    key = f"{name}:{component}"
    with _logger_lock:
        if key not in _loggers:
            _loggers[key] = StructuredLogger(name, component)
        return _loggers[key]


@dataclass
class LogAggregator:
    """
    Aggregates and summarizes log metrics.

    Useful for dashboards and alerting.
    """

    # Counters by level
    counts: dict[str, int] = field(default_factory=lambda: {
        'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0
    })

    # Error tracking
    recent_errors: list[dict] = field(default_factory=list)
    max_recent_errors: int = 100

    # Performance tracking
    operation_times: dict[str, list[float]] = field(default_factory=dict)

    # Lock for thread safety
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_log(self, level: str, message: str, duration_ms: float | None = None) -> None:
        """Record a log entry."""
        with self._lock:
            self.counts[level] = self.counts.get(level, 0) + 1

            if level in ('ERROR', 'CRITICAL'):
                self.recent_errors.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'level': level,
                    'message': message[:200],  # Truncate
                })

                # Trim old errors
                if len(self.recent_errors) > self.max_recent_errors:
                    self.recent_errors = self.recent_errors[-self.max_recent_errors:]

    def record_operation_time(self, operation: str, duration_ms: float) -> None:
        """Record operation timing."""
        with self._lock:
            if operation not in self.operation_times:
                self.operation_times[operation] = []

            self.operation_times[operation].append(duration_ms)

            # Keep only last 1000 samples
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-1000:]

    def get_summary(self) -> dict:
        """Get log summary for monitoring."""
        with self._lock:
            summary = {
                'counts': dict(self.counts),
                'error_rate': self._calculate_error_rate(),
                'recent_error_count': len(self.recent_errors),
                'operation_stats': {},
            }

            # Calculate operation statistics
            for op, times in self.operation_times.items():
                if times:
                    sorted_times = sorted(times)
                    summary['operation_stats'][op] = {
                        'count': len(times),
                        'avg_ms': sum(times) / len(times),
                        'min_ms': min(times),
                        'max_ms': max(times),
                        'p50_ms': sorted_times[len(times) // 2],
                        'p95_ms': sorted_times[int(len(times) * 0.95)],
                        'p99_ms': sorted_times[int(len(times) * 0.99)],
                    }

            return summary

    def _calculate_error_rate(self) -> float:
        """Calculate error rate as percentage."""
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        errors = self.counts.get('ERROR', 0) + self.counts.get('CRITICAL', 0)
        return (errors / total) * 100


# Global aggregator instance
_aggregator = LogAggregator()


def get_log_aggregator() -> LogAggregator:
    """Get global log aggregator."""
    return _aggregator


class ComponentLogFilter(logging.Filter):
    """Filter logs by component."""

    def __init__(self, allowed_components: list[str] | None = None,
                 excluded_components: list[str] | None = None):
        super().__init__()
        self.allowed = set(allowed_components) if allowed_components else None
        self.excluded = set(excluded_components) if excluded_components else set()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record by component."""
        component = getattr(record, 'component', '')

        if component in self.excluded:
            return False

        if self.allowed is not None:
            return component in self.allowed

        return True


def configure_global_logging(
    log_dir: str = "logs",
    json_file: str = "structured.jsonl",
    console_json: bool = False,
    level: str = "INFO",
    async_logging: bool = True,
) -> None:
    """
    Configure global structured logging.

    Call this once at application startup.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # JSON file handler
    json_formatter = JsonFormatter()
    json_handler = RotatingFileHandler(
        log_path / json_file,
        maxBytes=50 * 1024 * 1024,
        backupCount=30,
        encoding="utf-8",
    )
    json_handler.setFormatter(json_formatter)
    json_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if console_json:
        console_handler.setFormatter(json_formatter)
    else:
        text_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(text_formatter)
    console_handler.setLevel(getattr(logging, level.upper()))

    if async_logging:
        log_queue: Queue = Queue(-1)
        queue_handler = QueueHandler(log_queue)
        listener = QueueListener(
            log_queue, json_handler, console_handler, respect_handler_level=True
        )
        listener.start()
        root_logger.addHandler(queue_handler)
    else:
        root_logger.addHandler(json_handler)
        root_logger.addHandler(console_handler)

    logging.info("Structured logging configured", extra={
        'component': 'logging',
        'event_type': 'CONFIG',
        'context': {'log_dir': log_dir, 'level': level},
    })

# structured_logging

**Path**: `C:\Users\Alexa\ai-trading-firm\core\structured_logging.py`

## Overview

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

## Classes

### LogLevel

**Inherits from**: str, Enum

Standardized log levels.

### StructuredLogEntry

Structured log entry for JSON output.

#### Methods

##### `def to_json(self) -> str`

Convert to JSON string.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### JsonFormatter

**Inherits from**: logging.Formatter

JSON formatter for structured logging.

#### Methods

##### `def __init__(self, include_stack: bool)`

##### `def format(self, record: logging.LogRecord) -> str`

Format log record as JSON.

### StructuredLogger

Enhanced logger with structured JSON output.

Features:
- JSON-formatted logs for ELK/Splunk ingestion
- Correlation ID tracking
- Performance timing
- Context propagation
- Async logging support

#### Methods

##### `def __init__(self, name: str, component: str, log_dir: str, json_file: str, console_json: bool, max_bytes: int, backup_count: int, async_logging: bool)`

##### `def debug(self, message: str, event_type: str, context: ) -> None`

Log debug message.

##### `def info(self, message: str, event_type: str, context: , metrics: ) -> None`

Log info message.

##### `def warning(self, message: str, event_type: str, context: ) -> None`

Log warning message.

##### `def error(self, message: str, event_type: str, context: , exc_info: bool) -> None`

Log error message with optional exception info.

##### `def critical(self, message: str, event_type: str, context: , exc_info: bool) -> None`

Log critical message.

##### `def audit(self, message: str, event_type: str, context: , metrics: ) -> None`

Log audit event (compliance requirement).

##### `def start_timer(self, operation: str) -> str`

Start a performance timer.

##### `def stop_timer(self, timer_id: str) -> float`

Stop timer and return duration in milliseconds.

##### `def timed(self, message: str, event_type: str, context: )`

Context manager for timed operations.

##### `def with_context(self) -> LogContextManager`

Add context to all logs within the context manager.

### TimedOperation

Context manager for timing operations.

#### Methods

##### `def __init__(self, logger: StructuredLogger, message: str, event_type: str, context: )`

##### `def __enter__(self) -> TimedOperation`

##### `def __exit__(self, exc_type, exc_val, exc_tb) -> None`

### LogContextManager

Context manager for adding context to logs.

#### Methods

##### `def __init__(self, context: dict)`

##### `def __enter__(self) -> LogContextManager`

##### `def __exit__(self, exc_type, exc_val, exc_tb) -> None`

### LogAggregator

Aggregates and summarizes log metrics.

Useful for dashboards and alerting.

#### Methods

##### `def record_log(self, level: str, message: str, duration_ms: ) -> None`

Record a log entry.

##### `def record_operation_time(self, operation: str, duration_ms: float) -> None`

Record operation timing.

##### `def get_summary(self) -> dict`

Get log summary for monitoring.

### ComponentLogFilter

**Inherits from**: logging.Filter

Filter logs by component.

#### Methods

##### `def __init__(self, allowed_components: , excluded_components: )`

##### `def filter(self, record: logging.LogRecord) -> bool`

Filter log record by component.

## Functions

### `def set_correlation_id(correlation_id: ) -> str`

Set or generate correlation ID for request tracing.

Returns the correlation ID.

### `def get_correlation_id() -> str`

Get current correlation ID.

### `def add_log_context() -> contextvars.Token`

Add context to current log context.

### `def clear_log_context() -> None`

Clear log context.

### `def get_logger(name: str, component: str) -> StructuredLogger`

Get or create a structured logger.

### `def get_log_aggregator() -> LogAggregator`

Get global log aggregator.

### `def configure_global_logging(log_dir: str, json_file: str, console_json: bool, level: str, async_logging: bool) -> None`

Configure global structured logging.

Call this once at application startup.

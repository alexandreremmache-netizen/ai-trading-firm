# logging_config

**Path**: `C:\Users\Alexa\ai-trading-firm\core\logging_config.py`

## Overview

Logging Configuration Module
============================

Addresses issues:
- #Q24: Logging verbosity inconsistent
- #R30: Logging of risk calculations verbose

Features:
- Centralized logging configuration
- Consistent verbosity levels across modules
- Module-specific log level overrides
- Performance logging with timing
- Structured context injection
- Log filtering and sampling for high-frequency events

## Classes

### LogLevel

**Inherits from**: str, Enum

Standard log levels.

### LoggingConfig

Centralized logging configuration.

Provides consistent verbosity across the trading system.

#### Methods

##### `def apply(self) -> None`

Apply logging configuration to all handlers.

##### `def set_module_level(self, module_name: str, level: int) -> None`

Set log level for a specific module.

##### `def set_all_debug(self) -> None`

Set all loggers to DEBUG (for debugging).

##### `def set_production_mode(self) -> None`

Set production-appropriate log levels.

### LogSampler

Samples high-frequency log messages to reduce noise.

Useful for risk calculations and market data updates.

#### Methods

##### `def __init__(self, sample_rate: float, time_window_seconds: float)`

Initialize sampler.

Args:
    sample_rate: Fraction of messages to log (0.0-1.0)
    time_window_seconds: Window for rate limiting

##### `def should_log(self, message_key: str) -> bool`

Determine if a message should be logged.

Args:
    message_key: Unique key for the message type

Returns:
    True if message should be logged

##### `def get_suppressed_count(self, message_key: str) -> int`

Get count of suppressed messages for a key.

### RateLimitedLogger

Logger wrapper that rate-limits specific message types.

Prevents log flooding from high-frequency events.

#### Methods

##### `def __init__(self, logger: logging.Logger, max_per_minute: int, summary_interval_seconds: float)`

Initialize rate-limited logger.

Args:
    logger: Underlying logger
    max_per_minute: Max messages per minute per key
    summary_interval_seconds: Interval for summary messages

##### `def log(self, level: int, message: str, rate_limit_key: ) -> None`

Log message with optional rate limiting.

Args:
    level: Log level
    message: Log message
    rate_limit_key: Key for rate limiting (None = no limit)

##### `def info(self, message: str, rate_limit_key: )`

Log at INFO level.

##### `def debug(self, message: str, rate_limit_key: )`

Log at DEBUG level.

##### `def warning(self, message: str, rate_limit_key: )`

Log at WARNING level.

##### `def error(self, message: str, rate_limit_key: )`

Log at ERROR level.

### PerformanceLogger

Logs performance metrics for operations.

Useful for identifying slow operations and bottlenecks.

#### Methods

##### `def __init__(self, logger: logging.Logger, slow_threshold_ms: float, always_log_slow: bool)`

Initialize performance logger.

Args:
    logger: Underlying logger
    slow_threshold_ms: Threshold for slow operation warning
    always_log_slow: Always log operations exceeding threshold

##### `def measure(self, operation_name: str, log_always: bool)`

Context manager to measure operation duration.

Args:
    operation_name: Name of the operation
    log_always: Log even if under threshold

Yields:
    None

Example:
    with perf_logger.measure("var_calculation"):
        result = calculate_var()

##### `def get_stats(self, operation_name: str) -> dict[str, float]`

Get performance statistics for an operation.

##### `def log_summary(self) -> None`

Log summary of all operation statistics.

### ContextLogger

Logger with automatic context injection.

Adds consistent context (strategy, symbol, etc.) to all messages.

#### Methods

##### `def __init__(self, logger: logging.Logger, context: )`

Initialize context logger.

Args:
    logger: Underlying logger
    context: Default context to inject

##### `def with_context(self) -> ContextLogger`

Create new logger with additional context.

##### `def temporary_context(self)`

Temporarily add context for a block of code.

##### `def debug(self, message: str)`

Log at DEBUG level with context.

##### `def info(self, message: str)`

Log at INFO level with context.

##### `def warning(self, message: str)`

Log at WARNING level with context.

##### `def error(self, message: str)`

Log at ERROR level with context.

##### `def exception(self, message: str)`

Log exception with context.

## Functions

### `def timed(logger: , threshold_ms: float, operation_name: )`

Decorator to time function execution.

Args:
    logger: Logger to use (default: function's module logger)
    threshold_ms: Log warning if exceeds this threshold
    operation_name: Custom operation name (default: function name)

Example:
    @timed(threshold_ms=50.0)
    def calculate_var():
        ...

### `def configure_logging(config: ) -> LoggingConfig`

Configure logging for the trading system.

Args:
    config: Optional custom configuration

Returns:
    Applied configuration

### `def get_logging_config() -> LoggingConfig`

Get current logging configuration.

### `def get_rate_limited_logger(name: str, max_per_minute: int) -> RateLimitedLogger`

Get a rate-limited logger for a module.

Args:
    name: Logger name
    max_per_minute: Max messages per minute

Returns:
    Rate-limited logger

### `def get_performance_logger(name: str, slow_threshold_ms: float) -> PerformanceLogger`

Get a performance logger for a module.

Args:
    name: Logger name
    slow_threshold_ms: Threshold for slow operation warning

Returns:
    Performance logger

### `def get_context_logger(name: str) -> ContextLogger`

Get a context logger for a module.

Args:
    name: Logger name
    **context: Initial context

Returns:
    Context logger

## Constants

- `MODULE_LOG_LEVELS`

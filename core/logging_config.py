"""
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
"""

from __future__ import annotations

import logging
import time
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from collections import defaultdict
import threading


class LogLevel(str, Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Module-specific log level defaults for consistent verbosity (#Q24, #R30)
MODULE_LOG_LEVELS = {
    # Core modules - INFO level by default
    "core.var_calculator": logging.INFO,
    "core.risk_factors": logging.INFO,
    "core.correlation_manager": logging.INFO,
    "core.margin_optimizer": logging.INFO,
    "core.scenario_analysis": logging.INFO,

    # Risk calculations - reduce verbosity (#R30)
    "agents.risk_agent": logging.INFO,
    "core.stress_tester": logging.INFO,

    # Strategy modules
    "strategies.momentum_strategy": logging.INFO,
    "strategies.stat_arb_strategy": logging.INFO,
    "strategies.options_vol_strategy": logging.INFO,

    # Execution - WARNING for routine, INFO for important
    "agents.execution_agent": logging.INFO,
    "core.smart_order_router": logging.INFO,

    # Data feeds - WARNING to reduce noise
    "data.market_data": logging.WARNING,

    # Event bus - INFO
    "core.event_bus": logging.INFO,

    # Compliance - INFO for audit trail
    "agents.compliance_agent": logging.INFO,
    "agents.surveillance_agent": logging.INFO,
}


@dataclass
class LoggingConfig:
    """
    Centralized logging configuration.

    Provides consistent verbosity across the trading system.
    """
    root_level: int = logging.INFO
    format_string: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    module_levels: dict[str, int] = field(default_factory=lambda: MODULE_LOG_LEVELS.copy())

    # High-frequency event sampling
    sample_rate: float = 1.0  # 1.0 = log all, 0.1 = log 10%
    sampled_messages: set[str] = field(default_factory=set)

    # Performance threshold (log slow operations)
    slow_operation_threshold_ms: float = 100.0

    def apply(self) -> None:
        """Apply logging configuration to all handlers."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.root_level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add new handler with format
        handler = logging.StreamHandler()
        handler.setLevel(self.root_level)
        formatter = logging.Formatter(self.format_string, self.date_format)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # Apply module-specific levels
        for module_name, level in self.module_levels.items():
            logger = logging.getLogger(module_name)
            logger.setLevel(level)

    def set_module_level(self, module_name: str, level: int) -> None:
        """Set log level for a specific module."""
        self.module_levels[module_name] = level
        logger = logging.getLogger(module_name)
        logger.setLevel(level)

    def set_all_debug(self) -> None:
        """Set all loggers to DEBUG (for debugging)."""
        for module_name in self.module_levels:
            self.set_module_level(module_name, logging.DEBUG)

    def set_production_mode(self) -> None:
        """Set production-appropriate log levels."""
        # Reduce verbosity for high-frequency modules
        self.set_module_level("core.var_calculator", logging.WARNING)
        self.set_module_level("data.market_data", logging.WARNING)
        self.set_module_level("core.event_bus", logging.WARNING)


class LogSampler:
    """
    Samples high-frequency log messages to reduce noise.

    Useful for risk calculations and market data updates.
    """

    def __init__(self, sample_rate: float = 0.1, time_window_seconds: float = 60.0):
        """
        Initialize sampler.

        Args:
            sample_rate: Fraction of messages to log (0.0-1.0)
            time_window_seconds: Window for rate limiting
        """
        self.sample_rate = sample_rate
        self.time_window = time_window_seconds
        self._message_counts: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def should_log(self, message_key: str) -> bool:
        """
        Determine if a message should be logged.

        Args:
            message_key: Unique key for the message type

        Returns:
            True if message should be logged
        """
        import random

        with self._lock:
            now = time.time()

            # Clean old entries
            cutoff = now - self.time_window
            self._message_counts[message_key] = [
                t for t in self._message_counts[message_key] if t > cutoff
            ]

            # Sample based on rate
            if random.random() < self.sample_rate:
                self._message_counts[message_key].append(now)
                return True

            return False

    def get_suppressed_count(self, message_key: str) -> int:
        """Get count of suppressed messages for a key."""
        with self._lock:
            now = time.time()
            cutoff = now - self.time_window
            count = len([t for t in self._message_counts[message_key] if t > cutoff])
            expected = count / self.sample_rate if self.sample_rate > 0 else count
            return int(expected - count)


class RateLimitedLogger:
    """
    Logger wrapper that rate-limits specific message types.

    Prevents log flooding from high-frequency events.
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_per_minute: int = 10,
        summary_interval_seconds: float = 60.0,
    ):
        """
        Initialize rate-limited logger.

        Args:
            logger: Underlying logger
            max_per_minute: Max messages per minute per key
            summary_interval_seconds: Interval for summary messages
        """
        self.logger = logger
        self.max_per_minute = max_per_minute
        self.summary_interval = summary_interval_seconds
        self._counts: dict[str, int] = defaultdict(int)
        self._last_summary: dict[str, float] = {}
        self._suppressed: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def log(
        self,
        level: int,
        message: str,
        rate_limit_key: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Log message with optional rate limiting.

        Args:
            level: Log level
            message: Log message
            rate_limit_key: Key for rate limiting (None = no limit)
        """
        if rate_limit_key is None:
            self.logger.log(level, message, *args, **kwargs)
            return

        now = time.time()

        with self._lock:
            # Check if we should log
            window_start = now - 60.0
            key = rate_limit_key

            # Reset count if new window
            if key in self._last_summary:
                if now - self._last_summary[key] > self.summary_interval:
                    if self._suppressed[key] > 0:
                        self.logger.log(
                            level,
                            f"[Rate limited] {self._suppressed[key]} similar messages suppressed for '{key}'",
                        )
                    self._counts[key] = 0
                    self._suppressed[key] = 0
                    self._last_summary[key] = now
            else:
                self._last_summary[key] = now

            # Check rate limit
            if self._counts[key] < self.max_per_minute:
                self._counts[key] += 1
                self.logger.log(level, message, *args, **kwargs)
            else:
                self._suppressed[key] += 1

    def info(self, message: str, rate_limit_key: str | None = None, *args, **kwargs):
        """Log at INFO level."""
        self.log(logging.INFO, message, rate_limit_key, *args, **kwargs)

    def debug(self, message: str, rate_limit_key: str | None = None, *args, **kwargs):
        """Log at DEBUG level."""
        self.log(logging.DEBUG, message, rate_limit_key, *args, **kwargs)

    def warning(self, message: str, rate_limit_key: str | None = None, *args, **kwargs):
        """Log at WARNING level."""
        self.log(logging.WARNING, message, rate_limit_key, *args, **kwargs)

    def error(self, message: str, rate_limit_key: str | None = None, *args, **kwargs):
        """Log at ERROR level."""
        self.log(logging.ERROR, message, rate_limit_key, *args, **kwargs)


class PerformanceLogger:
    """
    Logs performance metrics for operations.

    Useful for identifying slow operations and bottlenecks.
    """

    def __init__(
        self,
        logger: logging.Logger,
        slow_threshold_ms: float = 100.0,
        always_log_slow: bool = True,
    ):
        """
        Initialize performance logger.

        Args:
            logger: Underlying logger
            slow_threshold_ms: Threshold for slow operation warning
            always_log_slow: Always log operations exceeding threshold
        """
        self.logger = logger
        self.slow_threshold_ms = slow_threshold_ms
        self.always_log_slow = always_log_slow
        self._stats: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, operation_name: str, log_always: bool = False):
        """
        Context manager to measure operation duration.

        Args:
            operation_name: Name of the operation
            log_always: Log even if under threshold

        Yields:
            None

        Example:
            with perf_logger.measure("var_calculation"):
                result = calculate_var()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            with self._lock:
                self._stats[operation_name].append(duration_ms)

            # Log if slow or always requested
            if duration_ms > self.slow_threshold_ms:
                self.logger.warning(
                    f"Slow operation: {operation_name} took {duration_ms:.2f}ms "
                    f"(threshold: {self.slow_threshold_ms}ms)"
                )
            elif log_always:
                self.logger.debug(f"{operation_name} completed in {duration_ms:.2f}ms")

    def get_stats(self, operation_name: str) -> dict[str, float]:
        """Get performance statistics for an operation."""
        with self._lock:
            times = self._stats.get(operation_name, [])

        if not times:
            return {}

        import statistics

        return {
            "count": len(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times),
        }

    def log_summary(self) -> None:
        """Log summary of all operation statistics."""
        with self._lock:
            operations = list(self._stats.keys())

        for op in operations:
            stats = self.get_stats(op)
            if stats:
                self.logger.info(
                    f"Performance stats for {op}: "
                    f"mean={stats['mean_ms']:.2f}ms, "
                    f"p95={stats['p95_ms']:.2f}ms, "
                    f"count={stats['count']}"
                )


def timed(
    logger: logging.Logger | None = None,
    threshold_ms: float = 100.0,
    operation_name: str | None = None,
):
    """
    Decorator to time function execution.

    Args:
        logger: Logger to use (default: function's module logger)
        threshold_ms: Log warning if exceeds this threshold
        operation_name: Custom operation name (default: function name)

    Example:
        @timed(threshold_ms=50.0)
        def calculate_var():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger, operation_name

            if logger is None:
                logger = logging.getLogger(func.__module__)
            if operation_name is None:
                operation_name = func.__name__

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                if duration_ms > threshold_ms:
                    logger.warning(
                        f"Slow operation: {operation_name} took {duration_ms:.2f}ms"
                    )
                else:
                    logger.debug(f"{operation_name} completed in {duration_ms:.2f}ms")

        return wrapper
    return decorator


class ContextLogger:
    """
    Logger with automatic context injection.

    Adds consistent context (strategy, symbol, etc.) to all messages.
    """

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None):
        """
        Initialize context logger.

        Args:
            logger: Underlying logger
            context: Default context to inject
        """
        self.logger = logger
        self.context = context or {}
        self._local = threading.local()

    def with_context(self, **kwargs) -> "ContextLogger":
        """Create new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return ContextLogger(self.logger, new_context)

    @contextmanager
    def temporary_context(self, **kwargs):
        """Temporarily add context for a block of code."""
        old_context = self.context.copy()
        self.context.update(kwargs)
        try:
            yield self
        finally:
            self.context = old_context

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if not self.context:
            return message

        context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
        return f"[{context_str}] {message}"

    def debug(self, message: str, *args, **kwargs):
        """Log at DEBUG level with context."""
        self.logger.debug(self._format_message(message), *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log at INFO level with context."""
        self.logger.info(self._format_message(message), *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log at WARNING level with context."""
        self.logger.warning(self._format_message(message), *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log at ERROR level with context."""
        self.logger.error(self._format_message(message), *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """Log exception with context."""
        self.logger.exception(self._format_message(message), *args, **kwargs)


# Global configuration instance
_logging_config: LoggingConfig | None = None


def configure_logging(config: LoggingConfig | None = None) -> LoggingConfig:
    """
    Configure logging for the trading system.

    Args:
        config: Optional custom configuration

    Returns:
        Applied configuration
    """
    global _logging_config

    if config is None:
        config = LoggingConfig()

    config.apply()
    _logging_config = config

    return config


def get_logging_config() -> LoggingConfig:
    """Get current logging configuration."""
    global _logging_config

    if _logging_config is None:
        _logging_config = LoggingConfig()
        _logging_config.apply()

    return _logging_config


def get_rate_limited_logger(
    name: str,
    max_per_minute: int = 10,
) -> RateLimitedLogger:
    """
    Get a rate-limited logger for a module.

    Args:
        name: Logger name
        max_per_minute: Max messages per minute

    Returns:
        Rate-limited logger
    """
    return RateLimitedLogger(logging.getLogger(name), max_per_minute=max_per_minute)


def get_performance_logger(
    name: str,
    slow_threshold_ms: float = 100.0,
) -> PerformanceLogger:
    """
    Get a performance logger for a module.

    Args:
        name: Logger name
        slow_threshold_ms: Threshold for slow operation warning

    Returns:
        Performance logger
    """
    return PerformanceLogger(logging.getLogger(name), slow_threshold_ms=slow_threshold_ms)


def get_context_logger(
    name: str,
    **context,
) -> ContextLogger:
    """
    Get a context logger for a module.

    Args:
        name: Logger name
        **context: Initial context

    Returns:
        Context logger
    """
    return ContextLogger(logging.getLogger(name), context)

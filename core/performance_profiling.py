"""
Performance Profiling Module
============================

Addresses issues:
- #S15: No performance profiling hooks
- #S17: No load testing framework

Features:
- Function-level profiling with minimal overhead
- Critical path identification
- Memory profiling
- Load testing framework
- Performance regression detection
"""

from __future__ import annotations

import asyncio
import functools
import gc
import logging
import statistics
import sys
import threading
import time
import tracemalloc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =========================================================================
# PERFORMANCE PROFILING (#S15)
# =========================================================================

@dataclass
class ProfileMetrics:
    """Metrics for a profiled function."""
    function_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    p50_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    std_dev_ms: float = 0.0
    last_call_time: datetime | None = None
    errors: int = 0
    _times: list[float] = field(default_factory=list, repr=False)

    def record(self, duration_ms: float, is_error: bool = False) -> None:
        """Record a call duration."""
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self._times.append(duration_ms)
        self.last_call_time = datetime.now(timezone.utc)

        if is_error:
            self.errors += 1

        # Keep only last 1000 samples for percentiles
        if len(self._times) > 1000:
            self._times = self._times[-1000:]

        # Update aggregates
        self.avg_time_ms = self.total_time_ms / self.call_count
        if len(self._times) > 1:
            self.std_dev_ms = statistics.stdev(self._times)
            sorted_times = sorted(self._times)
            n = len(sorted_times)
            self.p50_time_ms = sorted_times[n // 2]
            self.p95_time_ms = sorted_times[int(n * 0.95)]
            self.p99_time_ms = sorted_times[int(n * 0.99)]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "p50_time_ms": round(self.p50_time_ms, 2),
            "p95_time_ms": round(self.p95_time_ms, 2),
            "p99_time_ms": round(self.p99_time_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "errors": self.errors,
            "error_rate_pct": (self.errors / self.call_count * 100) if self.call_count > 0 else 0,
        }


class PerformanceProfiler:
    """
    Performance profiling system (#S15).

    Low-overhead profiling for production use.
    """

    def __init__(
        self,
        enabled: bool = True,
        slow_threshold_ms: float = 100.0,
        log_slow_calls: bool = True,
    ):
        """
        Initialize profiler.

        Args:
            enabled: Enable profiling
            slow_threshold_ms: Threshold for slow call logging
            log_slow_calls: Log calls exceeding threshold
        """
        self.enabled = enabled
        self.slow_threshold_ms = slow_threshold_ms
        self.log_slow_calls = log_slow_calls

        self._metrics: dict[str, ProfileMetrics] = {}
        self._lock = threading.Lock()
        self._start_time = datetime.now(timezone.utc)

    def profile(
        self,
        name: str | None = None,
        log_slow: bool | None = None,
        slow_threshold_ms: float | None = None,
    ):
        """
        Decorator to profile a function.

        Args:
            name: Custom name (default: function name)
            log_slow: Override slow logging setting
            slow_threshold_ms: Override slow threshold

        Example:
            @profiler.profile()
            def calculate_var():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or f"{func.__module__}.{func.__name__}"
            threshold = slow_threshold_ms or self.slow_threshold_ms
            should_log = log_slow if log_slow is not None else self.log_slow_calls

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                if not self.enabled:
                    return func(*args, **kwargs)

                start = time.perf_counter()
                is_error = False
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    is_error = True
                    logger.debug(f"Exception in profiled function {func_name}: {e}")
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    self._record(func_name, duration_ms, is_error)

                    if should_log and duration_ms > threshold:
                        logger.warning(
                            f"Slow call: {func_name} took {duration_ms:.2f}ms "
                            f"(threshold: {threshold}ms)"
                        )

            return wrapper
        return decorator

    @contextmanager
    def measure(self, name: str):
        """
        Context manager for profiling code blocks.

        Example:
            with profiler.measure("data_processing"):
                process_data()
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        is_error = False
        try:
            yield
        except Exception as e:
            is_error = True
            logger.debug(f"Exception in profiled block {name}: {e}")
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._record(name, duration_ms, is_error)

    def _record(self, name: str, duration_ms: float, is_error: bool) -> None:
        """Record a measurement."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = ProfileMetrics(function_name=name)
            self._metrics[name].record(duration_ms, is_error)

    def get_metrics(self, name: str) -> ProfileMetrics | None:
        """Get metrics for a specific function."""
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> dict[str, dict]:
        """Get all profiling metrics."""
        with self._lock:
            return {name: metrics.to_dict() for name, metrics in self._metrics.items()}

    def get_slowest_functions(self, n: int = 10, by: str = "avg") -> list[dict]:
        """Get slowest functions."""
        metrics = self.get_all_metrics()

        if by == "avg":
            sorted_metrics = sorted(
                metrics.values(),
                key=lambda x: x["avg_time_ms"],
                reverse=True
            )
        elif by == "p95":
            sorted_metrics = sorted(
                metrics.values(),
                key=lambda x: x["p95_time_ms"],
                reverse=True
            )
        elif by == "total":
            sorted_metrics = sorted(
                metrics.values(),
                key=lambda x: x["total_time_ms"],
                reverse=True
            )
        else:
            sorted_metrics = list(metrics.values())

        return sorted_metrics[:n]

    def get_most_called(self, n: int = 10) -> list[dict]:
        """Get most frequently called functions."""
        metrics = self.get_all_metrics()
        sorted_metrics = sorted(
            metrics.values(),
            key=lambda x: x["call_count"],
            reverse=True
        )
        return sorted_metrics[:n]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._start_time = datetime.now(timezone.utc)

    def get_summary(self) -> dict:
        """Get profiling summary."""
        metrics = self.get_all_metrics()
        total_calls = sum(m["call_count"] for m in metrics.values())
        total_time = sum(m["total_time_ms"] for m in metrics.values())

        return {
            "profiling_start": self._start_time.isoformat(),
            "total_functions_profiled": len(metrics),
            "total_calls": total_calls,
            "total_time_ms": total_time,
            "slowest_by_avg": self.get_slowest_functions(5, by="avg"),
            "most_called": self.get_most_called(5),
        }


class MemoryProfiler:
    """Memory profiling utilities."""

    def __init__(self):
        self._snapshots: list[tuple[datetime, int]] = []
        self._tracking = False

    def start_tracking(self) -> None:
        """Start memory tracking."""
        tracemalloc.start()
        self._tracking = True
        self._snapshots.clear()

    def stop_tracking(self) -> tracemalloc.Snapshot | None:
        """Stop tracking and return snapshot."""
        if self._tracking:
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            self._tracking = False
            return snapshot
        return None

    def take_snapshot(self) -> None:
        """Take a memory snapshot."""
        if self._tracking:
            current_memory = tracemalloc.get_traced_memory()[0]
            self._snapshots.append((datetime.now(timezone.utc), current_memory))

    def get_top_allocations(self, n: int = 10) -> list[dict]:
        """Get top memory allocations."""
        if not self._tracking:
            return []

        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("lineno")

        return [
            {
                "file": str(stat.traceback),
                "size_kb": stat.size / 1024,
                "count": stat.count,
            }
            for stat in stats[:n]
        ]

    @staticmethod
    def get_current_memory_mb() -> float:
        """Get current process memory usage."""
        import sys
        return sys.getsizeof(gc.get_objects()) / 1024 / 1024

    @contextmanager
    def measure_memory(self, label: str = "operation"):
        """Context manager to measure memory delta."""
        gc.collect()
        start_mem = self.get_current_memory_mb()

        yield

        gc.collect()
        end_mem = self.get_current_memory_mb()
        delta = end_mem - start_mem

        logger.debug(f"Memory delta for {label}: {delta:.2f} MB")


# =========================================================================
# LOAD TESTING (#S17)
# =========================================================================

@dataclass
class LoadTestResult:
    """Result of a load test run."""
    test_name: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_requests > 0:
            return self.successful_requests / self.total_requests * 100
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_pct": round(self.success_rate, 2),
            "requests_per_second": round(self.requests_per_second, 2),
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p95_ms": round(self.p95_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
                "max_ms": round(self.max_latency_ms, 2),
                "min_ms": round(self.min_latency_ms, 2),
            },
            "errors": self.errors[:10],  # Limit error samples
        }


class LoadTester:
    """
    Load testing framework (#S17).

    Supports:
    - Concurrent request simulation
    - Ramp-up patterns
    - Performance baseline comparison
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize load tester.

        Args:
            max_workers: Maximum concurrent workers
        """
        self.max_workers = max_workers
        self._results: list[LoadTestResult] = []

    def run_load_test(
        self,
        test_name: str,
        func: Callable[[], Any],
        num_requests: int,
        concurrent_users: int = 1,
        ramp_up_seconds: float = 0,
        think_time_ms: float = 0,
    ) -> LoadTestResult:
        """
        Run a load test.

        Args:
            test_name: Name of the test
            func: Function to test
            num_requests: Total number of requests
            concurrent_users: Number of concurrent users
            ramp_up_seconds: Time to ramp up to full concurrency
            think_time_ms: Delay between requests per user

        Returns:
            LoadTestResult with metrics
        """
        latencies: list[float] = []
        errors: list[str] = []
        successful = 0
        failed = 0

        start_time = time.perf_counter()

        # Calculate requests per worker
        requests_per_worker = num_requests // concurrent_users
        remainder = num_requests % concurrent_users

        def worker_task(worker_id: int, num_reqs: int) -> list[tuple[float, str | None]]:
            """Execute requests for a single worker."""
            worker_latencies = []

            # Ramp-up delay
            if ramp_up_seconds > 0:
                delay = (worker_id / concurrent_users) * ramp_up_seconds
                time.sleep(delay)

            for _ in range(num_reqs):
                req_start = time.perf_counter()
                error = None

                try:
                    func()
                except Exception as e:
                    error = str(e)

                req_duration = (time.perf_counter() - req_start) * 1000
                worker_latencies.append((req_duration, error))

                if think_time_ms > 0:
                    time.sleep(think_time_ms / 1000)

            return worker_latencies

        # Execute with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(concurrent_users):
                reqs = requests_per_worker + (1 if i < remainder else 0)
                futures.append(executor.submit(worker_task, i, reqs))

            for future in as_completed(futures):
                for latency, error in future.result():
                    latencies.append(latency)
                    if error:
                        failed += 1
                        if len(errors) < 100:
                            errors.append(error)
                    else:
                        successful += 1

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Calculate statistics
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        result = LoadTestResult(
            test_name=test_name,
            duration_seconds=duration,
            total_requests=len(latencies),
            successful_requests=successful,
            failed_requests=failed,
            requests_per_second=len(latencies) / duration if duration > 0 else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=sorted_latencies[n // 2] if n > 0 else 0,
            p95_latency_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            p99_latency_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            errors=errors,
        )

        self._results.append(result)
        return result

    def run_stress_test(
        self,
        test_name: str,
        func: Callable[[], Any],
        duration_seconds: float,
        start_users: int = 1,
        max_users: int = 100,
        step_users: int = 10,
        step_duration_seconds: float = 10,
    ) -> list[LoadTestResult]:
        """
        Run a stress test with increasing load.

        Returns results for each step.
        """
        results = []
        current_users = start_users

        while current_users <= max_users:
            # Estimate requests based on duration and expected throughput
            estimated_rps = current_users * 10  # Rough estimate
            num_requests = int(estimated_rps * step_duration_seconds)

            step_result = self.run_load_test(
                test_name=f"{test_name}_users_{current_users}",
                func=func,
                num_requests=num_requests,
                concurrent_users=current_users,
            )
            results.append(step_result)

            logger.info(
                f"Stress test step: {current_users} users, "
                f"RPS={step_result.requests_per_second:.1f}, "
                f"P95={step_result.p95_latency_ms:.1f}ms"
            )

            current_users += step_users

        return results

    def compare_with_baseline(
        self,
        current: LoadTestResult,
        baseline: LoadTestResult,
        latency_threshold_pct: float = 20,
        throughput_threshold_pct: float = 10,
    ) -> dict:
        """
        Compare test result with baseline.

        Returns comparison analysis.
        """
        latency_change = (
            (current.p95_latency_ms - baseline.p95_latency_ms) /
            baseline.p95_latency_ms * 100
        ) if baseline.p95_latency_ms > 0 else 0

        throughput_change = (
            (current.requests_per_second - baseline.requests_per_second) /
            baseline.requests_per_second * 100
        ) if baseline.requests_per_second > 0 else 0

        regression_detected = (
            latency_change > latency_threshold_pct or
            throughput_change < -throughput_threshold_pct
        )

        return {
            "baseline": baseline.to_dict(),
            "current": current.to_dict(),
            "comparison": {
                "latency_change_pct": round(latency_change, 2),
                "throughput_change_pct": round(throughput_change, 2),
                "regression_detected": regression_detected,
                "status": "FAIL" if regression_detected else "PASS",
            },
        }

    def get_all_results(self) -> list[dict]:
        """Get all test results."""
        return [r.to_dict() for r in self._results]


# Global profiler instance
_global_profiler: PerformanceProfiler | None = None


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profiled(
    name: str | None = None,
    slow_threshold_ms: float | None = None,
):
    """Decorator using global profiler."""
    return get_profiler().profile(name=name, slow_threshold_ms=slow_threshold_ms)


# =============================================================================
# P3: FUNCTION-LEVEL TIMING DECORATORS
# =============================================================================

class FunctionTimer:
    """
    Function-level timing decorators with detailed metrics (P3).

    Provides:
    - Simple timing decorator
    - Conditional timing based on thresholds
    - Timing with callback support
    - Cumulative timing across calls
    """

    def __init__(self):
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._call_counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def timed(
        self,
        name: str | None = None,
        log_level: str = "DEBUG",
        include_args: bool = False,
    ):
        """
        Decorator to time function execution.

        Args:
            name: Custom name (default: function.__qualname__)
            log_level: Logging level for timing output
            include_args: Whether to log function arguments

        Example:
            @timer.timed()
            def my_function():
                ...

            @timer.timed(name="custom_name", log_level="INFO")
            def another_function():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or func.__qualname__

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    with self._lock:
                        self._timings[func_name].append(elapsed_ms)
                        self._call_counts[func_name] += 1

                    log_msg = f"{func_name} completed in {elapsed_ms:.2f}ms"
                    if include_args:
                        log_msg += f" args={args}, kwargs={kwargs}"

                    log_func = getattr(logger, log_level.lower(), logger.debug)
                    log_func(log_msg)

            return wrapper
        return decorator

    def timed_async(
        self,
        name: str | None = None,
        log_level: str = "DEBUG",
    ):
        """
        Decorator to time async function execution.

        Args:
            name: Custom name (default: function.__qualname__)
            log_level: Logging level for timing output

        Example:
            @timer.timed_async()
            async def my_async_function():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or func.__qualname__

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    with self._lock:
                        self._timings[func_name].append(elapsed_ms)
                        self._call_counts[func_name] += 1

                    log_func = getattr(logger, log_level.lower(), logger.debug)
                    log_func(f"{func_name} completed in {elapsed_ms:.2f}ms")

            return wrapper
        return decorator

    def conditional_timed(
        self,
        threshold_ms: float,
        name: str | None = None,
    ):
        """
        Decorator that only logs if execution exceeds threshold.

        Args:
            threshold_ms: Minimum duration to trigger logging
            name: Custom name

        Example:
            @timer.conditional_timed(threshold_ms=100)
            def potentially_slow_function():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or func.__qualname__

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    with self._lock:
                        self._timings[func_name].append(elapsed_ms)
                        self._call_counts[func_name] += 1

                    if elapsed_ms >= threshold_ms:
                        logger.warning(
                            f"Slow execution: {func_name} took {elapsed_ms:.2f}ms "
                            f"(threshold: {threshold_ms}ms)"
                        )

            return wrapper
        return decorator

    def timed_with_callback(
        self,
        callback: Callable[[str, float], None],
        name: str | None = None,
    ):
        """
        Decorator that calls a callback with timing data.

        Args:
            callback: Function(func_name, elapsed_ms) to call after execution
            name: Custom name

        Example:
            def my_callback(func_name, elapsed_ms):
                send_metrics(func_name, elapsed_ms)

            @timer.timed_with_callback(my_callback)
            def monitored_function():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or func.__qualname__

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    with self._lock:
                        self._timings[func_name].append(elapsed_ms)
                        self._call_counts[func_name] += 1

                    try:
                        callback(func_name, elapsed_ms)
                    except Exception as e:
                        logger.error(f"Timing callback error: {e}")

            return wrapper
        return decorator

    def get_statistics(self, func_name: str) -> dict[str, Any]:
        """Get timing statistics for a function."""
        with self._lock:
            timings = self._timings.get(func_name, [])

        if not timings:
            return {"function": func_name, "error": "no_data"}

        return {
            "function": func_name,
            "call_count": len(timings),
            "total_ms": sum(timings),
            "avg_ms": statistics.mean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
            "p50_ms": statistics.median(timings),
            "p95_ms": sorted(timings)[int(len(timings) * 0.95)] if len(timings) >= 20 else max(timings),
        }

    def get_all_statistics(self) -> dict[str, dict]:
        """Get statistics for all timed functions."""
        with self._lock:
            func_names = list(self._timings.keys())
        return {name: self.get_statistics(name) for name in func_names}

    def reset(self, func_name: str | None = None) -> None:
        """Reset timing data."""
        with self._lock:
            if func_name:
                self._timings.pop(func_name, None)
                self._call_counts.pop(func_name, None)
            else:
                self._timings.clear()
                self._call_counts.clear()


# Global function timer instance
_function_timer: FunctionTimer | None = None


def get_function_timer() -> FunctionTimer:
    """Get global function timer instance."""
    global _function_timer
    if _function_timer is None:
        _function_timer = FunctionTimer()
    return _function_timer


# Convenience decorators using global timer
def timed(name: str | None = None, log_level: str = "DEBUG"):
    """Simple timing decorator using global timer."""
    return get_function_timer().timed(name=name, log_level=log_level)


def timed_if_slow(threshold_ms: float = 100.0, name: str | None = None):
    """Timing decorator that only logs slow executions."""
    return get_function_timer().conditional_timed(threshold_ms=threshold_ms, name=name)


# =============================================================================
# P3: MEMORY PROFILING HOOKS
# =============================================================================

class MemoryProfilingHooks:
    """
    Memory profiling hooks and decorators (P3).

    Provides:
    - Memory tracking decorators
    - Memory delta measurement
    - Memory leak detection helpers
    - Object count tracking
    """

    def __init__(self):
        self._memory_snapshots: list[dict] = []
        self._object_counts: dict[str, list[int]] = defaultdict(list)
        self._tracking_enabled = False

    def enable_tracking(self) -> None:
        """Enable memory tracking with tracemalloc."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._tracking_enabled = True

    def disable_tracking(self) -> None:
        """Disable memory tracking."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        self._tracking_enabled = False

    def memory_tracked(self, name: str | None = None, log_delta: bool = True):
        """
        Decorator to track memory usage of a function.

        Args:
            name: Custom name for the measurement
            log_delta: Whether to log memory delta

        Example:
            @memory_hooks.memory_tracked()
            def memory_intensive_function():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or func.__qualname__

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                if not self._tracking_enabled:
                    return func(*args, **kwargs)

                gc.collect()
                before = tracemalloc.get_traced_memory()[0]

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    gc.collect()
                    after = tracemalloc.get_traced_memory()[0]
                    delta_bytes = after - before
                    delta_mb = delta_bytes / (1024 * 1024)

                    self._memory_snapshots.append({
                        "function": func_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "before_bytes": before,
                        "after_bytes": after,
                        "delta_bytes": delta_bytes,
                        "delta_mb": delta_mb,
                    })

                    if log_delta:
                        if delta_mb > 10:
                            logger.warning(f"High memory usage in {func_name}: {delta_mb:.2f}MB")
                        elif delta_mb > 1:
                            logger.info(f"Memory delta in {func_name}: {delta_mb:.2f}MB")
                        else:
                            logger.debug(f"Memory delta in {func_name}: {delta_mb:.2f}MB")

            return wrapper
        return decorator

    def memory_tracked_async(self, name: str | None = None, log_delta: bool = True):
        """
        Decorator to track memory usage of async functions.

        Args:
            name: Custom name for the measurement
            log_delta: Whether to log memory delta
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or func.__qualname__

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                if not self._tracking_enabled:
                    return await func(*args, **kwargs)

                gc.collect()
                before = tracemalloc.get_traced_memory()[0]

                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    gc.collect()
                    after = tracemalloc.get_traced_memory()[0]
                    delta_bytes = after - before
                    delta_mb = delta_bytes / (1024 * 1024)

                    self._memory_snapshots.append({
                        "function": func_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "before_bytes": before,
                        "after_bytes": after,
                        "delta_bytes": delta_bytes,
                        "delta_mb": delta_mb,
                    })

                    if log_delta and delta_mb > 1:
                        logger.info(f"Memory delta in {func_name}: {delta_mb:.2f}MB")

            return wrapper
        return decorator

    @contextmanager
    def track_memory_block(self, label: str):
        """
        Context manager to track memory for a code block.

        Example:
            with memory_hooks.track_memory_block("data_processing"):
                process_large_data()
        """
        gc.collect()
        before = tracemalloc.get_traced_memory()[0] if self._tracking_enabled else 0

        yield

        gc.collect()
        after = tracemalloc.get_traced_memory()[0] if self._tracking_enabled else 0
        delta_mb = (after - before) / (1024 * 1024)

        self._memory_snapshots.append({
            "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "delta_mb": delta_mb,
        })

        logger.debug(f"Memory block '{label}': {delta_mb:.2f}MB")

    def track_object_counts(self, types_to_track: list[type] | None = None) -> dict[str, int]:
        """
        Track counts of objects by type.

        Args:
            types_to_track: Specific types to track (default: common types)

        Returns:
            Object counts by type name
        """
        if types_to_track is None:
            types_to_track = [list, dict, set, tuple, str, bytes]

        gc.collect()
        counts = {}

        for obj_type in types_to_track:
            count = len([obj for obj in gc.get_objects() if isinstance(obj, obj_type)])
            type_name = obj_type.__name__
            counts[type_name] = count
            self._object_counts[type_name].append(count)

        return counts

    def detect_potential_leaks(self, threshold_growth_pct: float = 50.0) -> list[dict]:
        """
        Detect potential memory leaks based on object count growth.

        Args:
            threshold_growth_pct: Growth percentage to flag as potential leak

        Returns:
            List of types with suspected leaks
        """
        leaks = []

        for type_name, counts in self._object_counts.items():
            if len(counts) < 3:
                continue

            initial = counts[0]
            final = counts[-1]

            if initial > 0:
                growth_pct = ((final - initial) / initial) * 100
                if growth_pct > threshold_growth_pct:
                    leaks.append({
                        "type": type_name,
                        "initial_count": initial,
                        "final_count": final,
                        "growth_pct": growth_pct,
                        "samples": len(counts),
                    })

        return leaks

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory profiling report."""
        current_mem = tracemalloc.get_traced_memory() if self._tracking_enabled else (0, 0)

        return {
            "tracking_enabled": self._tracking_enabled,
            "current_memory_mb": current_mem[0] / (1024 * 1024),
            "peak_memory_mb": current_mem[1] / (1024 * 1024),
            "snapshots_count": len(self._memory_snapshots),
            "recent_snapshots": self._memory_snapshots[-10:],
            "potential_leaks": self.detect_potential_leaks(),
        }

    def reset(self) -> None:
        """Reset all memory tracking data."""
        self._memory_snapshots.clear()
        self._object_counts.clear()


# Global memory hooks instance
_memory_hooks: MemoryProfilingHooks | None = None


def get_memory_hooks() -> MemoryProfilingHooks:
    """Get global memory hooks instance."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = MemoryProfilingHooks()
    return _memory_hooks


def memory_tracked(name: str | None = None):
    """Memory tracking decorator using global hooks."""
    return get_memory_hooks().memory_tracked(name=name)


# =============================================================================
# P3: ASYNC TASK PROFILING
# =============================================================================

@dataclass
class AsyncTaskMetrics:
    """Metrics for an async task."""
    task_name: str
    task_id: str
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: float | None = None
    status: str = "running"  # running, completed, failed, cancelled
    error: str | None = None
    parent_task_id: str | None = None
    children_task_ids: list[str] = field(default_factory=list)


class AsyncTaskProfiler:
    """
    Profiling system for async tasks (P3).

    Provides:
    - Task execution tracking
    - Concurrent task monitoring
    - Task hierarchy visualization
    - Async bottleneck detection
    """

    def __init__(self):
        self._tasks: dict[str, AsyncTaskMetrics] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._active_tasks: set[str] = set()

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        with self._lock:
            self._task_counter += 1
            return f"TASK-{self._task_counter:06d}"

    def profile_async(
        self,
        name: str | None = None,
        track_children: bool = True,
    ):
        """
        Decorator to profile async functions.

        Args:
            name: Custom task name
            track_children: Whether to track child tasks

        Example:
            @async_profiler.profile_async()
            async def fetch_data():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            task_name = name or func.__qualname__

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                task_id = self._generate_task_id()
                start_time = datetime.now(timezone.utc)

                # Get parent task if any
                parent_id = None
                current_task = asyncio.current_task()
                if current_task and hasattr(current_task, '_profiler_task_id'):
                    parent_id = current_task._profiler_task_id

                metrics = AsyncTaskMetrics(
                    task_name=task_name,
                    task_id=task_id,
                    start_time=start_time,
                    parent_task_id=parent_id,
                )

                with self._lock:
                    self._tasks[task_id] = metrics
                    self._active_tasks.add(task_id)
                    if parent_id and parent_id in self._tasks:
                        self._tasks[parent_id].children_task_ids.append(task_id)

                # Mark current task with profiler ID
                if current_task and track_children:
                    current_task._profiler_task_id = task_id

                try:
                    result = await func(*args, **kwargs)
                    metrics.status = "completed"
                    return result
                except asyncio.CancelledError:
                    metrics.status = "cancelled"
                    raise
                except Exception as e:
                    metrics.status = "failed"
                    metrics.error = str(e)
                    raise
                finally:
                    end_time = datetime.now(timezone.utc)
                    metrics.end_time = end_time
                    metrics.duration_ms = (end_time - start_time).total_seconds() * 1000

                    with self._lock:
                        self._active_tasks.discard(task_id)

                    logger.debug(
                        f"Async task {task_name} ({task_id}) "
                        f"completed in {metrics.duration_ms:.2f}ms "
                        f"status={metrics.status}"
                    )

            return wrapper
        return decorator

    @contextmanager
    def profile_task_group(self, name: str):
        """
        Context manager to profile a group of tasks.

        Example:
            async with async_profiler.profile_task_group("batch_fetch"):
                await asyncio.gather(fetch1(), fetch2(), fetch3())
        """
        task_id = self._generate_task_id()
        start_time = datetime.now(timezone.utc)

        metrics = AsyncTaskMetrics(
            task_name=name,
            task_id=task_id,
            start_time=start_time,
        )

        with self._lock:
            self._tasks[task_id] = metrics
            self._active_tasks.add(task_id)

        try:
            yield task_id
            metrics.status = "completed"
        except Exception as e:
            metrics.status = "failed"
            metrics.error = str(e)
            raise
        finally:
            end_time = datetime.now(timezone.utc)
            metrics.end_time = end_time
            metrics.duration_ms = (end_time - start_time).total_seconds() * 1000

            with self._lock:
                self._active_tasks.discard(task_id)

    def get_active_tasks(self) -> list[dict]:
        """Get currently running tasks."""
        with self._lock:
            active_ids = list(self._active_tasks)

        return [
            {
                "task_id": tid,
                "task_name": self._tasks[tid].task_name,
                "running_ms": (datetime.now(timezone.utc) - self._tasks[tid].start_time).total_seconds() * 1000,
                "parent_id": self._tasks[tid].parent_task_id,
            }
            for tid in active_ids
            if tid in self._tasks
        ]

    def get_task_hierarchy(self, task_id: str) -> dict | None:
        """Get task with its children recursively."""
        if task_id not in self._tasks:
            return None

        task = self._tasks[task_id]
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": task.status,
            "duration_ms": task.duration_ms,
            "children": [
                self.get_task_hierarchy(child_id)
                for child_id in task.children_task_ids
                if child_id in self._tasks
            ],
        }

    def get_slowest_tasks(self, n: int = 10) -> list[dict]:
        """Get slowest completed tasks."""
        completed = [
            t for t in self._tasks.values()
            if t.status == "completed" and t.duration_ms is not None
        ]

        sorted_tasks = sorted(completed, key=lambda t: t.duration_ms or 0, reverse=True)

        return [
            {
                "task_id": t.task_id,
                "task_name": t.task_name,
                "duration_ms": t.duration_ms,
                "start_time": t.start_time.isoformat(),
            }
            for t in sorted_tasks[:n]
        ]

    def get_task_statistics(self, task_name: str | None = None) -> dict[str, Any]:
        """Get statistics for tasks, optionally filtered by name."""
        tasks = list(self._tasks.values())
        if task_name:
            tasks = [t for t in tasks if t.task_name == task_name]

        if not tasks:
            return {"error": "no_tasks"}

        completed = [t for t in tasks if t.duration_ms is not None]
        durations = [t.duration_ms for t in completed if t.duration_ms]

        return {
            "total_tasks": len(tasks),
            "completed": len([t for t in tasks if t.status == "completed"]),
            "failed": len([t for t in tasks if t.status == "failed"]),
            "cancelled": len([t for t in tasks if t.status == "cancelled"]),
            "active": len([t for t in tasks if t.status == "running"]),
            "timing": {
                "avg_ms": statistics.mean(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
                "total_ms": sum(durations),
            } if durations else {},
        }

    def detect_bottlenecks(
        self,
        threshold_ms: float = 1000.0,
        min_occurrences: int = 3
    ) -> list[dict]:
        """
        Detect potential async bottlenecks.

        Args:
            threshold_ms: Duration threshold for slow tasks
            min_occurrences: Minimum occurrences to flag

        Returns:
            List of potential bottlenecks
        """
        # Group tasks by name
        task_groups: dict[str, list[AsyncTaskMetrics]] = defaultdict(list)
        for task in self._tasks.values():
            if task.duration_ms is not None:
                task_groups[task.task_name].append(task)

        bottlenecks = []
        for task_name, tasks in task_groups.items():
            slow_tasks = [t for t in tasks if (t.duration_ms or 0) > threshold_ms]

            if len(slow_tasks) >= min_occurrences:
                durations = [t.duration_ms for t in slow_tasks if t.duration_ms]
                bottlenecks.append({
                    "task_name": task_name,
                    "slow_count": len(slow_tasks),
                    "total_count": len(tasks),
                    "slow_pct": len(slow_tasks) / len(tasks) * 100,
                    "avg_slow_duration_ms": statistics.mean(durations) if durations else 0,
                    "recommendation": "Consider caching or parallel execution",
                })

        return sorted(bottlenecks, key=lambda b: b["avg_slow_duration_ms"], reverse=True)

    def get_concurrency_report(self) -> dict[str, Any]:
        """Get report on task concurrency patterns."""
        if not self._tasks:
            return {"error": "no_tasks"}

        # Group overlapping tasks
        timeline = []
        for task in self._tasks.values():
            timeline.append((task.start_time, "start", task.task_id))
            if task.end_time:
                timeline.append((task.end_time, "end", task.task_id))

        timeline.sort(key=lambda x: x[0])

        max_concurrent = 0
        current_concurrent = 0
        concurrent_history = []

        for ts, event_type, task_id in timeline:
            if event_type == "start":
                current_concurrent += 1
            else:
                current_concurrent -= 1

            max_concurrent = max(max_concurrent, current_concurrent)
            concurrent_history.append((ts, current_concurrent))

        return {
            "max_concurrent_tasks": max_concurrent,
            "total_tasks_tracked": len(self._tasks),
            "average_concurrent": (
                sum(c for _, c in concurrent_history) / len(concurrent_history)
                if concurrent_history else 0
            ),
        }

    def reset(self) -> None:
        """Reset all task tracking data."""
        with self._lock:
            self._tasks.clear()
            self._active_tasks.clear()


# Global async profiler instance
_async_profiler: AsyncTaskProfiler | None = None


def get_async_profiler() -> AsyncTaskProfiler:
    """Get global async profiler instance."""
    global _async_profiler
    if _async_profiler is None:
        _async_profiler = AsyncTaskProfiler()
    return _async_profiler


def profile_async(name: str | None = None):
    """Async profiling decorator using global profiler."""
    return get_async_profiler().profile_async(name=name)

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
                except Exception:
                    is_error = True
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
        except Exception:
            is_error = True
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

# performance_profiling

**Path**: `C:\Users\Alexa\ai-trading-firm\core\performance_profiling.py`

## Overview

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

## Classes

### ProfileMetrics

Metrics for a profiled function.

#### Methods

##### `def record(self, duration_ms: float, is_error: bool) -> None`

Record a call duration.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### PerformanceProfiler

Performance profiling system (#S15).

Low-overhead profiling for production use.

#### Methods

##### `def __init__(self, enabled: bool, slow_threshold_ms: float, log_slow_calls: bool)`

Initialize profiler.

Args:
    enabled: Enable profiling
    slow_threshold_ms: Threshold for slow call logging
    log_slow_calls: Log calls exceeding threshold

##### `def profile(self, name: , log_slow: , slow_threshold_ms: )`

Decorator to profile a function.

Args:
    name: Custom name (default: function name)
    log_slow: Override slow logging setting
    slow_threshold_ms: Override slow threshold

Example:
    @profiler.profile()
    def calculate_var():
        ...

##### `def measure(self, name: str)`

Context manager for profiling code blocks.

Example:
    with profiler.measure("data_processing"):
        process_data()

##### `def get_metrics(self, name: str)`

Get metrics for a specific function.

##### `def get_all_metrics(self) -> dict[str, dict]`

Get all profiling metrics.

##### `def get_slowest_functions(self, n: int, by: str) -> list[dict]`

Get slowest functions.

##### `def get_most_called(self, n: int) -> list[dict]`

Get most frequently called functions.

##### `def reset(self) -> None`

Reset all metrics.

##### `def get_summary(self) -> dict`

Get profiling summary.

### MemoryProfiler

Memory profiling utilities.

#### Methods

##### `def __init__(self)`

##### `def start_tracking(self) -> None`

Start memory tracking.

##### `def stop_tracking(self)`

Stop tracking and return snapshot.

##### `def take_snapshot(self) -> None`

Take a memory snapshot.

##### `def get_top_allocations(self, n: int) -> list[dict]`

Get top memory allocations.

##### `def get_current_memory_mb() -> float`

Get current process memory usage.

##### `def measure_memory(self, label: str)`

Context manager to measure memory delta.

### LoadTestResult

Result of a load test run.

#### Methods

##### `def success_rate(self) -> float`

Success rate as percentage.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### LoadTester

Load testing framework (#S17).

Supports:
- Concurrent request simulation
- Ramp-up patterns
- Performance baseline comparison

#### Methods

##### `def __init__(self, max_workers: int)`

Initialize load tester.

Args:
    max_workers: Maximum concurrent workers

##### `def run_load_test(self, test_name: str, func: Callable[, Any], num_requests: int, concurrent_users: int, ramp_up_seconds: float, think_time_ms: float) -> LoadTestResult`

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

##### `def run_stress_test(self, test_name: str, func: Callable[, Any], duration_seconds: float, start_users: int, max_users: int, step_users: int, step_duration_seconds: float) -> list[LoadTestResult]`

Run a stress test with increasing load.

Returns results for each step.

##### `def compare_with_baseline(self, current: LoadTestResult, baseline: LoadTestResult, latency_threshold_pct: float, throughput_threshold_pct: float) -> dict`

Compare test result with baseline.

Returns comparison analysis.

##### `def get_all_results(self) -> list[dict]`

Get all test results.

## Functions

### `def get_profiler() -> PerformanceProfiler`

Get global profiler instance.

### `def profiled(name: , slow_threshold_ms: )`

Decorator using global profiler.

## Constants

- `T`

"""
Execution Benchmark Comparison Module
=====================================

Addresses issues:
- #E32: No execution benchmark comparison
- #E33: Missing execution alert system

Features:
- Multiple benchmark comparisons (VWAP, TWAP, arrival price)
- Real-time execution quality monitoring
- Automated alerting on poor execution
- Historical comparison analysis
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class ExecutionBenchmark(str, Enum):
    """Standard execution benchmarks."""
    ARRIVAL_PRICE = "arrival_price"
    VWAP = "vwap"  # Volume-weighted average price
    TWAP = "twap"  # Time-weighted average price
    OPEN = "open"
    CLOSE = "close"
    MIDPOINT = "midpoint"
    BEST_BID = "best_bid"
    BEST_ASK = "best_ask"


class ExecutionAlertLevel(str, Enum):
    """Alert severity for execution issues."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BenchmarkPrice:
    """A single benchmark price observation."""
    timestamp: datetime
    benchmark: ExecutionBenchmark
    price: float
    symbol: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ExecutionComparison:
    """Comparison of execution against benchmarks."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    fill_price: float
    fill_time: datetime
    benchmarks: dict[ExecutionBenchmark, float]
    slippage_vs_benchmarks: dict[ExecutionBenchmark, float]  # In bps

    @property
    def vs_arrival_bps(self) -> float | None:
        """Slippage vs arrival price in bps."""
        return self.slippage_vs_benchmarks.get(ExecutionBenchmark.ARRIVAL_PRICE)

    @property
    def vs_vwap_bps(self) -> float | None:
        """Slippage vs VWAP in bps."""
        return self.slippage_vs_benchmarks.get(ExecutionBenchmark.VWAP)

    @property
    def best_benchmark_performance(self) -> tuple[ExecutionBenchmark, float] | None:
        """Best benchmark performance (lowest slippage)."""
        if not self.slippage_vs_benchmarks:
            return None
        return min(self.slippage_vs_benchmarks.items(), key=lambda x: abs(x[1]))

    @property
    def worst_benchmark_performance(self) -> tuple[ExecutionBenchmark, float] | None:
        """Worst benchmark performance (highest slippage)."""
        if not self.slippage_vs_benchmarks:
            return None
        return max(self.slippage_vs_benchmarks.items(), key=lambda x: abs(x[1]))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "fill_time": self.fill_time.isoformat(),
            "benchmarks": {b.value: p for b, p in self.benchmarks.items()},
            "slippage_bps": {b.value: s for b, s in self.slippage_vs_benchmarks.items()},
        }


@dataclass
class ExecutionAlert:
    """Alert for execution quality issues (#E33)."""
    alert_id: str
    level: ExecutionAlertLevel
    order_id: str
    symbol: str
    message: str
    timestamp: datetime
    details: dict = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


class ExecutionAlertThresholds:
    """Configurable thresholds for execution alerts (#E33)."""

    def __init__(
        self,
        slippage_warning_bps: float = 10.0,
        slippage_critical_bps: float = 25.0,
        fill_time_warning_ms: float = 500.0,
        fill_time_critical_ms: float = 2000.0,
        rejection_rate_warning_pct: float = 5.0,
        rejection_rate_critical_pct: float = 10.0,
        partial_fill_warning_pct: float = 50.0,  # Less than X% filled
        implementation_shortfall_warning_bps: float = 15.0,
        implementation_shortfall_critical_bps: float = 30.0,
    ):
        self.slippage_warning_bps = slippage_warning_bps
        self.slippage_critical_bps = slippage_critical_bps
        self.fill_time_warning_ms = fill_time_warning_ms
        self.fill_time_critical_ms = fill_time_critical_ms
        self.rejection_rate_warning_pct = rejection_rate_warning_pct
        self.rejection_rate_critical_pct = rejection_rate_critical_pct
        self.partial_fill_warning_pct = partial_fill_warning_pct
        self.implementation_shortfall_warning_bps = implementation_shortfall_warning_bps
        self.implementation_shortfall_critical_bps = implementation_shortfall_critical_bps


class ExecutionBenchmarkComparator:
    """
    Compares execution quality against multiple benchmarks (#E32).

    Provides comprehensive execution analysis and tracking.
    """

    def __init__(self):
        self._comparisons: list[ExecutionComparison] = []
        self._lock = threading.Lock()
        self._symbol_stats: dict[str, list[ExecutionComparison]] = defaultdict(list)

    def compare_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        fill_time: datetime,
        benchmarks: dict[ExecutionBenchmark, float],
    ) -> ExecutionComparison:
        """
        Compare an execution against benchmarks.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            fill_price: Actual fill price
            fill_time: Fill timestamp
            benchmarks: Dictionary of benchmark prices

        Returns:
            ExecutionComparison with slippage analysis
        """
        slippage = {}

        for benchmark, bench_price in benchmarks.items():
            if bench_price > 0:
                # Calculate slippage in bps
                # For buys: positive slippage = paid more than benchmark
                # For sells: positive slippage = received less than benchmark
                if side == "BUY":
                    slip_bps = (fill_price - bench_price) / bench_price * 10000
                else:
                    slip_bps = (bench_price - fill_price) / bench_price * 10000

                slippage[benchmark] = slip_bps

        comparison = ExecutionComparison(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            fill_time=fill_time,
            benchmarks=benchmarks,
            slippage_vs_benchmarks=slippage,
        )

        with self._lock:
            self._comparisons.append(comparison)
            self._symbol_stats[symbol].append(comparison)

        return comparison

    def get_aggregate_stats(
        self,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get aggregate execution statistics.

        Args:
            symbol: Optional symbol filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary of aggregate statistics
        """
        with self._lock:
            if symbol:
                comparisons = self._symbol_stats.get(symbol, [])
            else:
                comparisons = self._comparisons

        # Apply time filters
        if start_time:
            comparisons = [c for c in comparisons if c.fill_time >= start_time]
        if end_time:
            comparisons = [c for c in comparisons if c.fill_time <= end_time]

        if not comparisons:
            return {"error": "No comparisons found"}

        # Aggregate by benchmark
        stats_by_benchmark = {}

        for benchmark in ExecutionBenchmark:
            slippages = [
                c.slippage_vs_benchmarks.get(benchmark)
                for c in comparisons
                if benchmark in c.slippage_vs_benchmarks
            ]

            slippages = [s for s in slippages if s is not None]

            if slippages:
                stats_by_benchmark[benchmark.value] = {
                    "count": len(slippages),
                    "avg_slippage_bps": statistics.mean(slippages),
                    "median_slippage_bps": statistics.median(slippages),
                    "std_slippage_bps": statistics.stdev(slippages) if len(slippages) > 1 else 0,
                    "min_slippage_bps": min(slippages),
                    "max_slippage_bps": max(slippages),
                    "p25_slippage_bps": sorted(slippages)[int(len(slippages) * 0.25)],
                    "p75_slippage_bps": sorted(slippages)[int(len(slippages) * 0.75)],
                }

        # Overall stats
        all_arrival = [
            c.slippage_vs_benchmarks.get(ExecutionBenchmark.ARRIVAL_PRICE, 0)
            for c in comparisons
            if ExecutionBenchmark.ARRIVAL_PRICE in c.slippage_vs_benchmarks
        ]

        return {
            "period": {
                "start": min(c.fill_time for c in comparisons).isoformat() if comparisons else None,
                "end": max(c.fill_time for c in comparisons).isoformat() if comparisons else None,
            },
            "total_executions": len(comparisons),
            "total_quantity": sum(c.quantity for c in comparisons),
            "by_benchmark": stats_by_benchmark,
            "overall_vs_arrival_bps": statistics.mean(all_arrival) if all_arrival else 0,
            "symbols_traded": list(set(c.symbol for c in comparisons)),
        }

    def get_symbol_ranking(self, benchmark: ExecutionBenchmark = ExecutionBenchmark.ARRIVAL_PRICE) -> list[dict]:
        """
        Rank symbols by execution quality.

        Returns:
            List of symbols with avg slippage, sorted best to worst
        """
        rankings = []

        with self._lock:
            symbols = list(self._symbol_stats.keys())

        for symbol in symbols:
            stats = self.get_aggregate_stats(symbol=symbol)
            if benchmark.value in stats.get("by_benchmark", {}):
                rankings.append({
                    "symbol": symbol,
                    "avg_slippage_bps": stats["by_benchmark"][benchmark.value]["avg_slippage_bps"],
                    "execution_count": stats["total_executions"],
                })

        return sorted(rankings, key=lambda x: x["avg_slippage_bps"])


class ExecutionAlertManager:
    """
    Manages execution quality alerts (#E33).

    Monitors executions and generates alerts when thresholds are breached.
    """

    def __init__(
        self,
        thresholds: ExecutionAlertThresholds | None = None,
        alert_callback: Callable[[ExecutionAlert], None] | None = None,
    ):
        """
        Initialize alert manager.

        Args:
            thresholds: Alert thresholds
            alert_callback: Optional callback for alerts
        """
        self.thresholds = thresholds or ExecutionAlertThresholds()
        self.alert_callback = alert_callback

        self._alerts: list[ExecutionAlert] = []
        self._alert_counter = 0
        self._lock = threading.Lock()

        # Rate limiting
        self._alert_cooldowns: dict[str, datetime] = {}
        self._cooldown_seconds = 300  # 5 minutes between same alert type

    def check_execution(
        self,
        comparison: ExecutionComparison,
        fill_time_ms: float | None = None,
    ) -> list[ExecutionAlert]:
        """
        Check execution and generate alerts if needed.

        Args:
            comparison: Execution comparison result
            fill_time_ms: Fill time in milliseconds

        Returns:
            List of generated alerts
        """
        alerts = []

        # Check slippage
        arrival_slip = comparison.slippage_vs_benchmarks.get(ExecutionBenchmark.ARRIVAL_PRICE)
        if arrival_slip is not None:
            if abs(arrival_slip) > self.thresholds.slippage_critical_bps:
                alerts.append(self._create_alert(
                    level=ExecutionAlertLevel.CRITICAL,
                    order_id=comparison.order_id,
                    symbol=comparison.symbol,
                    message=f"Critical slippage: {arrival_slip:.1f} bps vs arrival price",
                    details={"slippage_bps": arrival_slip, "benchmark": "arrival_price"},
                    cooldown_key=f"slippage_critical:{comparison.symbol}",
                ))
            elif abs(arrival_slip) > self.thresholds.slippage_warning_bps:
                alerts.append(self._create_alert(
                    level=ExecutionAlertLevel.WARNING,
                    order_id=comparison.order_id,
                    symbol=comparison.symbol,
                    message=f"Elevated slippage: {arrival_slip:.1f} bps vs arrival price",
                    details={"slippage_bps": arrival_slip, "benchmark": "arrival_price"},
                    cooldown_key=f"slippage_warning:{comparison.symbol}",
                ))

        # Check fill time
        if fill_time_ms is not None:
            if fill_time_ms > self.thresholds.fill_time_critical_ms:
                alerts.append(self._create_alert(
                    level=ExecutionAlertLevel.CRITICAL,
                    order_id=comparison.order_id,
                    symbol=comparison.symbol,
                    message=f"Slow fill: {fill_time_ms:.0f}ms",
                    details={"fill_time_ms": fill_time_ms},
                    cooldown_key=f"fill_time_critical:{comparison.symbol}",
                ))
            elif fill_time_ms > self.thresholds.fill_time_warning_ms:
                alerts.append(self._create_alert(
                    level=ExecutionAlertLevel.WARNING,
                    order_id=comparison.order_id,
                    symbol=comparison.symbol,
                    message=f"Slow fill: {fill_time_ms:.0f}ms",
                    details={"fill_time_ms": fill_time_ms},
                    cooldown_key=f"fill_time_warning:{comparison.symbol}",
                ))

        # Store alerts
        with self._lock:
            self._alerts.extend(alerts)

        # Trigger callbacks
        for alert in alerts:
            if self.alert_callback:
                self.alert_callback(alert)

        return alerts

    def check_rejection(
        self,
        order_id: str,
        symbol: str,
        reason: str,
        recent_rejection_rate: float,
    ) -> ExecutionAlert | None:
        """
        Check for rejection rate alerts.

        Args:
            order_id: Rejected order ID
            symbol: Symbol
            reason: Rejection reason
            recent_rejection_rate: Recent rejection rate (0-100)

        Returns:
            Alert if threshold exceeded
        """
        if recent_rejection_rate > self.thresholds.rejection_rate_critical_pct:
            return self._create_alert(
                level=ExecutionAlertLevel.CRITICAL,
                order_id=order_id,
                symbol=symbol,
                message=f"High rejection rate: {recent_rejection_rate:.1f}%",
                details={"rejection_rate_pct": recent_rejection_rate, "reason": reason},
                cooldown_key=f"rejection_critical:{symbol}",
            )
        elif recent_rejection_rate > self.thresholds.rejection_rate_warning_pct:
            return self._create_alert(
                level=ExecutionAlertLevel.WARNING,
                order_id=order_id,
                symbol=symbol,
                message=f"Elevated rejection rate: {recent_rejection_rate:.1f}%",
                details={"rejection_rate_pct": recent_rejection_rate, "reason": reason},
                cooldown_key=f"rejection_warning:{symbol}",
            )

        return None

    def check_partial_fill(
        self,
        order_id: str,
        symbol: str,
        filled_pct: float,
        elapsed_time_minutes: float,
    ) -> ExecutionAlert | None:
        """
        Check for partial fill alerts.

        Args:
            order_id: Order ID
            symbol: Symbol
            filled_pct: Percentage filled (0-100)
            elapsed_time_minutes: Time since order submission

        Returns:
            Alert if conditions met
        """
        if filled_pct < self.thresholds.partial_fill_warning_pct and elapsed_time_minutes > 5:
            return self._create_alert(
                level=ExecutionAlertLevel.WARNING,
                order_id=order_id,
                symbol=symbol,
                message=f"Partial fill: {filled_pct:.1f}% after {elapsed_time_minutes:.0f} min",
                details={"filled_pct": filled_pct, "elapsed_minutes": elapsed_time_minutes},
                cooldown_key=f"partial_fill:{order_id}",
            )

        return None

    def _create_alert(
        self,
        level: ExecutionAlertLevel,
        order_id: str,
        symbol: str,
        message: str,
        details: dict,
        cooldown_key: str,
    ) -> ExecutionAlert | None:
        """Create alert with rate limiting."""
        now = datetime.now(timezone.utc)

        # Check cooldown
        with self._lock:
            if cooldown_key in self._alert_cooldowns:
                last_alert = self._alert_cooldowns[cooldown_key]
                if (now - last_alert).total_seconds() < self._cooldown_seconds:
                    return None

            self._alert_counter += 1
            alert_id = f"EXEC-{now.strftime('%Y%m%d')}-{self._alert_counter:05d}"
            self._alert_cooldowns[cooldown_key] = now

        alert = ExecutionAlert(
            alert_id=alert_id,
            level=level,
            order_id=order_id,
            symbol=symbol,
            message=message,
            timestamp=now,
            details=details,
        )

        logger.log(
            logging.WARNING if level == ExecutionAlertLevel.WARNING else logging.ERROR,
            f"Execution alert: {alert.message} [Order: {order_id}, Symbol: {symbol}]"
        )

        return alert

    def get_alerts(
        self,
        level: ExecutionAlertLevel | None = None,
        symbol: str | None = None,
        unacknowledged_only: bool = False,
        limit: int = 100,
    ) -> list[ExecutionAlert]:
        """
        Get alerts with optional filters.

        Args:
            level: Filter by level
            symbol: Filter by symbol
            unacknowledged_only: Only unacknowledged alerts
            limit: Max alerts to return

        Returns:
            List of matching alerts
        """
        with self._lock:
            alerts = list(self._alerts)

        if level:
            alerts = [a for a in alerts if a.level == level]
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        # Sort by timestamp descending
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledge(user)
                    return True
        return False

    def get_alert_summary(self) -> dict:
        """Get summary of alert counts."""
        with self._lock:
            alerts = list(self._alerts)

        return {
            "total": len(alerts),
            "by_level": {
                level.value: sum(1 for a in alerts if a.level == level)
                for level in ExecutionAlertLevel
            },
            "unacknowledged": sum(1 for a in alerts if not a.acknowledged),
            "last_24h": sum(
                1 for a in alerts
                if a.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
            ),
        }


class ExecutionQualityMonitor:
    """
    Comprehensive execution quality monitoring.

    Combines benchmark comparison and alerting.
    """

    def __init__(
        self,
        thresholds: ExecutionAlertThresholds | None = None,
        alert_callback: Callable[[ExecutionAlert], None] | None = None,
    ):
        self.comparator = ExecutionBenchmarkComparator()
        self.alert_manager = ExecutionAlertManager(
            thresholds=thresholds,
            alert_callback=alert_callback,
        )

    def record_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        fill_time: datetime,
        benchmarks: dict[ExecutionBenchmark, float],
        fill_latency_ms: float | None = None,
    ) -> tuple[ExecutionComparison, list[ExecutionAlert]]:
        """
        Record and analyze an execution.

        Returns:
            Tuple of (comparison, alerts)
        """
        # Compare against benchmarks
        comparison = self.comparator.compare_execution(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            fill_time=fill_time,
            benchmarks=benchmarks,
        )

        # Check for alerts
        alerts = self.alert_manager.check_execution(
            comparison=comparison,
            fill_time_ms=fill_latency_ms,
        )

        return comparison, alerts

    def get_dashboard_data(self) -> dict:
        """Get data for execution dashboard."""
        return {
            "aggregate_stats": self.comparator.get_aggregate_stats(),
            "symbol_ranking": self.comparator.get_symbol_ranking(),
            "alert_summary": self.alert_manager.get_alert_summary(),
            "recent_alerts": [
                a.to_dict() for a in self.alert_manager.get_alerts(limit=10)
            ],
        }

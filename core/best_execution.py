"""
Best Execution Analysis
=======================

Implements MiFID II / RTS 27/28 best execution requirements.
Tracks execution quality benchmarks and generates compliance reports.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class ExecutionBenchmark(Enum):
    """Benchmark types for execution quality."""
    ARRIVAL = "arrival"  # Price at decision time
    VWAP = "vwap"  # Volume-Weighted Average Price
    TWAP = "twap"  # Time-Weighted Average Price
    CLOSE = "close"  # Closing price
    OPEN = "open"  # Opening price
    MIDPOINT = "midpoint"  # Bid-ask midpoint at execution


@dataclass
class ExecutionRecord:
    """Record of a single execution for analysis."""
    execution_id: str
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    fill_price: float
    commission: float

    # Benchmark prices (captured at relevant times)
    arrival_price: float | None = None
    vwap_price: float | None = None
    twap_price: float | None = None
    close_price: float | None = None
    midpoint_at_execution: float | None = None

    # Venue information
    venue: str = ""
    algo_used: str = ""

    # Order metadata
    order_id: str = ""
    decision_timestamp: datetime | None = None
    order_timestamp: datetime | None = None

    @property
    def arrival_slippage_bps(self) -> float | None:
        """Calculate slippage vs arrival price in basis points."""
        if self.arrival_price is None or self.arrival_price == 0:
            return None

        if self.side == "buy":
            slippage = (self.fill_price - self.arrival_price) / self.arrival_price
        else:
            slippage = (self.arrival_price - self.fill_price) / self.arrival_price

        return slippage * 10000  # Convert to bps

    @property
    def vwap_slippage_bps(self) -> float | None:
        """Calculate slippage vs VWAP in basis points."""
        if self.vwap_price is None or self.vwap_price == 0:
            return None

        if self.side == "buy":
            slippage = (self.fill_price - self.vwap_price) / self.vwap_price
        else:
            slippage = (self.vwap_price - self.fill_price) / self.vwap_price

        return slippage * 10000

    @property
    def implementation_shortfall(self) -> float | None:
        """
        Calculate implementation shortfall.

        IS = (Execution Price - Decision Price) * Quantity
        For sells, sign is reversed.
        """
        if self.arrival_price is None:
            return None

        if self.side == "buy":
            return (self.fill_price - self.arrival_price) * self.quantity
        else:
            return (self.arrival_price - self.fill_price) * self.quantity

    @property
    def total_cost(self) -> float:
        """Total execution cost including commission and slippage."""
        is_cost = self.implementation_shortfall or 0
        return is_cost + self.commission

    @property
    def execution_latency_ms(self) -> float | None:
        """Latency from order to fill in milliseconds."""
        if self.order_timestamp is None:
            return None
        delta = self.timestamp - self.order_timestamp
        return delta.total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "commission": self.commission,
            "arrival_price": self.arrival_price,
            "vwap_price": self.vwap_price,
            "arrival_slippage_bps": self.arrival_slippage_bps,
            "vwap_slippage_bps": self.vwap_slippage_bps,
            "implementation_shortfall": self.implementation_shortfall,
            "total_cost": self.total_cost,
            "venue": self.venue,
            "algo_used": self.algo_used,
        }


@dataclass
class ExecutionStats:
    """Aggregated execution statistics."""
    period: str  # e.g., "2025-Q1", "2025-01"
    symbol: str | None  # None for aggregate
    total_executions: int = 0
    total_volume: int = 0
    total_notional: float = 0.0
    total_commission: float = 0.0

    # Slippage metrics (in bps)
    avg_arrival_slippage_bps: float = 0.0
    avg_vwap_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    min_slippage_bps: float = 0.0
    std_slippage_bps: float = 0.0

    # Implementation shortfall
    total_is: float = 0.0
    avg_is_per_trade: float = 0.0

    # Latency
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # By venue breakdown
    volume_by_venue: dict[str, int] = field(default_factory=dict)

    # By algo breakdown
    volume_by_algo: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": self.period,
            "symbol": self.symbol,
            "total_executions": self.total_executions,
            "total_volume": self.total_volume,
            "total_notional": self.total_notional,
            "total_commission": self.total_commission,
            "avg_arrival_slippage_bps": self.avg_arrival_slippage_bps,
            "avg_vwap_slippage_bps": self.avg_vwap_slippage_bps,
            "max_slippage_bps": self.max_slippage_bps,
            "total_implementation_shortfall": self.total_is,
            "avg_latency_ms": self.avg_latency_ms,
            "volume_by_venue": self.volume_by_venue,
            "volume_by_algo": self.volume_by_algo,
        }


class BestExecutionAnalyzer:
    """
    Best execution analysis and reporting.

    Compliant with MiFID II RTS 27/28 requirements:
    - Execution quality monitoring
    - Benchmark tracking (Arrival, VWAP, TWAP)
    - Slippage analysis
    - Quarterly reporting
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize best execution analyzer.

        Args:
            config: Configuration with:
                - benchmark: Primary benchmark (default: "vwap")
                - slippage_alert_bps: Alert threshold in bps (default: 50)
                - report_retention_quarters: Quarters to retain (default: 8)
        """
        self._config = config or {}
        self._primary_benchmark = ExecutionBenchmark[
            self._config.get("benchmark", "vwap").upper()
        ]
        self._slippage_alert_bps = self._config.get("slippage_alert_bps", 50)
        self._retention_quarters = self._config.get("report_retention_quarters", 8)

        # Execution records
        self._executions: list[ExecutionRecord] = []
        self._execution_counter = 0

        # Statistics cache
        self._stats_cache: dict[str, ExecutionStats] = {}
        self._last_stats_update: datetime | None = None

        # Alerts
        self._alerts: list[dict[str, Any]] = []

        logger.info(
            f"BestExecutionAnalyzer initialized: benchmark={self._primary_benchmark.value}, "
            f"alert_threshold={self._slippage_alert_bps}bps"
        )

    def record_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        commission: float,
        arrival_price: float | None = None,
        vwap_price: float | None = None,
        twap_price: float | None = None,
        venue: str = "",
        algo_used: str = "",
        order_id: str = "",
        decision_timestamp: datetime | None = None,
        order_timestamp: datetime | None = None
    ) -> ExecutionRecord:
        """
        Record an execution for analysis.

        Args:
            symbol: Instrument symbol
            side: "buy" or "sell"
            quantity: Executed quantity
            fill_price: Fill price
            commission: Commission paid
            arrival_price: Price at decision time
            vwap_price: VWAP benchmark
            twap_price: TWAP benchmark
            venue: Execution venue
            algo_used: Algorithm used
            order_id: Original order ID
            decision_timestamp: When decision was made
            order_timestamp: When order was sent

        Returns:
            ExecutionRecord
        """
        self._execution_counter += 1
        execution_id = f"EXEC-{self._execution_counter:08d}"

        record = ExecutionRecord(
            execution_id=execution_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            arrival_price=arrival_price,
            vwap_price=vwap_price,
            twap_price=twap_price,
            venue=venue,
            algo_used=algo_used,
            order_id=order_id,
            decision_timestamp=decision_timestamp,
            order_timestamp=order_timestamp,
        )

        self._executions.append(record)

        # Check for alerts
        self._check_slippage_alert(record)

        # Invalidate cache
        self._last_stats_update = None

        logger.debug(
            f"Recorded execution: {symbol} {side} {quantity} @ {fill_price}, "
            f"arrival_slip={record.arrival_slippage_bps:.1f}bps" if record.arrival_slippage_bps else ""
        )

        return record

    def _check_slippage_alert(self, record: ExecutionRecord) -> None:
        """Check if execution exceeds slippage threshold."""
        slippage = None

        if self._primary_benchmark == ExecutionBenchmark.ARRIVAL:
            slippage = record.arrival_slippage_bps
        elif self._primary_benchmark == ExecutionBenchmark.VWAP:
            slippage = record.vwap_slippage_bps

        if slippage is not None and slippage > self._slippage_alert_bps:
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "slippage_threshold_exceeded",
                "severity": "warning",
                "execution_id": record.execution_id,
                "symbol": record.symbol,
                "slippage_bps": slippage,
                "threshold_bps": self._slippage_alert_bps,
            }
            self._alerts.append(alert)

            logger.warning(
                f"Slippage alert: {record.symbol} {slippage:.1f}bps > {self._slippage_alert_bps}bps threshold"
            )

    def calculate_stats(
        self,
        period: str | None = None,
        symbol: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None
    ) -> ExecutionStats:
        """
        Calculate execution statistics.

        Args:
            period: Period label (e.g., "2025-Q1")
            symbol: Filter by symbol (None for all)
            start_date: Start of period
            end_date: End of period

        Returns:
            ExecutionStats for the period
        """
        # Filter executions
        filtered = self._executions

        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]

        if start_date:
            filtered = [e for e in filtered if e.timestamp.date() >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.timestamp.date() <= end_date]

        if not filtered:
            return ExecutionStats(period=period or "N/A", symbol=symbol)

        # Calculate stats
        total_executions = len(filtered)
        total_volume = sum(e.quantity for e in filtered)
        total_notional = sum(e.quantity * e.fill_price for e in filtered)
        total_commission = sum(e.commission for e in filtered)

        # Slippage calculations
        arrival_slippages = [e.arrival_slippage_bps for e in filtered if e.arrival_slippage_bps is not None]
        vwap_slippages = [e.vwap_slippage_bps for e in filtered if e.vwap_slippage_bps is not None]

        avg_arrival_slip = np.mean(arrival_slippages) if arrival_slippages else 0
        avg_vwap_slip = np.mean(vwap_slippages) if vwap_slippages else 0

        all_slippages = arrival_slippages + vwap_slippages
        max_slip = max(all_slippages) if all_slippages else 0
        min_slip = min(all_slippages) if all_slippages else 0
        std_slip = np.std(all_slippages) if all_slippages else 0

        # Implementation shortfall
        is_values = [e.implementation_shortfall for e in filtered if e.implementation_shortfall is not None]
        total_is = sum(is_values) if is_values else 0
        avg_is = np.mean(is_values) if is_values else 0

        # Latency
        latencies = [e.execution_latency_ms for e in filtered if e.execution_latency_ms is not None]
        avg_latency = np.mean(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0

        # By venue
        volume_by_venue: dict[str, int] = defaultdict(int)
        for e in filtered:
            if e.venue:
                volume_by_venue[e.venue] += e.quantity

        # By algo
        volume_by_algo: dict[str, int] = defaultdict(int)
        for e in filtered:
            if e.algo_used:
                volume_by_algo[e.algo_used] += e.quantity

        return ExecutionStats(
            period=period or "custom",
            symbol=symbol,
            total_executions=total_executions,
            total_volume=total_volume,
            total_notional=total_notional,
            total_commission=total_commission,
            avg_arrival_slippage_bps=avg_arrival_slip,
            avg_vwap_slippage_bps=avg_vwap_slip,
            max_slippage_bps=max_slip,
            min_slippage_bps=min_slip,
            std_slippage_bps=std_slip,
            total_is=total_is,
            avg_is_per_trade=avg_is,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            volume_by_venue=dict(volume_by_venue),
            volume_by_algo=dict(volume_by_algo),
        )

    def generate_quarterly_report(
        self,
        year: int,
        quarter: int
    ) -> dict[str, Any]:
        """
        Generate RTS 27/28 compliant quarterly report.

        Args:
            year: Report year
            quarter: Quarter (1-4)

        Returns:
            Report dictionary
        """
        # Calculate quarter dates
        quarter_starts = {
            1: date(year, 1, 1),
            2: date(year, 4, 1),
            3: date(year, 7, 1),
            4: date(year, 10, 1),
        }
        quarter_ends = {
            1: date(year, 3, 31),
            2: date(year, 6, 30),
            3: date(year, 9, 30),
            4: date(year, 12, 31),
        }

        start_date = quarter_starts[quarter]
        end_date = quarter_ends[quarter]
        period_label = f"{year}-Q{quarter}"

        # Overall stats
        overall_stats = self.calculate_stats(
            period=period_label,
            start_date=start_date,
            end_date=end_date
        )

        # Stats by symbol
        symbols = set(e.symbol for e in self._executions)
        symbol_stats = {}
        for symbol in symbols:
            stats = self.calculate_stats(
                period=period_label,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if stats.total_executions > 0:
                symbol_stats[symbol] = stats.to_dict()

        # Top 5 venues by volume
        top_venues = sorted(
            overall_stats.volume_by_venue.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Execution quality summary
        quality_rating = "Good"
        if overall_stats.avg_arrival_slippage_bps > 20:
            quality_rating = "Needs Improvement"
        if overall_stats.avg_arrival_slippage_bps > 50:
            quality_rating = "Poor"

        report = {
            "report_type": "RTS_27_28_Best_Execution",
            "period": period_label,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "firm_identifier": "AI_TRADING_FIRM",  # Should come from config

            "summary": {
                "total_executions": overall_stats.total_executions,
                "total_volume": overall_stats.total_volume,
                "total_notional": overall_stats.total_notional,
                "total_commission": overall_stats.total_commission,
                "execution_quality_rating": quality_rating,
            },

            "execution_quality": {
                "primary_benchmark": self._primary_benchmark.value,
                "avg_arrival_slippage_bps": overall_stats.avg_arrival_slippage_bps,
                "avg_vwap_slippage_bps": overall_stats.avg_vwap_slippage_bps,
                "max_slippage_bps": overall_stats.max_slippage_bps,
                "slippage_std_bps": overall_stats.std_slippage_bps,
                "total_implementation_shortfall": overall_stats.total_is,
            },

            "venue_analysis": {
                "top_venues": [
                    {"venue": v, "volume": vol, "pct": vol / overall_stats.total_volume * 100 if overall_stats.total_volume > 0 else 0}
                    for v, vol in top_venues
                ],
                "venue_count": len(overall_stats.volume_by_venue),
            },

            "algorithm_analysis": {
                "algorithms_used": list(overall_stats.volume_by_algo.keys()),
                "volume_by_algo": overall_stats.volume_by_algo,
            },

            "latency_metrics": {
                "avg_latency_ms": overall_stats.avg_latency_ms,
                "max_latency_ms": overall_stats.max_latency_ms,
            },

            "by_symbol": symbol_stats,

            "compliance_statement": (
                f"This report is generated in accordance with MiFID II RTS 27/28 requirements. "
                f"Data covers {overall_stats.total_executions} executions in {period_label}."
            ),
        }

        return report

    def get_slippage_distribution(
        self,
        benchmark: ExecutionBenchmark = ExecutionBenchmark.ARRIVAL,
        symbol: str | None = None
    ) -> dict[str, Any]:
        """
        Get slippage distribution statistics.

        Args:
            benchmark: Benchmark to use
            symbol: Filter by symbol

        Returns:
            Distribution statistics
        """
        filtered = self._executions
        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]

        if benchmark == ExecutionBenchmark.ARRIVAL:
            slippages = [e.arrival_slippage_bps for e in filtered if e.arrival_slippage_bps is not None]
        else:
            slippages = [e.vwap_slippage_bps for e in filtered if e.vwap_slippage_bps is not None]

        if not slippages:
            return {"error": "No slippage data available"}

        slippages = np.array(slippages)

        return {
            "benchmark": benchmark.value,
            "symbol": symbol,
            "count": len(slippages),
            "mean": float(np.mean(slippages)),
            "median": float(np.median(slippages)),
            "std": float(np.std(slippages)),
            "min": float(np.min(slippages)),
            "max": float(np.max(slippages)),
            "percentile_25": float(np.percentile(slippages, 25)),
            "percentile_75": float(np.percentile(slippages, 75)),
            "percentile_95": float(np.percentile(slippages, 95)),
            "pct_negative": float(np.sum(slippages < 0) / len(slippages) * 100),  # Price improvement
            "pct_within_10bps": float(np.sum(np.abs(slippages) <= 10) / len(slippages) * 100),
        }

    def get_recent_executions(
        self,
        limit: int = 100,
        symbol: str | None = None
    ) -> list[ExecutionRecord]:
        """Get recent executions."""
        filtered = self._executions
        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]
        return filtered[-limit:]

    def get_alerts(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent alerts."""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        return [
            a for a in self._alerts
            if datetime.fromisoformat(a["timestamp"]).timestamp() > cutoff
        ]

    def compare_algos(
        self,
        start_date: date | None = None,
        end_date: date | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Compare execution quality across algorithms.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            Comparison by algorithm
        """
        filtered = self._executions

        if start_date:
            filtered = [e for e in filtered if e.timestamp.date() >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.timestamp.date() <= end_date]

        # Group by algo
        by_algo: dict[str, list[ExecutionRecord]] = defaultdict(list)
        for e in filtered:
            algo = e.algo_used or "UNKNOWN"
            by_algo[algo].append(e)

        results = {}
        for algo, execs in by_algo.items():
            slippages = [e.arrival_slippage_bps for e in execs if e.arrival_slippage_bps is not None]

            results[algo] = {
                "executions": len(execs),
                "total_volume": sum(e.quantity for e in execs),
                "avg_slippage_bps": np.mean(slippages) if slippages else None,
                "std_slippage_bps": np.std(slippages) if slippages else None,
                "pct_price_improvement": sum(1 for s in slippages if s < 0) / len(slippages) * 100 if slippages else 0,
            }

        return results

    def get_status(self) -> dict[str, Any]:
        """Get analyzer status for monitoring."""
        return {
            "total_executions_recorded": len(self._executions),
            "primary_benchmark": self._primary_benchmark.value,
            "slippage_alert_threshold_bps": self._slippage_alert_bps,
            "recent_alerts": len(self.get_alerts(24)),
            "algos_tracked": len(set(e.algo_used for e in self._executions if e.algo_used)),
            "venues_tracked": len(set(e.venue for e in self._executions if e.venue)),
        }

"""
Dashboard Metrics Module
========================

Addresses issues:
- #R26: Risk dashboard metrics incomplete
- #C41: Compliance dashboard metrics incomplete
- #E31: Execution statistics dashboard incomplete
- #O17: Option analytics dashboard incomplete

Features:
- Comprehensive real-time metrics for dashboards
- Risk metrics with alerts and thresholds
- Compliance status tracking
- Execution quality metrics
- Options analytics summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricStatus(str, Enum):
    """Metric status indicators."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DashboardAlert:
    """Alert for dashboard display."""
    level: AlertLevel
    category: str
    message: str
    timestamp: datetime
    metric_name: str | None = None
    current_value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
        }


# =========================================================================
# RISK DASHBOARD METRICS (#R26)
# =========================================================================

@dataclass
class RiskDashboardMetrics:
    """
    Comprehensive risk metrics for dashboard display (#R26).

    Provides real-time risk monitoring with thresholds and alerts.
    """
    # Portfolio Risk
    portfolio_var_95: float = 0.0
    portfolio_var_99: float = 0.0
    portfolio_cvar_95: float = 0.0
    portfolio_volatility: float = 0.0

    # Exposure
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    leverage_ratio: float = 0.0

    # Drawdown
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0

    # P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    mtd_pnl: float = 0.0
    ytd_pnl: float = 0.0

    # Greeks (if options)
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0

    # Concentration
    top_position_weight: float = 0.0
    top_5_weight: float = 0.0
    hhi_concentration: float = 0.0
    sector_concentration: dict[str, float] = field(default_factory=dict)

    # Limits
    limit_utilization: dict[str, float] = field(default_factory=dict)

    # Stress Test Summary
    worst_case_loss: float = 0.0
    worst_case_scenario: str = ""

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_staleness_seconds: float = 0.0

    # Thresholds for alerts
    var_warning_threshold: float = 0.02  # 2% VaR warning
    var_critical_threshold: float = 0.03  # 3% VaR critical
    leverage_warning: float = 2.0
    leverage_critical: float = 3.0
    drawdown_warning: float = 0.05  # 5%
    drawdown_critical: float = 0.10  # 10%
    concentration_warning: float = 0.15  # 15% single position

    def get_alerts(self) -> list[DashboardAlert]:
        """Generate alerts based on current metrics."""
        alerts = []
        now = datetime.now(timezone.utc)

        # VaR alerts
        if self.portfolio_var_95 > self.var_critical_threshold:
            alerts.append(DashboardAlert(
                level=AlertLevel.CRITICAL,
                category="risk",
                message=f"VaR (95%) at {self.portfolio_var_95*100:.2f}% exceeds critical threshold",
                timestamp=now,
                metric_name="portfolio_var_95",
                current_value=self.portfolio_var_95,
                threshold=self.var_critical_threshold,
            ))
        elif self.portfolio_var_95 > self.var_warning_threshold:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="risk",
                message=f"VaR (95%) at {self.portfolio_var_95*100:.2f}% exceeds warning threshold",
                timestamp=now,
                metric_name="portfolio_var_95",
                current_value=self.portfolio_var_95,
                threshold=self.var_warning_threshold,
            ))

        # Leverage alerts
        if self.leverage_ratio > self.leverage_critical:
            alerts.append(DashboardAlert(
                level=AlertLevel.CRITICAL,
                category="risk",
                message=f"Leverage at {self.leverage_ratio:.2f}x exceeds critical limit",
                timestamp=now,
                metric_name="leverage_ratio",
                current_value=self.leverage_ratio,
                threshold=self.leverage_critical,
            ))
        elif self.leverage_ratio > self.leverage_warning:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="risk",
                message=f"Leverage at {self.leverage_ratio:.2f}x exceeds warning limit",
                timestamp=now,
                metric_name="leverage_ratio",
                current_value=self.leverage_ratio,
                threshold=self.leverage_warning,
            ))

        # Drawdown alerts
        if self.current_drawdown > self.drawdown_critical:
            alerts.append(DashboardAlert(
                level=AlertLevel.CRITICAL,
                category="risk",
                message=f"Drawdown at {self.current_drawdown*100:.2f}% exceeds critical limit",
                timestamp=now,
                metric_name="current_drawdown",
                current_value=self.current_drawdown,
                threshold=self.drawdown_critical,
            ))
        elif self.current_drawdown > self.drawdown_warning:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="risk",
                message=f"Drawdown at {self.current_drawdown*100:.2f}% exceeds warning limit",
                timestamp=now,
                metric_name="current_drawdown",
                current_value=self.current_drawdown,
                threshold=self.drawdown_warning,
            ))

        # Concentration alerts
        if self.top_position_weight > self.concentration_warning:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="concentration",
                message=f"Top position concentration at {self.top_position_weight*100:.1f}%",
                timestamp=now,
                metric_name="top_position_weight",
                current_value=self.top_position_weight,
                threshold=self.concentration_warning,
            ))

        # Data staleness alert
        if self.data_staleness_seconds > 60:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="data",
                message=f"Risk data is {self.data_staleness_seconds:.0f}s stale",
                timestamp=now,
                metric_name="data_staleness_seconds",
                current_value=self.data_staleness_seconds,
                threshold=60.0,
            ))

        return alerts

    def get_status(self, metric_name: str) -> MetricStatus:
        """Get status indicator for a specific metric."""
        metric_value = getattr(self, metric_name, None)
        if metric_value is None:
            return MetricStatus.NORMAL

        if metric_name == "portfolio_var_95":
            if metric_value > self.var_critical_threshold:
                return MetricStatus.CRITICAL
            elif metric_value > self.var_warning_threshold:
                return MetricStatus.HIGH
            elif metric_value > self.var_warning_threshold * 0.8:
                return MetricStatus.ELEVATED
        elif metric_name == "leverage_ratio":
            if metric_value > self.leverage_critical:
                return MetricStatus.CRITICAL
            elif metric_value > self.leverage_warning:
                return MetricStatus.HIGH
        elif metric_name == "current_drawdown":
            if metric_value > self.drawdown_critical:
                return MetricStatus.CRITICAL
            elif metric_value > self.drawdown_warning:
                return MetricStatus.HIGH

        return MetricStatus.NORMAL

    def to_dict(self) -> dict:
        """Convert to dictionary for API/JSON."""
        return {
            "portfolio_risk": {
                "var_95_pct": self.portfolio_var_95 * 100,
                "var_99_pct": self.portfolio_var_99 * 100,
                "cvar_95_pct": self.portfolio_cvar_95 * 100,
                "volatility_pct": self.portfolio_volatility * 100,
                "var_status": self.get_status("portfolio_var_95").value,
            },
            "exposure": {
                "gross": self.gross_exposure,
                "net": self.net_exposure,
                "long": self.long_exposure,
                "short": self.short_exposure,
                "leverage_ratio": self.leverage_ratio,
                "leverage_status": self.get_status("leverage_ratio").value,
            },
            "drawdown": {
                "current_pct": self.current_drawdown * 100,
                "max_pct": self.max_drawdown * 100,
                "duration_days": self.drawdown_duration_days,
                "status": self.get_status("current_drawdown").value,
            },
            "pnl": {
                "daily": self.daily_pnl,
                "daily_pct": self.daily_pnl_pct * 100,
                "mtd": self.mtd_pnl,
                "ytd": self.ytd_pnl,
            },
            "greeks": {
                "delta": self.portfolio_delta,
                "gamma": self.portfolio_gamma,
                "vega": self.portfolio_vega,
                "theta": self.portfolio_theta,
            },
            "concentration": {
                "top_position_pct": self.top_position_weight * 100,
                "top_5_pct": self.top_5_weight * 100,
                "hhi": self.hhi_concentration,
                "by_sector": {k: v * 100 for k, v in self.sector_concentration.items()},
            },
            "limits": self.limit_utilization,
            "stress": {
                "worst_case_loss": self.worst_case_loss,
                "worst_case_scenario": self.worst_case_scenario,
            },
            "metadata": {
                "last_updated": self.last_updated.isoformat(),
                "staleness_seconds": self.data_staleness_seconds,
            },
            "alerts": [a.to_dict() for a in self.get_alerts()],
        }


# =========================================================================
# COMPLIANCE DASHBOARD METRICS (#C41)
# =========================================================================

@dataclass
class ComplianceDashboardMetrics:
    """
    Compliance metrics for dashboard display (#C41).

    Tracks regulatory compliance status and upcoming deadlines.
    """
    # Overall Status
    compliance_score: float = 100.0  # 0-100
    compliance_status: str = "compliant"  # compliant, warning, breach

    # Reporting Status
    reports_due_today: int = 0
    reports_overdue: int = 0
    reports_submitted_today: int = 0

    # Transaction Reporting
    emir_pending_reports: int = 0
    emir_failed_reports: int = 0
    mifid_pending_records: int = 0

    # Position Limits
    position_limit_breaches: int = 0
    position_limit_warnings: int = 0
    concentration_warnings: int = 0

    # Market Abuse
    surveillance_alerts_today: int = 0
    surveillance_alerts_pending: int = 0
    stor_reports_submitted: int = 0

    # Best Execution
    best_execution_compliance_pct: float = 100.0
    execution_quality_score: float = 0.0

    # Training & Certification
    training_overdue: int = 0
    certifications_expiring: int = 0

    # Audit
    last_audit_date: datetime | None = None
    audit_findings_open: int = 0
    days_since_last_audit: int = 0

    # Kill Switch
    kill_switch_status: str = "inactive"
    kill_switch_tests_due: int = 0

    # Upcoming Deadlines
    upcoming_deadlines: list[dict] = field(default_factory=list)

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_alerts(self) -> list[DashboardAlert]:
        """Generate compliance alerts."""
        alerts = []
        now = datetime.now(timezone.utc)

        # Overdue reports
        if self.reports_overdue > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.CRITICAL,
                category="compliance",
                message=f"{self.reports_overdue} regulatory reports are overdue",
                timestamp=now,
                metric_name="reports_overdue",
                current_value=self.reports_overdue,
            ))

        # EMIR failures
        if self.emir_failed_reports > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.CRITICAL,
                category="compliance",
                message=f"{self.emir_failed_reports} EMIR reports failed submission",
                timestamp=now,
                metric_name="emir_failed_reports",
                current_value=self.emir_failed_reports,
            ))

        # Position breaches
        if self.position_limit_breaches > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.CRITICAL,
                category="compliance",
                message=f"{self.position_limit_breaches} position limit breaches",
                timestamp=now,
                metric_name="position_limit_breaches",
                current_value=self.position_limit_breaches,
            ))

        # Surveillance alerts
        if self.surveillance_alerts_pending > 10:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="compliance",
                message=f"{self.surveillance_alerts_pending} surveillance alerts pending review",
                timestamp=now,
                metric_name="surveillance_alerts_pending",
                current_value=self.surveillance_alerts_pending,
            ))

        # Training overdue
        if self.training_overdue > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="compliance",
                message=f"{self.training_overdue} staff have overdue compliance training",
                timestamp=now,
                metric_name="training_overdue",
                current_value=self.training_overdue,
            ))

        # Audit findings
        if self.audit_findings_open > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="compliance",
                message=f"{self.audit_findings_open} audit findings remain open",
                timestamp=now,
                metric_name="audit_findings_open",
                current_value=self.audit_findings_open,
            ))

        return alerts

    def calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        score = 100.0

        # Deductions for issues
        score -= self.reports_overdue * 10
        score -= self.emir_failed_reports * 5
        score -= self.position_limit_breaches * 15
        score -= self.position_limit_warnings * 2
        score -= self.surveillance_alerts_pending * 1
        score -= self.training_overdue * 3
        score -= self.audit_findings_open * 5

        # Best execution impact
        if self.best_execution_compliance_pct < 95:
            score -= (95 - self.best_execution_compliance_pct)

        return max(0, min(100, score))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overall": {
                "score": self.calculate_compliance_score(),
                "status": self.compliance_status,
            },
            "reporting": {
                "due_today": self.reports_due_today,
                "overdue": self.reports_overdue,
                "submitted_today": self.reports_submitted_today,
            },
            "transaction_reporting": {
                "emir_pending": self.emir_pending_reports,
                "emir_failed": self.emir_failed_reports,
                "mifid_pending": self.mifid_pending_records,
            },
            "position_limits": {
                "breaches": self.position_limit_breaches,
                "warnings": self.position_limit_warnings,
                "concentration_warnings": self.concentration_warnings,
            },
            "surveillance": {
                "alerts_today": self.surveillance_alerts_today,
                "alerts_pending": self.surveillance_alerts_pending,
                "stor_submitted": self.stor_reports_submitted,
            },
            "best_execution": {
                "compliance_pct": self.best_execution_compliance_pct,
                "quality_score": self.execution_quality_score,
            },
            "training": {
                "overdue": self.training_overdue,
                "certifications_expiring": self.certifications_expiring,
            },
            "audit": {
                "last_date": self.last_audit_date.isoformat() if self.last_audit_date else None,
                "findings_open": self.audit_findings_open,
                "days_since": self.days_since_last_audit,
            },
            "kill_switch": {
                "status": self.kill_switch_status,
                "tests_due": self.kill_switch_tests_due,
            },
            "deadlines": self.upcoming_deadlines,
            "alerts": [a.to_dict() for a in self.get_alerts()],
        }


# =========================================================================
# EXECUTION DASHBOARD METRICS (#E31)
# =========================================================================

@dataclass
class ExecutionDashboardMetrics:
    """
    Execution statistics for dashboard display (#E31).

    Tracks execution quality and order flow metrics.
    """
    # Order Statistics
    orders_today: int = 0
    orders_filled: int = 0
    orders_pending: int = 0
    orders_cancelled: int = 0
    orders_rejected: int = 0

    # Fill Rates
    fill_rate_pct: float = 0.0
    partial_fill_rate_pct: float = 0.0

    # Execution Quality
    avg_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    price_improvement_rate: float = 0.0
    avg_price_improvement_bps: float = 0.0

    # Implementation Shortfall
    total_implementation_shortfall: float = 0.0
    avg_shortfall_bps: float = 0.0

    # Timing
    avg_fill_time_ms: float = 0.0
    avg_order_latency_ms: float = 0.0
    p95_fill_time_ms: float = 0.0

    # Volume
    total_volume_traded: float = 0.0
    total_notional: float = 0.0

    # Venue Statistics
    venue_fill_rates: dict[str, float] = field(default_factory=dict)
    venue_latencies: dict[str, float] = field(default_factory=dict)

    # Algorithm Performance
    twap_orders: int = 0
    vwap_orders: int = 0
    market_orders: int = 0
    limit_orders: int = 0

    # Cost Analysis
    total_commission: float = 0.0
    total_slippage_cost: float = 0.0
    total_market_impact: float = 0.0

    # Benchmarks
    vs_vwap_bps: float = 0.0
    vs_twap_bps: float = 0.0
    vs_arrival_bps: float = 0.0

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    measurement_period_start: datetime | None = None

    def get_alerts(self) -> list[DashboardAlert]:
        """Generate execution alerts."""
        alerts = []
        now = datetime.now(timezone.utc)

        # High rejection rate
        if self.orders_today > 0:
            rejection_rate = self.orders_rejected / self.orders_today
            if rejection_rate > 0.05:  # >5% rejection
                alerts.append(DashboardAlert(
                    level=AlertLevel.WARNING,
                    category="execution",
                    message=f"High order rejection rate: {rejection_rate*100:.1f}%",
                    timestamp=now,
                    metric_name="rejection_rate",
                    current_value=rejection_rate,
                    threshold=0.05,
                ))

        # High slippage
        if self.avg_slippage_bps > 10:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="execution",
                message=f"High average slippage: {self.avg_slippage_bps:.1f} bps",
                timestamp=now,
                metric_name="avg_slippage_bps",
                current_value=self.avg_slippage_bps,
                threshold=10.0,
            ))

        # Slow fills
        if self.avg_fill_time_ms > 500:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="execution",
                message=f"Slow average fill time: {self.avg_fill_time_ms:.0f}ms",
                timestamp=now,
                metric_name="avg_fill_time_ms",
                current_value=self.avg_fill_time_ms,
                threshold=500.0,
            ))

        # Pending orders
        if self.orders_pending > 10:
            alerts.append(DashboardAlert(
                level=AlertLevel.INFO,
                category="execution",
                message=f"{self.orders_pending} orders pending execution",
                timestamp=now,
                metric_name="orders_pending",
                current_value=self.orders_pending,
            ))

        return alerts

    def calculate_execution_quality_score(self) -> float:
        """Calculate overall execution quality score (0-100)."""
        score = 100.0

        # Penalize slippage
        score -= min(20, self.avg_slippage_bps * 2)

        # Penalize rejections
        if self.orders_today > 0:
            rejection_rate = self.orders_rejected / self.orders_today
            score -= rejection_rate * 50

        # Penalize slow fills
        if self.avg_fill_time_ms > 100:
            score -= min(10, (self.avg_fill_time_ms - 100) / 100)

        # Bonus for price improvement
        score += min(10, self.price_improvement_rate * 20)

        return max(0, min(100, score))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "orders": {
                "total": self.orders_today,
                "filled": self.orders_filled,
                "pending": self.orders_pending,
                "cancelled": self.orders_cancelled,
                "rejected": self.orders_rejected,
            },
            "fill_rates": {
                "fill_rate_pct": self.fill_rate_pct,
                "partial_fill_pct": self.partial_fill_rate_pct,
            },
            "quality": {
                "score": self.calculate_execution_quality_score(),
                "avg_slippage_bps": self.avg_slippage_bps,
                "max_slippage_bps": self.max_slippage_bps,
                "price_improvement_rate": self.price_improvement_rate,
                "avg_improvement_bps": self.avg_price_improvement_bps,
            },
            "implementation_shortfall": {
                "total": self.total_implementation_shortfall,
                "avg_bps": self.avg_shortfall_bps,
            },
            "timing": {
                "avg_fill_ms": self.avg_fill_time_ms,
                "avg_latency_ms": self.avg_order_latency_ms,
                "p95_fill_ms": self.p95_fill_time_ms,
            },
            "volume": {
                "shares": self.total_volume_traded,
                "notional": self.total_notional,
            },
            "venues": {
                "fill_rates": self.venue_fill_rates,
                "latencies": self.venue_latencies,
            },
            "order_types": {
                "twap": self.twap_orders,
                "vwap": self.vwap_orders,
                "market": self.market_orders,
                "limit": self.limit_orders,
            },
            "costs": {
                "commission": self.total_commission,
                "slippage": self.total_slippage_cost,
                "market_impact": self.total_market_impact,
                "total": self.total_commission + self.total_slippage_cost + self.total_market_impact,
            },
            "benchmarks": {
                "vs_vwap_bps": self.vs_vwap_bps,
                "vs_twap_bps": self.vs_twap_bps,
                "vs_arrival_bps": self.vs_arrival_bps,
            },
            "alerts": [a.to_dict() for a in self.get_alerts()],
        }


# =========================================================================
# OPTIONS ANALYTICS DASHBOARD (#O17)
# =========================================================================

@dataclass
class OptionsDashboardMetrics:
    """
    Options analytics for dashboard display (#O17).

    Comprehensive options portfolio metrics and risk indicators.
    """
    # Portfolio Greeks
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_rho: float = 0.0

    # Dollar Greeks
    delta_dollars: float = 0.0
    gamma_dollars: float = 0.0  # Per 1% move
    vega_dollars: float = 0.0  # Per 1% vol move
    theta_dollars: float = 0.0  # Per day

    # Position Counts
    total_contracts: int = 0
    long_calls: int = 0
    short_calls: int = 0
    long_puts: int = 0
    short_puts: int = 0

    # Expiration Profile
    contracts_expiring_today: int = 0
    contracts_expiring_this_week: int = 0
    contracts_expiring_this_month: int = 0
    weighted_avg_dte: float = 0.0

    # Volatility Analysis
    portfolio_weighted_iv: float = 0.0
    iv_vs_hv_ratio: float = 0.0
    avg_iv_rank: float = 0.0
    avg_iv_percentile: float = 0.0

    # P&L Attribution
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0
    total_pnl: float = 0.0

    # Risk Metrics
    max_loss: float = 0.0
    max_profit: float = 0.0
    breakeven_prices: list[float] = field(default_factory=list)

    # Pin Risk
    positions_with_pin_risk: int = 0
    total_pin_risk_exposure: float = 0.0

    # Assignment Risk
    positions_with_assignment_risk: int = 0
    total_assignment_risk_exposure: float = 0.0

    # Strategy Breakdown
    strategies: dict[str, int] = field(default_factory=dict)

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    underlying_price: float = 0.0

    def get_alerts(self) -> list[DashboardAlert]:
        """Generate options-specific alerts."""
        alerts = []
        now = datetime.now(timezone.utc)

        # Expiring contracts
        if self.contracts_expiring_today > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="options",
                message=f"{self.contracts_expiring_today} contracts expiring today",
                timestamp=now,
                metric_name="contracts_expiring_today",
                current_value=self.contracts_expiring_today,
            ))

        # High gamma
        if abs(self.gamma_dollars) > 10000:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="options",
                message=f"High gamma exposure: ${self.gamma_dollars:,.0f} per 1% move",
                timestamp=now,
                metric_name="gamma_dollars",
                current_value=self.gamma_dollars,
            ))

        # Pin risk
        if self.positions_with_pin_risk > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="options",
                message=f"{self.positions_with_pin_risk} positions with pin risk near expiry",
                timestamp=now,
                metric_name="positions_with_pin_risk",
                current_value=self.positions_with_pin_risk,
            ))

        # Assignment risk
        if self.positions_with_assignment_risk > 0:
            alerts.append(DashboardAlert(
                level=AlertLevel.INFO,
                category="options",
                message=f"{self.positions_with_assignment_risk} positions with assignment risk",
                timestamp=now,
                metric_name="positions_with_assignment_risk",
                current_value=self.positions_with_assignment_risk,
            ))

        # IV vs HV
        if self.iv_vs_hv_ratio > 1.5:
            alerts.append(DashboardAlert(
                level=AlertLevel.INFO,
                category="options",
                message=f"IV significantly above HV (ratio: {self.iv_vs_hv_ratio:.2f})",
                timestamp=now,
                metric_name="iv_vs_hv_ratio",
                current_value=self.iv_vs_hv_ratio,
            ))

        return alerts

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "greeks": {
                "delta": self.portfolio_delta,
                "gamma": self.portfolio_gamma,
                "vega": self.portfolio_vega,
                "theta": self.portfolio_theta,
                "rho": self.portfolio_rho,
            },
            "dollar_greeks": {
                "delta": self.delta_dollars,
                "gamma_per_pct": self.gamma_dollars,
                "vega_per_pct": self.vega_dollars,
                "theta_per_day": self.theta_dollars,
            },
            "positions": {
                "total_contracts": self.total_contracts,
                "long_calls": self.long_calls,
                "short_calls": self.short_calls,
                "long_puts": self.long_puts,
                "short_puts": self.short_puts,
            },
            "expiration": {
                "today": self.contracts_expiring_today,
                "this_week": self.contracts_expiring_this_week,
                "this_month": self.contracts_expiring_this_month,
                "weighted_avg_dte": self.weighted_avg_dte,
            },
            "volatility": {
                "portfolio_iv": self.portfolio_weighted_iv * 100,
                "iv_hv_ratio": self.iv_vs_hv_ratio,
                "avg_iv_rank": self.avg_iv_rank * 100,
                "avg_iv_percentile": self.avg_iv_percentile * 100,
            },
            "pnl_attribution": {
                "delta": self.delta_pnl,
                "gamma": self.gamma_pnl,
                "theta": self.theta_pnl,
                "vega": self.vega_pnl,
                "total": self.total_pnl,
            },
            "risk": {
                "max_loss": self.max_loss,
                "max_profit": self.max_profit,
                "breakevens": self.breakeven_prices,
            },
            "special_risks": {
                "pin_risk_positions": self.positions_with_pin_risk,
                "pin_risk_exposure": self.total_pin_risk_exposure,
                "assignment_risk_positions": self.positions_with_assignment_risk,
                "assignment_risk_exposure": self.total_assignment_risk_exposure,
            },
            "strategies": self.strategies,
            "alerts": [a.to_dict() for a in self.get_alerts()],
        }


# =========================================================================
# REAL-TIME P&L METRICS (P3: Add real-time P&L metrics)
# =========================================================================

@dataclass
class RealTimePnLMetrics:
    """
    Real-time P&L metrics for dashboard display (P3 monitoring improvement).

    Provides granular P&L tracking with attribution and time series.
    """
    # Current P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0

    # Daily P&L
    daily_unrealized_pnl: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_total_pnl: float = 0.0
    daily_pnl_pct: float = 0.0

    # P&L by time period
    hourly_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    ytd_pnl: float = 0.0

    # P&L attribution by source
    pnl_by_strategy: dict[str, float] = field(default_factory=dict)
    pnl_by_asset_class: dict[str, float] = field(default_factory=dict)
    pnl_by_symbol: dict[str, float] = field(default_factory=dict)

    # P&L time series (last N data points for charting)
    pnl_history: list[dict] = field(default_factory=list)

    # Benchmarks
    vs_benchmark_pnl: float = 0.0
    benchmark_name: str = "SPY"

    # High/Low tracking
    daily_high_pnl: float = 0.0
    daily_low_pnl: float = 0.0
    intraday_swing: float = 0.0

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trading_day_start: datetime | None = None

    def record_pnl_point(self, pnl: float, timestamp: datetime | None = None) -> None:
        """Record a P&L data point for time series."""
        ts = timestamp or datetime.now(timezone.utc)
        self.pnl_history.append({
            "timestamp": ts.isoformat(),
            "pnl": pnl,
        })
        # Keep last 500 points
        if len(self.pnl_history) > 500:
            self.pnl_history = self.pnl_history[-500:]

        # Update high/low
        if pnl > self.daily_high_pnl:
            self.daily_high_pnl = pnl
        if pnl < self.daily_low_pnl:
            self.daily_low_pnl = pnl
        self.intraday_swing = self.daily_high_pnl - self.daily_low_pnl

    def get_alerts(self) -> list[DashboardAlert]:
        """Generate P&L-specific alerts."""
        alerts = []
        now = datetime.now(timezone.utc)

        # Large daily loss
        if self.daily_pnl_pct < -0.02:  # -2%
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING if self.daily_pnl_pct > -0.03 else AlertLevel.CRITICAL,
                category="pnl",
                message=f"Daily P&L at {self.daily_pnl_pct*100:.2f}%",
                timestamp=now,
                metric_name="daily_pnl_pct",
                current_value=self.daily_pnl_pct,
                threshold=-0.02,
            ))

        # Large intraday swing
        if self.intraday_swing > abs(self.total_pnl) * 0.5 and self.intraday_swing > 1000:
            alerts.append(DashboardAlert(
                level=AlertLevel.INFO,
                category="pnl",
                message=f"Large intraday P&L swing: ${self.intraday_swing:,.0f}",
                timestamp=now,
                metric_name="intraday_swing",
                current_value=self.intraday_swing,
            ))

        return alerts

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current": {
                "unrealized": self.unrealized_pnl,
                "realized": self.realized_pnl,
                "total": self.total_pnl,
            },
            "daily": {
                "unrealized": self.daily_unrealized_pnl,
                "realized": self.daily_realized_pnl,
                "total": self.daily_total_pnl,
                "pct": self.daily_pnl_pct * 100,
                "high": self.daily_high_pnl,
                "low": self.daily_low_pnl,
                "swing": self.intraday_swing,
            },
            "periods": {
                "hourly": self.hourly_pnl,
                "weekly": self.weekly_pnl,
                "monthly": self.monthly_pnl,
                "ytd": self.ytd_pnl,
            },
            "attribution": {
                "by_strategy": self.pnl_by_strategy,
                "by_asset_class": self.pnl_by_asset_class,
                "by_symbol": dict(sorted(
                    self.pnl_by_symbol.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]),  # Top 10 by absolute P&L
            },
            "benchmark": {
                "name": self.benchmark_name,
                "vs_benchmark": self.vs_benchmark_pnl,
            },
            "history": self.pnl_history[-50:],  # Last 50 points for chart
            "alerts": [a.to_dict() for a in self.get_alerts()],
            "last_updated": self.last_updated.isoformat(),
        }


# =========================================================================
# POSITION SUMMARY METRICS (P3: Add position summary metrics)
# =========================================================================

@dataclass
class PositionSummaryMetrics:
    """
    Position summary metrics for dashboard display (P3 monitoring improvement).

    Provides overview of all positions with key stats.
    """
    # Position counts
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0

    # Value metrics
    total_market_value: float = 0.0
    long_market_value: float = 0.0
    short_market_value: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    # P&L summary
    total_unrealized_pnl: float = 0.0
    positions_in_profit: int = 0
    positions_in_loss: int = 0
    win_rate: float = 0.0

    # Top positions
    top_positions: list[dict] = field(default_factory=list)
    largest_winner: dict = field(default_factory=dict)
    largest_loser: dict = field(default_factory=dict)

    # Concentration
    largest_position_pct: float = 0.0
    top_5_concentration_pct: float = 0.0

    # By asset class
    positions_by_asset_class: dict[str, int] = field(default_factory=dict)
    exposure_by_asset_class: dict[str, float] = field(default_factory=dict)

    # By sector (for equities)
    positions_by_sector: dict[str, int] = field(default_factory=dict)
    exposure_by_sector: dict[str, float] = field(default_factory=dict)

    # Age metrics
    avg_holding_period_days: float = 0.0
    oldest_position_days: int = 0
    newest_position_days: int = 0

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_alerts(self) -> list[DashboardAlert]:
        """Generate position-specific alerts."""
        alerts = []
        now = datetime.now(timezone.utc)

        # High concentration
        if self.largest_position_pct > 0.15:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="positions",
                message=f"Largest position is {self.largest_position_pct*100:.1f}% of portfolio",
                timestamp=now,
                metric_name="largest_position_pct",
                current_value=self.largest_position_pct,
                threshold=0.15,
            ))

        # Low win rate
        if self.total_positions > 5 and self.win_rate < 0.4:
            alerts.append(DashboardAlert(
                level=AlertLevel.INFO,
                category="positions",
                message=f"Position win rate at {self.win_rate*100:.1f}%",
                timestamp=now,
                metric_name="win_rate",
                current_value=self.win_rate,
                threshold=0.4,
            ))

        return alerts

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "counts": {
                "total": self.total_positions,
                "long": self.long_positions,
                "short": self.short_positions,
                "in_profit": self.positions_in_profit,
                "in_loss": self.positions_in_loss,
            },
            "values": {
                "total_market_value": self.total_market_value,
                "long_value": self.long_market_value,
                "short_value": self.short_market_value,
                "gross_exposure": self.gross_exposure,
                "net_exposure": self.net_exposure,
            },
            "pnl": {
                "total_unrealized": self.total_unrealized_pnl,
                "win_rate_pct": self.win_rate * 100,
            },
            "top_positions": self.top_positions[:10],
            "winners_losers": {
                "largest_winner": self.largest_winner,
                "largest_loser": self.largest_loser,
            },
            "concentration": {
                "largest_pct": self.largest_position_pct * 100,
                "top_5_pct": self.top_5_concentration_pct * 100,
            },
            "by_asset_class": {
                "positions": self.positions_by_asset_class,
                "exposure": self.exposure_by_asset_class,
            },
            "by_sector": {
                "positions": self.positions_by_sector,
                "exposure": self.exposure_by_sector,
            },
            "holding_period": {
                "avg_days": self.avg_holding_period_days,
                "oldest_days": self.oldest_position_days,
                "newest_days": self.newest_position_days,
            },
            "alerts": [a.to_dict() for a in self.get_alerts()],
            "last_updated": self.last_updated.isoformat(),
        }


# =========================================================================
# RISK EXPOSURE METRICS (P3: Add risk exposure metrics)
# =========================================================================

@dataclass
class RiskExposureMetrics:
    """
    Risk exposure metrics for dashboard display (P3 monitoring improvement).

    Provides detailed risk exposure breakdown.
    """
    # Market exposure
    beta_adjusted_exposure: float = 0.0
    dollar_delta: float = 0.0

    # Factor exposures
    factor_exposures: dict[str, float] = field(default_factory=dict)

    # Geographic exposure
    exposure_by_country: dict[str, float] = field(default_factory=dict)
    exposure_by_region: dict[str, float] = field(default_factory=dict)

    # Currency exposure
    exposure_by_currency: dict[str, float] = field(default_factory=dict)
    fx_hedged_pct: float = 0.0

    # Sector/Industry exposure
    exposure_by_sector: dict[str, float] = field(default_factory=dict)
    exposure_by_industry: dict[str, float] = field(default_factory=dict)

    # Market cap exposure
    large_cap_exposure: float = 0.0
    mid_cap_exposure: float = 0.0
    small_cap_exposure: float = 0.0

    # Style exposures
    value_exposure: float = 0.0
    growth_exposure: float = 0.0
    momentum_exposure: float = 0.0

    # Liquidity
    avg_daily_volume_coverage: float = 0.0  # Days to liquidate at ADV
    illiquid_positions_pct: float = 0.0

    # Correlation
    avg_position_correlation: float = 0.0
    max_correlated_pair: tuple[str, str, float] = ("", "", 0.0)

    # Stress scenarios
    scenario_impacts: dict[str, float] = field(default_factory=dict)

    # Limits utilization
    limit_utilization: dict[str, dict[str, float]] = field(default_factory=dict)

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_alerts(self) -> list[DashboardAlert]:
        """Generate exposure-specific alerts."""
        alerts = []
        now = datetime.now(timezone.utc)

        # High factor exposure
        for factor, exposure in self.factor_exposures.items():
            if abs(exposure) > 0.3:  # >30% factor exposure
                alerts.append(DashboardAlert(
                    level=AlertLevel.WARNING,
                    category="exposure",
                    message=f"High {factor} factor exposure: {exposure*100:.1f}%",
                    timestamp=now,
                    metric_name=f"factor_{factor}",
                    current_value=exposure,
                    threshold=0.3,
                ))

        # High geographic concentration
        for country, exposure in self.exposure_by_country.items():
            if exposure > 0.5:  # >50% in single country
                alerts.append(DashboardAlert(
                    level=AlertLevel.INFO,
                    category="exposure",
                    message=f"High {country} geographic exposure: {exposure*100:.1f}%",
                    timestamp=now,
                    metric_name=f"country_{country}",
                    current_value=exposure,
                    threshold=0.5,
                ))

        # High currency exposure (unhedged)
        for currency, exposure in self.exposure_by_currency.items():
            if currency != "USD" and exposure > 0.2:  # >20% non-USD
                alerts.append(DashboardAlert(
                    level=AlertLevel.INFO,
                    category="exposure",
                    message=f"Unhedged {currency} exposure: {exposure*100:.1f}%",
                    timestamp=now,
                    metric_name=f"currency_{currency}",
                    current_value=exposure,
                    threshold=0.2,
                ))

        # Liquidity concern
        if self.avg_daily_volume_coverage > 5:
            alerts.append(DashboardAlert(
                level=AlertLevel.WARNING,
                category="exposure",
                message=f"Portfolio takes {self.avg_daily_volume_coverage:.1f} days to liquidate at ADV",
                timestamp=now,
                metric_name="liquidity_days",
                current_value=self.avg_daily_volume_coverage,
                threshold=5.0,
            ))

        return alerts

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market": {
                "beta_adjusted": self.beta_adjusted_exposure,
                "dollar_delta": self.dollar_delta,
            },
            "factors": self.factor_exposures,
            "geographic": {
                "by_country": self.exposure_by_country,
                "by_region": self.exposure_by_region,
            },
            "currency": {
                "by_currency": self.exposure_by_currency,
                "hedged_pct": self.fx_hedged_pct * 100,
            },
            "sector": {
                "by_sector": self.exposure_by_sector,
                "by_industry": self.exposure_by_industry,
            },
            "market_cap": {
                "large_cap": self.large_cap_exposure,
                "mid_cap": self.mid_cap_exposure,
                "small_cap": self.small_cap_exposure,
            },
            "style": {
                "value": self.value_exposure,
                "growth": self.growth_exposure,
                "momentum": self.momentum_exposure,
            },
            "liquidity": {
                "days_to_liquidate": self.avg_daily_volume_coverage,
                "illiquid_pct": self.illiquid_positions_pct * 100,
            },
            "correlation": {
                "avg_correlation": self.avg_position_correlation,
                "max_correlated_pair": {
                    "symbol1": self.max_correlated_pair[0],
                    "symbol2": self.max_correlated_pair[1],
                    "correlation": self.max_correlated_pair[2],
                },
            },
            "stress_scenarios": self.scenario_impacts,
            "limits": self.limit_utilization,
            "alerts": [a.to_dict() for a in self.get_alerts()],
            "last_updated": self.last_updated.isoformat(),
        }


class DashboardMetricsCollector:
    """
    Collects and aggregates metrics for all dashboards.

    Provides unified interface for dashboard data.
    """

    def __init__(self):
        self.risk_metrics = RiskDashboardMetrics()
        self.compliance_metrics = ComplianceDashboardMetrics()
        self.execution_metrics = ExecutionDashboardMetrics()
        self.options_metrics = OptionsDashboardMetrics()

        # P3: Additional metrics
        self.pnl_metrics = RealTimePnLMetrics()
        self.position_metrics = PositionSummaryMetrics()
        self.exposure_metrics = RiskExposureMetrics()

    def get_all_alerts(self) -> list[DashboardAlert]:
        """Get all alerts from all dashboards."""
        alerts = []
        alerts.extend(self.risk_metrics.get_alerts())
        alerts.extend(self.compliance_metrics.get_alerts())
        alerts.extend(self.execution_metrics.get_alerts())
        alerts.extend(self.options_metrics.get_alerts())

        # P3: Include new metric alerts
        alerts.extend(self.pnl_metrics.get_alerts())
        alerts.extend(self.position_metrics.get_alerts())
        alerts.extend(self.exposure_metrics.get_alerts())

        # Sort by level (critical first)
        level_order = {
            AlertLevel.EMERGENCY: 0,
            AlertLevel.CRITICAL: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 3,
        }
        return sorted(alerts, key=lambda a: (level_order[a.level], a.timestamp))

    def get_summary(self) -> dict:
        """Get summary for all dashboards."""
        return {
            "risk": self.risk_metrics.to_dict(),
            "compliance": self.compliance_metrics.to_dict(),
            "execution": self.execution_metrics.to_dict(),
            "options": self.options_metrics.to_dict(),
            "pnl": self.pnl_metrics.to_dict(),
            "positions": self.position_metrics.to_dict(),
            "exposure": self.exposure_metrics.to_dict(),
            "all_alerts": [a.to_dict() for a in self.get_all_alerts()],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_quick_summary(self) -> dict:
        """Get a quick summary for dashboard header."""
        return {
            "pnl": {
                "daily_total": self.pnl_metrics.daily_total_pnl,
                "daily_pct": self.pnl_metrics.daily_pnl_pct * 100,
                "unrealized": self.pnl_metrics.unrealized_pnl,
            },
            "positions": {
                "total": self.position_metrics.total_positions,
                "long": self.position_metrics.long_positions,
                "short": self.position_metrics.short_positions,
            },
            "risk": {
                "drawdown_pct": self.risk_metrics.current_drawdown * 100,
                "var_95_pct": self.risk_metrics.portfolio_var_95 * 100,
                "leverage": self.risk_metrics.leverage_ratio,
            },
            "execution": {
                "orders_today": self.execution_metrics.orders_today,
                "fill_rate_pct": self.execution_metrics.fill_rate_pct,
            },
            "compliance": {
                "score": self.compliance_metrics.calculate_compliance_score(),
                "alerts_pending": self.compliance_metrics.surveillance_alerts_pending,
            },
            "alert_counts": {
                "critical": len([a for a in self.get_all_alerts() if a.level == AlertLevel.CRITICAL]),
                "warning": len([a for a in self.get_all_alerts() if a.level == AlertLevel.WARNING]),
                "info": len([a for a in self.get_all_alerts() if a.level == AlertLevel.INFO]),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

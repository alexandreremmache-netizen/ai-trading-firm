"""
Daily Report Generator
======================

Generates comprehensive daily trading reports with performance metrics,
risk analysis, decision summaries, and AI-generated insights.

Features:
- Executive summary with key metrics and highlights
- Performance tracking vs benchmarks
- Position and decision analysis
- Risk metrics and limit utilization
- AI-generated recommendations (with graceful fallback)
- Template-based HTML export
- Historical report storage and comparison
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportStatus(Enum):
    """Report generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class AlertSeverity(Enum):
    """Alert severity level."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ExecutiveSummary:
    """Executive summary section of the daily report."""
    report_date: date
    trading_day_number: int = 0  # Days since inception
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    trades_executed: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    highlights: list[str] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    market_summary: str = ""
    portfolio_value: float = 0.0
    cash_balance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_date": self.report_date.isoformat(),
            "trading_day_number": self.trading_day_number,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "trades_executed": self.trades_executed,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "highlights": self.highlights,
            "alerts": self.alerts,
            "market_summary": self.market_summary,
            "portfolio_value": self.portfolio_value,
            "cash_balance": self.cash_balance,
        }


@dataclass
class PerformanceSection:
    """Performance analysis section of the daily report."""
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    mtd_pnl: float = 0.0
    mtd_return_pct: float = 0.0
    ytd_pnl: float = 0.0
    ytd_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    pnl_by_strategy: dict[str, float] = field(default_factory=dict)
    pnl_by_asset_class: dict[str, float] = field(default_factory=dict)
    pnl_by_symbol: dict[str, float] = field(default_factory=dict)
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "daily_pnl": self.daily_pnl,
            "daily_return_pct": self.daily_return_pct,
            "mtd_pnl": self.mtd_pnl,
            "mtd_return_pct": self.mtd_return_pct,
            "ytd_pnl": self.ytd_pnl,
            "ytd_return_pct": self.ytd_return_pct,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "pnl_by_strategy": self.pnl_by_strategy,
            "pnl_by_asset_class": self.pnl_by_asset_class,
            "pnl_by_symbol": self.pnl_by_symbol,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
        }


@dataclass
class PositionEntry:
    """Single position entry."""
    symbol: str
    quantity: int
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_pct: float
    asset_class: str = ""
    sector: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_cost": self.average_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "weight_pct": self.weight_pct,
            "asset_class": self.asset_class,
            "sector": self.sector,
        }


@dataclass
class PositionsSection:
    """Positions analysis section of the daily report."""
    current_positions: list[PositionEntry] = field(default_factory=list)
    positions_opened: list[dict[str, Any]] = field(default_factory=list)
    positions_closed: list[dict[str, Any]] = field(default_factory=list)
    total_long_exposure: float = 0.0
    total_short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    concentration_top5_pct: float = 0.0
    sector_allocation: dict[str, float] = field(default_factory=dict)
    asset_class_allocation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_positions": [p.to_dict() for p in self.current_positions],
            "positions_opened": self.positions_opened,
            "positions_closed": self.positions_closed,
            "total_long_exposure": self.total_long_exposure,
            "total_short_exposure": self.total_short_exposure,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "concentration_top5_pct": self.concentration_top5_pct,
            "sector_allocation": self.sector_allocation,
            "asset_class_allocation": self.asset_class_allocation,
        }


@dataclass
class DecisionEntry:
    """Single CIO decision entry."""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str
    quantity: int
    rationale: str
    conviction_score: float
    contributing_signals: list[str] = field(default_factory=list)
    outcome: str = ""  # "executed", "rejected_risk", "rejected_compliance", "pending"
    realized_pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "rationale": self.rationale,
            "conviction_score": self.conviction_score,
            "contributing_signals": self.contributing_signals,
            "outcome": self.outcome,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class DecisionsSection:
    """CIO decisions section of the daily report."""
    decisions: list[DecisionEntry] = field(default_factory=list)
    total_decisions: int = 0
    executed_count: int = 0
    rejected_by_risk: int = 0
    rejected_by_compliance: int = 0
    average_conviction: float = 0.0
    decision_accuracy: float = 0.0  # % of decisions that resulted in profit
    top_performing_decision: DecisionEntry | None = None
    worst_performing_decision: DecisionEntry | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "total_decisions": self.total_decisions,
            "executed_count": self.executed_count,
            "rejected_by_risk": self.rejected_by_risk,
            "rejected_by_compliance": self.rejected_by_compliance,
            "average_conviction": self.average_conviction,
            "decision_accuracy": self.decision_accuracy,
            "top_performing_decision": self.top_performing_decision.to_dict() if self.top_performing_decision else None,
            "worst_performing_decision": self.worst_performing_decision.to_dict() if self.worst_performing_decision else None,
        }


@dataclass
class SignalSummary:
    """Summary of signals from a specific agent."""
    agent_name: str
    signals_generated: int = 0
    long_signals: int = 0
    short_signals: int = 0
    flat_signals: int = 0
    average_strength: float = 0.0
    signals_acted_upon: int = 0
    accuracy_rate: float = 0.0
    top_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "signals_generated": self.signals_generated,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
            "flat_signals": self.flat_signals,
            "average_strength": self.average_strength,
            "signals_acted_upon": self.signals_acted_upon,
            "accuracy_rate": self.accuracy_rate,
            "top_symbols": self.top_symbols,
        }


@dataclass
class SignalsSection:
    """Signals analysis section of the daily report."""
    signal_summaries: list[SignalSummary] = field(default_factory=list)
    total_signals: int = 0
    signal_agreement_rate: float = 0.0  # How often agents agree
    most_active_agent: str = ""
    most_accurate_agent: str = ""
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_summaries": [s.to_dict() for s in self.signal_summaries],
            "total_signals": self.total_signals,
            "signal_agreement_rate": self.signal_agreement_rate,
            "most_active_agent": self.most_active_agent,
            "most_accurate_agent": self.most_accurate_agent,
            "correlation_matrix": self.correlation_matrix,
        }


@dataclass
class RiskSection:
    """Risk analysis section of the daily report."""
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    current_leverage: float = 0.0
    max_leverage_limit: float = 2.0
    leverage_utilization_pct: float = 0.0
    position_limit_utilization: dict[str, float] = field(default_factory=dict)
    sector_limit_utilization: dict[str, float] = field(default_factory=dict)
    daily_loss_limit_pct: float = 0.0
    daily_loss_utilization_pct: float = 0.0
    risk_alerts: list[dict[str, Any]] = field(default_factory=list)
    kill_switch_triggered: bool = False
    kill_switch_reason: str = ""
    stress_test_results: dict[str, float] = field(default_factory=dict)
    correlation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "expected_shortfall": self.expected_shortfall,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "current_leverage": self.current_leverage,
            "max_leverage_limit": self.max_leverage_limit,
            "leverage_utilization_pct": self.leverage_utilization_pct,
            "position_limit_utilization": self.position_limit_utilization,
            "sector_limit_utilization": self.sector_limit_utilization,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "daily_loss_utilization_pct": self.daily_loss_utilization_pct,
            "risk_alerts": self.risk_alerts,
            "kill_switch_triggered": self.kill_switch_triggered,
            "kill_switch_reason": self.kill_switch_reason,
            "stress_test_results": self.stress_test_results,
            "correlation_warnings": self.correlation_warnings,
        }


@dataclass
class Recommendation:
    """AI-generated recommendation."""
    category: str  # "risk", "performance", "portfolio", "strategy"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    rationale: str
    suggested_action: str
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "suggested_action": self.suggested_action,
            "confidence": self.confidence,
        }


@dataclass
class RecommendationsSection:
    """AI-generated recommendations section."""
    recommendations: list[Recommendation] = field(default_factory=list)
    generated_by: str = "rule_based"  # "llm" or "rule_based"
    generation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "generated_by": self.generated_by,
            "generation_timestamp": self.generation_timestamp.isoformat(),
        }


@dataclass
class Issue:
    """Problem or warning encountered."""
    severity: AlertSeverity
    category: str  # "system", "execution", "data", "compliance", "risk"
    timestamp: datetime
    title: str
    description: str
    affected_components: list[str] = field(default_factory=list)
    resolution_status: str = "open"  # "open", "acknowledged", "resolved"
    resolution_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "description": self.description,
            "affected_components": self.affected_components,
            "resolution_status": self.resolution_status,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class IssuesSection:
    """Issues and warnings section."""
    issues: list[Issue] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    unresolved_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issues": [i.to_dict() for i in self.issues],
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "unresolved_count": self.unresolved_count,
        }


@dataclass
class DailyReport:
    """Complete daily trading report."""
    report_id: str
    report_date: date
    generated_at: datetime
    status: ReportStatus
    executive_summary: ExecutiveSummary
    performance: PerformanceSection
    positions: PositionsSection
    decisions: DecisionsSection
    signals: SignalsSection
    risk: RiskSection
    recommendations: RecommendationsSection
    issues: IssuesSection
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "report_date": self.report_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "executive_summary": self.executive_summary.to_dict(),
            "performance": self.performance.to_dict(),
            "positions": self.positions.to_dict(),
            "decisions": self.decisions.to_dict(),
            "signals": self.signals.to_dict(),
            "risk": self.risk.to_dict(),
            "recommendations": self.recommendations.to_dict(),
            "issues": self.issues.to_dict(),
            "metadata": self.metadata,
        }


class DailyReportGenerator:
    """
    Generates comprehensive daily trading reports.

    Features:
    - Aggregates data from audit logs, positions, and metrics
    - AI-generated insights (with graceful fallback to rule-based)
    - Template-based HTML export
    - Historical report storage and comparison

    Usage:
        generator = DailyReportGenerator(audit_logger, llm_client)
        report = await generator.generate_report(date.today())
        html = generator.export_html(report)
        generator.export_json(report, "reports/daily_2024_01_15.json")
    """

    # HTML template for report generation
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Trading Report - {report_date}</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #d97706;
            --danger-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .header {{
            background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        .header .meta {{ opacity: 0.9; font-size: 0.95rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }}
        .card h2 {{
            font-size: 1.25rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .metric {{
            background: var(--bg-color);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .positive {{ color: var(--success-color); }}
        .negative {{ color: var(--danger-color); }}
        .warning {{ color: var(--warning-color); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background: var(--bg-color);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{ background: var(--bg-color); }}
        .alert {{
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.75rem;
        }}
        .alert-critical {{ background: #fef2f2; border-left: 4px solid var(--danger-color); }}
        .alert-warning {{ background: #fffbeb; border-left: 4px solid var(--warning-color); }}
        .alert-info {{ background: #eff6ff; border-left: 4px solid var(--primary-color); }}
        .recommendation {{
            background: #f0fdf4;
            border-left: 4px solid var(--success-color);
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 0 8px 8px 0;
        }}
        .recommendation-title {{ font-weight: 600; margin-bottom: 0.5rem; }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-high {{ background: #fef2f2; color: var(--danger-color); }}
        .badge-medium {{ background: #fffbeb; color: var(--warning-color); }}
        .badge-low {{ background: #f0fdf4; color: var(--success-color); }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
        .highlight-box {{
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        .progress-bar {{
            background: var(--border-color);
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .footer {{
            text-align: center;
            color: var(--text-secondary);
            padding: 2rem;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Daily Trading Report</h1>
            <div class="meta">
                {report_date} | Generated at {generated_at} | Report ID: {report_id}
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="card">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value {pnl_class}">{total_pnl}</div>
                    <div class="metric-label">Daily P&L</div>
                </div>
                <div class="metric">
                    <div class="metric-value {pnl_pct_class}">{total_pnl_pct}%</div>
                    <div class="metric-label">Daily Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{trades_executed}</div>
                    <div class="metric-label">Trades Executed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{win_rate}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{portfolio_value}</div>
                    <div class="metric-label">Portfolio Value</div>
                </div>
                <div class="metric">
                    <div class="metric-value {drawdown_class}">{max_drawdown}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
            </div>
            {highlights_html}
            {alerts_html}
        </div>

        <!-- Performance -->
        <div class="card">
            <h2>Performance Analysis</h2>
            <div class="two-col">
                <div>
                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--text-secondary);">Returns</h3>
                    <table>
                        <tr><td>Daily Return</td><td class="{daily_return_class}">{daily_return_pct}%</td></tr>
                        <tr><td>MTD Return</td><td class="{mtd_return_class}">{mtd_return_pct}%</td></tr>
                        <tr><td>YTD Return</td><td class="{ytd_return_class}">{ytd_return_pct}%</td></tr>
                        <tr><td>Benchmark</td><td>{benchmark_return_pct}%</td></tr>
                        <tr><td>Alpha</td><td class="{alpha_class}">{alpha}%</td></tr>
                    </table>
                </div>
                <div>
                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--text-secondary);">Risk-Adjusted</h3>
                    <table>
                        <tr><td>Sharpe Ratio</td><td>{sharpe_ratio}</td></tr>
                        <tr><td>Sortino Ratio</td><td>{sortino_ratio}</td></tr>
                        <tr><td>Calmar Ratio</td><td>{calmar_ratio}</td></tr>
                        <tr><td>Information Ratio</td><td>{information_ratio}</td></tr>
                        <tr><td>Profit Factor</td><td>{profit_factor}</td></tr>
                    </table>
                </div>
            </div>
            {pnl_by_strategy_html}
        </div>

        <!-- Positions -->
        <div class="card">
            <h2>Portfolio Positions</h2>
            <div class="metrics-grid" style="margin-bottom: 1rem;">
                <div class="metric">
                    <div class="metric-value">{long_exposure}</div>
                    <div class="metric-label">Long Exposure</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{short_exposure}</div>
                    <div class="metric-label">Short Exposure</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{net_exposure}</div>
                    <div class="metric-label">Net Exposure</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{gross_exposure}</div>
                    <div class="metric-label">Gross Exposure</div>
                </div>
            </div>
            {positions_table_html}
        </div>

        <!-- Risk -->
        <div class="card">
            <h2>Risk Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{var_95}</div>
                    <div class="metric-label">VaR (95%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{var_99}</div>
                    <div class="metric-label">VaR (99%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{expected_shortfall}</div>
                    <div class="metric-label">Expected Shortfall</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{current_leverage}x</div>
                    <div class="metric-label">Leverage ({leverage_util}% used)</div>
                </div>
            </div>
            {risk_alerts_html}
            {limit_utilization_html}
        </div>

        <!-- Decisions -->
        <div class="card">
            <h2>CIO Decisions</h2>
            <div class="metrics-grid" style="margin-bottom: 1rem;">
                <div class="metric">
                    <div class="metric-value">{total_decisions}</div>
                    <div class="metric-label">Total Decisions</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{executed_count}</div>
                    <div class="metric-label">Executed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{rejected_risk}</div>
                    <div class="metric-label">Rejected (Risk)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{rejected_compliance}</div>
                    <div class="metric-label">Rejected (Compliance)</div>
                </div>
            </div>
            {decisions_table_html}
        </div>

        <!-- Signals -->
        <div class="card">
            <h2>Signal Analysis</h2>
            {signals_summary_html}
        </div>

        <!-- Recommendations -->
        <div class="card">
            <h2>Recommendations</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.9rem;">
                Generated by: {recommendations_source}
            </p>
            {recommendations_html}
        </div>

        <!-- Issues -->
        <div class="card">
            <h2>Issues & Warnings</h2>
            {issues_html}
        </div>

        <div class="footer">
            AI Trading Firm - Daily Report | Generated automatically | Confidential
        </div>
    </div>
</body>
</html>"""

    def __init__(
        self,
        audit_logger: Any | None = None,
        llm_client: Any | None = None,
        reports_dir: str = "reports/daily",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the daily report generator.

        Args:
            audit_logger: AuditLogger instance for reading audit data
            llm_client: LLMClient instance for AI-generated insights (optional)
            reports_dir: Directory for storing generated reports
            config: Optional configuration dictionary
        """
        self._audit_logger = audit_logger
        self._llm_client = llm_client
        self._reports_dir = Path(reports_dir)
        self._config = config or {}

        # Ensure reports directory exists
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        # Cache for historical reports
        self._report_cache: dict[str, DailyReport] = {}

        logger.info(f"DailyReportGenerator initialized, reports_dir={reports_dir}")

    async def generate_report(
        self,
        report_date: date | None = None,
        include_llm_insights: bool = True,
    ) -> DailyReport:
        """
        Generate a comprehensive daily report.

        Args:
            report_date: Date for the report (defaults to today)
            include_llm_insights: Whether to include AI-generated insights

        Returns:
            DailyReport with all sections populated
        """
        if report_date is None:
            report_date = date.today()

        report_id = f"DR-{report_date.isoformat()}-{datetime.now(timezone.utc).strftime('%H%M%S')}"

        logger.info(f"Generating daily report for {report_date}, id={report_id}")

        try:
            # Build report sections
            executive_summary = await self._build_executive_summary(report_date)
            performance = await self._build_performance_section(report_date)
            positions = await self._build_positions_section(report_date)
            decisions = await self._build_decisions_section(report_date)
            signals = await self._build_signals_section(report_date)
            risk = await self._build_risk_section(report_date)
            issues = await self._build_issues_section(report_date)

            # Generate recommendations (with LLM if available)
            recommendations = await self._build_recommendations_section(
                report_date,
                executive_summary,
                performance,
                risk,
                include_llm_insights,
            )

            report = DailyReport(
                report_id=report_id,
                report_date=report_date,
                generated_at=datetime.now(timezone.utc),
                status=ReportStatus.COMPLETED,
                executive_summary=executive_summary,
                performance=performance,
                positions=positions,
                decisions=decisions,
                signals=signals,
                risk=risk,
                recommendations=recommendations,
                issues=issues,
                metadata={
                    "generator_version": "1.0.0",
                    "llm_enabled": include_llm_insights and self._llm_client is not None,
                },
            )

            # Store in cache
            self._report_cache[report_date.isoformat()] = report

            logger.info(f"Daily report generated successfully: {report_id}")
            return report

        except Exception as e:
            logger.exception(f"Failed to generate daily report: {e}")
            # Return a minimal failed report
            return DailyReport(
                report_id=report_id,
                report_date=report_date,
                generated_at=datetime.now(timezone.utc),
                status=ReportStatus.FAILED,
                executive_summary=ExecutiveSummary(report_date=report_date),
                performance=PerformanceSection(),
                positions=PositionsSection(),
                decisions=DecisionsSection(),
                signals=SignalsSection(),
                risk=RiskSection(),
                recommendations=RecommendationsSection(),
                issues=IssuesSection(
                    issues=[
                        Issue(
                            severity=AlertSeverity.CRITICAL,
                            category="system",
                            timestamp=datetime.now(timezone.utc),
                            title="Report Generation Failed",
                            description=str(e),
                        )
                    ],
                    critical_count=1,
                ),
                metadata={"error": str(e)},
            )

    async def _build_executive_summary(self, report_date: date) -> ExecutiveSummary:
        """Build the executive summary section."""
        summary = ExecutiveSummary(report_date=report_date)

        # Get data from audit logger if available
        if self._audit_logger:
            try:
                start = datetime.combine(report_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                end = datetime.combine(report_date, datetime.max.time()).replace(tzinfo=timezone.utc)

                trades = self._audit_logger.get_trades(start_date=start, end_date=end)
                decisions = self._audit_logger.get_decisions(start_date=start, end_date=end)

                summary.trades_executed = len(trades)

                # Calculate P&L from trades
                total_pnl = sum(
                    t.get("details", {}).get("total_value", 0) *
                    (1 if t.get("details", {}).get("side") == "sell" else -1)
                    for t in trades
                )
                summary.total_pnl = total_pnl

                # Win rate
                if trades:
                    wins = sum(1 for t in trades if t.get("details", {}).get("total_value", 0) > 0)
                    summary.win_rate = (wins / len(trades)) * 100

            except Exception as e:
                logger.warning(f"Error building executive summary from audit logs: {e}")

        return summary

    async def _build_performance_section(self, report_date: date) -> PerformanceSection:
        """Build the performance analysis section."""
        performance = PerformanceSection()

        if self._audit_logger:
            try:
                start = datetime.combine(report_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                end = datetime.combine(report_date, datetime.max.time()).replace(tzinfo=timezone.utc)

                trades = self._audit_logger.get_trades(start_date=start, end_date=end)

                # Calculate P&L by symbol
                pnl_by_symbol: dict[str, float] = {}
                for trade in trades:
                    details = trade.get("details", {})
                    symbol = details.get("symbol", "UNKNOWN")
                    side = details.get("side", "buy")
                    value = details.get("total_value", 0)
                    pnl = value if side == "sell" else -value
                    pnl_by_symbol[symbol] = pnl_by_symbol.get(symbol, 0) + pnl

                performance.pnl_by_symbol = pnl_by_symbol
                performance.daily_pnl = sum(pnl_by_symbol.values())

                # Profit factor
                gross_profit = sum(v for v in pnl_by_symbol.values() if v > 0)
                gross_loss = abs(sum(v for v in pnl_by_symbol.values() if v < 0))
                performance.gross_profit = gross_profit
                performance.gross_loss = gross_loss
                if gross_loss > 0:
                    performance.profit_factor = gross_profit / gross_loss

            except Exception as e:
                logger.warning(f"Error building performance section: {e}")

        return performance

    async def _build_positions_section(self, report_date: date) -> PositionsSection:
        """Build the positions analysis section."""
        return PositionsSection()

    async def _build_decisions_section(self, report_date: date) -> DecisionsSection:
        """Build the CIO decisions section."""
        section = DecisionsSection()

        if self._audit_logger:
            try:
                start = datetime.combine(report_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                end = datetime.combine(report_date, datetime.max.time()).replace(tzinfo=timezone.utc)

                raw_decisions = self._audit_logger.get_decisions(start_date=start, end_date=end)

                decisions = []
                total_conviction = 0.0

                for d in raw_decisions:
                    details = d.get("details", {})
                    entry = DecisionEntry(
                        decision_id=d.get("event_id", ""),
                        timestamp=datetime.fromisoformat(d.get("timestamp", datetime.now(timezone.utc).isoformat())),
                        symbol=details.get("symbol", ""),
                        action=details.get("action", ""),
                        quantity=details.get("quantity", 0),
                        rationale=details.get("rationale", ""),
                        conviction_score=details.get("conviction_score", 0.5),
                        contributing_signals=details.get("contributing_signals", []),
                    )
                    decisions.append(entry)
                    total_conviction += entry.conviction_score

                section.decisions = decisions
                section.total_decisions = len(decisions)
                section.executed_count = len(decisions)  # Simplified
                if decisions:
                    section.average_conviction = total_conviction / len(decisions)

            except Exception as e:
                logger.warning(f"Error building decisions section: {e}")

        return section

    async def _build_signals_section(self, report_date: date) -> SignalsSection:
        """Build the signals analysis section."""
        section = SignalsSection()

        # Define signal agents
        signal_agents = [
            "MacroAgent",
            "StatArbAgent",
            "MomentumAgent",
            "MarketMakingAgent",
            "OptionsVolAgent",
        ]

        for agent in signal_agents:
            summary = SignalSummary(agent_name=agent)
            section.signal_summaries.append(summary)

        return section

    async def _build_risk_section(self, report_date: date) -> RiskSection:
        """Build the risk analysis section."""
        section = RiskSection()

        # Read config for limits
        section.max_leverage_limit = self._config.get("risk", {}).get("max_leverage", 2.0)
        section.daily_loss_limit_pct = self._config.get("risk", {}).get("max_daily_loss_pct", 3.0)

        return section

    async def _build_issues_section(self, report_date: date) -> IssuesSection:
        """Build the issues and warnings section."""
        section = IssuesSection()

        # Parse system logs for issues if available
        try:
            log_path = Path("logs/system.log")
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-1000:]  # Last 1000 lines

                for line in lines:
                    if "ERROR" in line or "CRITICAL" in line:
                        severity = AlertSeverity.CRITICAL if "CRITICAL" in line else AlertSeverity.WARNING
                        section.issues.append(
                            Issue(
                                severity=severity,
                                category="system",
                                timestamp=datetime.now(timezone.utc),
                                title="System Log Entry",
                                description=line.strip()[:200],
                            )
                        )

                section.critical_count = sum(1 for i in section.issues if i.severity == AlertSeverity.CRITICAL)
                section.warning_count = sum(1 for i in section.issues if i.severity == AlertSeverity.WARNING)
                section.unresolved_count = len(section.issues)

        except Exception as e:
            logger.warning(f"Error parsing system logs: {e}")

        return section

    async def _build_recommendations_section(
        self,
        report_date: date,
        summary: ExecutiveSummary,
        performance: PerformanceSection,
        risk: RiskSection,
        include_llm: bool,
    ) -> RecommendationsSection:
        """Build the recommendations section with AI insights."""
        section = RecommendationsSection()

        # Try LLM-based recommendations if available
        if include_llm and self._llm_client:
            try:
                llm_recommendations = await self._generate_llm_recommendations(
                    summary, performance, risk
                )
                if llm_recommendations:
                    section.recommendations = llm_recommendations
                    section.generated_by = "llm"
                    return section
            except Exception as e:
                logger.warning(f"LLM recommendations failed, falling back to rule-based: {e}")

        # Fallback to rule-based recommendations
        section.recommendations = self._generate_rule_based_recommendations(
            summary, performance, risk
        )
        section.generated_by = "rule_based"
        return section

    async def _generate_llm_recommendations(
        self,
        summary: ExecutiveSummary,
        performance: PerformanceSection,
        risk: RiskSection,
    ) -> list[Recommendation]:
        """Generate recommendations using LLM."""
        if not self._llm_client:
            return []

        try:
            # Prepare context for LLM
            context = f"""
Daily Trading Report Summary:
- Total P&L: ${summary.total_pnl:,.2f} ({summary.total_pnl_pct:.2f}%)
- Trades Executed: {summary.trades_executed}
- Win Rate: {summary.win_rate:.1f}%
- Max Drawdown: {summary.max_drawdown_pct:.2f}%
- Sharpe Ratio: {summary.sharpe_ratio:.2f}

Risk Metrics:
- VaR 95%: ${risk.var_95:,.2f}
- Current Leverage: {risk.current_leverage:.2f}x (limit: {risk.max_leverage_limit:.2f}x)
- Kill Switch Triggered: {risk.kill_switch_triggered}

Performance by Strategy: {json.dumps(performance.pnl_by_strategy)}
"""
            # Use LLM client's analyze_sentiment method as a proxy
            # In production, you'd have a dedicated method for recommendations
            result = await self._llm_client.analyze_sentiment(
                f"Based on this trading day's performance, provide 3 recommendations:\n{context}",
                "PORTFOLIO"
            )

            if result.is_valid:
                # Parse recommendations from LLM response
                rec = Recommendation(
                    category="portfolio",
                    priority="medium",
                    title="AI-Generated Insight",
                    description=result.rationale,
                    rationale="Based on daily performance analysis",
                    suggested_action="Review portfolio allocation",
                    confidence=result.confidence,
                )
                return [rec]

        except Exception as e:
            logger.warning(f"LLM recommendation generation failed: {e}")

        return []

    def _generate_rule_based_recommendations(
        self,
        summary: ExecutiveSummary,
        performance: PerformanceSection,
        risk: RiskSection,
    ) -> list[Recommendation]:
        """Generate rule-based recommendations."""
        recommendations = []

        # Check for high drawdown
        if summary.max_drawdown_pct > 5.0:
            recommendations.append(
                Recommendation(
                    category="risk",
                    priority="high",
                    title="Elevated Drawdown Warning",
                    description=f"Current drawdown of {summary.max_drawdown_pct:.2f}% exceeds 5% threshold.",
                    rationale="High drawdown increases risk of hitting kill-switch limits.",
                    suggested_action="Consider reducing position sizes or increasing hedge ratios.",
                    confidence=0.9,
                )
            )

        # Check for low win rate
        if summary.win_rate < 40 and summary.trades_executed > 5:
            recommendations.append(
                Recommendation(
                    category="performance",
                    priority="medium",
                    title="Low Win Rate Alert",
                    description=f"Win rate of {summary.win_rate:.1f}% is below target of 40%.",
                    rationale="Consistently low win rate may indicate signal quality issues.",
                    suggested_action="Review signal agent performance and conviction thresholds.",
                    confidence=0.7,
                )
            )

        # Check leverage utilization
        if risk.leverage_utilization_pct > 80:
            recommendations.append(
                Recommendation(
                    category="risk",
                    priority="high",
                    title="High Leverage Utilization",
                    description=f"Leverage at {risk.leverage_utilization_pct:.1f}% of limit.",
                    rationale="Operating close to leverage limits reduces flexibility.",
                    suggested_action="Consider reducing positions to maintain buffer.",
                    confidence=0.85,
                )
            )

        # Positive performance acknowledgment
        if summary.total_pnl > 0 and summary.win_rate >= 50:
            recommendations.append(
                Recommendation(
                    category="performance",
                    priority="low",
                    title="Strong Performance Day",
                    description="Positive P&L with solid win rate.",
                    rationale="Strategy appears to be working well.",
                    suggested_action="Maintain current approach while monitoring for regime changes.",
                    confidence=0.6,
                )
            )

        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                Recommendation(
                    category="portfolio",
                    priority="low",
                    title="Continue Monitoring",
                    description="No significant issues detected.",
                    rationale="Metrics are within normal ranges.",
                    suggested_action="Continue normal operations with standard monitoring.",
                    confidence=0.5,
                )
            )

        return recommendations

    def get_recommendations(self, report: DailyReport) -> list[Recommendation]:
        """Get recommendations from a report."""
        return report.recommendations.recommendations

    def get_issues(self, report: DailyReport) -> list[Issue]:
        """Get issues from a report."""
        return report.issues.issues

    def export_html(self, report: DailyReport) -> str:
        """
        Export report as HTML.

        Args:
            report: DailyReport to export

        Returns:
            HTML string
        """
        def format_currency(value: float) -> str:
            return f"${value:,.2f}"

        def format_pct(value: float) -> str:
            return f"{value:.2f}"

        def get_value_class(value: float) -> str:
            if value > 0:
                return "positive"
            elif value < 0:
                return "negative"
            return ""

        # Build highlights HTML
        highlights_html = ""
        if report.executive_summary.highlights:
            highlights_html = '<div class="highlight-box"><h4>Highlights</h4><ul>'
            for h in report.executive_summary.highlights:
                highlights_html += f"<li>{h}</li>"
            highlights_html += "</ul></div>"

        # Build alerts HTML
        alerts_html = ""
        for alert in report.executive_summary.alerts:
            severity = alert.get("severity", "info")
            alerts_html += f'<div class="alert alert-{severity}">{alert.get("message", "")}</div>'

        # Build positions table
        positions_table_html = ""
        if report.positions.current_positions:
            positions_table_html = """
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Avg Cost</th>
                        <th>Current</th>
                        <th>Market Value</th>
                        <th>Unrealized P&L</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
            """
            for pos in report.positions.current_positions:
                pnl_class = get_value_class(pos.unrealized_pnl)
                positions_table_html += f"""
                <tr>
                    <td><strong>{pos.symbol}</strong></td>
                    <td>{pos.quantity:,}</td>
                    <td>{format_currency(pos.average_cost)}</td>
                    <td>{format_currency(pos.current_price)}</td>
                    <td>{format_currency(pos.market_value)}</td>
                    <td class="{pnl_class}">{format_currency(pos.unrealized_pnl)} ({format_pct(pos.unrealized_pnl_pct)}%)</td>
                    <td>{format_pct(pos.weight_pct)}%</td>
                </tr>
                """
            positions_table_html += "</tbody></table>"
        else:
            positions_table_html = '<p style="color: var(--text-secondary);">No open positions.</p>'

        # Build P&L by strategy
        pnl_by_strategy_html = ""
        if report.performance.pnl_by_strategy:
            pnl_by_strategy_html = '<h3 style="font-size: 1rem; margin: 1.5rem 0 1rem;">P&L by Strategy</h3><table>'
            for strategy, pnl in report.performance.pnl_by_strategy.items():
                pnl_class = get_value_class(pnl)
                pnl_by_strategy_html += f'<tr><td>{strategy}</td><td class="{pnl_class}">{format_currency(pnl)}</td></tr>'
            pnl_by_strategy_html += "</table>"

        # Build risk alerts
        risk_alerts_html = ""
        if report.risk.risk_alerts:
            risk_alerts_html = '<div style="margin-top: 1rem;">'
            for alert in report.risk.risk_alerts:
                severity = alert.get("severity", "warning")
                risk_alerts_html += f'<div class="alert alert-{severity}">{alert.get("message", "")}</div>'
            risk_alerts_html += "</div>"

        # Build limit utilization
        limit_utilization_html = ""
        if report.risk.position_limit_utilization:
            limit_utilization_html = '<h3 style="font-size: 1rem; margin: 1.5rem 0 1rem;">Limit Utilization</h3>'
            for limit_name, util in report.risk.position_limit_utilization.items():
                bar_color = "#16a34a" if util < 70 else "#d97706" if util < 90 else "#dc2626"
                limit_utilization_html += f"""
                <div style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span>{limit_name}</span>
                        <span>{util:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-bar-fill" style="width: {min(util, 100):.1f}%; background: {bar_color};"></div>
                    </div>
                </div>
                """

        # Build decisions table
        decisions_table_html = ""
        if report.decisions.decisions:
            decisions_table_html = """
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Action</th>
                        <th>Quantity</th>
                        <th>Conviction</th>
                        <th>Rationale</th>
                    </tr>
                </thead>
                <tbody>
            """
            for dec in report.decisions.decisions[:10]:  # Show top 10
                decisions_table_html += f"""
                <tr>
                    <td>{dec.timestamp.strftime('%H:%M:%S')}</td>
                    <td><strong>{dec.symbol}</strong></td>
                    <td>{dec.action}</td>
                    <td>{dec.quantity:,}</td>
                    <td>{dec.conviction_score:.2f}</td>
                    <td>{dec.rationale[:50]}...</td>
                </tr>
                """
            decisions_table_html += "</tbody></table>"
        else:
            decisions_table_html = '<p style="color: var(--text-secondary);">No decisions recorded.</p>'

        # Build signals summary
        signals_summary_html = ""
        if report.signals.signal_summaries:
            signals_summary_html = """
            <table>
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Signals</th>
                        <th>Long</th>
                        <th>Short</th>
                        <th>Avg Strength</th>
                        <th>Acted Upon</th>
                    </tr>
                </thead>
                <tbody>
            """
            for sig in report.signals.signal_summaries:
                signals_summary_html += f"""
                <tr>
                    <td><strong>{sig.agent_name}</strong></td>
                    <td>{sig.signals_generated}</td>
                    <td>{sig.long_signals}</td>
                    <td>{sig.short_signals}</td>
                    <td>{sig.average_strength:.2f}</td>
                    <td>{sig.signals_acted_upon}</td>
                </tr>
                """
            signals_summary_html += "</tbody></table>"

        # Build recommendations
        recommendations_html = ""
        if report.recommendations.recommendations:
            for rec in report.recommendations.recommendations:
                badge_class = f"badge-{rec.priority}"
                recommendations_html += f"""
                <div class="recommendation">
                    <div class="recommendation-title">
                        <span class="badge {badge_class}">{rec.priority.upper()}</span>
                        {rec.title}
                    </div>
                    <p>{rec.description}</p>
                    <p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">
                        <strong>Action:</strong> {rec.suggested_action}
                    </p>
                </div>
                """
        else:
            recommendations_html = '<p style="color: var(--text-secondary);">No recommendations at this time.</p>'

        # Build issues
        issues_html = ""
        if report.issues.issues:
            for issue in report.issues.issues[:10]:  # Show top 10
                alert_class = f"alert-{issue.severity.value}"
                issues_html += f"""
                <div class="alert {alert_class}">
                    <strong>[{issue.severity.value.upper()}]</strong> {issue.title}
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">{issue.description[:200]}</p>
                </div>
                """
        else:
            issues_html = '<p style="color: var(--text-secondary);">No issues detected.</p>'

        # Fill template
        html = self.HTML_TEMPLATE.format(
            report_date=report.report_date.strftime("%B %d, %Y"),
            generated_at=report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            report_id=report.report_id,
            total_pnl=format_currency(report.executive_summary.total_pnl),
            pnl_class=get_value_class(report.executive_summary.total_pnl),
            total_pnl_pct=format_pct(report.executive_summary.total_pnl_pct),
            pnl_pct_class=get_value_class(report.executive_summary.total_pnl_pct),
            trades_executed=report.executive_summary.trades_executed,
            win_rate=format_pct(report.executive_summary.win_rate),
            portfolio_value=format_currency(report.executive_summary.portfolio_value),
            max_drawdown=format_pct(report.executive_summary.max_drawdown_pct),
            drawdown_class="negative" if report.executive_summary.max_drawdown_pct > 5 else "",
            highlights_html=highlights_html,
            alerts_html=alerts_html,
            daily_return_pct=format_pct(report.performance.daily_return_pct),
            daily_return_class=get_value_class(report.performance.daily_return_pct),
            mtd_return_pct=format_pct(report.performance.mtd_return_pct),
            mtd_return_class=get_value_class(report.performance.mtd_return_pct),
            ytd_return_pct=format_pct(report.performance.ytd_return_pct),
            ytd_return_class=get_value_class(report.performance.ytd_return_pct),
            benchmark_return_pct=format_pct(report.performance.benchmark_return_pct),
            alpha=format_pct(report.performance.alpha),
            alpha_class=get_value_class(report.performance.alpha),
            sharpe_ratio=f"{report.executive_summary.sharpe_ratio:.2f}",
            sortino_ratio=f"{report.performance.sortino_ratio:.2f}",
            calmar_ratio=f"{report.performance.calmar_ratio:.2f}",
            information_ratio=f"{report.performance.information_ratio:.2f}",
            profit_factor=f"{report.performance.profit_factor:.2f}",
            pnl_by_strategy_html=pnl_by_strategy_html,
            long_exposure=format_currency(report.positions.total_long_exposure),
            short_exposure=format_currency(report.positions.total_short_exposure),
            net_exposure=format_currency(report.positions.net_exposure),
            gross_exposure=format_currency(report.positions.gross_exposure),
            positions_table_html=positions_table_html,
            var_95=format_currency(report.risk.var_95),
            var_99=format_currency(report.risk.var_99),
            expected_shortfall=format_currency(report.risk.expected_shortfall),
            current_leverage=f"{report.risk.current_leverage:.2f}",
            leverage_util=format_pct(report.risk.leverage_utilization_pct),
            risk_alerts_html=risk_alerts_html,
            limit_utilization_html=limit_utilization_html,
            total_decisions=report.decisions.total_decisions,
            executed_count=report.decisions.executed_count,
            rejected_risk=report.decisions.rejected_by_risk,
            rejected_compliance=report.decisions.rejected_by_compliance,
            decisions_table_html=decisions_table_html,
            signals_summary_html=signals_summary_html,
            recommendations_source=report.recommendations.generated_by,
            recommendations_html=recommendations_html,
            issues_html=issues_html,
        )

        return html

    def export_json(
        self,
        report: DailyReport,
        file_path: str | None = None,
    ) -> str:
        """
        Export report as JSON.

        Args:
            report: DailyReport to export
            file_path: Optional path to write file

        Returns:
            JSON string
        """
        json_str = json.dumps(report.to_dict(), indent=2, default=str)

        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Report exported to {file_path}")

        return json_str

    def save_report(self, report: DailyReport) -> str:
        """
        Save report to storage directory.

        Args:
            report: DailyReport to save

        Returns:
            Path to saved report
        """
        # Save JSON
        json_path = self._reports_dir / f"{report.report_id}.json"
        self.export_json(report, str(json_path))

        # Save HTML
        html_path = self._reports_dir / f"{report.report_id}.html"
        html_content = self.export_html(report)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Report saved: {json_path}, {html_path}")
        return str(json_path)

    def load_report(self, report_date: date) -> DailyReport | None:
        """
        Load a previously generated report.

        Args:
            report_date: Date of the report to load

        Returns:
            DailyReport or None if not found
        """
        # Check cache first
        cache_key = report_date.isoformat()
        if cache_key in self._report_cache:
            return self._report_cache[cache_key]

        # Look for report files
        pattern = f"DR-{report_date.isoformat()}-*.json"
        matching_files = list(self._reports_dir.glob(pattern))

        if not matching_files:
            return None

        # Load most recent
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct report (simplified - full reconstruction would need more work)
            report = DailyReport(
                report_id=data.get("report_id", ""),
                report_date=date.fromisoformat(data.get("report_date", "")),
                generated_at=datetime.fromisoformat(data.get("generated_at", "")),
                status=ReportStatus(data.get("status", "completed")),
                executive_summary=ExecutiveSummary(report_date=report_date),
                performance=PerformanceSection(),
                positions=PositionsSection(),
                decisions=DecisionsSection(),
                signals=SignalsSection(),
                risk=RiskSection(),
                recommendations=RecommendationsSection(),
                issues=IssuesSection(),
                metadata=data.get("metadata", {}),
            )

            self._report_cache[cache_key] = report
            return report

        except Exception as e:
            logger.exception(f"Error loading report from {latest_file}: {e}")
            return None

    def compare_reports(
        self,
        report1: DailyReport,
        report2: DailyReport,
    ) -> dict[str, Any]:
        """
        Compare two reports and highlight differences.

        Args:
            report1: First report (typically earlier)
            report2: Second report (typically later)

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "date_range": {
                "from": report1.report_date.isoformat(),
                "to": report2.report_date.isoformat(),
            },
            "performance": {
                "pnl_change": report2.executive_summary.total_pnl - report1.executive_summary.total_pnl,
                "win_rate_change": report2.executive_summary.win_rate - report1.executive_summary.win_rate,
                "sharpe_change": report2.executive_summary.sharpe_ratio - report1.executive_summary.sharpe_ratio,
            },
            "risk": {
                "drawdown_change": report2.risk.max_drawdown_pct - report1.risk.max_drawdown_pct,
                "var_change": report2.risk.var_95 - report1.risk.var_95,
                "leverage_change": report2.risk.current_leverage - report1.risk.current_leverage,
            },
            "activity": {
                "trades_change": report2.executive_summary.trades_executed - report1.executive_summary.trades_executed,
                "decisions_change": report2.decisions.total_decisions - report1.decisions.total_decisions,
            },
        }

        return comparison

    def list_available_reports(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """
        List available reports in storage.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of report metadata
        """
        reports = []

        for json_file in self._reports_dir.glob("DR-*.json"):
            try:
                # Extract date from filename
                parts = json_file.stem.split("-")
                if len(parts) >= 4:
                    report_date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    report_date = date.fromisoformat(report_date_str)

                    # Apply filters
                    if start_date and report_date < start_date:
                        continue
                    if end_date and report_date > end_date:
                        continue

                    reports.append({
                        "report_id": json_file.stem,
                        "report_date": report_date.isoformat(),
                        "file_path": str(json_file),
                        "file_size": json_file.stat().st_size,
                        "modified_at": datetime.fromtimestamp(
                            json_file.stat().st_mtime, tz=timezone.utc
                        ).isoformat(),
                    })

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse report filename {json_file}: {e}")

        # Sort by date descending
        reports.sort(key=lambda r: r["report_date"], reverse=True)
        return reports

"""
Risk Report Generation Module
=============================

Automated risk report generation (Issue #R25).

Features:
- Daily risk summary reports
- Position risk breakdown
- VaR/CVaR analysis
- Greeks exposure reports
- Limit utilization tracking
- Stress test summaries
- Export to multiple formats (JSON, CSV, HTML)
"""

from __future__ import annotations

import json
import logging
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from io import StringIO

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Type of risk report."""
    DAILY_SUMMARY = "daily_summary"
    POSITION_RISK = "position_risk"
    VAR_ANALYSIS = "var_analysis"
    GREEKS_EXPOSURE = "greeks_exposure"
    LIMIT_UTILIZATION = "limit_utilization"
    STRESS_TEST = "stress_test"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"


class ReportFormat(str, Enum):
    """Output format for reports."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "text"


@dataclass
class ReportSection:
    """A section within a report."""
    title: str
    data: dict | list
    summary: str = ""
    alerts: list[str] = field(default_factory=list)


@dataclass
class RiskReport:
    """Complete risk report."""
    report_id: str
    report_type: ReportType
    generated_at: datetime
    as_of_date: datetime
    title: str
    sections: list[ReportSection] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'generated_at': self.generated_at.isoformat(),
            'as_of_date': self.as_of_date.isoformat(),
            'title': self.title,
            'sections': [
                {
                    'title': s.title,
                    'data': s.data,
                    'summary': s.summary,
                    'alerts': s.alerts,
                }
                for s in self.sections
            ],
            'metadata': self.metadata,
            'alerts': self.alerts,
        }

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
        self.alerts.extend(section.alerts)


class RiskReportGenerator:
    """
    Generates various risk reports (#R25).

    Provides comprehensive risk reporting for compliance and monitoring.
    """

    def __init__(
        self,
        output_dir: str = "reports/risk",
        firm_name: str = "AI Trading Firm",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.firm_name = firm_name
        self._report_counter = 0

    def _generate_report_id(self, report_type: ReportType) -> str:
        """Generate unique report ID."""
        self._report_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{report_type.value}_{timestamp}_{self._report_counter}"

    def generate_daily_summary(
        self,
        risk_state: Any,
        positions: dict[str, Any],
        var_results: dict | None = None,
        stress_results: list[dict] | None = None,
    ) -> RiskReport:
        """
        Generate daily risk summary report.

        Comprehensive overview of portfolio risk for the day.
        """
        now = datetime.now(timezone.utc)

        report = RiskReport(
            report_id=self._generate_report_id(ReportType.DAILY_SUMMARY),
            report_type=ReportType.DAILY_SUMMARY,
            generated_at=now,
            as_of_date=now,
            title=f"Daily Risk Summary - {now.strftime('%Y-%m-%d')}",
            metadata={
                'firm_name': self.firm_name,
                'generated_by': 'RiskReportGenerator',
            },
        )

        # Portfolio Overview Section
        portfolio_data = {
            'net_liquidation': getattr(risk_state, 'net_liquidation', 0),
            'gross_exposure': getattr(risk_state, 'gross_exposure', 0),
            'net_exposure': getattr(risk_state, 'net_exposure', 0),
            'cash': getattr(risk_state, 'cash', 0),
            'leverage': getattr(risk_state, 'leverage', 0),
            'position_count': len(positions),
        }

        portfolio_alerts = []
        leverage = portfolio_data['leverage']
        if leverage > 2.0:
            portfolio_alerts.append(f"HIGH LEVERAGE: {leverage:.2f}x exceeds 2.0x threshold")

        report.add_section(ReportSection(
            title="Portfolio Overview",
            data=portfolio_data,
            summary=f"Portfolio NLV: ${portfolio_data['net_liquidation']:,.0f}",
            alerts=portfolio_alerts,
        ))

        # P&L Section
        pnl_data = {
            'daily_pnl': getattr(risk_state, 'daily_pnl', 0),
            'daily_pnl_pct': (getattr(risk_state, 'daily_pnl', 0) / portfolio_data['net_liquidation'] * 100)
                if portfolio_data['net_liquidation'] > 0 else 0,
            'mtd_pnl': getattr(risk_state, 'mtd_pnl', 0),
            'ytd_pnl': getattr(risk_state, 'ytd_pnl', 0),
            'current_drawdown': getattr(risk_state, 'current_drawdown', 0),
            'max_drawdown': getattr(risk_state, 'max_drawdown', 0),
        }

        pnl_alerts = []
        if pnl_data['daily_pnl_pct'] < -2.0:
            pnl_alerts.append(f"DAILY LOSS: {pnl_data['daily_pnl_pct']:.2f}% exceeds -2% threshold")
        if pnl_data['current_drawdown'] > 0.05:
            pnl_alerts.append(f"DRAWDOWN WARNING: {pnl_data['current_drawdown']*100:.1f}% drawdown")

        report.add_section(ReportSection(
            title="P&L Summary",
            data=pnl_data,
            summary=f"Daily P&L: ${pnl_data['daily_pnl']:,.0f} ({pnl_data['daily_pnl_pct']:.2f}%)",
            alerts=pnl_alerts,
        ))

        # VaR Section
        var_data = {
            'var_95': getattr(risk_state, 'var_95', 0),
            'var_99': getattr(risk_state, 'var_99', 0),
            'expected_shortfall': getattr(risk_state, 'expected_shortfall', 0),
            'var_95_dollars': getattr(risk_state, 'var_95', 0) * portfolio_data['net_liquidation'],
            'var_99_dollars': getattr(risk_state, 'var_99', 0) * portfolio_data['net_liquidation'],
        }

        if var_results:
            var_data.update(var_results)

        var_alerts = []
        if var_data['var_95'] > 0.02:
            var_alerts.append(f"VAR WARNING: 95% VaR of {var_data['var_95']*100:.2f}% exceeds 2% limit")

        report.add_section(ReportSection(
            title="Value at Risk",
            data=var_data,
            summary=f"95% VaR: ${var_data['var_95_dollars']:,.0f} ({var_data['var_95']*100:.2f}%)",
            alerts=var_alerts,
        ))

        # Greeks Section (if applicable)
        greeks_data = {
            'portfolio_delta': getattr(risk_state, 'portfolio_delta', 0),
            'portfolio_gamma': getattr(risk_state, 'portfolio_gamma', 0),
            'portfolio_vega': getattr(risk_state, 'portfolio_vega', 0),
            'portfolio_theta': getattr(risk_state, 'portfolio_theta', 0),
        }

        if any(v != 0 for v in greeks_data.values()):
            report.add_section(ReportSection(
                title="Greeks Exposure",
                data=greeks_data,
                summary=f"Net Delta: {greeks_data['portfolio_delta']:.0f}",
            ))

        # Position Concentration Section
        concentration_data = self._calculate_concentration(positions, portfolio_data['net_liquidation'])
        concentration_alerts = []

        if concentration_data['top_position_pct'] > 10:
            concentration_alerts.append(
                f"CONCENTRATION: Top position ({concentration_data['top_position']}) "
                f"at {concentration_data['top_position_pct']:.1f}% exceeds 10%"
            )

        report.add_section(ReportSection(
            title="Position Concentration",
            data=concentration_data,
            summary=f"Top 5 positions: {concentration_data['top_5_pct']:.1f}% of portfolio",
            alerts=concentration_alerts,
        ))

        # Stress Test Section
        if stress_results:
            stress_data = {
                'scenarios': stress_results,
                'worst_case_scenario': max(stress_results, key=lambda x: abs(x.get('pnl_impact', 0))),
                'average_impact': sum(s.get('pnl_impact', 0) for s in stress_results) / len(stress_results),
            }

            stress_alerts = []
            worst_impact = stress_data['worst_case_scenario'].get('pnl_impact', 0)
            if abs(worst_impact) > portfolio_data['net_liquidation'] * 0.25:
                stress_alerts.append(
                    f"STRESS WARNING: Worst case scenario could lose "
                    f"${abs(worst_impact):,.0f} (>25% of NLV)"
                )

            report.add_section(ReportSection(
                title="Stress Test Results",
                data=stress_data,
                summary=f"Worst case: {stress_data['worst_case_scenario'].get('name', 'Unknown')}",
                alerts=stress_alerts,
            ))

        logger.info(f"Generated daily risk summary report: {report.report_id}")
        return report

    def generate_position_risk_report(
        self,
        positions: dict[str, Any],
        risk_contributions: dict[str, float] | None = None,
    ) -> RiskReport:
        """Generate detailed position risk breakdown report."""
        now = datetime.now(timezone.utc)

        report = RiskReport(
            report_id=self._generate_report_id(ReportType.POSITION_RISK),
            report_type=ReportType.POSITION_RISK,
            generated_at=now,
            as_of_date=now,
            title=f"Position Risk Report - {now.strftime('%Y-%m-%d %H:%M')}",
        )

        position_data = []
        for symbol, pos in positions.items():
            pos_info = {
                'symbol': symbol,
                'quantity': getattr(pos, 'quantity', 0),
                'market_value': getattr(pos, 'market_value', 0),
                'unrealized_pnl': getattr(pos, 'unrealized_pnl', 0),
                'sector': getattr(pos, 'sector', 'Unknown'),
                'risk_contribution': risk_contributions.get(symbol, 0) if risk_contributions else 0,
            }
            position_data.append(pos_info)

        # Sort by absolute market value
        position_data.sort(key=lambda x: abs(x['market_value']), reverse=True)

        report.add_section(ReportSection(
            title="Position Details",
            data=position_data,
            summary=f"Total positions: {len(position_data)}",
        ))

        return report

    def generate_limit_utilization_report(
        self,
        current_values: dict[str, float],
        limits: dict[str, float],
    ) -> RiskReport:
        """Generate limit utilization report."""
        now = datetime.now(timezone.utc)

        report = RiskReport(
            report_id=self._generate_report_id(ReportType.LIMIT_UTILIZATION),
            report_type=ReportType.LIMIT_UTILIZATION,
            generated_at=now,
            as_of_date=now,
            title=f"Limit Utilization Report - {now.strftime('%Y-%m-%d %H:%M')}",
        )

        utilization_data = []
        alerts = []

        for limit_name, limit_value in limits.items():
            current = current_values.get(limit_name, 0)
            utilization = (current / limit_value * 100) if limit_value > 0 else 0

            utilization_data.append({
                'limit_name': limit_name,
                'current_value': current,
                'limit_value': limit_value,
                'utilization_pct': utilization,
                'remaining': limit_value - current,
                'status': 'OK' if utilization < 80 else ('WARNING' if utilization < 100 else 'BREACH'),
            })

            if utilization >= 100:
                alerts.append(f"BREACH: {limit_name} at {utilization:.1f}% (limit: {limit_value})")
            elif utilization >= 80:
                alerts.append(f"WARNING: {limit_name} at {utilization:.1f}% (approaching limit)")

        report.add_section(ReportSection(
            title="Limit Utilization",
            data=utilization_data,
            summary=f"Tracking {len(limits)} limits",
            alerts=alerts,
        ))

        return report

    def _calculate_concentration(
        self,
        positions: dict[str, Any],
        total_value: float,
    ) -> dict:
        """Calculate position concentration metrics."""
        if not positions or total_value == 0:
            return {
                'top_position': None,
                'top_position_pct': 0,
                'top_5_pct': 0,
                'hhi': 0,
            }

        # Sort by absolute value
        sorted_positions = sorted(
            positions.items(),
            key=lambda x: abs(getattr(x[1], 'market_value', 0)),
            reverse=True
        )

        # Top position
        top_symbol = sorted_positions[0][0] if sorted_positions else None
        top_value = abs(getattr(sorted_positions[0][1], 'market_value', 0)) if sorted_positions else 0
        top_pct = (top_value / total_value * 100) if total_value > 0 else 0

        # Top 5
        top_5_value = sum(
            abs(getattr(pos, 'market_value', 0))
            for _, pos in sorted_positions[:5]
        )
        top_5_pct = (top_5_value / total_value * 100) if total_value > 0 else 0

        # HHI (Herfindahl-Hirschman Index)
        weights = [
            (abs(getattr(pos, 'market_value', 0)) / total_value) ** 2
            for _, pos in sorted_positions
        ]
        hhi = sum(weights) * 10000  # Scale to 0-10000

        return {
            'top_position': top_symbol,
            'top_position_pct': top_pct,
            'top_5_pct': top_5_pct,
            'hhi': hhi,
            'position_count': len(positions),
        }

    def export_report(
        self,
        report: RiskReport,
        format: ReportFormat = ReportFormat.JSON,
        filename: str | None = None,
    ) -> str:
        """
        Export report to file.

        Returns the filepath of the exported report.
        """
        if filename is None:
            filename = f"{report.report_id}.{format.value}"

        filepath = self.output_dir / filename

        if format == ReportFormat.JSON:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        elif format == ReportFormat.CSV:
            self._export_csv(report, filepath)

        elif format == ReportFormat.HTML:
            self._export_html(report, filepath)

        elif format == ReportFormat.TEXT:
            self._export_text(report, filepath)

        logger.info(f"Exported report to {filepath}")
        return str(filepath)

    def _export_csv(self, report: RiskReport, filepath: Path) -> None:
        """Export report to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Report ID', report.report_id])
            writer.writerow(['Type', report.report_type.value])
            writer.writerow(['Generated', report.generated_at.isoformat()])
            writer.writerow([])

            # Sections
            for section in report.sections:
                writer.writerow([section.title])
                writer.writerow([section.summary])

                if isinstance(section.data, list) and section.data:
                    # Table format
                    headers = section.data[0].keys()
                    writer.writerow(headers)
                    for row in section.data:
                        writer.writerow(row.values())
                elif isinstance(section.data, dict):
                    for key, value in section.data.items():
                        writer.writerow([key, value])

                writer.writerow([])

    def _export_html(self, report: RiskReport, filepath: Path) -> None:
        """Export report to HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .alert {{ background-color: #ffcccc; padding: 10px; margin: 10px 0; border-left: 4px solid #ff0000; }}
        .summary {{ font-weight: bold; margin: 10px 0; }}
        .metadata {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p class="metadata">
        Generated: {report.generated_at.isoformat()}<br>
        Report ID: {report.report_id}
    </p>
"""

        # Alerts at top
        if report.alerts:
            html += "<div class='alert'><strong>Alerts:</strong><ul>"
            for alert in report.alerts:
                html += f"<li>{alert}</li>"
            html += "</ul></div>"

        # Sections
        for section in report.sections:
            html += f"<h2>{section.title}</h2>"
            html += f"<p class='summary'>{section.summary}</p>"

            if isinstance(section.data, list) and section.data:
                html += "<table><tr>"
                for key in section.data[0].keys():
                    html += f"<th>{key}</th>"
                html += "</tr>"
                for row in section.data:
                    html += "<tr>"
                    for value in row.values():
                        html += f"<td>{value}</td>"
                    html += "</tr>"
                html += "</table>"
            elif isinstance(section.data, dict):
                html += "<table>"
                for key, value in section.data.items():
                    html += f"<tr><th>{key}</th><td>{value}</td></tr>"
                html += "</table>"

        html += "</body></html>"

        with open(filepath, 'w') as f:
            f.write(html)

    def _export_text(self, report: RiskReport, filepath: Path) -> None:
        """Export report to plain text."""
        lines = [
            "=" * 60,
            report.title,
            "=" * 60,
            f"Generated: {report.generated_at.isoformat()}",
            f"Report ID: {report.report_id}",
            "",
        ]

        if report.alerts:
            lines.append("ALERTS:")
            for alert in report.alerts:
                lines.append(f"  ! {alert}")
            lines.append("")

        for section in report.sections:
            lines.append("-" * 40)
            lines.append(section.title)
            lines.append("-" * 40)
            lines.append(section.summary)
            lines.append("")

            if isinstance(section.data, dict):
                for key, value in section.data.items():
                    lines.append(f"  {key}: {value}")
            elif isinstance(section.data, list):
                for item in section.data[:10]:  # Limit for text
                    lines.append(f"  {item}")

            lines.append("")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))


class ScheduledReportManager:
    """Manages scheduled report generation."""

    def __init__(self, generator: RiskReportGenerator):
        self.generator = generator
        self._last_daily_report: datetime | None = None

    def should_generate_daily_report(self) -> bool:
        """Check if daily report should be generated."""
        now = datetime.now(timezone.utc)

        if self._last_daily_report is None:
            return True

        # Generate if last report was from previous day
        return self._last_daily_report.date() < now.date()

    def mark_daily_report_generated(self) -> None:
        """Mark that daily report was generated."""
        self._last_daily_report = datetime.now(timezone.utc)

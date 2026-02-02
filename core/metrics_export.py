"""
Strategy Performance Metrics Export
===================================

Addresses issue #Q23: No strategy performance metrics export.

Features:
- Export performance metrics to multiple formats (JSON, CSV, HTML, Excel)
- Standardized metric definitions
- Time-series export support
- Compliance-ready reporting formats
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from io import StringIO

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    EXCEL_CSV = "excel_csv"  # Excel-compatible CSV
    MARKDOWN = "markdown"


class MetricCategory(str, Enum):
    """Categories for organizing metrics."""
    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    TRADING = "trading"
    EXECUTION = "execution"
    EXPOSURE = "exposure"


@dataclass
class MetricDefinition:
    """
    Definition of a performance metric.

    Provides standardized descriptions and formatting for compliance reporting.
    """
    name: str
    display_name: str
    category: MetricCategory
    description: str
    unit: str  # "percent", "ratio", "currency", "count", "days"
    decimal_places: int = 2
    is_higher_better: bool = True

    def format_value(self, value: float) -> str:
        """Format value according to metric definition."""
        if value is None:
            return "N/A"

        if self.unit == "percent":
            return f"{value * 100:.{self.decimal_places}f}%"
        elif self.unit == "currency":
            return f"${value:,.{self.decimal_places}f}"
        elif self.unit == "count":
            return f"{int(value):,}"
        elif self.unit == "days":
            return f"{value:.0f} days"
        else:
            return f"{value:.{self.decimal_places}f}"


# Standard metric definitions
METRIC_DEFINITIONS = {
    "total_return": MetricDefinition(
        name="total_return",
        display_name="Total Return",
        category=MetricCategory.RETURN,
        description="Total return over the analysis period",
        unit="percent",
    ),
    "annualized_return": MetricDefinition(
        name="annualized_return",
        display_name="Annualized Return",
        category=MetricCategory.RETURN,
        description="Return annualized to yearly basis",
        unit="percent",
    ),
    "cagr": MetricDefinition(
        name="cagr",
        display_name="CAGR",
        category=MetricCategory.RETURN,
        description="Compound Annual Growth Rate",
        unit="percent",
    ),
    "volatility": MetricDefinition(
        name="volatility",
        display_name="Volatility",
        category=MetricCategory.RISK,
        description="Annualized standard deviation of returns",
        unit="percent",
        is_higher_better=False,
    ),
    "max_drawdown": MetricDefinition(
        name="max_drawdown",
        display_name="Maximum Drawdown",
        category=MetricCategory.RISK,
        description="Largest peak-to-trough decline",
        unit="percent",
        is_higher_better=False,
    ),
    "var_95": MetricDefinition(
        name="var_95",
        display_name="VaR (95%)",
        category=MetricCategory.RISK,
        description="Value at Risk at 95% confidence level",
        unit="percent",
        is_higher_better=False,
    ),
    "cvar_95": MetricDefinition(
        name="cvar_95",
        display_name="CVaR (95%)",
        category=MetricCategory.RISK,
        description="Conditional Value at Risk (Expected Shortfall)",
        unit="percent",
        is_higher_better=False,
    ),
    "sharpe_ratio": MetricDefinition(
        name="sharpe_ratio",
        display_name="Sharpe Ratio",
        category=MetricCategory.RISK_ADJUSTED,
        description="Risk-adjusted return using volatility",
        unit="ratio",
    ),
    "sortino_ratio": MetricDefinition(
        name="sortino_ratio",
        display_name="Sortino Ratio",
        category=MetricCategory.RISK_ADJUSTED,
        description="Risk-adjusted return using downside deviation",
        unit="ratio",
    ),
    "calmar_ratio": MetricDefinition(
        name="calmar_ratio",
        display_name="Calmar Ratio",
        category=MetricCategory.RISK_ADJUSTED,
        description="Return relative to maximum drawdown",
        unit="ratio",
    ),
    "information_ratio": MetricDefinition(
        name="information_ratio",
        display_name="Information Ratio",
        category=MetricCategory.RISK_ADJUSTED,
        description="Active return relative to tracking error",
        unit="ratio",
    ),
    "total_trades": MetricDefinition(
        name="total_trades",
        display_name="Total Trades",
        category=MetricCategory.TRADING,
        description="Number of trades executed",
        unit="count",
    ),
    "win_rate": MetricDefinition(
        name="win_rate",
        display_name="Win Rate",
        category=MetricCategory.TRADING,
        description="Percentage of winning trades",
        unit="percent",
    ),
    "profit_factor": MetricDefinition(
        name="profit_factor",
        display_name="Profit Factor",
        category=MetricCategory.TRADING,
        description="Gross profit divided by gross loss",
        unit="ratio",
    ),
    "avg_win": MetricDefinition(
        name="avg_win",
        display_name="Average Win",
        category=MetricCategory.TRADING,
        description="Average profit on winning trades",
        unit="currency",
    ),
    "avg_loss": MetricDefinition(
        name="avg_loss",
        display_name="Average Loss",
        category=MetricCategory.TRADING,
        description="Average loss on losing trades",
        unit="currency",
        is_higher_better=False,
    ),
    "total_costs": MetricDefinition(
        name="total_costs",
        display_name="Total Transaction Costs",
        category=MetricCategory.EXECUTION,
        description="Total commissions and slippage",
        unit="currency",
        is_higher_better=False,
    ),
    "avg_exposure": MetricDefinition(
        name="avg_exposure",
        display_name="Average Exposure",
        category=MetricCategory.EXPOSURE,
        description="Average portfolio exposure",
        unit="percent",
    ),
    "time_in_market": MetricDefinition(
        name="time_in_market",
        display_name="Time in Market",
        category=MetricCategory.EXPOSURE,
        description="Percentage of time with active positions",
        unit="percent",
    ),
}


@dataclass
class PerformanceSnapshot:
    """
    Snapshot of strategy performance at a point in time.

    Supports time-series export of performance metrics.
    """
    timestamp: datetime
    strategy_name: str
    metrics: dict[str, float]
    positions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "metrics": self.metrics,
            "positions": self.positions,
            "metadata": self.metadata,
        }


class PerformanceMetricsExporter:
    """
    Exports strategy performance metrics to various formats (#Q23).

    Supports:
    - Single snapshot export
    - Time-series export
    - Multi-strategy comparison
    - Compliance-ready formats
    """

    def __init__(
        self,
        output_dir: str = "reports/performance",
        include_definitions: bool = True,
    ):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for exported files
            include_definitions: Include metric definitions in exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_definitions = include_definitions
        self._snapshots: list[PerformanceSnapshot] = []

    def add_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Add performance snapshot for time-series export."""
        self._snapshots.append(snapshot)

    def export_metrics(
        self,
        metrics: dict[str, float],
        strategy_name: str,
        format: ExportFormat = ExportFormat.JSON,
        filename: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Export metrics to file.

        Args:
            metrics: Dictionary of metric name to value
            strategy_name: Name of the strategy
            format: Export format
            filename: Optional custom filename
            metadata: Additional metadata to include

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_{timestamp}.{format.value}"

        filepath = self.output_dir / filename

        if format == ExportFormat.JSON:
            self._export_json(metrics, strategy_name, filepath, metadata)
        elif format == ExportFormat.CSV:
            self._export_csv(metrics, strategy_name, filepath, metadata)
        elif format == ExportFormat.HTML:
            self._export_html(metrics, strategy_name, filepath, metadata)
        elif format == ExportFormat.EXCEL_CSV:
            self._export_excel_csv(metrics, strategy_name, filepath, metadata)
        elif format == ExportFormat.MARKDOWN:
            self._export_markdown(metrics, strategy_name, filepath, metadata)

        logger.info(f"Exported metrics to {filepath}")
        return str(filepath)

    def _export_json(
        self,
        metrics: dict[str, float],
        strategy_name: str,
        filepath: Path,
        metadata: dict | None,
    ) -> None:
        """Export to JSON format."""
        export_data = {
            "strategy_name": strategy_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
        }

        if metadata:
            export_data["metadata"] = metadata

        # Add metrics with definitions
        for name, value in metrics.items():
            definition = METRIC_DEFINITIONS.get(name)
            metric_data = {"value": value}

            if definition and self.include_definitions:
                metric_data.update({
                    "display_name": definition.display_name,
                    "category": definition.category.value,
                    "description": definition.description,
                    "formatted_value": definition.format_value(value),
                })

            export_data["metrics"][name] = metric_data

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    def _export_csv(
        self,
        metrics: dict[str, float],
        strategy_name: str,
        filepath: Path,
        metadata: dict | None,
    ) -> None:
        """Export to CSV format."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["Strategy", strategy_name])
            writer.writerow(["Generated", datetime.now(timezone.utc).isoformat()])
            writer.writerow([])

            # Metrics by category
            by_category = {}
            for name, value in metrics.items():
                definition = METRIC_DEFINITIONS.get(name)
                category = definition.category.value if definition else "other"
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append((name, value, definition))

            for category, items in by_category.items():
                writer.writerow([category.upper()])
                writer.writerow(["Metric", "Value", "Formatted", "Description"])
                for name, value, definition in items:
                    formatted = definition.format_value(value) if definition else str(value)
                    description = definition.description if definition else ""
                    writer.writerow([name, value, formatted, description])
                writer.writerow([])

    def _export_excel_csv(
        self,
        metrics: dict[str, float],
        strategy_name: str,
        filepath: Path,
        metadata: dict | None,
    ) -> None:
        """Export to Excel-compatible CSV format."""
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # Excel-friendly header with BOM
            writer.writerow(["Performance Report"])
            writer.writerow(["Strategy:", strategy_name])
            writer.writerow(["Date:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")])
            writer.writerow([])

            # Main metrics table
            writer.writerow(["Metric Name", "Value", "Unit", "Category"])

            for name, value in sorted(metrics.items()):
                definition = METRIC_DEFINITIONS.get(name)
                if definition:
                    writer.writerow([
                        definition.display_name,
                        definition.format_value(value),
                        definition.unit,
                        definition.category.value,
                    ])
                else:
                    writer.writerow([name, value, "", ""])

    def _export_html(
        self,
        metrics: dict[str, float],
        strategy_name: str,
        filepath: Path,
        metadata: dict | None,
    ) -> None:
        """Export to HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Performance Report - {strategy_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
            margin-top: 30px;
        }}
        .metadata {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .positive {{
            color: #4CAF50;
        }}
        .negative {{
            color: #f44336;
        }}
        .category-header {{
            background-color: #e0e0e0;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Report: {strategy_name}</h1>
        <div class="metadata">
            Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
        </div>
"""

        # Group by category
        by_category = {}
        for name, value in metrics.items():
            definition = METRIC_DEFINITIONS.get(name)
            category = definition.category if definition else MetricCategory.RETURN
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((name, value, definition))

        for category in MetricCategory:
            if category not in by_category:
                continue

            items = by_category[category]
            html += f"""
        <h2>{category.value.replace('_', ' ').title()} Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
"""

            for name, value, definition in items:
                if definition:
                    formatted = definition.format_value(value)
                    description = definition.description
                    display_name = definition.display_name

                    # Color coding
                    is_good = (definition.is_higher_better and value > 0) or \
                              (not definition.is_higher_better and value < 0)
                    css_class = "positive" if is_good else "negative" if value != 0 else ""
                else:
                    formatted = str(value)
                    description = ""
                    display_name = name
                    css_class = ""

                html += f"""
            <tr>
                <td>{display_name}</td>
                <td class="{css_class}">{formatted}</td>
                <td>{description}</td>
            </tr>
"""

            html += """
        </table>
"""

        html += """
    </div>
</body>
</html>
"""

        with open(filepath, "w") as f:
            f.write(html)

    def _export_markdown(
        self,
        metrics: dict[str, float],
        strategy_name: str,
        filepath: Path,
        metadata: dict | None,
    ) -> None:
        """Export to Markdown format."""
        lines = [
            f"# Performance Report: {strategy_name}",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
        ]

        # Group by category
        by_category = {}
        for name, value in metrics.items():
            definition = METRIC_DEFINITIONS.get(name)
            category = definition.category if definition else MetricCategory.RETURN
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((name, value, definition))

        for category in MetricCategory:
            if category not in by_category:
                continue

            items = by_category[category]
            lines.append(f"## {category.value.replace('_', ' ').title()} Metrics")
            lines.append("")
            lines.append("| Metric | Value | Description |")
            lines.append("|--------|-------|-------------|")

            for name, value, definition in items:
                if definition:
                    formatted = definition.format_value(value)
                    description = definition.description
                    display_name = definition.display_name
                else:
                    formatted = str(value)
                    description = ""
                    display_name = name

                lines.append(f"| {display_name} | {formatted} | {description} |")

            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def export_time_series(
        self,
        format: ExportFormat = ExportFormat.CSV,
        filename: str | None = None,
        metrics_to_include: list[str] | None = None,
    ) -> str:
        """
        Export time-series of performance snapshots.

        Args:
            format: Export format
            filename: Optional custom filename
            metrics_to_include: Specific metrics to include (default: all)

        Returns:
            Path to exported file
        """
        if not self._snapshots:
            raise ValueError("No snapshots to export")

        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"performance_timeseries_{timestamp}.{format.value}"

        filepath = self.output_dir / filename

        # Determine metrics to include
        if metrics_to_include is None:
            all_metrics = set()
            for snapshot in self._snapshots:
                all_metrics.update(snapshot.metrics.keys())
            metrics_to_include = sorted(all_metrics)

        if format == ExportFormat.CSV:
            self._export_timeseries_csv(filepath, metrics_to_include)
        elif format == ExportFormat.JSON:
            self._export_timeseries_json(filepath, metrics_to_include)
        else:
            raise ValueError(f"Time series export not supported for {format}")

        logger.info(f"Exported time series to {filepath}")
        return str(filepath)

    def _export_timeseries_csv(
        self,
        filepath: Path,
        metrics: list[str],
    ) -> None:
        """Export time series to CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["timestamp", "strategy"] + metrics
            writer.writerow(header)

            # Data rows
            for snapshot in sorted(self._snapshots, key=lambda s: s.timestamp):
                row = [
                    snapshot.timestamp.isoformat(),
                    snapshot.strategy_name,
                ]
                for metric in metrics:
                    row.append(snapshot.metrics.get(metric, ""))
                writer.writerow(row)

    def _export_timeseries_json(
        self,
        filepath: Path,
        metrics: list[str],
    ) -> None:
        """Export time series to JSON."""
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics_included": metrics,
            "snapshots": [s.to_dict() for s in self._snapshots],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def export_comparison(
        self,
        strategies: dict[str, dict[str, float]],
        format: ExportFormat = ExportFormat.HTML,
        filename: str | None = None,
    ) -> str:
        """
        Export multi-strategy comparison.

        Args:
            strategies: Dict mapping strategy name to metrics dict
            format: Export format
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_comparison_{timestamp}.{format.value}"

        filepath = self.output_dir / filename

        if format == ExportFormat.HTML:
            self._export_comparison_html(strategies, filepath)
        elif format == ExportFormat.CSV:
            self._export_comparison_csv(strategies, filepath)
        elif format == ExportFormat.MARKDOWN:
            self._export_comparison_markdown(strategies, filepath)
        else:
            raise ValueError(f"Comparison export not supported for {format}")

        logger.info(f"Exported comparison to {filepath}")
        return str(filepath)

    def _export_comparison_html(
        self,
        strategies: dict[str, dict[str, float]],
        filepath: Path,
    ) -> None:
        """Export comparison to HTML."""
        # Get all unique metrics
        all_metrics = set()
        for metrics in strategies.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: right; }}
        th {{ background-color: #4CAF50; color: white; }}
        td:first-child {{ text-align: left; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .best {{ background-color: #c8e6c9; }}
        .worst {{ background-color: #ffcdd2; }}
    </style>
</head>
<body>
    <h1>Strategy Comparison</h1>
    <p>Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
    <table>
        <tr>
            <th>Metric</th>
"""

        for strategy_name in strategies.keys():
            html += f"            <th>{strategy_name}</th>\n"

        html += "        </tr>\n"

        for metric in all_metrics:
            definition = METRIC_DEFINITIONS.get(metric)
            display_name = definition.display_name if definition else metric

            values = []
            for strategy_name in strategies.keys():
                value = strategies[strategy_name].get(metric)
                values.append(value)

            html += f"        <tr>\n            <td>{display_name}</td>\n"

            # Find best/worst
            numeric_values = [v for v in values if v is not None]
            if numeric_values and definition:
                if definition.is_higher_better:
                    best_val = max(numeric_values)
                    worst_val = min(numeric_values)
                else:
                    best_val = min(numeric_values)
                    worst_val = max(numeric_values)
            else:
                best_val = worst_val = None

            for value in values:
                if value is not None:
                    formatted = definition.format_value(value) if definition else str(value)
                    css_class = ""
                    if value == best_val and len(numeric_values) > 1:
                        css_class = "best"
                    elif value == worst_val and len(numeric_values) > 1:
                        css_class = "worst"
                    html += f'            <td class="{css_class}">{formatted}</td>\n'
                else:
                    html += "            <td>N/A</td>\n"

            html += "        </tr>\n"

        html += """
    </table>
</body>
</html>
"""

        with open(filepath, "w") as f:
            f.write(html)

    def _export_comparison_csv(
        self,
        strategies: dict[str, dict[str, float]],
        filepath: Path,
    ) -> None:
        """Export comparison to CSV."""
        all_metrics = set()
        for metrics in strategies.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["Metric"] + list(strategies.keys())
            writer.writerow(header)

            # Data
            for metric in all_metrics:
                row = [metric]
                for strategy_name in strategies.keys():
                    value = strategies[strategy_name].get(metric, "")
                    row.append(value)
                writer.writerow(row)

    def _export_comparison_markdown(
        self,
        strategies: dict[str, dict[str, float]],
        filepath: Path,
    ) -> None:
        """Export comparison to Markdown."""
        all_metrics = set()
        for metrics in strategies.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        strategy_names = list(strategies.keys())

        lines = [
            "# Strategy Comparison",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "| Metric | " + " | ".join(strategy_names) + " |",
            "|--------|" + "|".join(["-------" for _ in strategy_names]) + "|",
        ]

        for metric in all_metrics:
            definition = METRIC_DEFINITIONS.get(metric)
            display_name = definition.display_name if definition else metric

            row = [display_name]
            for strategy_name in strategy_names:
                value = strategies[strategy_name].get(metric)
                if value is not None:
                    formatted = definition.format_value(value) if definition else str(value)
                else:
                    formatted = "N/A"
                row.append(formatted)

            lines.append("| " + " | ".join(row) + " |")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

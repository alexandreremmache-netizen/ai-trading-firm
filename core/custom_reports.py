"""
Custom Reporting Module
=======================

Flexible custom reporting framework for portfolio management.

Issues Addressed:
- #P19: Missing custom reporting
- #E30: No execution venue selection logic (integrated here)
"""

from __future__ import annotations

import logging
import json
import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM REPORTING FRAMEWORK (#P19)
# =============================================================================

class ReportFormat(str, Enum):
    """Output format for reports."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "text"


class ReportFrequency(str, Enum):
    """Report generation frequency."""
    REALTIME = "realtime"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"


class ReportCategory(str, Enum):
    """Report category."""
    PERFORMANCE = "performance"
    RISK = "risk"
    COMPLIANCE = "compliance"
    EXECUTION = "execution"
    POSITION = "position"
    ATTRIBUTION = "attribution"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


@dataclass
class ReportDefinition:
    """Definition of a custom report (#P19)."""
    report_id: str
    name: str
    description: str
    category: ReportCategory
    frequency: ReportFrequency
    formats: list[ReportFormat]
    data_sources: list[str]
    filters: dict[str, Any]
    columns: list[dict]  # {name, source, transform, format}
    grouping: list[str] | None
    sorting: list[dict] | None  # {column, direction}
    aggregations: list[dict] | None  # {column, function}
    created_by: str
    created_at: datetime
    last_modified: datetime
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            'report_id': self.report_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'frequency': self.frequency.value,
            'formats': [f.value for f in self.formats],
            'data_sources': self.data_sources,
            'filters': self.filters,
            'columns': self.columns,
            'grouping': self.grouping,
            'sorting': self.sorting,
            'aggregations': self.aggregations,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'is_active': self.is_active,
        }


@dataclass
class ReportExecution:
    """Record of report execution (#P19)."""
    execution_id: str
    report_id: str
    started_at: datetime
    completed_at: datetime | None
    status: str  # "running", "completed", "failed"
    row_count: int
    output_format: ReportFormat
    output_path: str | None
    error_message: str | None

    def to_dict(self) -> dict:
        return {
            'execution_id': self.execution_id,
            'report_id': self.report_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'row_count': self.row_count,
            'output_format': self.output_format.value,
            'output_path': self.output_path,
            'error_message': self.error_message,
        }


class CustomReportBuilder:
    """
    Builder for custom reports (#P19).

    Allows defining custom reports with:
    - Multiple data sources
    - Custom filters and transformations
    - Flexible column definitions
    - Grouping and aggregation
    - Multiple output formats
    """

    def __init__(self, report_name: str, category: ReportCategory = ReportCategory.CUSTOM):
        self._report_name = report_name
        self._category = category
        self._description = ""
        self._frequency = ReportFrequency.ON_DEMAND
        self._formats = [ReportFormat.JSON]
        self._data_sources: list[str] = []
        self._filters: dict[str, Any] = {}
        self._columns: list[dict] = []
        self._grouping: list[str] | None = None
        self._sorting: list[dict] | None = None
        self._aggregations: list[dict] | None = None

    def with_description(self, description: str) -> "CustomReportBuilder":
        """Set report description."""
        self._description = description
        return self

    def with_frequency(self, frequency: ReportFrequency) -> "CustomReportBuilder":
        """Set report frequency."""
        self._frequency = frequency
        return self

    def with_formats(self, formats: list[ReportFormat]) -> "CustomReportBuilder":
        """Set output formats."""
        self._formats = formats
        return self

    def add_data_source(self, source: str) -> "CustomReportBuilder":
        """Add data source."""
        self._data_sources.append(source)
        return self

    def add_filter(self, field: str, operator: str, value: Any) -> "CustomReportBuilder":
        """Add filter condition."""
        self._filters[field] = {'operator': operator, 'value': value}
        return self

    def add_column(
        self,
        name: str,
        source: str,
        transform: str | None = None,
        format_spec: str | None = None,
    ) -> "CustomReportBuilder":
        """Add column definition."""
        self._columns.append({
            'name': name,
            'source': source,
            'transform': transform,
            'format': format_spec,
        })
        return self

    def with_grouping(self, fields: list[str]) -> "CustomReportBuilder":
        """Set grouping fields."""
        self._grouping = fields
        return self

    def with_sorting(self, column: str, direction: str = "asc") -> "CustomReportBuilder":
        """Add sorting."""
        if self._sorting is None:
            self._sorting = []
        self._sorting.append({'column': column, 'direction': direction})
        return self

    def with_aggregation(self, column: str, function: str) -> "CustomReportBuilder":
        """Add aggregation."""
        if self._aggregations is None:
            self._aggregations = []
        self._aggregations.append({'column': column, 'function': function})
        return self

    def build(self, created_by: str) -> ReportDefinition:
        """Build report definition."""
        report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return ReportDefinition(
            report_id=report_id,
            name=self._report_name,
            description=self._description,
            category=self._category,
            frequency=self._frequency,
            formats=self._formats,
            data_sources=self._data_sources,
            filters=self._filters,
            columns=self._columns,
            grouping=self._grouping,
            sorting=self._sorting,
            aggregations=self._aggregations,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
        )


class CustomReportEngine:
    """
    Custom report execution engine (#P19).

    Features:
    - Report definition management
    - Data source integration
    - Report execution and scheduling
    - Multiple output formats
    - Execution history
    """

    def __init__(self):
        self._reports: dict[str, ReportDefinition] = {}
        self._executions: list[ReportExecution] = []
        self._execution_counter = 0

        # Data source handlers
        self._data_handlers: dict[str, Callable] = {}

        # Transform functions
        self._transforms: dict[str, Callable] = {
            'abs': abs,
            'round2': lambda x: round(x, 2) if isinstance(x, (int, float)) else x,
            'round4': lambda x: round(x, 4) if isinstance(x, (int, float)) else x,
            'pct': lambda x: x * 100 if isinstance(x, (int, float)) else x,
            'upper': lambda x: str(x).upper(),
            'lower': lambda x: str(x).lower(),
        }

        # Format functions
        self._formatters: dict[str, Callable] = {
            'currency': lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x),
            'percent': lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x),
            'number': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'decimal2': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x),
            'decimal4': lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x),
        }

        # Aggregation functions
        self._aggregators: dict[str, Callable] = {
            'sum': sum,
            'avg': lambda x: sum(x) / len(x) if x else 0,
            'min': min,
            'max': max,
            'count': len,
            'first': lambda x: x[0] if x else None,
            'last': lambda x: x[-1] if x else None,
        }

    def register_data_handler(
        self,
        source_name: str,
        handler: Callable[[], list[dict]],
    ) -> None:
        """Register data source handler."""
        self._data_handlers[source_name] = handler

    def register_transform(self, name: str, func: Callable) -> None:
        """Register custom transform function."""
        self._transforms[name] = func

    def save_report(self, report: ReportDefinition) -> None:
        """Save report definition."""
        self._reports[report.report_id] = report
        logger.info(f"Report saved: {report.report_id} - {report.name}")

    def get_report(self, report_id: str) -> ReportDefinition | None:
        """Get report definition."""
        return self._reports.get(report_id)

    def list_reports(
        self,
        category: ReportCategory | None = None,
    ) -> list[ReportDefinition]:
        """List available reports."""
        reports = list(self._reports.values())
        if category:
            reports = [r for r in reports if r.category == category]
        return reports

    def _fetch_data(self, data_sources: list[str]) -> list[dict]:
        """Fetch data from sources."""
        all_data = []
        for source in data_sources:
            handler = self._data_handlers.get(source)
            if handler:
                try:
                    data = handler()
                    all_data.extend(data)
                except Exception as e:
                    logger.error(f"Error fetching from {source}: {e}")
        return all_data

    def _apply_filters(
        self,
        data: list[dict],
        filters: dict[str, Any],
    ) -> list[dict]:
        """Apply filters to data."""
        if not filters:
            return data

        filtered = []
        for row in data:
            include = True
            for field, condition in filters.items():
                value = row.get(field)
                op = condition.get('operator', '=')
                target = condition.get('value')

                if op == '=' and value != target:
                    include = False
                elif op == '!=' and value == target:
                    include = False
                elif op == '>' and not (isinstance(value, (int, float)) and value > target):
                    include = False
                elif op == '<' and not (isinstance(value, (int, float)) and value < target):
                    include = False
                elif op == '>=' and not (isinstance(value, (int, float)) and value >= target):
                    include = False
                elif op == '<=' and not (isinstance(value, (int, float)) and value <= target):
                    include = False
                elif op == 'in' and value not in target:
                    include = False
                elif op == 'contains' and target not in str(value):
                    include = False

            if include:
                filtered.append(row)

        return filtered

    def _apply_columns(
        self,
        data: list[dict],
        columns: list[dict],
    ) -> list[dict]:
        """Apply column definitions to data."""
        result = []
        for row in data:
            new_row = {}
            for col in columns:
                name = col['name']
                source = col['source']
                transform = col.get('transform')
                format_spec = col.get('format')

                # Get value
                value = row.get(source)

                # Apply transform
                if transform and transform in self._transforms:
                    value = self._transforms[transform](value)

                # Store raw value (formatting applied at output)
                new_row[name] = value
                if format_spec:
                    new_row[f'_format_{name}'] = format_spec

            result.append(new_row)

        return result

    def _apply_grouping_and_aggregation(
        self,
        data: list[dict],
        grouping: list[str] | None,
        aggregations: list[dict] | None,
    ) -> list[dict]:
        """Apply grouping and aggregation."""
        if not grouping or not aggregations:
            return data

        # Group data
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in data:
            key = tuple(row.get(g) for g in grouping)
            groups[key].append(row)

        # Aggregate
        result = []
        for key, group_rows in groups.items():
            new_row = {}

            # Include grouping fields
            for i, field in enumerate(grouping):
                new_row[field] = key[i]

            # Apply aggregations
            for agg in aggregations:
                col = agg['column']
                func_name = agg['function']
                func = self._aggregators.get(func_name)

                if func:
                    values = [r.get(col) for r in group_rows if r.get(col) is not None]
                    if values:
                        new_row[f"{col}_{func_name}"] = func(values)

            result.append(new_row)

        return result

    def _apply_sorting(
        self,
        data: list[dict],
        sorting: list[dict] | None,
    ) -> list[dict]:
        """Apply sorting to data."""
        if not sorting:
            return data

        for sort_spec in reversed(sorting):
            col = sort_spec['column']
            reverse = sort_spec.get('direction', 'asc').lower() == 'desc'
            data = sorted(data, key=lambda x: x.get(col, ''), reverse=reverse)

        return data

    def _format_output(
        self,
        data: list[dict],
        format: ReportFormat,
    ) -> str:
        """Format data for output."""
        if format == ReportFormat.JSON:
            return json.dumps(data, indent=2, default=str)

        elif format == ReportFormat.CSV:
            if not data:
                return ""
            output = io.StringIO()
            # Filter out format metadata columns
            fieldnames = [k for k in data[0].keys() if not k.startswith('_format_')]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                clean_row = {k: v for k, v in row.items() if not k.startswith('_format_')}
                writer.writerow(clean_row)
            return output.getvalue()

        elif format == ReportFormat.HTML:
            if not data:
                return "<table><tr><td>No data</td></tr></table>"

            fieldnames = [k for k in data[0].keys() if not k.startswith('_format_')]

            html = ["<table border='1' style='border-collapse: collapse;'>"]
            html.append("<tr>")
            for field in fieldnames:
                html.append(f"<th style='padding: 8px;'>{field}</th>")
            html.append("</tr>")

            for row in data:
                html.append("<tr>")
                for field in fieldnames:
                    value = row.get(field, '')
                    format_spec = row.get(f'_format_{field}')
                    if format_spec and format_spec in self._formatters:
                        value = self._formatters[format_spec](value)
                    html.append(f"<td style='padding: 8px;'>{value}</td>")
                html.append("</tr>")

            html.append("</table>")
            return "\n".join(html)

        else:  # TEXT
            if not data:
                return "No data"

            fieldnames = [k for k in data[0].keys() if not k.startswith('_format_')]
            lines = []
            lines.append(" | ".join(fieldnames))
            lines.append("-" * len(lines[0]))

            for row in data:
                values = []
                for field in fieldnames:
                    value = row.get(field, '')
                    format_spec = row.get(f'_format_{field}')
                    if format_spec and format_spec in self._formatters:
                        value = self._formatters[format_spec](value)
                    values.append(str(value))
                lines.append(" | ".join(values))

            return "\n".join(lines)

    def execute_report(
        self,
        report_id: str,
        output_format: ReportFormat | None = None,
        override_filters: dict | None = None,
    ) -> tuple[str, ReportExecution]:
        """
        Execute a report (#P19).

        Args:
            report_id: Report to execute
            output_format: Override default format
            override_filters: Additional filters

        Returns:
            (output_data, execution_record)
        """
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report not found: {report_id}")

        self._execution_counter += 1
        execution_id = f"EXEC-{self._execution_counter:06d}"

        execution = ReportExecution(
            execution_id=execution_id,
            report_id=report_id,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            status="running",
            row_count=0,
            output_format=output_format or report.formats[0],
            output_path=None,
            error_message=None,
        )

        try:
            # Fetch data
            data = self._fetch_data(report.data_sources)

            # Merge filters
            filters = dict(report.filters)
            if override_filters:
                filters.update(override_filters)

            # Apply filters
            data = self._apply_filters(data, filters)

            # Apply columns
            data = self._apply_columns(data, report.columns)

            # Apply grouping/aggregation
            data = self._apply_grouping_and_aggregation(
                data, report.grouping, report.aggregations
            )

            # Apply sorting
            data = self._apply_sorting(data, report.sorting)

            # Format output
            output = self._format_output(data, execution.output_format)

            # Update execution
            execution.completed_at = datetime.now(timezone.utc)
            execution.status = "completed"
            execution.row_count = len(data)

            logger.info(
                f"Report executed: {report.name} - {execution.row_count} rows"
            )

        except Exception as e:
            execution.completed_at = datetime.now(timezone.utc)
            execution.status = "failed"
            execution.error_message = str(e)
            output = ""
            logger.error(f"Report execution failed: {e}")

        self._executions.append(execution)
        return output, execution

    def get_execution_history(
        self,
        report_id: str | None = None,
        limit: int = 100,
    ) -> list[ReportExecution]:
        """Get report execution history."""
        history = self._executions
        if report_id:
            history = [e for e in history if e.report_id == report_id]
        return history[-limit:]


# =============================================================================
# EXECUTION VENUE SELECTION LOGIC (#E30)
# =============================================================================

class VenueType(str, Enum):
    """Execution venue types (#E30)."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ATS = "ats"
    INTERNALIZER = "internalizer"
    MARKET_MAKER = "market_maker"


class OrderCharacteristic(str, Enum):
    """Order characteristics for venue selection (#E30)."""
    SMALL = "small"  # < 1% ADV
    MEDIUM = "medium"  # 1-5% ADV
    LARGE = "large"  # 5-20% ADV
    BLOCK = "block"  # > 20% ADV
    URGENT = "urgent"
    PATIENT = "patient"
    PRICE_SENSITIVE = "price_sensitive"
    INFORMATION_SENSITIVE = "information_sensitive"


@dataclass
class VenueCharacteristics:
    """Characteristics of an execution venue (#E30)."""
    venue_id: str
    venue_type: VenueType
    name: str

    # Cost metrics
    maker_fee_bps: float
    taker_fee_bps: float
    typical_spread_bps: float

    # Quality metrics
    fill_rate_pct: float
    avg_fill_time_ms: float
    price_improvement_bps: float

    # Capacity
    avg_daily_volume: float
    max_order_size: float
    min_order_size: float

    # Features
    supports_midpoint: bool
    supports_hidden: bool
    supports_ioc: bool
    supports_block: bool

    # Restrictions
    restricted_symbols: list[str] = field(default_factory=list)
    min_notional: float = 0.0

    def to_dict(self) -> dict:
        return {
            'venue_id': self.venue_id,
            'venue_type': self.venue_type.value,
            'name': self.name,
            'maker_fee_bps': self.maker_fee_bps,
            'taker_fee_bps': self.taker_fee_bps,
            'typical_spread_bps': self.typical_spread_bps,
            'fill_rate_pct': self.fill_rate_pct,
            'avg_fill_time_ms': self.avg_fill_time_ms,
            'price_improvement_bps': self.price_improvement_bps,
            'avg_daily_volume': self.avg_daily_volume,
            'supports_midpoint': self.supports_midpoint,
            'supports_hidden': self.supports_hidden,
        }


@dataclass
class VenueSelectionResult:
    """Result of venue selection (#E30)."""
    symbol: str
    order_size: int
    order_characteristic: OrderCharacteristic
    recommended_venues: list[dict]  # [{venue_id, allocation_pct, reason}]
    estimated_cost_bps: float
    estimated_fill_time_ms: float
    routing_strategy: str
    considerations: list[str]

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'order_size': self.order_size,
            'order_characteristic': self.order_characteristic.value,
            'recommended_venues': self.recommended_venues,
            'estimated_cost_bps': self.estimated_cost_bps,
            'estimated_fill_time_ms': self.estimated_fill_time_ms,
            'routing_strategy': self.routing_strategy,
            'considerations': self.considerations,
        }


class ExecutionVenueSelector:
    """
    Execution venue selection logic (#E30).

    Features:
    - Venue characteristic analysis
    - Order-based venue matching
    - Cost optimization
    - Fill rate optimization
    - Information leakage mitigation
    """

    def __init__(self):
        self._venues: dict[str, VenueCharacteristics] = {}
        self._venue_performance: dict[str, list[dict]] = defaultdict(list)

        # Symbol-specific ADV
        self._symbol_adv: dict[str, float] = {}

    def register_venue(self, venue: VenueCharacteristics) -> None:
        """Register execution venue."""
        self._venues[venue.venue_id] = venue
        logger.info(f"Venue registered: {venue.venue_id} ({venue.venue_type.value})")

    def update_symbol_adv(self, symbol: str, adv: float) -> None:
        """Update average daily volume for symbol."""
        self._symbol_adv[symbol] = adv

    def classify_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        is_urgent: bool = False,
        is_info_sensitive: bool = False,
    ) -> OrderCharacteristic:
        """
        Classify order by characteristics (#E30).

        Args:
            symbol: Symbol
            quantity: Order quantity
            price: Current price
            is_urgent: Urgency flag
            is_info_sensitive: Information sensitivity

        Returns:
            Order characteristic
        """
        if is_urgent:
            return OrderCharacteristic.URGENT

        if is_info_sensitive:
            return OrderCharacteristic.INFORMATION_SENSITIVE

        # Size-based classification
        adv = self._symbol_adv.get(symbol, 1_000_000)
        order_value = quantity * price
        adv_pct = (quantity / adv) * 100

        if adv_pct < 1:
            return OrderCharacteristic.SMALL
        elif adv_pct < 5:
            return OrderCharacteristic.MEDIUM
        elif adv_pct < 20:
            return OrderCharacteristic.LARGE
        else:
            return OrderCharacteristic.BLOCK

    def _score_venue(
        self,
        venue: VenueCharacteristics,
        order_char: OrderCharacteristic,
        quantity: int,
        is_passive: bool,
    ) -> tuple[float, list[str]]:
        """
        Score venue for order (#E30).

        Returns: (score, reasons)
        """
        score = 100.0
        reasons = []

        # Cost scoring (lower is better)
        if is_passive:
            cost_bps = venue.maker_fee_bps - venue.price_improvement_bps
        else:
            cost_bps = venue.taker_fee_bps + venue.typical_spread_bps / 2

        cost_penalty = cost_bps * 2  # 2 points per bp
        score -= cost_penalty

        # Fill rate scoring
        fill_bonus = (venue.fill_rate_pct - 50) * 0.5  # Up to 25 points for 100% fill rate
        score += fill_bonus

        # Speed scoring (for urgent orders)
        if order_char == OrderCharacteristic.URGENT:
            if venue.avg_fill_time_ms < 10:
                score += 20
                reasons.append("fast_execution")
            elif venue.avg_fill_time_ms > 100:
                score -= 20

        # Information sensitivity scoring
        if order_char == OrderCharacteristic.INFORMATION_SENSITIVE:
            if venue.venue_type == VenueType.DARK_POOL:
                score += 30
                reasons.append("dark_pool_for_info_protection")
            elif venue.supports_hidden:
                score += 15
                reasons.append("hidden_order_support")

        # Block order scoring
        if order_char == OrderCharacteristic.BLOCK:
            if venue.supports_block:
                score += 25
                reasons.append("block_trading_support")
            if venue.venue_type == VenueType.DARK_POOL:
                score += 15
                reasons.append("dark_pool_for_block")

        # Capacity check
        if quantity > venue.max_order_size:
            score -= 50
            reasons.append("exceeds_max_size")
        elif quantity < venue.min_order_size:
            score -= 100  # Disqualify
            reasons.append("below_min_size")

        # Price improvement bonus
        if venue.price_improvement_bps > 0:
            score += venue.price_improvement_bps * 5
            reasons.append(f"price_improvement_{venue.price_improvement_bps:.1f}bps")

        return score, reasons

    def select_venues(
        self,
        symbol: str,
        quantity: int,
        price: float,
        is_passive: bool = True,
        is_urgent: bool = False,
        is_info_sensitive: bool = False,
        max_venues: int = 3,
    ) -> VenueSelectionResult:
        """
        Select optimal execution venues (#E30).

        Args:
            symbol: Symbol to trade
            quantity: Order quantity
            price: Current price
            is_passive: Passive (make) vs aggressive (take)
            is_urgent: Urgency flag
            is_info_sensitive: Information sensitivity
            max_venues: Maximum venues to use

        Returns:
            VenueSelectionResult with recommendations
        """
        order_char = self.classify_order(
            symbol, quantity, price, is_urgent, is_info_sensitive
        )

        # Score all venues
        venue_scores = []
        for venue in self._venues.values():
            # Skip restricted
            if symbol in venue.restricted_symbols:
                continue

            score, reasons = self._score_venue(venue, order_char, quantity, is_passive)
            if score > 0:
                venue_scores.append({
                    'venue': venue,
                    'score': score,
                    'reasons': reasons,
                })

        # Sort by score
        venue_scores.sort(key=lambda x: x['score'], reverse=True)

        # Select top venues and allocate
        selected = venue_scores[:max_venues]
        total_score = sum(v['score'] for v in selected)

        recommendations = []
        for vs in selected:
            allocation_pct = (vs['score'] / total_score * 100) if total_score > 0 else 0
            recommendations.append({
                'venue_id': vs['venue'].venue_id,
                'venue_name': vs['venue'].name,
                'venue_type': vs['venue'].venue_type.value,
                'allocation_pct': round(allocation_pct, 1),
                'score': round(vs['score'], 1),
                'reasons': vs['reasons'],
            })

        # Estimate overall metrics
        if selected:
            weights = [v['score'] / total_score for v in selected]
            est_cost = sum(
                w * (selected[i]['venue'].taker_fee_bps if not is_passive else selected[i]['venue'].maker_fee_bps)
                for i, w in enumerate(weights)
            )
            est_time = sum(
                w * selected[i]['venue'].avg_fill_time_ms
                for i, w in enumerate(weights)
            )
        else:
            est_cost = 0
            est_time = 0

        # Determine routing strategy
        if order_char == OrderCharacteristic.BLOCK:
            strategy = "block_crossing"
        elif order_char == OrderCharacteristic.URGENT:
            strategy = "immediate_or_cancel"
        elif order_char == OrderCharacteristic.INFORMATION_SENSITIVE:
            strategy = "dark_first"
        elif is_passive:
            strategy = "passive_rebate_capture"
        else:
            strategy = "smart_order_routing"

        # Considerations
        considerations = []
        if order_char == OrderCharacteristic.LARGE or order_char == OrderCharacteristic.BLOCK:
            considerations.append("Consider TWAP/VWAP algo to minimize market impact")
        if order_char == OrderCharacteristic.INFORMATION_SENSITIVE:
            considerations.append("Route to dark pools first to minimize information leakage")
        if not is_passive:
            considerations.append("Aggressive order - expect full spread cost")

        return VenueSelectionResult(
            symbol=symbol,
            order_size=quantity,
            order_characteristic=order_char,
            recommended_venues=recommendations,
            estimated_cost_bps=est_cost,
            estimated_fill_time_ms=est_time,
            routing_strategy=strategy,
            considerations=considerations,
        )

    def record_execution(
        self,
        venue_id: str,
        symbol: str,
        quantity: int,
        fill_pct: float,
        fill_time_ms: float,
        price_improvement_bps: float,
    ) -> None:
        """Record execution for venue performance tracking."""
        self._venue_performance[venue_id].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'quantity': quantity,
            'fill_pct': fill_pct,
            'fill_time_ms': fill_time_ms,
            'price_improvement_bps': price_improvement_bps,
        })

        # Trim to last 1000 executions
        self._venue_performance[venue_id] = self._venue_performance[venue_id][-1000:]

    def get_venue_statistics(self, venue_id: str) -> dict:
        """Get venue performance statistics."""
        executions = self._venue_performance.get(venue_id, [])
        if not executions:
            return {'venue_id': venue_id, 'no_data': True}

        fill_rates = [e['fill_pct'] for e in executions]
        fill_times = [e['fill_time_ms'] for e in executions]
        improvements = [e['price_improvement_bps'] for e in executions]

        return {
            'venue_id': venue_id,
            'execution_count': len(executions),
            'avg_fill_rate': sum(fill_rates) / len(fill_rates),
            'avg_fill_time_ms': sum(fill_times) / len(fill_times),
            'avg_price_improvement_bps': sum(improvements) / len(improvements),
        }

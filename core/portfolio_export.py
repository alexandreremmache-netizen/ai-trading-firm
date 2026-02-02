"""
Portfolio Export Module
=======================

Addresses issues:
- #P20: Portfolio metrics caching suboptimal
- #P21: No portfolio export formats (IBKR, etc.)

Features:
- Export portfolio to IBKR flex query format
- CSV/Excel export with multiple templates
- FIX protocol format
- Portfolio metrics caching with intelligent invalidation
"""

from __future__ import annotations

import csv
import json
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from collections import OrderedDict
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Portfolio export formats."""
    CSV = "csv"
    JSON = "json"
    IBKR_FLEX = "ibkr_flex"
    FIX = "fix"
    EXCEL_CSV = "excel_csv"


@dataclass
class PortfolioPosition:
    """Position for export."""
    symbol: str
    quantity: int
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    asset_class: str
    currency: str
    exchange: str | None = None
    conid: int | None = None  # IBKR contract ID
    account: str = ""
    sector: str = ""
    last_updated: datetime | None = None

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage."""
        cost_basis = abs(self.quantity * self.avg_cost)
        if cost_basis > 0:
            return self.unrealized_pnl / cost_basis * 100
        return 0.0


@dataclass
class PortfolioSummary:
    """Portfolio summary for export."""
    account_id: str
    as_of_date: datetime
    net_liquidation: float
    gross_position_value: float
    cash: float
    buying_power: float
    excess_liquidity: float
    maintenance_margin: float
    available_funds: float
    currency: str = "USD"
    positions: list[PortfolioPosition] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def position_count(self) -> int:
        """Number of positions."""
        return len(self.positions)

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions)


class IBKRFlexExporter:
    """
    Export to IBKR Flex Query format (#P21).

    Generates XML compatible with IBKR's reporting tools.
    """

    def __init__(self, account_id: str):
        self.account_id = account_id

    def export_positions(
        self,
        positions: list[PortfolioPosition],
        as_of_date: datetime | None = None,
    ) -> str:
        """
        Export positions to IBKR Flex XML format.

        Args:
            positions: List of positions
            as_of_date: Report date

        Returns:
            XML string
        """
        as_of_date = as_of_date or datetime.now(timezone.utc)

        root = ET.Element("FlexQueryResponse")
        root.set("queryName", "Portfolio Export")
        root.set("type", "AF")

        flex_statements = ET.SubElement(root, "FlexStatements")
        flex_statements.set("count", "1")

        statement = ET.SubElement(flex_statements, "FlexStatement")
        statement.set("accountId", self.account_id)
        statement.set("fromDate", as_of_date.strftime("%Y%m%d"))
        statement.set("toDate", as_of_date.strftime("%Y%m%d"))
        statement.set("period", "Daily")
        statement.set("whenGenerated", datetime.now(timezone.utc).strftime("%Y%m%d;%H%M%S"))

        # Open positions section
        open_positions = ET.SubElement(statement, "OpenPositions")

        for pos in positions:
            pos_elem = ET.SubElement(open_positions, "OpenPosition")
            pos_elem.set("accountId", self.account_id)
            pos_elem.set("acctAlias", "")
            pos_elem.set("model", "")
            pos_elem.set("currency", pos.currency)
            pos_elem.set("fxRateToBase", "1.0")
            pos_elem.set("assetCategory", self._map_asset_class(pos.asset_class))
            pos_elem.set("symbol", pos.symbol)
            pos_elem.set("description", pos.symbol)
            pos_elem.set("conid", str(pos.conid or 0))
            pos_elem.set("securityID", "")
            pos_elem.set("securityIDType", "")
            pos_elem.set("cusip", "")
            pos_elem.set("isin", "")
            pos_elem.set("listingExchange", pos.exchange or "SMART")
            pos_elem.set("multiplier", "1")
            pos_elem.set("reportDate", as_of_date.strftime("%Y%m%d"))
            pos_elem.set("position", str(pos.quantity))
            pos_elem.set("markPrice", f"{pos.market_price:.4f}")
            pos_elem.set("positionValue", f"{pos.market_value:.2f}")
            pos_elem.set("openPrice", f"{pos.avg_cost:.4f}")
            pos_elem.set("costBasisPrice", f"{pos.avg_cost:.4f}")
            pos_elem.set("costBasisMoney", f"{abs(pos.quantity * pos.avg_cost):.2f}")
            pos_elem.set("fifoPnlUnrealized", f"{pos.unrealized_pnl:.2f}")
            pos_elem.set("side", "Long" if pos.quantity > 0 else "Short")
            pos_elem.set("levelOfDetail", "SUMMARY")

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def export_account_info(self, summary: PortfolioSummary) -> str:
        """Export account information to IBKR format."""
        root = ET.Element("FlexQueryResponse")

        flex_statements = ET.SubElement(root, "FlexStatements")
        statement = ET.SubElement(flex_statements, "FlexStatement")
        statement.set("accountId", self.account_id)

        account_info = ET.SubElement(statement, "AccountInformation")
        account_info.set("accountId", self.account_id)
        account_info.set("dateOpened", "")
        account_info.set("currency", summary.currency)
        account_info.set("primaryEmail", "")

        # Cash report
        cash_report = ET.SubElement(statement, "CashReport")
        cash_elem = ET.SubElement(cash_report, "CashReportCurrency")
        cash_elem.set("currency", summary.currency)
        cash_elem.set("endingCash", f"{summary.cash:.2f}")
        cash_elem.set("endingSettledCash", f"{summary.cash:.2f}")

        # Net asset value
        nav = ET.SubElement(statement, "EquitySummaryInBase")
        nav_elem = ET.SubElement(nav, "EquitySummaryByReportDateInBase")
        nav_elem.set("total", f"{summary.net_liquidation:.2f}")
        nav_elem.set("cash", f"{summary.cash:.2f}")

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _map_asset_class(self, asset_class: str) -> str:
        """Map asset class to IBKR category."""
        mapping = {
            "stock": "STK",
            "equity": "STK",
            "option": "OPT",
            "future": "FUT",
            "forex": "CASH",
            "bond": "BOND",
            "etf": "STK",
        }
        return mapping.get(asset_class.lower(), "STK")


class CSVPortfolioExporter:
    """
    Export portfolio to CSV formats (#P21).

    Supports multiple templates for different use cases.
    """

    def __init__(self, delimiter: str = ","):
        self.delimiter = delimiter

    def export_positions(
        self,
        positions: list[PortfolioPosition],
        template: str = "standard",
    ) -> str:
        """
        Export positions to CSV.

        Templates:
        - standard: Basic position info
        - detailed: Full position data
        - pnl: P&L focused
        - reconciliation: For broker reconciliation
        """
        output = StringIO()

        if template == "standard":
            columns = ["Symbol", "Quantity", "Price", "Market Value", "P&L", "P&L %"]
            writer = csv.writer(output, delimiter=self.delimiter)
            writer.writerow(columns)

            for pos in positions:
                writer.writerow([
                    pos.symbol,
                    pos.quantity,
                    f"{pos.market_price:.4f}",
                    f"{pos.market_value:.2f}",
                    f"{pos.unrealized_pnl:.2f}",
                    f"{pos.pnl_pct:.2f}",
                ])

        elif template == "detailed":
            columns = [
                "Symbol", "Asset Class", "Currency", "Exchange",
                "Quantity", "Avg Cost", "Market Price", "Market Value",
                "Unrealized P&L", "Realized P&L", "P&L %", "Sector",
                "Account", "Last Updated"
            ]
            writer = csv.writer(output, delimiter=self.delimiter)
            writer.writerow(columns)

            for pos in positions:
                writer.writerow([
                    pos.symbol,
                    pos.asset_class,
                    pos.currency,
                    pos.exchange or "",
                    pos.quantity,
                    f"{pos.avg_cost:.4f}",
                    f"{pos.market_price:.4f}",
                    f"{pos.market_value:.2f}",
                    f"{pos.unrealized_pnl:.2f}",
                    f"{pos.realized_pnl:.2f}",
                    f"{pos.pnl_pct:.2f}",
                    pos.sector,
                    pos.account,
                    pos.last_updated.isoformat() if pos.last_updated else "",
                ])

        elif template == "pnl":
            columns = [
                "Symbol", "Quantity", "Cost Basis", "Market Value",
                "Unrealized P&L", "Realized P&L", "Total P&L", "P&L %"
            ]
            writer = csv.writer(output, delimiter=self.delimiter)
            writer.writerow(columns)

            for pos in positions:
                cost_basis = abs(pos.quantity * pos.avg_cost)
                total_pnl = pos.unrealized_pnl + pos.realized_pnl
                writer.writerow([
                    pos.symbol,
                    pos.quantity,
                    f"{cost_basis:.2f}",
                    f"{pos.market_value:.2f}",
                    f"{pos.unrealized_pnl:.2f}",
                    f"{pos.realized_pnl:.2f}",
                    f"{total_pnl:.2f}",
                    f"{pos.pnl_pct:.2f}",
                ])

        elif template == "reconciliation":
            columns = [
                "Account", "Symbol", "CONID", "Quantity",
                "Price", "Value", "Currency"
            ]
            writer = csv.writer(output, delimiter=self.delimiter)
            writer.writerow(columns)

            for pos in positions:
                writer.writerow([
                    pos.account,
                    pos.symbol,
                    pos.conid or "",
                    pos.quantity,
                    f"{pos.market_price:.6f}",
                    f"{pos.market_value:.2f}",
                    pos.currency,
                ])

        return output.getvalue()

    def export_summary(self, summary: PortfolioSummary) -> str:
        """Export portfolio summary to CSV."""
        output = StringIO()
        writer = csv.writer(output, delimiter=self.delimiter)

        writer.writerow(["Portfolio Summary"])
        writer.writerow(["Account ID", summary.account_id])
        writer.writerow(["As Of", summary.as_of_date.isoformat()])
        writer.writerow(["Currency", summary.currency])
        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Net Liquidation", f"{summary.net_liquidation:.2f}"])
        writer.writerow(["Gross Position Value", f"{summary.gross_position_value:.2f}"])
        writer.writerow(["Cash", f"{summary.cash:.2f}"])
        writer.writerow(["Buying Power", f"{summary.buying_power:.2f}"])
        writer.writerow(["Excess Liquidity", f"{summary.excess_liquidity:.2f}"])
        writer.writerow(["Maintenance Margin", f"{summary.maintenance_margin:.2f}"])
        writer.writerow(["Position Count", summary.position_count])
        writer.writerow(["Total Unrealized P&L", f"{summary.total_unrealized_pnl:.2f}"])

        return output.getvalue()


class FIXPortfolioExporter:
    """
    Export portfolio in FIX protocol format (#P21).

    Generates FIX-style messages for position reporting.
    """

    def __init__(self, sender_comp_id: str = "FIRM", target_comp_id: str = "BROKER"):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self._seq_num = 0

    def export_position_report(self, position: PortfolioPosition) -> str:
        """
        Generate FIX Position Report (AP) message.

        Args:
            position: Position to export

        Returns:
            FIX message string
        """
        self._seq_num += 1
        fields = [
            ("8", "FIX.4.4"),  # BeginString
            ("9", "0"),  # BodyLength (placeholder)
            ("35", "AP"),  # MsgType: Position Report
            ("49", self.sender_comp_id),  # SenderCompID
            ("56", self.target_comp_id),  # TargetCompID
            ("34", str(self._seq_num)),  # MsgSeqNum
            ("52", datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]),  # SendingTime
            ("721", f"POS{self._seq_num:08d}"),  # PosMaintRptID
            ("710", f"REQ{self._seq_num:08d}"),  # PosReqID
            ("724", "0"),  # PosReqType: Positions
            ("728", "0"),  # PosReqResult: Valid
            ("727", "2"),  # TotalNumPosReports
            ("325", "N"),  # UnsolicitedIndicator
            ("55", position.symbol),  # Symbol
            ("460", self._map_product(position.asset_class)),  # Product
            ("453", "0"),  # NoPartyIDs
            ("1", position.account),  # Account
            ("581", "1"),  # AccountType
            ("15", position.currency),  # Currency
            ("730", f"{position.market_price:.6f}"),  # SettlPrice
            ("731", "1"),  # SettlPriceType: Final
            ("734", f"{position.avg_cost:.6f}"),  # PriorSettlPrice
            ("702", "1"),  # NoPositions
            ("703", "TOT"),  # PosType: Total
            ("704", str(abs(position.quantity))),  # LongQty
            ("705", "0"),  # ShortQty
            ("706", "0"),  # PosQtyStatus
        ]

        # Build message body
        body = "\x01".join(f"{tag}={value}" for tag, value in fields[2:-1])
        body_length = len(body) + 1

        # Update body length
        fields[1] = ("9", str(body_length))

        # Calculate checksum
        full_msg = "\x01".join(f"{tag}={value}" for tag, value in fields)
        checksum = sum(ord(c) for c in full_msg + "\x01") % 256

        return full_msg + f"\x0110={checksum:03d}\x01"

    def _map_product(self, asset_class: str) -> str:
        """Map asset class to FIX product code."""
        mapping = {
            "stock": "4",  # EQUITY
            "equity": "4",
            "option": "2",  # OPTION
            "future": "1",  # COMMODITY
            "forex": "4",  # CURRENCY
        }
        return mapping.get(asset_class.lower(), "4")


# =========================================================================
# PORTFOLIO METRICS CACHING (#P20)
# =========================================================================

@dataclass
class CachedPortfolioMetrics:
    """Cached portfolio metrics."""
    nav: float
    total_pnl: float
    total_pnl_pct: float
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    cash: float
    leverage: float
    position_count: int
    sector_weights: dict[str, float]
    computed_at: datetime

    @property
    def age_seconds(self) -> float:
        """Age of cache in seconds."""
        return (datetime.now(timezone.utc) - self.computed_at).total_seconds()


class PortfolioMetricsCache:
    """
    Intelligent caching for portfolio metrics (#P20).

    Features:
    - Position-change aware invalidation
    - Tiered TTL based on metric volatility
    - Memory-efficient storage
    - Thread-safe access
    """

    # TTLs for different metric types (seconds)
    METRIC_TTLS = {
        "nav": 30,  # Net liquidation changes frequently
        "exposure": 60,  # Exposure relatively stable
        "pnl": 30,  # P&L changes with prices
        "sector_weights": 300,  # Sector weights stable
        "risk_metrics": 120,  # Risk metrics moderately stable
    }

    def __init__(self, default_ttl: float = 60.0):
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[Any, float, float]] = {}  # key -> (value, computed_at, ttl)
        self._lock = threading.RLock()
        self._position_version = 0
        self._last_position_change = time.time()

        # Stats
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get cached value if valid."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, computed_at, ttl = self._cache[key]

            # Check TTL
            if time.time() - computed_at > ttl:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        metric_type: str | None = None,
    ) -> None:
        """Store value in cache."""
        if ttl is None:
            if metric_type and metric_type in self.METRIC_TTLS:
                ttl = self.METRIC_TTLS[metric_type]
            else:
                ttl = self.default_ttl

        with self._lock:
            self._cache[key] = (value, time.time(), ttl)

    def invalidate(self, key: str) -> bool:
        """Invalidate specific key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern prefix."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(pattern)]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)

    def on_position_change(self) -> None:
        """Handle position change - invalidate relevant caches."""
        with self._lock:
            self._position_version += 1
            self._last_position_change = time.time()

            # Invalidate position-dependent metrics
            patterns = ["nav:", "exposure:", "pnl:", "sector:"]
            for pattern in patterns:
                self.invalidate_pattern(pattern)

    def on_price_update(self, symbols: list[str]) -> None:
        """Handle price update - selective invalidation."""
        with self._lock:
            # Invalidate P&L and NAV
            self.invalidate_pattern("nav:")
            self.invalidate_pattern("pnl:")

            # Symbol-specific invalidations
            for symbol in symbols:
                self.invalidate_pattern(f"position:{symbol}")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: float | None = None,
        metric_type: str | None = None,
    ) -> Any:
        """
        Get from cache or compute with double-check pattern.

        Uses double-check locking to prevent deadlock while avoiding
        duplicate computation by multiple threads.
        """
        # First check without holding lock during compute
        value = self.get(key)
        if value is not None:
            return value

        # Compute value outside of lock to prevent deadlock
        computed_value = compute_fn()

        # Double-check: another thread may have computed and cached
        with self._lock:
            # Check again under lock
            if key in self._cache:
                existing_value, computed_at, cached_ttl = self._cache[key]
                # Check if cached value is still valid
                if time.time() - computed_at <= cached_ttl:
                    self._hits += 1
                    return existing_value

            # Store our computed value
            if ttl is None:
                if metric_type and metric_type in self.METRIC_TTLS:
                    ttl = self.METRIC_TTLS[metric_type]
                else:
                    ttl = self.default_ttl
            self._cache[key] = (computed_value, time.time(), ttl)

        return computed_value

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": (self._hits / total * 100) if total > 0 else 0,
                "cache_size": len(self._cache),
                "position_version": self._position_version,
            }


class PortfolioExporter:
    """
    High-level portfolio export manager.

    Supports multiple formats with caching.
    """

    def __init__(
        self,
        account_id: str,
        output_dir: str = "exports",
    ):
        self.account_id = account_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Exporters
        self.ibkr_exporter = IBKRFlexExporter(account_id)
        self.csv_exporter = CSVPortfolioExporter()
        self.fix_exporter = FIXPortfolioExporter()

        # Caching
        self.metrics_cache = PortfolioMetricsCache()

    def export(
        self,
        positions: list[PortfolioPosition],
        format: ExportFormat,
        filename: str | None = None,
        template: str = "standard",
    ) -> str:
        """
        Export portfolio to file.

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            ext = "xml" if format == ExportFormat.IBKR_FLEX else format.value
            filename = f"portfolio_{timestamp}.{ext}"

        filepath = self.output_dir / filename

        if format == ExportFormat.CSV:
            content = self.csv_exporter.export_positions(positions, template=template)
        elif format == ExportFormat.IBKR_FLEX:
            content = self.ibkr_exporter.export_positions(positions)
        elif format == ExportFormat.JSON:
            content = json.dumps([
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "realized_pnl": p.realized_pnl,
                    "asset_class": p.asset_class,
                    "currency": p.currency,
                }
                for p in positions
            ], indent=2)
        elif format == ExportFormat.FIX:
            content = "\n".join(
                self.fix_exporter.export_position_report(p)
                for p in positions
            )
        elif format == ExportFormat.EXCEL_CSV:
            self.csv_exporter.delimiter = ","
            content = self.csv_exporter.export_positions(positions, template="detailed")
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(filepath, "w") as f:
            f.write(content)

        logger.info(f"Exported portfolio to {filepath}")
        return str(filepath)

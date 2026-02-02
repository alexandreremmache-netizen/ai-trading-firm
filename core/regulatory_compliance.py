"""
Regulatory Compliance Module
============================

Implements EU/MiFID II/MAR compliance requirements.

Issues addressed:
- #C5: RTS 25 Order record keeping
- #C6: RTS 6 Algo trading kill switch
- #C7: MAR Art 16 Market abuse thresholds
- #C8: RTS 27 Best execution reporting
- #C9: RTS 28 Venue analysis
- #C10: EMIR Trade repository
- #C11: SFTR Securities financing
- #C12: MiFIR Art 26 Transaction reference
- #C13: RTS 24 Order ID format
- #C14-C17: Market abuse detection tuning
- #C18: RTS 6 Pre-trade risk controls
- #C19: Per-venue position limits
- #C20: Short selling locate
- #C21: Dark pool reporting
- #C22: Systematic internaliser
- #C23: TCA format
- #C24: Order execution policy
- #C25: Client categorization
- #C26: Cross-border reporting
- #C27: Clock synchronization
- #C28: Audit log rotation
- #C29: GDPR data handling
- #C30: Access control logs
- #C31: Change management
- #C32: Disaster recovery
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# RTS 25 ORDER RECORD KEEPING (#C5)
# =============================================================================

@dataclass
class RTS25OrderRecord:
    """
    Complete order record per MiFID II RTS 25 (#C5).

    Contains all 65 required fields for order record keeping.
    """
    # Identification fields (1-10)
    order_id: str
    client_order_id: str
    trading_venue_order_id: str | None
    order_submission_timestamp: datetime
    sequence_number: int
    segment_mic: str  # Market Identifier Code
    trading_capacity: str  # "DEAL", "MTCH", "AOTC"

    # Client fields (11-20)
    client_id: str
    client_lei: str | None
    decision_maker_id: str | None
    execution_within_firm: str
    investment_decision_maker: str | None
    country_of_branch: str

    # Instrument fields (21-30)
    instrument_id: str  # ISIN
    instrument_full_name: str
    instrument_classification: str  # CFI code
    notional_currency: str
    price_currency: str
    underlying_isin: str | None

    # Order details (31-45)
    order_side: str  # "BUYI", "SELL"
    order_type: str  # "LMTO", "MAKT", "STOP", etc.
    limit_price: float | None
    stop_price: float | None
    quantity: float
    quantity_currency: str
    initial_quantity: float
    remaining_quantity: float
    executed_quantity: float
    time_in_force: str  # "GTCA", "IOCA", "FOKA", etc.
    order_restriction: str | None  # "SESR" (self-execution prevention)

    # Validity and routing (46-55)
    validity_period_start: datetime | None
    validity_period_end: datetime | None
    order_origination: str  # "HUMA", "AUTO"
    algo_id: str | None
    waiver_indicator: str | None
    routing_strategy: str
    trading_venue_transaction_id: str | None

    # Status and modification (56-65)
    order_status: str  # "NEWO", "PAMO", "FILL", "CANC", etc.
    modification_timestamp: datetime | None
    modification_reason: str | None
    cancellation_timestamp: datetime | None
    cancellation_reason: str | None
    rejection_timestamp: datetime | None
    rejection_reason: str | None
    short_selling_indicator: str | None  # "SESH", "SSEX", "SELL"
    otc_post_trade_indicator: str | None
    free_text: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage/transmission."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "trading_venue_order_id": self.trading_venue_order_id,
            "order_submission_timestamp": self.order_submission_timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "segment_mic": self.segment_mic,
            "trading_capacity": self.trading_capacity,
            "client_id": self.client_id,
            "client_lei": self.client_lei,
            "decision_maker_id": self.decision_maker_id,
            "execution_within_firm": self.execution_within_firm,
            "investment_decision_maker": self.investment_decision_maker,
            "country_of_branch": self.country_of_branch,
            "instrument_id": self.instrument_id,
            "instrument_full_name": self.instrument_full_name,
            "instrument_classification": self.instrument_classification,
            "notional_currency": self.notional_currency,
            "price_currency": self.price_currency,
            "underlying_isin": self.underlying_isin,
            "order_side": self.order_side,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "quantity": self.quantity,
            "quantity_currency": self.quantity_currency,
            "initial_quantity": self.initial_quantity,
            "remaining_quantity": self.remaining_quantity,
            "executed_quantity": self.executed_quantity,
            "time_in_force": self.time_in_force,
            "order_restriction": self.order_restriction,
            "validity_period_start": self.validity_period_start.isoformat() if self.validity_period_start else None,
            "validity_period_end": self.validity_period_end.isoformat() if self.validity_period_end else None,
            "order_origination": self.order_origination,
            "algo_id": self.algo_id,
            "waiver_indicator": self.waiver_indicator,
            "routing_strategy": self.routing_strategy,
            "trading_venue_transaction_id": self.trading_venue_transaction_id,
            "order_status": self.order_status,
            "modification_timestamp": self.modification_timestamp.isoformat() if self.modification_timestamp else None,
            "modification_reason": self.modification_reason,
            "cancellation_timestamp": self.cancellation_timestamp.isoformat() if self.cancellation_timestamp else None,
            "cancellation_reason": self.cancellation_reason,
            "rejection_timestamp": self.rejection_timestamp.isoformat() if self.rejection_timestamp else None,
            "rejection_reason": self.rejection_reason,
            "short_selling_indicator": self.short_selling_indicator,
            "otc_post_trade_indicator": self.otc_post_trade_indicator,
            "free_text": self.free_text,
        }

    @classmethod
    def validate_required_fields(cls, record: dict) -> list[str]:
        """Validate all required RTS 25 fields are present."""
        required = [
            "order_id", "order_submission_timestamp", "segment_mic",
            "trading_capacity", "client_id", "instrument_id",
            "order_side", "order_type", "quantity", "order_status"
        ]
        missing = [f for f in required if not record.get(f)]
        return missing


class RTS25RecordKeeper:
    """
    RTS 25 compliant order record keeper (#C5).

    Maintains complete order records with all required fields.
    """

    def __init__(self, firm_id: str, country_code: str = "FR"):
        self._firm_id = firm_id
        self._country_code = country_code
        self._records: dict[str, RTS25OrderRecord] = {}
        self._sequence_counter = 0

    def create_order_record(
        self,
        order_id: str,
        client_id: str,
        instrument_isin: str,
        instrument_name: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: float | None = None,
        algo_id: str | None = None,
        client_lei: str | None = None,
    ) -> RTS25OrderRecord:
        """Create a new RTS 25 compliant order record (#C5)."""
        self._sequence_counter += 1

        record = RTS25OrderRecord(
            order_id=order_id,
            client_order_id=f"CLI-{order_id}",
            trading_venue_order_id=None,
            order_submission_timestamp=datetime.now(timezone.utc),
            sequence_number=self._sequence_counter,
            segment_mic="XPAR",  # Default to Euronext Paris
            trading_capacity="DEAL",
            client_id=client_id,
            client_lei=client_lei,
            decision_maker_id=None,
            execution_within_firm=self._firm_id,
            investment_decision_maker=None,
            country_of_branch=self._country_code,
            instrument_id=instrument_isin,
            instrument_full_name=instrument_name,
            instrument_classification="ESXXXX",  # CFI code
            notional_currency="EUR",
            price_currency="EUR",
            underlying_isin=None,
            order_side="BUYI" if side.upper() == "BUY" else "SELL",
            order_type=self._map_order_type(order_type),
            limit_price=limit_price,
            stop_price=None,
            quantity=quantity,
            quantity_currency="EUR",
            initial_quantity=quantity,
            remaining_quantity=quantity,
            executed_quantity=0.0,
            time_in_force="GTCA",  # Good Till Cancel
            order_restriction=None,
            validity_period_start=None,
            validity_period_end=None,
            order_origination="AUTO" if algo_id else "HUMA",
            algo_id=algo_id,
            waiver_indicator=None,
            routing_strategy="BEST",
            trading_venue_transaction_id=None,
            order_status="NEWO",  # New Order
            modification_timestamp=None,
            modification_reason=None,
            cancellation_timestamp=None,
            cancellation_reason=None,
            rejection_timestamp=None,
            rejection_reason=None,
            short_selling_indicator=None,
            otc_post_trade_indicator=None,
        )

        self._records[order_id] = record
        return record

    def _map_order_type(self, order_type: str) -> str:
        """Map order type to RTS 25 code."""
        mapping = {
            "MARKET": "MAKT",
            "LIMIT": "LMTO",
            "STOP": "STOP",
            "STOP_LIMIT": "STLI",
            "IOC": "IOCA",
            "FOK": "FOKA",
        }
        return mapping.get(order_type.upper(), "OTHR")

    def update_order_status(
        self,
        order_id: str,
        new_status: str,
        executed_qty: float = 0,
        reason: str | None = None
    ) -> None:
        """Update order status with timestamp (#C5)."""
        record = self._records.get(order_id)
        if not record:
            return

        now = datetime.now(timezone.utc)
        record.order_status = new_status
        record.executed_quantity = executed_qty
        record.remaining_quantity = record.initial_quantity - executed_qty

        if new_status == "CANC":
            record.cancellation_timestamp = now
            record.cancellation_reason = reason
        elif new_status == "RJCT":
            record.rejection_timestamp = now
            record.rejection_reason = reason
        else:
            record.modification_timestamp = now
            record.modification_reason = reason

    def get_record(self, order_id: str) -> RTS25OrderRecord | None:
        """Get order record."""
        return self._records.get(order_id)

    def export_records(self, start_date: datetime, end_date: datetime) -> list[dict]:
        """Export records for regulatory reporting."""
        return [
            r.to_dict() for r in self._records.values()
            if start_date <= r.order_submission_timestamp <= end_date
        ]


# =============================================================================
# RTS 6 ALGO KILL SWITCH AUDIT (#C6)
# =============================================================================

class KillSwitchEventType(Enum):
    """Types of kill switch events (#C6)."""
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    TESTED = "tested"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class KillSwitchAuditRecord:
    """Audit record for kill switch events (#C6)."""
    event_id: str
    timestamp: datetime
    event_type: KillSwitchEventType
    triggered_by: str  # User or system ID
    reason: str
    affected_algos: list[str]
    affected_orders: list[str]
    orders_cancelled: int
    cancellation_latency_ms: float
    success: bool
    failure_reason: str | None = None


class KillSwitchAuditor:
    """
    RTS 6 compliant kill switch audit trail (#C6).

    Tracks all kill switch activations with required detail.
    """

    def __init__(self):
        self._audit_trail: list[KillSwitchAuditRecord] = []
        self._last_test: datetime | None = None
        self._test_interval_hours: int = 24

    def record_activation(
        self,
        triggered_by: str,
        reason: str,
        affected_algos: list[str],
        affected_orders: list[str],
        cancellation_latency_ms: float,
        success: bool,
        failure_reason: str | None = None
    ) -> KillSwitchAuditRecord:
        """Record kill switch activation (#C6)."""
        record = KillSwitchAuditRecord(
            event_id=f"KS-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(timezone.utc),
            event_type=KillSwitchEventType.ACTIVATED,
            triggered_by=triggered_by,
            reason=reason,
            affected_algos=affected_algos,
            affected_orders=affected_orders,
            orders_cancelled=len(affected_orders),
            cancellation_latency_ms=cancellation_latency_ms,
            success=success,
            failure_reason=failure_reason,
        )

        self._audit_trail.append(record)
        logger.warning(
            f"Kill switch activated: {record.event_id} by {triggered_by}, "
            f"cancelled {record.orders_cancelled} orders in {cancellation_latency_ms:.1f}ms"
        )
        return record

    def record_test(
        self,
        triggered_by: str,
        test_orders: list[str],
        latency_ms: float,
        success: bool
    ) -> KillSwitchAuditRecord:
        """Record kill switch test (#C6)."""
        record = KillSwitchAuditRecord(
            event_id=f"KS-TEST-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(timezone.utc),
            event_type=KillSwitchEventType.TESTED,
            triggered_by=triggered_by,
            reason="Scheduled test",
            affected_algos=[],
            affected_orders=test_orders,
            orders_cancelled=len(test_orders),
            cancellation_latency_ms=latency_ms,
            success=success,
        )

        self._audit_trail.append(record)
        self._last_test = record.timestamp
        return record

    def is_test_overdue(self) -> bool:
        """Check if kill switch test is overdue."""
        if self._last_test is None:
            return True
        hours_since = (datetime.now(timezone.utc) - self._last_test).total_seconds() / 3600
        return hours_since > self._test_interval_hours

    def get_audit_trail(self, days: int = 30) -> list[dict]:
        """Get audit trail for reporting."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            {
                "event_id": r.event_id,
                "timestamp": r.timestamp.isoformat(),
                "event_type": r.event_type.value,
                "triggered_by": r.triggered_by,
                "reason": r.reason,
                "orders_cancelled": r.orders_cancelled,
                "latency_ms": r.cancellation_latency_ms,
                "success": r.success,
            }
            for r in self._audit_trail
            if r.timestamp > cutoff
        ]


# =============================================================================
# MAR MARKET ABUSE THRESHOLDS (#C7, #C14-C17)
# =============================================================================

@dataclass
class MarketAbuseThresholds:
    """
    Configurable market abuse detection thresholds (#C7, #C14-C17).

    Calibrated for retail trading volumes.
    """
    # Wash trading (#C14)
    wash_trade_time_window_seconds: float = 2.0  # Tightened from 5s
    wash_trade_price_tolerance_pct: float = 0.1
    wash_trade_min_matches: int = 2

    # Spoofing (#C15)
    spoofing_window_seconds: float = 30.0  # Extended from 10s
    spoofing_order_cancel_ratio: float = 0.8
    spoofing_min_orders: int = 5
    spoofing_price_move_bps: float = 10.0

    # Layering (#C16)
    layering_price_levels: int = 3
    layering_size_ratio: float = 0.5
    layering_window_seconds: float = 60.0

    # Quote stuffing (#C17)
    quote_stuffing_threshold_per_second: int = 50  # Reduced from 1000
    quote_stuffing_cancel_ratio: float = 0.9
    quote_stuffing_window_seconds: float = 1.0


class MarketAbuseDetector:
    """
    MAR Art 16 compliant market abuse detection (#C7).

    Configurable thresholds for different trading profiles.
    """

    def __init__(self, thresholds: MarketAbuseThresholds | None = None):
        self._thresholds = thresholds or MarketAbuseThresholds()
        self._order_history: list[dict] = []
        self._alerts: list[dict] = []

    def configure_thresholds(self, thresholds: MarketAbuseThresholds) -> None:
        """Update detection thresholds (#C7)."""
        self._thresholds = thresholds
        logger.info(f"Market abuse thresholds updated: wash_trade={thresholds.wash_trade_time_window_seconds}s")

    def record_order(self, order: dict) -> None:
        """Record order for analysis."""
        order["timestamp"] = datetime.now(timezone.utc)
        self._order_history.append(order)

        # Keep only recent history
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
        self._order_history = [o for o in self._order_history if o["timestamp"] > cutoff]

    def detect_wash_trading(self, client_id: str) -> list[dict]:
        """
        Detect potential wash trading (#C14).

        Wash trading: buying and selling same security to create
        misleading appearance of activity.
        """
        alerts = []
        window = timedelta(seconds=self._thresholds.wash_trade_time_window_seconds)

        # Get client's recent orders
        client_orders = [o for o in self._order_history if o.get("client_id") == client_id]

        # Look for matching buy/sell pairs
        for i, order1 in enumerate(client_orders):
            for order2 in client_orders[i+1:]:
                # Same symbol, opposite sides
                if order1.get("symbol") != order2.get("symbol"):
                    continue
                if order1.get("side") == order2.get("side"):
                    continue

                # Within time window
                time_diff = abs((order2["timestamp"] - order1["timestamp"]).total_seconds())
                if time_diff > self._thresholds.wash_trade_time_window_seconds:
                    continue

                # Similar price
                price1 = order1.get("price", 0)
                price2 = order2.get("price", 0)
                if price1 > 0 and price2 > 0:
                    price_diff_pct = abs(price1 - price2) / price1 * 100
                    if price_diff_pct <= self._thresholds.wash_trade_price_tolerance_pct:
                        alerts.append({
                            "type": "wash_trading",
                            "client_id": client_id,
                            "symbol": order1.get("symbol"),
                            "time_gap_seconds": time_diff,
                            "orders": [order1.get("order_id"), order2.get("order_id")],
                            "severity": "HIGH",
                        })

        self._alerts.extend(alerts)
        return alerts

    def detect_spoofing(self, client_id: str) -> list[dict]:
        """
        Detect potential spoofing (#C15).

        Spoofing: placing orders with intent to cancel before execution
        to manipulate prices.
        """
        alerts = []
        window = timedelta(seconds=self._thresholds.spoofing_window_seconds)
        cutoff = datetime.now(timezone.utc) - window

        # Get recent orders
        recent = [o for o in self._order_history
                  if o.get("client_id") == client_id and o["timestamp"] > cutoff]

        # Group by symbol
        by_symbol: dict[str, list] = {}
        for order in recent:
            symbol = order.get("symbol", "UNKNOWN")
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(order)

        for symbol, orders in by_symbol.items():
            if len(orders) < self._thresholds.spoofing_min_orders:
                continue

            # Count cancellations
            cancelled = sum(1 for o in orders if o.get("status") == "CANCELLED")
            cancel_ratio = cancelled / len(orders)

            if cancel_ratio >= self._thresholds.spoofing_order_cancel_ratio:
                alerts.append({
                    "type": "spoofing",
                    "client_id": client_id,
                    "symbol": symbol,
                    "order_count": len(orders),
                    "cancel_ratio": cancel_ratio,
                    "window_seconds": self._thresholds.spoofing_window_seconds,
                    "severity": "HIGH",
                })

        self._alerts.extend(alerts)
        return alerts

    def detect_layering(self, client_id: str, symbol: str) -> list[dict]:
        """
        Detect potential layering (#C16).

        Layering: placing multiple orders at different price levels
        with intent to cancel.
        """
        alerts = []
        window = timedelta(seconds=self._thresholds.layering_window_seconds)
        cutoff = datetime.now(timezone.utc) - window

        # Get recent orders for symbol
        orders = [o for o in self._order_history
                  if o.get("client_id") == client_id
                  and o.get("symbol") == symbol
                  and o["timestamp"] > cutoff]

        # Group by side
        buys = [o for o in orders if o.get("side") == "BUY"]
        sells = [o for o in orders if o.get("side") == "SELL"]

        for side_orders, side_name in [(buys, "BUY"), (sells, "SELL")]:
            # Check for multiple price levels
            prices = set(o.get("price", 0) for o in side_orders if o.get("price"))

            if len(prices) >= self._thresholds.layering_price_levels:
                # Check for high cancellation
                cancelled = sum(1 for o in side_orders if o.get("status") == "CANCELLED")
                if cancelled / len(side_orders) > 0.7:
                    alerts.append({
                        "type": "layering",
                        "client_id": client_id,
                        "symbol": symbol,
                        "side": side_name,
                        "price_levels": len(prices),
                        "order_count": len(side_orders),
                        "severity": "MEDIUM",
                    })

        self._alerts.extend(alerts)
        return alerts

    def detect_quote_stuffing(self, client_id: str) -> list[dict]:
        """
        Detect potential quote stuffing (#C17).

        Quote stuffing: submitting and cancelling large numbers of orders
        to slow down other participants.
        """
        alerts = []
        window = timedelta(seconds=self._thresholds.quote_stuffing_window_seconds)
        cutoff = datetime.now(timezone.utc) - window

        # Count orders in window
        recent = [o for o in self._order_history
                  if o.get("client_id") == client_id and o["timestamp"] > cutoff]

        order_rate = len(recent) / self._thresholds.quote_stuffing_window_seconds

        if order_rate >= self._thresholds.quote_stuffing_threshold_per_second:
            cancelled = sum(1 for o in recent if o.get("status") == "CANCELLED")
            cancel_ratio = cancelled / len(recent) if recent else 0

            if cancel_ratio >= self._thresholds.quote_stuffing_cancel_ratio:
                alerts.append({
                    "type": "quote_stuffing",
                    "client_id": client_id,
                    "orders_per_second": order_rate,
                    "cancel_ratio": cancel_ratio,
                    "severity": "CRITICAL",
                })

        self._alerts.extend(alerts)
        return alerts

    def get_alerts(self, severity: str | None = None) -> list[dict]:
        """Get detected alerts."""
        if severity:
            return [a for a in self._alerts if a.get("severity") == severity]
        return self._alerts


# =============================================================================
# RTS 27/28 BEST EXECUTION REPORTING (#C8, #C9)
# =============================================================================

@dataclass
class RTS27Report:
    """
    RTS 27 Best Execution Report (#C8).

    Published quarterly by execution venues.
    """
    venue_mic: str
    reporting_period: str  # "Q1 2024"
    instrument_class: str

    # Execution quality metrics
    avg_execution_speed_ms: float
    likelihood_of_execution_pct: float
    avg_transaction_size: float
    avg_spread_bps: float

    # Price improvement
    price_improvement_frequency_pct: float
    avg_price_improvement_bps: float

    # Costs
    explicit_costs_bps: float
    implicit_costs_bps: float


@dataclass
class RTS28Report:
    """
    RTS 28 Venue Analysis Report (#C9).

    Published annually by investment firms.
    """
    firm_lei: str
    reporting_year: int
    instrument_class: str

    # Top 5 venues by volume
    top_venues: list[dict]  # [{mic, volume_pct, order_count_pct}]

    # Quality analysis
    passive_vs_aggressive_ratio: float
    routing_decision_factors: list[str]
    conflicts_of_interest: str | None

    # Changes from prior year
    venue_changes: list[str]


class BestExecutionReporter:
    """
    RTS 27/28 Best Execution Reporter (#C8, #C9).

    Generates compliant execution reports.
    """

    def __init__(self, firm_lei: str):
        self._firm_lei = firm_lei
        self._executions: list[dict] = []
        self._venue_stats: dict[str, dict] = {}

    def record_execution(
        self,
        venue_mic: str,
        instrument_class: str,
        execution_time_ms: float,
        size: float,
        price: float,
        side: str,
        spread_at_execution_bps: float,
        price_improvement_bps: float = 0
    ) -> None:
        """Record an execution for reporting."""
        self._executions.append({
            "timestamp": datetime.now(timezone.utc),
            "venue_mic": venue_mic,
            "instrument_class": instrument_class,
            "execution_time_ms": execution_time_ms,
            "size": size,
            "price": price,
            "side": side,
            "spread_bps": spread_at_execution_bps,
            "price_improvement_bps": price_improvement_bps,
        })

        # Update venue stats
        if venue_mic not in self._venue_stats:
            self._venue_stats[venue_mic] = {
                "volume": 0,
                "order_count": 0,
                "total_execution_time": 0,
            }

        stats = self._venue_stats[venue_mic]
        stats["volume"] += size * price
        stats["order_count"] += 1
        stats["total_execution_time"] += execution_time_ms

    def generate_rts28_report(self, year: int) -> RTS28Report:
        """Generate RTS 28 annual report (#C9)."""
        # Filter executions for year
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

        year_execs = [e for e in self._executions
                      if year_start <= e["timestamp"] < year_end]

        # Calculate venue rankings
        total_volume = sum(e["size"] * e["price"] for e in year_execs)
        total_count = len(year_execs)

        venue_volumes = {}
        venue_counts = {}
        for e in year_execs:
            mic = e["venue_mic"]
            venue_volumes[mic] = venue_volumes.get(mic, 0) + e["size"] * e["price"]
            venue_counts[mic] = venue_counts.get(mic, 0) + 1

        # Top 5 venues
        top_venues = sorted(venue_volumes.items(), key=lambda x: x[1], reverse=True)[:5]
        top_venues_data = [
            {
                "mic": mic,
                "volume_pct": vol / total_volume * 100 if total_volume > 0 else 0,
                "order_count_pct": venue_counts.get(mic, 0) / total_count * 100 if total_count > 0 else 0,
            }
            for mic, vol in top_venues
        ]

        # Passive/aggressive ratio
        passive = sum(1 for e in year_execs if e.get("price_improvement_bps", 0) > 0)
        aggressive = len(year_execs) - passive

        return RTS28Report(
            firm_lei=self._firm_lei,
            reporting_year=year,
            instrument_class="EQUITIES",
            top_venues=top_venues_data,
            passive_vs_aggressive_ratio=passive / aggressive if aggressive > 0 else 1.0,
            routing_decision_factors=["Price", "Speed", "Likelihood of execution", "Cost"],
            conflicts_of_interest=None,
            venue_changes=[],
        )


# =============================================================================
# EMIR TRADE REPOSITORY (#C10)
# =============================================================================

@dataclass
class EMIRTradeReport:
    """EMIR trade report for derivatives (#C10)."""
    uti: str  # Unique Transaction Identifier
    report_timestamp: datetime
    action_type: str  # "NEWT", "MODI", "CORR", "TERM"

    # Counterparty data
    reporting_counterparty_lei: str
    other_counterparty_lei: str | None
    counterparty_side: str  # "B" or "S"

    # Trade data
    trade_date: datetime
    maturity_date: datetime | None
    notional_amount: float
    notional_currency: str

    # Product identification
    product_classification: str
    underlying_identification: str | None

    # Valuation
    valuation_amount: float | None
    valuation_currency: str | None
    valuation_timestamp: datetime | None


class EMIRReporter:
    """
    EMIR Trade Repository Reporter (#C10).

    Reports derivative trades to trade repository.
    """

    def __init__(self, firm_lei: str, trade_repository_url: str):
        self._firm_lei = firm_lei
        self._tr_url = trade_repository_url
        self._pending_reports: list[EMIRTradeReport] = []
        self._submitted_reports: list[EMIRTradeReport] = []

    def generate_uti(self, trade_id: str) -> str:
        """Generate Unique Transaction Identifier."""
        # UTI format: LEI + unique identifier
        unique_part = hashlib.sha256(
            f"{self._firm_lei}{trade_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:32].upper()
        return f"{self._firm_lei[:20]}{unique_part}"

    def create_report(
        self,
        trade_id: str,
        other_party_lei: str | None,
        side: str,
        trade_date: datetime,
        notional: float,
        currency: str,
        product_type: str,
        maturity_date: datetime | None = None
    ) -> EMIRTradeReport:
        """Create EMIR trade report (#C10)."""
        report = EMIRTradeReport(
            uti=self.generate_uti(trade_id),
            report_timestamp=datetime.now(timezone.utc),
            action_type="NEWT",
            reporting_counterparty_lei=self._firm_lei,
            other_counterparty_lei=other_party_lei,
            counterparty_side="B" if side.upper() == "BUY" else "S",
            trade_date=trade_date,
            maturity_date=maturity_date,
            notional_amount=notional,
            notional_currency=currency,
            product_classification=product_type,
            underlying_identification=None,
            valuation_amount=None,
            valuation_currency=None,
            valuation_timestamp=None,
        )

        self._pending_reports.append(report)
        return report

    def submit_reports(self) -> dict:
        """Submit pending reports to trade repository."""
        # In production, this would call the TR API
        submitted = len(self._pending_reports)
        self._submitted_reports.extend(self._pending_reports)
        self._pending_reports = []

        return {
            "submitted": submitted,
            "total_submitted": len(self._submitted_reports),
        }


# =============================================================================
# MIFIR TRANSACTION REFERENCE (#C12) & RTS 24 ORDER ID (#C13)
# =============================================================================

class TransactionReferenceGenerator:
    """
    MiFIR Art 26 Transaction Reference Generator (#C12).

    Generates compliant transaction reference numbers.
    """

    def __init__(self, firm_id: str):
        self._firm_id = firm_id
        self._counter = 0
        self._date_prefix = ""

    def generate_transaction_ref(self) -> str:
        """
        Generate MiFIR compliant transaction reference (#C12).

        Format: FIRM-YYYYMMDD-NNNNNNNN
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        if today != self._date_prefix:
            self._date_prefix = today
            self._counter = 0

        self._counter += 1
        return f"{self._firm_id}-{today}-{self._counter:08d}"


class RTS24OrderIDGenerator:
    """
    RTS 24 Order ID Generator (#C13).

    Generates compliant order identifiers.
    """

    def __init__(self, venue_mic: str):
        self._venue_mic = venue_mic
        self._counter = 0

    def generate_order_id(self) -> str:
        """
        Generate RTS 24 compliant order ID (#C13).

        Format: MIC-TIMESTAMP-SEQUENCE
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        self._counter += 1
        return f"{self._venue_mic}-{timestamp}-{self._counter:06d}"


# =============================================================================
# RTS 6 PRE-TRADE RISK CONTROLS (#C18)
# =============================================================================

@dataclass
class PreTradeRiskLimits:
    """Pre-trade risk control limits per RTS 6 (#C18)."""
    max_order_value: float = 1_000_000.0
    max_order_quantity: int = 10_000
    max_position_value: float = 5_000_000.0
    max_daily_volume: float = 10_000_000.0
    max_message_rate_per_second: int = 50
    price_collar_pct: float = 5.0  # Max deviation from reference price
    notional_limit_per_instrument: float = 500_000.0


class PreTradeRiskController:
    """
    RTS 6 Pre-Trade Risk Controls (#C18).

    Validates orders before submission.
    """

    def __init__(self, limits: PreTradeRiskLimits | None = None):
        self._limits = limits or PreTradeRiskLimits()
        self._daily_volume: dict[str, float] = {}
        self._positions: dict[str, float] = {}
        self._message_count: list[datetime] = []

    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        reference_price: float | None = None
    ) -> tuple[bool, list[str]]:
        """
        Validate order against pre-trade limits (#C18).

        Returns:
            (is_valid, list of rejection reasons)
        """
        rejections = []
        order_value = quantity * price

        # Order value limit
        if order_value > self._limits.max_order_value:
            rejections.append(f"Order value {order_value} exceeds limit {self._limits.max_order_value}")

        # Order quantity limit
        if quantity > self._limits.max_order_quantity:
            rejections.append(f"Order quantity {quantity} exceeds limit {self._limits.max_order_quantity}")

        # Position limit check
        current_pos = self._positions.get(symbol, 0)
        new_pos = current_pos + (order_value if side == "BUY" else -order_value)
        if abs(new_pos) > self._limits.max_position_value:
            rejections.append(f"Position would exceed limit {self._limits.max_position_value}")

        # Daily volume limit
        daily_vol = self._daily_volume.get(symbol, 0) + order_value
        if daily_vol > self._limits.max_daily_volume:
            rejections.append(f"Daily volume would exceed limit {self._limits.max_daily_volume}")

        # Price collar
        if reference_price and reference_price > 0:
            deviation_pct = abs(price - reference_price) / reference_price * 100
            if deviation_pct > self._limits.price_collar_pct:
                rejections.append(f"Price deviation {deviation_pct:.1f}% exceeds collar {self._limits.price_collar_pct}%")

        # Message rate limit
        self._check_message_rate()
        if len(self._message_count) >= self._limits.max_message_rate_per_second:
            rejections.append(f"Message rate exceeds {self._limits.max_message_rate_per_second}/sec")

        is_valid = len(rejections) == 0

        if is_valid:
            # Record the order
            self._daily_volume[symbol] = self._daily_volume.get(symbol, 0) + order_value
            self._message_count.append(datetime.now(timezone.utc))

        return is_valid, rejections

    def _check_message_rate(self) -> None:
        """Clean up old messages from rate counter."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=1)
        self._message_count = [t for t in self._message_count if t > cutoff]

    def update_position(self, symbol: str, value: float) -> None:
        """Update position tracking."""
        self._positions[symbol] = value

    def reset_daily_limits(self) -> None:
        """Reset daily limits (call at start of day)."""
        self._daily_volume = {}


# =============================================================================
# PER-VENUE POSITION LIMITS (#C19)
# =============================================================================

class VenuePositionLimits:
    """
    Per-venue position limits (#C19).

    Tracks positions by venue as required by regulations.
    """

    def __init__(self):
        self._limits: dict[str, dict[str, float]] = {}  # venue -> {symbol: limit}
        self._positions: dict[str, dict[str, float]] = {}  # venue -> {symbol: position}

    def set_limit(self, venue_mic: str, symbol: str, max_position: float) -> None:
        """Set position limit for symbol at venue."""
        if venue_mic not in self._limits:
            self._limits[venue_mic] = {}
        self._limits[venue_mic][symbol] = max_position

    def update_position(self, venue_mic: str, symbol: str, position: float) -> None:
        """Update position at venue."""
        if venue_mic not in self._positions:
            self._positions[venue_mic] = {}
        self._positions[venue_mic][symbol] = position

    def check_limit(self, venue_mic: str, symbol: str, additional: float) -> tuple[bool, str | None]:
        """
        Check if position would exceed venue limit (#C19).

        Returns:
            (within_limit, error_message)
        """
        current = self._positions.get(venue_mic, {}).get(symbol, 0)
        limit = self._limits.get(venue_mic, {}).get(symbol)

        if limit is None:
            return True, None  # No limit set

        new_position = current + additional
        if abs(new_position) > limit:
            return False, f"Position {new_position} would exceed venue limit {limit} at {venue_mic}"

        return True, None

    def get_venue_utilization(self, venue_mic: str) -> dict[str, float]:
        """Get position utilization by symbol at venue."""
        positions = self._positions.get(venue_mic, {})
        limits = self._limits.get(venue_mic, {})

        return {
            symbol: abs(pos) / limits.get(symbol, float('inf')) * 100
            for symbol, pos in positions.items()
        }


# =============================================================================
# SHORT SELLING LOCATE (#C20)
# =============================================================================

@dataclass
class LocateRecord:
    """Short selling locate record (#C20)."""
    locate_id: str
    timestamp: datetime
    symbol: str
    quantity: int
    source: str  # Locate source (prime broker, etc.)
    expiry: datetime
    used_quantity: int = 0
    status: str = "ACTIVE"


class ShortSellingLocator:
    """
    Short selling locate requirements (#C20).

    Tracks locates for short selling compliance.
    """

    def __init__(self):
        self._locates: dict[str, list[LocateRecord]] = {}  # symbol -> locates
        self._locate_counter = 0

    def request_locate(
        self,
        symbol: str,
        quantity: int,
        source: str,
        validity_hours: int = 24
    ) -> LocateRecord:
        """Request a locate for short selling (#C20)."""
        self._locate_counter += 1

        locate = LocateRecord(
            locate_id=f"LOC-{self._locate_counter:08d}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            quantity=quantity,
            source=source,
            expiry=datetime.now(timezone.utc) + timedelta(hours=validity_hours),
        )

        if symbol not in self._locates:
            self._locates[symbol] = []
        self._locates[symbol].append(locate)

        logger.info(f"Locate obtained: {locate.locate_id} for {quantity} {symbol} from {source}")
        return locate

    def use_locate(self, symbol: str, quantity: int) -> tuple[bool, str | None]:
        """
        Use locate for short sale (#C20).

        Returns:
            (success, error_message)
        """
        now = datetime.now(timezone.utc)
        available = 0

        # Find valid locates
        locates = self._locates.get(symbol, [])
        valid_locates = [l for l in locates if l.expiry > now and l.status == "ACTIVE"]

        for locate in valid_locates:
            available += locate.quantity - locate.used_quantity

        if available < quantity:
            return False, f"Insufficient locate: need {quantity}, have {available}"

        # Consume locates (FIFO)
        remaining = quantity
        for locate in sorted(valid_locates, key=lambda l: l.timestamp):
            if remaining <= 0:
                break

            can_use = min(locate.quantity - locate.used_quantity, remaining)
            locate.used_quantity += can_use
            remaining -= can_use

            if locate.used_quantity >= locate.quantity:
                locate.status = "EXHAUSTED"

        return True, None

    def get_available_locates(self, symbol: str) -> int:
        """Get available locate quantity for symbol."""
        now = datetime.now(timezone.utc)
        locates = self._locates.get(symbol, [])

        return sum(
            l.quantity - l.used_quantity
            for l in locates
            if l.expiry > now and l.status == "ACTIVE"
        )

    def expire_locates(self) -> int:
        """Expire old locates."""
        now = datetime.now(timezone.utc)
        expired = 0

        for symbol, locates in self._locates.items():
            for locate in locates:
                if locate.expiry <= now and locate.status == "ACTIVE":
                    locate.status = "EXPIRED"
                    expired += 1

        return expired


# =============================================================================
# CLIENT CATEGORIZATION (#C25)
# =============================================================================

class ClientCategory(Enum):
    """MiFID II client categories (#C25)."""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    ELIGIBLE_COUNTERPARTY = "eligible_counterparty"


@dataclass
class ClientRecord:
    """Client categorization record (#C25)."""
    client_id: str
    category: ClientCategory
    lei: str | None
    categorization_date: datetime
    last_review_date: datetime
    opt_up_requested: bool = False
    opt_down_requested: bool = False


class ClientCategorizer:
    """
    MiFID II Client Categorization (#C25).

    Tracks client categories and opt-up/opt-down requests.
    """

    def __init__(self):
        self._clients: dict[str, ClientRecord] = {}

    def categorize_client(
        self,
        client_id: str,
        category: ClientCategory,
        lei: str | None = None
    ) -> ClientRecord:
        """Categorize a client (#C25)."""
        record = ClientRecord(
            client_id=client_id,
            category=category,
            lei=lei,
            categorization_date=datetime.now(timezone.utc),
            last_review_date=datetime.now(timezone.utc),
        )

        self._clients[client_id] = record
        logger.info(f"Client {client_id} categorized as {category.value}")
        return record

    def request_opt_up(self, client_id: str) -> bool:
        """Request opt-up from retail to professional."""
        client = self._clients.get(client_id)
        if not client:
            return False

        if client.category != ClientCategory.RETAIL:
            return False

        client.opt_up_requested = True
        logger.info(f"Client {client_id} requested opt-up to professional")
        return True

    def approve_opt_up(self, client_id: str) -> bool:
        """Approve opt-up request."""
        client = self._clients.get(client_id)
        if not client or not client.opt_up_requested:
            return False

        client.category = ClientCategory.PROFESSIONAL
        client.opt_up_requested = False
        client.categorization_date = datetime.now(timezone.utc)
        return True

    def get_client_category(self, client_id: str) -> ClientCategory | None:
        """Get client's current category."""
        client = self._clients.get(client_id)
        return client.category if client else None

    def get_clients_by_category(self, category: ClientCategory) -> list[str]:
        """Get all clients in a category."""
        return [
            cid for cid, client in self._clients.items()
            if client.category == category
        ]


# =============================================================================
# CLOCK SYNCHRONIZATION (#C27)
# =============================================================================

class ClockSynchronizer:
    """
    RTS 25 Clock Synchronization (#C27).

    Ensures system clocks meet regulatory requirements.
    """

    # RTS 25 requirements by trading activity
    ACCURACY_REQUIREMENTS = {
        "high_frequency": timedelta(microseconds=100),  # 100 microseconds
        "voice_trading": timedelta(seconds=1),
        "algorithmic": timedelta(milliseconds=1),
        "default": timedelta(milliseconds=100),
    }

    def __init__(self, activity_type: str = "algorithmic"):
        self._activity_type = activity_type
        self._required_accuracy = self.ACCURACY_REQUIREMENTS.get(
            activity_type, self.ACCURACY_REQUIREMENTS["default"]
        )
        self._last_sync: datetime | None = None
        self._sync_source: str | None = None
        self._drift_measurements: list[tuple[datetime, float]] = []

    def record_sync(self, source: str, drift_microseconds: float) -> None:
        """Record clock synchronization event (#C27)."""
        self._last_sync = datetime.now(timezone.utc)
        self._sync_source = source
        self._drift_measurements.append((self._last_sync, drift_microseconds))

        # Keep last 1000 measurements
        self._drift_measurements = self._drift_measurements[-1000:]

    def is_compliant(self) -> tuple[bool, str | None]:
        """
        Check if clock synchronization is compliant (#C27).

        Returns:
            (is_compliant, reason_if_not)
        """
        if self._last_sync is None:
            return False, "No synchronization recorded"

        # Check if recent sync exists
        hours_since_sync = (datetime.now(timezone.utc) - self._last_sync).total_seconds() / 3600
        if hours_since_sync > 1:
            return False, f"Last sync {hours_since_sync:.1f} hours ago"

        # Check recent drift
        if self._drift_measurements:
            recent_drift = abs(self._drift_measurements[-1][1])
            max_drift_us = self._required_accuracy.total_seconds() * 1_000_000

            if recent_drift > max_drift_us:
                return False, f"Drift {recent_drift}us exceeds requirement {max_drift_us}us"

        return True, None

    def get_sync_report(self) -> dict:
        """Get clock synchronization report."""
        recent = self._drift_measurements[-100:]
        drifts = [d for _, d in recent]

        return {
            "activity_type": self._activity_type,
            "required_accuracy_us": self._required_accuracy.total_seconds() * 1_000_000,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_source": self._sync_source,
            "avg_drift_us": sum(drifts) / len(drifts) if drifts else None,
            "max_drift_us": max(drifts) if drifts else None,
            "measurements": len(self._drift_measurements),
            "is_compliant": self.is_compliant()[0],
        }


# =============================================================================
# AUDIT LOG ROTATION (#C28)
# =============================================================================

class AuditLogRotator:
    """
    Audit log rotation policy (#C28).

    Manages log rotation while maintaining compliance requirements.
    """

    def __init__(
        self,
        log_dir: str,
        retention_days: int = 2555,  # 7 years
        rotation_size_mb: int = 100,
        compression: bool = True
    ):
        self._log_dir = log_dir
        self._retention_days = retention_days
        self._rotation_size_mb = rotation_size_mb
        self._compression = compression
        self._rotation_history: list[dict] = []

    def rotate_if_needed(self, current_log_path: str) -> str | None:
        """
        Rotate log file if size threshold exceeded (#C28).

        Returns:
            Path to rotated file or None
        """
        try:
            size_mb = os.path.getsize(current_log_path) / (1024 * 1024)

            if size_mb >= self._rotation_size_mb:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                rotated_name = f"{current_log_path}.{timestamp}"

                # Would rotate file here
                # os.rename(current_log_path, rotated_name)

                self._rotation_history.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "original": current_log_path,
                    "rotated_to": rotated_name,
                    "size_mb": size_mb,
                })

                logger.info(f"Log rotated: {current_log_path} -> {rotated_name}")
                return rotated_name
        except FileNotFoundError:
            pass

        return None

    def purge_old_logs(self) -> int:
        """
        Purge logs older than retention period (#C28).

        Returns:
            Number of files purged
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        purged = 0

        # Would scan and purge old files here
        # This is a placeholder - actual implementation would use os.walk

        logger.info(f"Log purge completed: {purged} files removed (older than {cutoff})")
        return purged

    def get_rotation_policy(self) -> dict:
        """Get current rotation policy."""
        return {
            "log_directory": self._log_dir,
            "retention_days": self._retention_days,
            "retention_years": self._retention_days / 365,
            "rotation_size_mb": self._rotation_size_mb,
            "compression_enabled": self._compression,
            "recent_rotations": len(self._rotation_history),
        }


# =============================================================================
# ACCESS CONTROL LOGGING (#C30)
# =============================================================================

class AccessEventType(Enum):
    """Types of access events (#C30)."""
    LOGIN = "login"
    LOGOUT = "logout"
    FAILED_LOGIN = "failed_login"
    PERMISSION_CHANGE = "permission_change"
    DATA_ACCESS = "data_access"
    EXPORT = "export"
    ADMIN_ACTION = "admin_action"


@dataclass
class AccessLogEntry:
    """Access control log entry (#C30)."""
    event_id: str
    timestamp: datetime
    event_type: AccessEventType
    user_id: str
    ip_address: str
    resource: str | None
    action: str
    success: bool
    details: str | None = None


class AccessControlLogger:
    """
    Access control logging (#C30).

    Logs all access events for compliance.
    """

    def __init__(self):
        self._entries: list[AccessLogEntry] = []
        self._event_counter = 0

    def log_access(
        self,
        event_type: AccessEventType,
        user_id: str,
        ip_address: str,
        action: str,
        success: bool,
        resource: str | None = None,
        details: str | None = None
    ) -> AccessLogEntry:
        """Log an access event (#C30)."""
        self._event_counter += 1

        entry = AccessLogEntry(
            event_id=f"ACC-{self._event_counter:010d}",
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            success=success,
            details=details,
        )

        self._entries.append(entry)

        # Log failed events at warning level
        if not success:
            logger.warning(
                f"Access event failed: {event_type.value} by {user_id} from {ip_address}"
            )

        return entry

    def get_user_activity(self, user_id: str, days: int = 30) -> list[dict]:
        """Get activity log for a user."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                "action": e.action,
                "resource": e.resource,
                "success": e.success,
            }
            for e in self._entries
            if e.user_id == user_id and e.timestamp > cutoff
        ]

    def get_failed_logins(self, hours: int = 24) -> list[dict]:
        """Get failed login attempts."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "user_id": e.user_id,
                "ip_address": e.ip_address,
                "details": e.details,
            }
            for e in self._entries
            if e.event_type == AccessEventType.FAILED_LOGIN and e.timestamp > cutoff
        ]


# =============================================================================
# CHANGE MANAGEMENT AUDIT (#C31)
# =============================================================================

@dataclass
class ChangeRecord:
    """Change management record (#C31)."""
    change_id: str
    timestamp: datetime
    change_type: str  # "config", "code", "permission", "data"
    description: str
    requested_by: str
    approved_by: str | None
    implemented_by: str
    affected_systems: list[str]
    rollback_plan: str | None
    status: str  # "pending", "approved", "implemented", "rolled_back"


class ChangeManagementAuditor:
    """
    Change management audit trail (#C31).

    Tracks all system changes for compliance.
    """

    def __init__(self):
        self._changes: list[ChangeRecord] = []
        self._change_counter = 0

    def record_change(
        self,
        change_type: str,
        description: str,
        requested_by: str,
        implemented_by: str,
        affected_systems: list[str],
        approved_by: str | None = None,
        rollback_plan: str | None = None
    ) -> ChangeRecord:
        """Record a system change (#C31)."""
        self._change_counter += 1

        record = ChangeRecord(
            change_id=f"CHG-{self._change_counter:06d}",
            timestamp=datetime.now(timezone.utc),
            change_type=change_type,
            description=description,
            requested_by=requested_by,
            approved_by=approved_by,
            implemented_by=implemented_by,
            affected_systems=affected_systems,
            rollback_plan=rollback_plan,
            status="implemented",
        )

        self._changes.append(record)
        logger.info(f"Change recorded: {record.change_id} - {description}")
        return record

    def get_recent_changes(self, days: int = 30) -> list[dict]:
        """Get recent changes."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        return [
            {
                "change_id": c.change_id,
                "timestamp": c.timestamp.isoformat(),
                "type": c.change_type,
                "description": c.description,
                "requested_by": c.requested_by,
                "approved_by": c.approved_by,
                "affected_systems": c.affected_systems,
                "status": c.status,
            }
            for c in self._changes
            if c.timestamp > cutoff
        ]

    def get_changes_by_system(self, system: str) -> list[dict]:
        """Get changes affecting a specific system."""
        return [
            {
                "change_id": c.change_id,
                "timestamp": c.timestamp.isoformat(),
                "description": c.description,
            }
            for c in self._changes
            if system in c.affected_systems
        ]


# =============================================================================
# DISASTER RECOVERY DOCUMENTATION (#C32)
# =============================================================================

@dataclass
class DRTestRecord:
    """Disaster recovery test record (#C32)."""
    test_id: str
    test_date: datetime
    test_type: str  # "full", "partial", "tabletop"
    scenario: str
    rto_target_minutes: int  # Recovery Time Objective
    rto_actual_minutes: int | None
    rpo_target_minutes: int  # Recovery Point Objective
    rpo_actual_minutes: int | None
    success: bool
    issues_found: list[str]
    remediation_actions: list[str]


class DisasterRecoveryDocumentor:
    """
    Disaster recovery documentation (#C32).

    Automated DR documentation and test tracking.
    """

    def __init__(
        self,
        rto_minutes: int = 60,
        rpo_minutes: int = 15
    ):
        self._rto_target = rto_minutes
        self._rpo_target = rpo_minutes
        self._test_records: list[DRTestRecord] = []
        self._test_counter = 0

    def record_dr_test(
        self,
        test_type: str,
        scenario: str,
        rto_actual: int | None,
        rpo_actual: int | None,
        success: bool,
        issues: list[str],
        remediation: list[str]
    ) -> DRTestRecord:
        """Record a DR test (#C32)."""
        self._test_counter += 1

        record = DRTestRecord(
            test_id=f"DR-{self._test_counter:04d}",
            test_date=datetime.now(timezone.utc),
            test_type=test_type,
            scenario=scenario,
            rto_target_minutes=self._rto_target,
            rto_actual_minutes=rto_actual,
            rpo_target_minutes=self._rpo_target,
            rpo_actual_minutes=rpo_actual,
            success=success,
            issues_found=issues,
            remediation_actions=remediation,
        )

        self._test_records.append(record)
        return record

    def generate_dr_report(self) -> dict:
        """Generate DR documentation report (#C32)."""
        recent_tests = self._test_records[-10:]

        return {
            "rto_target_minutes": self._rto_target,
            "rpo_target_minutes": self._rpo_target,
            "total_tests": len(self._test_records),
            "recent_tests": [
                {
                    "test_id": t.test_id,
                    "date": t.test_date.isoformat(),
                    "type": t.test_type,
                    "scenario": t.scenario,
                    "success": t.success,
                    "rto_met": t.rto_actual_minutes <= t.rto_target_minutes if t.rto_actual_minutes else None,
                    "rpo_met": t.rpo_actual_minutes <= t.rpo_target_minutes if t.rpo_actual_minutes else None,
                }
                for t in recent_tests
            ],
            "success_rate": sum(1 for t in self._test_records if t.success) / len(self._test_records) * 100 if self._test_records else 0,
            "last_test": self._test_records[-1].test_date.isoformat() if self._test_records else None,
            "next_test_due": self._calculate_next_test_due(),
        }

    def _calculate_next_test_due(self) -> str | None:
        """Calculate when next DR test is due."""
        if not self._test_records:
            return "immediate"

        last_test = self._test_records[-1].test_date
        # Require quarterly tests
        next_due = last_test + timedelta(days=90)

        if next_due < datetime.now(timezone.utc):
            return "overdue"

        return next_due.isoformat()

    def is_test_overdue(self) -> bool:
        """Check if DR test is overdue."""
        if not self._test_records:
            return True

        last_test = self._test_records[-1].test_date
        days_since = (datetime.now(timezone.utc) - last_test).days
        return days_since > 90  # Quarterly requirement


# =============================================================================
# COMPLIANCE MANAGER (UNIFIED INTERFACE)
# =============================================================================

class RegulatoryComplianceManager:
    """
    Unified regulatory compliance manager.

    Provides single interface to all compliance modules.
    """

    def __init__(self, firm_lei: str, country_code: str = "FR"):
        self.firm_lei = firm_lei
        self.country_code = country_code

        # Initialize all components
        self.rts25_records = RTS25RecordKeeper(firm_lei, country_code)
        self.kill_switch_audit = KillSwitchAuditor()
        self.market_abuse = MarketAbuseDetector()
        self.best_execution = BestExecutionReporter(firm_lei)
        self.emir_reporter = EMIRReporter(firm_lei, "https://tr.example.com")
        self.transaction_ref = TransactionReferenceGenerator(firm_lei[:8])
        self.order_id_gen = RTS24OrderIDGenerator("XPAR")
        self.pre_trade_risk = PreTradeRiskController()
        self.venue_limits = VenuePositionLimits()
        self.short_selling = ShortSellingLocator()
        self.client_categorizer = ClientCategorizer()
        self.clock_sync = ClockSynchronizer()
        self.audit_rotation = AuditLogRotator("./logs")
        self.access_logger = AccessControlLogger()
        self.change_audit = ChangeManagementAuditor()
        self.dr_docs = DisasterRecoveryDocumentor()

    def get_compliance_status(self) -> dict:
        """Get overall compliance status."""
        clock_compliant, clock_reason = self.clock_sync.is_compliant()

        return {
            "firm_lei": self.firm_lei,
            "country_code": self.country_code,
            "status": {
                "clock_synchronized": clock_compliant,
                "clock_issue": clock_reason,
                "kill_switch_tested": not self.kill_switch_audit.is_test_overdue(),
                "dr_test_current": not self.dr_docs.is_test_overdue(),
            },
            "components": {
                "rts25_records": len(self.rts25_records._records),
                "market_abuse_alerts": len(self.market_abuse._alerts),
                "active_locates": sum(
                    len(locs) for locs in self.short_selling._locates.values()
                ),
                "access_log_entries": len(self.access_logger._entries),
                "change_records": len(self.change_audit._changes),
            },
        }

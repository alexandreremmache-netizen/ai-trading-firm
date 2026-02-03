"""
Market Surveillance Agent
=========================

Implements market abuse surveillance per MAR 2014/596/EU.
Detects potential manipulation patterns for compliance.

Monitors for:
- Wash trading
- Spoofing
- Quote stuffing
- Layering
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from core.agent_base import BaseAgent, AgentConfig
from core.events import Event, EventType, OrderEvent, FillEvent


logger = logging.getLogger(__name__)


class SurveillanceAlertType(Enum):
    """Types of surveillance alerts."""
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"
    QUOTE_STUFFING = "quote_stuffing"
    LAYERING = "layering"
    UNUSUAL_VOLUME = "unusual_volume"
    PRICE_MANIPULATION = "price_manipulation"
    # P2: Cross-market surveillance patterns
    CROSS_MARKET_MANIPULATION = "cross_market_manipulation"
    COORDINATED_TRADING = "coordinated_trading"
    # P2: Advanced pattern recognition
    MOMENTUM_IGNITION = "momentum_ignition"
    PAINTING_THE_TAPE = "painting_the_tape"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class STORStatus(Enum):
    """STOR submission status per MAR Article 16."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"


@dataclass
class SurveillanceAlert:
    """Market abuse surveillance alert."""
    alert_id: str
    timestamp: datetime
    alert_type: SurveillanceAlertType
    severity: AlertSeverity
    symbol: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    orders_involved: list[str] = field(default_factory=list)
    recommended_action: str = ""
    requires_review: bool = True
    # P2: Alert prioritization
    priority_score: float = 0.0  # 0-100, higher = more urgent
    related_alerts: list[str] = field(default_factory=list)
    symbols_involved: list[str] = field(default_factory=list)  # For cross-market

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "description": self.description,
            "evidence": self.evidence,
            "orders_involved": self.orders_involved,
            "recommended_action": self.recommended_action,
            "requires_review": self.requires_review,
        }


@dataclass
class STORReport:
    """
    Suspicious Transaction and Order Report (STOR) per MAR Article 16 (#C2).

    Required fields per ESMA MAR Guidelines (2016/1452):
    - Reporting entity details
    - Suspicious order/transaction details
    - Description of suspicious behavior
    - Supporting documentation
    """
    stor_id: str
    created_timestamp: datetime
    status: STORStatus = STORStatus.DRAFT

    # Reporting entity (our firm)
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    reporting_entity_country: str = ""
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Suspicious activity details
    instrument_isin: str = ""
    instrument_name: str = ""
    trading_venue_mic: str = ""

    # Description of suspicious behavior
    description_of_suspicion: str = ""
    type_of_suspicious_activity: str = ""  # e.g., "wash_trading", "spoofing"
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Orders/transactions involved
    orders_involved: list[str] = field(default_factory=list)
    transactions_involved: list[str] = field(default_factory=list)
    total_volume_involved: float = 0.0
    total_value_involved: float = 0.0

    # Evidence and analysis
    evidence_summary: str = ""
    analysis_methodology: str = ""
    related_alerts: list[str] = field(default_factory=list)

    # Regulatory submission
    submitted_timestamp: datetime | None = None
    nca_reference: str = ""  # Reference from National Competent Authority
    nca_acknowledgement_timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for submission."""
        return {
            "stor_id": self.stor_id,
            "created_timestamp": self.created_timestamp.isoformat(),
            "status": self.status.value,
            "reporting_entity": {
                "lei": self.reporting_entity_lei,
                "name": self.reporting_entity_name,
                "country": self.reporting_entity_country,
                "contact": {
                    "name": self.contact_person_name,
                    "email": self.contact_person_email,
                    "phone": self.contact_person_phone,
                },
            },
            "instrument": {
                "isin": self.instrument_isin,
                "name": self.instrument_name,
                "trading_venue": self.trading_venue_mic,
            },
            "suspicion": {
                "type": self.type_of_suspicious_activity,
                "description": self.description_of_suspicion,
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
            },
            "involved": {
                "orders": self.orders_involved,
                "transactions": self.transactions_involved,
                "total_volume": self.total_volume_involved,
                "total_value": self.total_value_involved,
            },
            "evidence": {
                "summary": self.evidence_summary,
                "methodology": self.analysis_methodology,
                "related_alerts": self.related_alerts,
            },
            "submission": {
                "submitted_timestamp": self.submitted_timestamp.isoformat() if self.submitted_timestamp else None,
                "nca_reference": self.nca_reference,
                "acknowledged": self.nca_acknowledgement_timestamp.isoformat() if self.nca_acknowledgement_timestamp else None,
            },
        }

    def to_xml(self) -> str:
        """
        Convert to XML format for NCA submission.

        Format follows ESMA MAR STOR reporting schema.
        """
        # Simplified XML - production would use full ESMA schema
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<STOR xmlns="urn:esma:stor:1.0">
    <Header>
        <STORId>{self.stor_id}</STORId>
        <CreationDateTime>{self.created_timestamp.isoformat()}</CreationDateTime>
    </Header>
    <ReportingEntity>
        <LEI>{self.reporting_entity_lei}</LEI>
        <Name>{self.reporting_entity_name}</Name>
        <Country>{self.reporting_entity_country}</Country>
    </ReportingEntity>
    <Instrument>
        <ISIN>{self.instrument_isin}</ISIN>
        <Name>{self.instrument_name}</Name>
        <TradingVenue>{self.trading_venue_mic}</TradingVenue>
    </Instrument>
    <SuspiciousActivity>
        <Type>{self.type_of_suspicious_activity}</Type>
        <Description>{self.description_of_suspicion}</Description>
        <StartDate>{self.start_date.isoformat() if self.start_date else ''}</StartDate>
        <EndDate>{self.end_date.isoformat() if self.end_date else ''}</EndDate>
    </SuspiciousActivity>
    <Evidence>
        <Summary>{self.evidence_summary}</Summary>
        <OrdersInvolved>{','.join(self.orders_involved)}</OrdersInvolved>
    </Evidence>
</STOR>"""
        return xml


@dataclass
class OrderRecord:
    """Record of an order for surveillance analysis."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: float | None
    status: str  # "submitted", "filled", "cancelled", "modified"
    fill_price: float | None = None
    fill_quantity: int = 0
    cancel_timestamp: datetime | None = None
    modify_count: int = 0

    @property
    def order_lifetime_seconds(self) -> float | None:
        """Calculate order lifetime in seconds (from submission to cancellation)."""
        if self.cancel_timestamp is None:
            return None
        return (self.cancel_timestamp - self.timestamp).total_seconds()


class SurveillanceAgent(BaseAgent):
    """
    Market abuse surveillance agent.

    Monitors trading activity for potential market manipulation
    patterns as required by MAR 2014/596/EU.
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus,
        audit_logger,
        surveillance_config: dict[str, Any] | None = None
    ):
        """
        Initialize surveillance agent.

        Args:
            config: Agent configuration
            event_bus: Event bus for subscriptions
            audit_logger: Audit logger
            surveillance_config: Surveillance-specific configuration
        """
        super().__init__(config, event_bus, audit_logger)

        self._surveillance_config = surveillance_config or {}

        # Detection thresholds
        self._wash_trading_window_seconds = self._surveillance_config.get(
            "wash_trading_window_seconds", 60
        )
        # Configurable lookback window for wash trading (SURV-P0-1)
        self._wash_trading_lookback_minutes = self._surveillance_config.get(
            "wash_trading_lookback_minutes", 60
        )
        # Wash trading net position threshold (% of total volume)
        self._wash_trading_net_position_threshold = self._surveillance_config.get(
            "wash_trading_net_position_threshold", 0.1
        )  # Net position < 10% of volume is suspicious
        # Wash trading minimum volume for detection
        self._wash_trading_min_volume = self._surveillance_config.get(
            "wash_trading_min_volume", 100
        )

        self._spoofing_cancel_threshold = self._surveillance_config.get(
            "spoofing_cancel_threshold", 0.8
        )  # 80% cancel rate
        # Spoofing: rapid cancellation threshold in seconds (SURV-P1-1)
        self._spoofing_rapid_cancel_seconds = self._surveillance_config.get(
            "spoofing_rapid_cancel_seconds", 1.0
        )
        # Spoofing: minimum repeat pattern count in window
        self._spoofing_min_pattern_count = self._surveillance_config.get(
            "spoofing_min_pattern_count", 3
        )
        # Spoofing: pattern detection window in minutes
        self._spoofing_pattern_window_minutes = self._surveillance_config.get(
            "spoofing_pattern_window_minutes", 5
        )

        self._quote_stuffing_rate_per_second = self._surveillance_config.get(
            "quote_stuffing_rate_per_second", 10
        )

        self._layering_level_threshold = self._surveillance_config.get(
            "layering_level_threshold", 3
        )
        # Layering: time window for detecting coordinated cancellations (SURV-P1-2)
        self._layering_cancel_window_seconds = self._surveillance_config.get(
            "layering_cancel_window_seconds", 2.0
        )
        # Layering: minimum orders cancelled together
        self._layering_min_orders_cancelled = self._surveillance_config.get(
            "layering_min_orders_cancelled", 3
        )

        # Enable/disable specific detections
        self._wash_trading_detection = self._surveillance_config.get(
            "wash_trading_detection", True
        )
        self._spoofing_detection = self._surveillance_config.get(
            "spoofing_detection", True
        )
        self._quote_stuffing_detection = self._surveillance_config.get(
            "quote_stuffing_detection", True
        )
        self._layering_detection = self._surveillance_config.get(
            "layering_detection", True
        )

        # Order tracking
        self._orders: dict[str, OrderRecord] = {}
        self._order_history: list[OrderRecord] = []
        self._fills: list[FillEvent] = []

        # Alert tracking
        self._alerts: list[SurveillanceAlert] = []
        self._alert_counter = 0

        # STOR tracking (#C2)
        self._stor_reports: list[STORReport] = []
        self._stor_counter = 0
        self._stor_auto_generate = self._surveillance_config.get("stor_auto_generate", True)
        self._stor_severity_threshold = AlertSeverity(
            self._surveillance_config.get("stor_severity_threshold", "high")
        )

        # Firm details for STOR
        self._firm_lei = self._surveillance_config.get("firm_lei", "")
        self._firm_name = self._surveillance_config.get("firm_name", "")
        self._firm_country = self._surveillance_config.get("firm_country", "FR")
        self._compliance_contact_name = self._surveillance_config.get("compliance_contact_name", "")
        self._compliance_contact_email = self._surveillance_config.get("compliance_contact_email", "")
        self._compliance_contact_phone = self._surveillance_config.get("compliance_contact_phone", "")

        # ISIN mappings for STOR reporting
        self._isin_map: dict[str, str] = self._surveillance_config.get("isin_map", {})
        self._venue_map: dict[str, str] = self._surveillance_config.get("venue_map", {})

        # Metrics by symbol
        self._symbol_metrics: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "orders_submitted": 0,
            "orders_cancelled": 0,
            "orders_filled": 0,
            "total_volume": 0,
            "cancel_rate": 0.0,
        })

        # Compliance notifier for suspicious activity alerts (#C33)
        self._compliance_notifier = None

        # P2: Cross-market surveillance
        self._cross_market_surveillance = self._surveillance_config.get(
            "cross_market_surveillance", True
        )
        self._cross_market_correlation_threshold = self._surveillance_config.get(
            "cross_market_correlation_threshold", 0.8
        )
        self._cross_market_time_window_seconds = self._surveillance_config.get(
            "cross_market_time_window_seconds", 60
        )
        # Symbol correlation map for related instruments
        self._symbol_correlations: dict[str, list[str]] = self._surveillance_config.get(
            "symbol_correlations", {}
        )

        # P2: Pattern recognition improvements
        self._momentum_ignition_detection = self._surveillance_config.get(
            "momentum_ignition_detection", True
        )
        self._momentum_ignition_threshold = self._surveillance_config.get(
            "momentum_ignition_threshold", 0.02
        )  # 2% price move
        self._painting_tape_detection = self._surveillance_config.get(
            "painting_tape_detection", True
        )

        # P2: Alert prioritization
        self._alert_priority_enabled = self._surveillance_config.get(
            "alert_priority_enabled", True
        )
        self._priority_weights = {
            "severity": self._surveillance_config.get("priority_weight_severity", 0.4),
            "volume": self._surveillance_config.get("priority_weight_volume", 0.2),
            "frequency": self._surveillance_config.get("priority_weight_frequency", 0.2),
            "recency": self._surveillance_config.get("priority_weight_recency", 0.2),
        }
        # Track alert frequency by type for prioritization
        self._alert_frequency: dict[str, int] = defaultdict(int)

        logger.info(
            f"SurveillanceAgent initialized: "
            f"wash={self._wash_trading_detection}, "
            f"spoof={self._spoofing_detection}, "
            f"quote_stuff={self._quote_stuffing_detection}, "
            f"layer={self._layering_detection}, "
            f"stor_auto={self._stor_auto_generate}, "
            f"cross_market={self._cross_market_surveillance}, "
            f"priority={self._alert_priority_enabled}"
        )

    async def initialize(self) -> None:
        """Initialize the agent."""
        logger.info("SurveillanceAgent initialized")

    def set_compliance_notifier(self, compliance_notifier) -> None:
        """Set compliance officer notifier (#C33)."""
        self._compliance_notifier = compliance_notifier
        logger.info("ComplianceOfficerNotifier connected to SurveillanceAgent")

    def get_subscribed_events(self) -> list[EventType]:
        """Subscribe to order and fill events."""
        return [EventType.ORDER, EventType.FILL]

    async def process_event(self, event: Event) -> None:
        """Process incoming events for surveillance."""
        if event.event_type == EventType.ORDER:
            await self._process_order(event)
        elif event.event_type == EventType.FILL:
            await self._process_fill(event)

    async def _process_order(self, event: OrderEvent) -> None:
        """Process an order event."""
        order = OrderRecord(
            order_id=event.event_id,
            timestamp=event.timestamp,
            symbol=event.symbol,
            side=event.side.value,
            quantity=event.quantity,
            order_type=event.order_type.value,
            price=event.limit_price,
            status="submitted",
        )

        self._orders[order.order_id] = order
        self._order_history.append(order)

        # Update metrics
        self._symbol_metrics[event.symbol]["orders_submitted"] += 1

        # Run detection algorithms
        await self._run_detections(event.symbol)

        # Trim old history
        self._trim_history()

    async def _process_fill(self, event: FillEvent) -> None:
        """Process a fill event."""
        self._fills.append(event)

        # Update order status
        if event.order_id in self._orders:
            order = self._orders[event.order_id]
            order.status = "filled"
            order.fill_price = event.fill_price
            order.fill_quantity = event.filled_quantity

        # Update metrics
        self._symbol_metrics[event.symbol]["orders_filled"] += 1
        self._symbol_metrics[event.symbol]["total_volume"] += event.filled_quantity

        # Run detection
        await self._run_detections(event.symbol)

    def record_order_cancelled(self, order_id: str) -> None:
        """Record an order cancellation."""
        if order_id in self._orders:
            order = self._orders[order_id]
            order.status = "cancelled"
            order.cancel_timestamp = datetime.now(timezone.utc)

            self._symbol_metrics[order.symbol]["orders_cancelled"] += 1

            # Update cancel rate
            metrics = self._symbol_metrics[order.symbol]
            if metrics["orders_submitted"] > 0:
                metrics["cancel_rate"] = metrics["orders_cancelled"] / metrics["orders_submitted"]

    async def _run_detections(self, symbol: str) -> None:
        """Run all enabled detection algorithms for a symbol."""
        if self._wash_trading_detection:
            await self._detect_wash_trading(symbol)

        if self._spoofing_detection:
            await self._detect_spoofing(symbol)

        if self._quote_stuffing_detection:
            await self._detect_quote_stuffing(symbol)

        if self._layering_detection:
            await self._detect_layering(symbol)

        # P2: Advanced pattern recognition
        if self._momentum_ignition_detection:
            await self._detect_momentum_ignition(symbol)

        if self._painting_tape_detection:
            await self._detect_painting_the_tape(symbol)

        # P2: Cross-market surveillance
        if self._cross_market_surveillance:
            await self._detect_cross_market_manipulation(symbol)

    async def _detect_wash_trading(self, symbol: str) -> None:
        """
        Detect potential wash trading (SURV-P0-1 - Enhanced).

        Wash trading: Simultaneous or near-simultaneous buy and sell
        orders in the same instrument that result in no change in
        beneficial ownership.

        Enhanced detection:
        - Configurable lookback window
        - Group by symbol and time window
        - Check for offsetting trades (buy/sell pairs within window)
        - Flag if net position change is small but volume is high
        """
        now = datetime.now(timezone.utc)
        # Use configurable lookback window (SURV-P0-1)
        lookback_start = now - timedelta(minutes=self._wash_trading_lookback_minutes)

        # Get all filled orders for this symbol within lookback window
        filled_orders = [
            o for o in self._order_history
            if o.symbol == symbol
            and o.timestamp >= lookback_start
            and o.status == "filled"
        ]

        if len(filled_orders) < 2:
            return

        # Group by time windows for pattern detection
        window_size_seconds = self._wash_trading_window_seconds
        time_windows: dict[int, list[OrderRecord]] = defaultdict(list)

        for order in filled_orders:
            # Group orders into time buckets
            seconds_from_start = (order.timestamp - lookback_start).total_seconds()
            window_index = int(seconds_from_start // window_size_seconds)
            time_windows[window_index].append(order)

        # Analyze each time window for offsetting patterns
        for window_idx, orders_in_window in time_windows.items():
            buys = [o for o in orders_in_window if o.side == "buy"]
            sells = [o for o in orders_in_window if o.side == "sell"]

            if not buys or not sells:
                continue

            # Calculate total volume and net position
            total_buy_volume = sum(o.fill_quantity for o in buys)
            total_sell_volume = sum(o.fill_quantity for o in sells)
            total_volume = total_buy_volume + total_sell_volume
            net_position = abs(total_buy_volume - total_sell_volume)

            # Skip if volume below threshold
            if total_volume < self._wash_trading_min_volume:
                continue

            # Check if net position change is small relative to total volume (SURV-P0-1)
            net_position_ratio = net_position / total_volume if total_volume > 0 else 1.0

            if net_position_ratio <= self._wash_trading_net_position_threshold:
                # Further validate with price similarity check
                buy_prices = [o.fill_price for o in buys if o.fill_price]
                sell_prices = [o.fill_price for o in sells if o.fill_price]

                if buy_prices and sell_prices:
                    avg_buy_price = sum(buy_prices) / len(buy_prices)
                    avg_sell_price = sum(sell_prices) / len(sell_prices)

                    if avg_buy_price > 0:
                        price_diff_pct = abs(avg_buy_price - avg_sell_price) / avg_buy_price

                        # Suspicious if prices are similar (within 2%)
                        if price_diff_pct < 0.02:
                            severity = AlertSeverity.HIGH if net_position_ratio < 0.05 else AlertSeverity.MEDIUM
                            order_ids = [o.order_id for o in orders_in_window]

                            self._generate_alert(
                                alert_type=SurveillanceAlertType.WASH_TRADING,
                                severity=severity,
                                symbol=symbol,
                                description=(
                                    f"Potential wash trading detected: "
                                    f"High volume ({total_volume}) with minimal net position change "
                                    f"({net_position_ratio:.1%}). Buy vol: {total_buy_volume}, "
                                    f"Sell vol: {total_sell_volume}, Price diff: {price_diff_pct:.2%}"
                                ),
                                evidence={
                                    "total_volume": total_volume,
                                    "net_position": net_position,
                                    "net_position_ratio": net_position_ratio,
                                    "total_buy_volume": total_buy_volume,
                                    "total_sell_volume": total_sell_volume,
                                    "avg_buy_price": avg_buy_price,
                                    "avg_sell_price": avg_sell_price,
                                    "price_diff_pct": price_diff_pct,
                                    "window_index": window_idx,
                                    "orders_count": len(orders_in_window),
                                    "lookback_minutes": self._wash_trading_lookback_minutes,
                                },
                                orders_involved=order_ids,
                                recommended_action="Review trades for legitimate business purpose - high volume with offsetting positions",
                            )

        # Also check for specific pair matches (original logic enhanced)
        buys = [o for o in filled_orders if o.side == "buy"]
        sells = [o for o in filled_orders if o.side == "sell"]

        for buy in buys:
            for sell in sells:
                # Check for similar quantity and timing
                time_diff = abs((buy.timestamp - sell.timestamp).total_seconds())

                if time_diff <= self._wash_trading_window_seconds:
                    # Check quantity similarity (within 10%)
                    max_qty = max(buy.quantity, sell.quantity)
                    if max_qty > 0:
                        qty_ratio = min(buy.quantity, sell.quantity) / max_qty
                    else:
                        continue  # Skip if both quantities are 0

                    if qty_ratio >= 0.9:
                        # Check price similarity
                        if buy.fill_price and sell.fill_price and buy.fill_price > 0:
                            price_diff_pct = abs(buy.fill_price - sell.fill_price) / buy.fill_price

                            if price_diff_pct < 0.01:  # Within 1%
                                self._generate_alert(
                                    alert_type=SurveillanceAlertType.WASH_TRADING,
                                    severity=AlertSeverity.MEDIUM,
                                    symbol=symbol,
                                    description=(
                                        f"Potential wash trading detected: "
                                        f"Buy {buy.quantity} @ {buy.fill_price:.2f} and "
                                        f"Sell {sell.quantity} @ {sell.fill_price:.2f} "
                                        f"within {time_diff:.1f}s"
                                    ),
                                    evidence={
                                        "buy_order": buy.order_id,
                                        "sell_order": sell.order_id,
                                        "time_diff_seconds": time_diff,
                                        "qty_ratio": qty_ratio,
                                        "price_diff_pct": price_diff_pct,
                                    },
                                    orders_involved=[buy.order_id, sell.order_id],
                                    recommended_action="Review trades for legitimate business purpose",
                                )

    async def _detect_spoofing(self, symbol: str) -> None:
        """
        Detect potential spoofing (SURV-P1-1 - Enhanced).

        Spoofing: Entering orders with intent to cancel before execution
        to create false impression of demand/supply.

        Enhanced detection:
        - Track order lifetime (time from submission to cancellation)
        - Flag orders cancelled < 1 second after placement (configurable)
        - Flag if pattern repeats > 3 times in 5 minutes (configurable)
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=self._spoofing_pattern_window_minutes)

        # Get recent orders for this symbol
        recent = [
            o for o in self._order_history
            if o.symbol == symbol and o.timestamp >= window_start
        ]

        if not recent:
            return

        # SURV-P1-1: Detect rapid cancellations
        rapid_cancellations = self._detect_rapid_cancellations(recent, symbol)

        # Original spoofing detection based on cancel rate
        await self._detect_spoofing_by_cancel_rate(symbol, recent)

    def _detect_rapid_cancellations(self, orders: list[OrderRecord], symbol: str) -> list[SurveillanceAlert]:
        """
        Detect spoofing via rapid order cancellations (SURV-P1-1).

        Flags orders cancelled within configurable threshold (default < 1 second)
        and alerts if pattern repeats more than threshold times in the window.
        """
        alerts = []

        # Find all rapidly cancelled orders
        rapid_cancelled = [
            o for o in orders
            if o.status == "cancelled"
            and o.order_lifetime_seconds is not None
            and o.order_lifetime_seconds < self._spoofing_rapid_cancel_seconds
        ]

        if len(rapid_cancelled) >= self._spoofing_min_pattern_count:
            # Pattern detected: multiple rapid cancellations
            order_ids = [o.order_id for o in rapid_cancelled]
            avg_lifetime = sum(o.order_lifetime_seconds or 0 for o in rapid_cancelled) / len(rapid_cancelled)

            # Group by side to detect manipulation direction
            rapid_buy_cancels = [o for o in rapid_cancelled if o.side == "buy"]
            rapid_sell_cancels = [o for o in rapid_cancelled if o.side == "sell"]

            # Check for one-sided pattern (stronger indication of manipulation)
            total_rapid = len(rapid_cancelled)
            buy_ratio = len(rapid_buy_cancels) / total_rapid if total_rapid > 0 else 0
            sell_ratio = len(rapid_sell_cancels) / total_rapid if total_rapid > 0 else 0

            # Severity based on pattern strength
            if total_rapid >= self._spoofing_min_pattern_count * 2:
                severity = AlertSeverity.CRITICAL
            elif buy_ratio > 0.8 or sell_ratio > 0.8:
                # One-sided rapid cancellations are more suspicious
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.MEDIUM

            total_rapid_volume = sum(o.quantity for o in rapid_cancelled)

            alert = self._generate_alert(
                alert_type=SurveillanceAlertType.SPOOFING,
                severity=severity,
                symbol=symbol,
                description=(
                    f"Rapid cancellation pattern detected: {total_rapid} orders cancelled "
                    f"within {self._spoofing_rapid_cancel_seconds}s (avg lifetime: {avg_lifetime:.3f}s). "
                    f"Buy cancels: {len(rapid_buy_cancels)}, Sell cancels: {len(rapid_sell_cancels)}"
                ),
                evidence={
                    "rapid_cancellation_count": total_rapid,
                    "rapid_cancel_threshold_seconds": self._spoofing_rapid_cancel_seconds,
                    "avg_order_lifetime_seconds": avg_lifetime,
                    "rapid_buy_cancels": len(rapid_buy_cancels),
                    "rapid_sell_cancels": len(rapid_sell_cancels),
                    "total_volume_cancelled": total_rapid_volume,
                    "detection_window_minutes": self._spoofing_pattern_window_minutes,
                    "min_pattern_count": self._spoofing_min_pattern_count,
                },
                orders_involved=order_ids,
                recommended_action="Investigate intent behind rapid order cancellations - potential spoofing activity",
            )
            alerts.append(alert)

        return alerts

    async def _detect_spoofing_by_cancel_rate(self, symbol: str, recent_orders: list[OrderRecord]) -> None:
        """Original spoofing detection based on cancel rate patterns."""
        metrics = self._symbol_metrics[symbol]

        # Need minimum order count for meaningful analysis
        if metrics["orders_submitted"] < 10:
            return

        cancel_rate = metrics["cancel_rate"]

        if cancel_rate >= self._spoofing_cancel_threshold:
            cancelled = [o for o in recent_orders if o.status == "cancelled"]
            filled = [o for o in recent_orders if o.status == "filled"]

            # Check for pattern: cancelled orders on one side, fills on opposite
            cancelled_buys = sum(o.quantity for o in cancelled if o.side == "buy")
            cancelled_sells = sum(o.quantity for o in cancelled if o.side == "sell")
            filled_buys = sum(o.fill_quantity for o in filled if o.side == "buy")
            filled_sells = sum(o.fill_quantity for o in filled if o.side == "sell")

            # Pattern: Many cancelled buys with sells executed (or vice versa)
            if cancelled_buys > filled_sells * 3 and filled_sells > 0:
                self._generate_alert(
                    alert_type=SurveillanceAlertType.SPOOFING,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    description=(
                        f"Potential spoofing detected: High cancel rate ({cancel_rate:.1%}) "
                        f"with cancelled buys ({cancelled_buys}) vs executed sells ({filled_sells})"
                    ),
                    evidence={
                        "cancel_rate": cancel_rate,
                        "cancelled_buy_qty": cancelled_buys,
                        "cancelled_sell_qty": cancelled_sells,
                        "filled_buy_qty": filled_buys,
                        "filled_sell_qty": filled_sells,
                    },
                    recommended_action="Escalate to compliance for review",
                )

            elif cancelled_sells > filled_buys * 3 and filled_buys > 0:
                self._generate_alert(
                    alert_type=SurveillanceAlertType.SPOOFING,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    description=(
                        f"Potential spoofing detected: High cancel rate ({cancel_rate:.1%}) "
                        f"with cancelled sells ({cancelled_sells}) vs executed buys ({filled_buys})"
                    ),
                    evidence={
                        "cancel_rate": cancel_rate,
                        "cancelled_buy_qty": cancelled_buys,
                        "cancelled_sell_qty": cancelled_sells,
                        "filled_buy_qty": filled_buys,
                        "filled_sell_qty": filled_sells,
                    },
                    recommended_action="Escalate to compliance for review",
                )

    async def _detect_quote_stuffing(self, symbol: str) -> None:
        """
        Detect potential quote stuffing.

        Quote stuffing: Rapidly entering and canceling orders to
        slow down other traders or create confusion.
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=1)

        # Count orders in last second
        recent_orders = [
            o for o in self._order_history
            if o.symbol == symbol and o.timestamp >= window_start
        ]

        order_rate = len(recent_orders)

        if order_rate >= self._quote_stuffing_rate_per_second:
            # Check if most are cancelled
            cancelled_count = sum(1 for o in recent_orders if o.status == "cancelled")

            if cancelled_count >= order_rate * 0.9:
                self._generate_alert(
                    alert_type=SurveillanceAlertType.QUOTE_STUFFING,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    description=(
                        f"Potential quote stuffing detected: {order_rate} orders/second "
                        f"with {cancelled_count} cancelled"
                    ),
                    evidence={
                        "order_rate_per_second": order_rate,
                        "cancelled_count": cancelled_count,
                        "threshold": self._quote_stuffing_rate_per_second,
                    },
                    recommended_action="Immediate review required - potential market manipulation",
                )

    async def _detect_layering(self, symbol: str) -> None:
        """
        Detect potential layering (SURV-P1-2 - Enhanced).

        Layering: Entering multiple orders at different price levels
        to create artificial depth, then cancelling after execution
        on the opposite side.

        Enhanced detection:
        - Detect multiple orders at different prices cancelled together
        - Track coordinated cancellation timing
        - Identify layering with opposite side execution correlation
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=1)

        recent_orders = [
            o for o in self._order_history
            if o.symbol == symbol
            and o.timestamp >= window_start
            and o.price is not None
        ]

        if len(recent_orders) < self._layering_level_threshold * 2:
            return

        # Group by side and price
        buy_levels: dict[float, list[OrderRecord]] = defaultdict(list)
        sell_levels: dict[float, list[OrderRecord]] = defaultdict(list)

        for order in recent_orders:
            if order.side == "buy":
                buy_levels[order.price].append(order)
            else:
                sell_levels[order.price].append(order)

        # SURV-P1-2: Detect coordinated cancellations at multiple price levels
        self._detect_coordinated_layering_cancellations(symbol, recent_orders, buy_levels, sell_levels)

        # Original layering detection
        for levels, side in [(buy_levels, "buy"), (sell_levels, "sell")]:
            if len(levels) >= self._layering_level_threshold:
                # Check if most are cancelled
                total_orders = sum(len(orders) for orders in levels.values())
                cancelled_orders = sum(
                    1 for orders in levels.values()
                    for o in orders if o.status == "cancelled"
                )

                if cancelled_orders >= total_orders * 0.8:
                    self._generate_alert(
                        alert_type=SurveillanceAlertType.LAYERING,
                        severity=AlertSeverity.MEDIUM,
                        symbol=symbol,
                        description=(
                            f"Potential layering detected on {side} side: "
                            f"{len(levels)} price levels with {cancelled_orders}/{total_orders} cancelled"
                        ),
                        evidence={
                            "side": side,
                            "price_levels": len(levels),
                            "total_orders": total_orders,
                            "cancelled_orders": cancelled_orders,
                        },
                        recommended_action="Review order pattern for manipulation intent",
                    )

    def _detect_coordinated_layering_cancellations(
        self,
        symbol: str,
        orders: list[OrderRecord],
        buy_levels: dict[float, list[OrderRecord]],
        sell_levels: dict[float, list[OrderRecord]]
    ) -> None:
        """
        Detect coordinated cancellations at multiple price levels (SURV-P1-2).

        Identifies patterns where multiple orders at different prices are
        cancelled within a short time window, indicating potential layering.
        """
        # Get all cancelled orders with cancel timestamps
        cancelled_orders = [
            o for o in orders
            if o.status == "cancelled" and o.cancel_timestamp is not None
        ]

        if len(cancelled_orders) < self._layering_min_orders_cancelled:
            return

        # Sort by cancel timestamp
        cancelled_orders.sort(key=lambda o: o.cancel_timestamp)  # type: ignore

        # Find clusters of cancellations within the time window
        clusters: list[list[OrderRecord]] = []
        current_cluster: list[OrderRecord] = []

        for order in cancelled_orders:
            if not current_cluster:
                current_cluster.append(order)
            else:
                # Check if this cancellation is within the window of the cluster start
                first_cancel = current_cluster[0].cancel_timestamp
                this_cancel = order.cancel_timestamp
                if first_cancel and this_cancel:
                    time_diff = (this_cancel - first_cancel).total_seconds()
                    if time_diff <= self._layering_cancel_window_seconds:
                        current_cluster.append(order)
                    else:
                        # Start new cluster
                        if len(current_cluster) >= self._layering_min_orders_cancelled:
                            clusters.append(current_cluster)
                        current_cluster = [order]

        # Don't forget the last cluster
        if len(current_cluster) >= self._layering_min_orders_cancelled:
            clusters.append(current_cluster)

        # Analyze each cluster for layering pattern
        for cluster in clusters:
            # Get unique price levels in this cluster
            unique_prices = set(o.price for o in cluster if o.price is not None)

            if len(unique_prices) < self._layering_level_threshold:
                continue  # Not enough price levels for layering

            # Check if cluster is predominantly one-sided
            buy_orders = [o for o in cluster if o.side == "buy"]
            sell_orders = [o for o in cluster if o.side == "sell"]

            total_in_cluster = len(cluster)
            buy_ratio = len(buy_orders) / total_in_cluster if total_in_cluster > 0 else 0

            # Layering is typically one-sided
            if buy_ratio > 0.7 or buy_ratio < 0.3:
                dominant_side = "buy" if buy_ratio > 0.7 else "sell"
                dominant_orders = buy_orders if dominant_side == "buy" else sell_orders

                # Calculate price spread
                prices = [o.price for o in dominant_orders if o.price is not None]
                if prices:
                    price_spread = max(prices) - min(prices)
                    price_spread_pct = price_spread / min(prices) if min(prices) > 0 else 0

                    # Check for fills on opposite side around the same time
                    cluster_start = cluster[0].cancel_timestamp
                    cluster_end = cluster[-1].cancel_timestamp
                    if cluster_start and cluster_end:
                        # Look for opposite side fills within extended window
                        opposite_side = "sell" if dominant_side == "buy" else "buy"
                        extended_start = cluster_start - timedelta(seconds=5)
                        extended_end = cluster_end + timedelta(seconds=5)

                        opposite_fills = [
                            o for o in orders
                            if o.side == opposite_side
                            and o.status == "filled"
                            and o.timestamp >= extended_start
                            and o.timestamp <= extended_end
                        ]

                        # Severity based on pattern strength
                        if opposite_fills and len(unique_prices) >= self._layering_level_threshold * 2:
                            severity = AlertSeverity.CRITICAL
                        elif opposite_fills:
                            severity = AlertSeverity.HIGH
                        else:
                            severity = AlertSeverity.MEDIUM

                        total_cancelled_volume = sum(o.quantity for o in dominant_orders)
                        opposite_fill_volume = sum(o.fill_quantity for o in opposite_fills)

                        order_ids = [o.order_id for o in cluster]

                        self._generate_alert(
                            alert_type=SurveillanceAlertType.LAYERING,
                            severity=severity,
                            symbol=symbol,
                            description=(
                                f"Coordinated layering cancellation detected: {len(cluster)} {dominant_side} orders "
                                f"at {len(unique_prices)} price levels cancelled within "
                                f"{self._layering_cancel_window_seconds}s. "
                                f"Opposite side fills: {len(opposite_fills)} ({opposite_fill_volume} qty)"
                            ),
                            evidence={
                                "coordinated_cancellation": True,
                                "dominant_side": dominant_side,
                                "orders_cancelled": len(cluster),
                                "price_levels": len(unique_prices),
                                "price_spread": price_spread,
                                "price_spread_pct": price_spread_pct,
                                "cancel_window_seconds": self._layering_cancel_window_seconds,
                                "total_cancelled_volume": total_cancelled_volume,
                                "opposite_side_fills": len(opposite_fills),
                                "opposite_fill_volume": opposite_fill_volume,
                                "prices": sorted(prices),
                            },
                            orders_involved=order_ids,
                            recommended_action=(
                                "Investigate coordinated order cancellation pattern - "
                                "potential layering manipulation with opposite side execution"
                            ),
                        )

    def _generate_alert(
        self,
        alert_type: SurveillanceAlertType,
        severity: AlertSeverity,
        symbol: str,
        description: str,
        evidence: dict[str, Any],
        orders_involved: list[str] | None = None,
        recommended_action: str = "",
        symbols_involved: list[str] | None = None,
    ) -> SurveillanceAlert:
        """Generate and store a surveillance alert."""
        self._alert_counter += 1
        alert_id = f"SURV-{self._alert_counter:06d}"

        # P2: Calculate priority score
        priority_score = self._calculate_priority_score(
            alert_type=alert_type,
            severity=severity,
            symbol=symbol,
            evidence=evidence,
        )

        # P2: Find related alerts
        related_alerts = self._find_related_alerts(symbol, alert_type)

        alert = SurveillanceAlert(
            alert_id=alert_id,
            timestamp=datetime.now(timezone.utc),
            alert_type=alert_type,
            severity=severity,
            symbol=symbol,
            description=description,
            evidence=evidence,
            orders_involved=orders_involved or [],
            recommended_action=recommended_action,
            requires_review=severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL],
            priority_score=priority_score,
            related_alerts=related_alerts,
            symbols_involved=symbols_involved or [symbol],
        )

        self._alerts.append(alert)

        # P2: Update alert frequency tracking
        self._alert_frequency[f"{symbol}:{alert_type.value}"] += 1

        # Log the alert
        self._audit_logger.log_agent_event(
            agent_name=self.name,
            event_type="surveillance_alert",
            details=alert.to_dict(),
        )

        logger.warning(
            f"Surveillance alert: [{severity.value}] {alert_type.value} - {description} "
            f"(priority={priority_score:.1f})"
        )

        # Auto-generate STOR for HIGH/CRITICAL alerts (#C2)
        if self._stor_auto_generate and self._should_generate_stor(severity):
            self._create_stor_from_alert(alert)

        return alert

    # =========================================================================
    # P2: ALERT PRIORITIZATION
    # =========================================================================

    def _calculate_priority_score(
        self,
        alert_type: SurveillanceAlertType,
        severity: AlertSeverity,
        symbol: str,
        evidence: dict[str, Any],
    ) -> float:
        """
        Calculate priority score for alert ordering (P2).

        Score is 0-100, with higher scores being more urgent.

        Factors:
        1. Severity (40%): CRITICAL=100, HIGH=75, MEDIUM=50, LOW=25, INFO=10
        2. Volume involved (20%): Higher volume = higher priority
        3. Frequency (20%): Repeated patterns are more concerning
        4. Recency (20%): Recent similar alerts boost priority
        """
        if not self._alert_priority_enabled:
            return 50.0  # Default middle priority

        # Factor 1: Severity score (0-100)
        severity_scores = {
            AlertSeverity.CRITICAL: 100,
            AlertSeverity.HIGH: 75,
            AlertSeverity.MEDIUM: 50,
            AlertSeverity.LOW: 25,
            AlertSeverity.INFO: 10,
        }
        severity_score = severity_scores.get(severity, 50)

        # Factor 2: Volume score (0-100)
        total_volume = evidence.get("total_volume", 0)
        if total_volume == 0:
            total_volume = evidence.get("total_cancelled_volume", 0)
        if total_volume == 0:
            total_volume = sum(
                self._orders[oid].quantity
                for oid in evidence.get("orders_involved", [])
                if oid in self._orders
            )

        # Scale volume: 1000 shares = 50, 10000 = 100
        volume_score = min(100, 50 + (total_volume / 200))

        # Factor 3: Frequency score (0-100)
        freq_key = f"{symbol}:{alert_type.value}"
        frequency = self._alert_frequency.get(freq_key, 0)
        # 5+ occurrences = max score
        frequency_score = min(100, frequency * 20)

        # Factor 4: Recency score (0-100)
        # Check for similar alerts in last hour
        recent_similar = len([
            a for a in self._alerts[-100:]
            if a.symbol == symbol
            and a.alert_type == alert_type
            and (datetime.now(timezone.utc) - a.timestamp).total_seconds() < 3600
        ])
        recency_score = min(100, recent_similar * 25)

        # Weighted combination
        priority_score = (
            severity_score * self._priority_weights["severity"] +
            volume_score * self._priority_weights["volume"] +
            frequency_score * self._priority_weights["frequency"] +
            recency_score * self._priority_weights["recency"]
        )

        return round(priority_score, 1)

    def _find_related_alerts(
        self,
        symbol: str,
        alert_type: SurveillanceAlertType,
    ) -> list[str]:
        """
        Find related alerts for correlation analysis (P2).

        Returns alert IDs of potentially related alerts.
        """
        related = []

        # Look for alerts in last 24 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        for alert in self._alerts[-200:]:
            if alert.timestamp < cutoff:
                continue

            # Same symbol and type
            if alert.symbol == symbol and alert.alert_type == alert_type:
                related.append(alert.alert_id)

            # Same symbol, different type (might be related manipulation)
            elif alert.symbol == symbol:
                related.append(alert.alert_id)

            # Related symbol (cross-market)
            elif symbol in self._symbol_correlations:
                if alert.symbol in self._symbol_correlations[symbol]:
                    related.append(alert.alert_id)

        return related[-10:]  # Limit to 10 most recent

    def get_prioritized_alerts(
        self,
        hours: int = 24,
        limit: int = 20,
    ) -> list[SurveillanceAlert]:
        """
        Get alerts sorted by priority score (P2).

        Args:
            hours: Lookback period
            limit: Maximum number of alerts to return

        Returns:
            List of alerts sorted by priority (highest first)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        recent_alerts = [a for a in self._alerts if a.timestamp >= cutoff]
        recent_alerts.sort(key=lambda a: a.priority_score, reverse=True)

        return recent_alerts[:limit]

    # =========================================================================
    # P2: CROSS-MARKET SURVEILLANCE
    # =========================================================================

    async def _detect_cross_market_manipulation(self, symbol: str) -> None:
        """
        Detect potential cross-market manipulation (P2).

        Monitors for coordinated activity across related instruments:
        - Stock and its options
        - Stock and related ETFs
        - Correlated pairs

        MAR Article 12(1)(c) prohibits manipulation across related instruments.
        """
        if symbol not in self._symbol_correlations:
            return

        related_symbols = self._symbol_correlations[symbol]
        if not related_symbols:
            return

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self._cross_market_time_window_seconds)

        # Get orders for this symbol and related symbols
        primary_orders = [
            o for o in self._order_history
            if o.symbol == symbol and o.timestamp >= window_start
        ]

        related_orders = [
            o for o in self._order_history
            if o.symbol in related_symbols and o.timestamp >= window_start
        ]

        if len(primary_orders) < 3 or len(related_orders) < 3:
            return

        # Check for coordinated cancellation pattern
        primary_cancels = [o for o in primary_orders if o.status == "cancelled"]
        related_cancels = [o for o in related_orders if o.status == "cancelled"]

        if len(primary_cancels) >= 3 and len(related_cancels) >= 3:
            # Check if cancellations are synchronized (within 5 seconds)
            synchronized_pairs = 0
            for pc in primary_cancels:
                if pc.cancel_timestamp is None:
                    continue
                for rc in related_cancels:
                    if rc.cancel_timestamp is None:
                        continue
                    time_diff = abs((pc.cancel_timestamp - rc.cancel_timestamp).total_seconds())
                    if time_diff < 5:
                        synchronized_pairs += 1
                        break

            if synchronized_pairs >= 3:
                all_symbols = [symbol] + related_symbols
                order_ids = [o.order_id for o in primary_cancels + related_cancels]

                self._generate_alert(
                    alert_type=SurveillanceAlertType.CROSS_MARKET_MANIPULATION,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    description=(
                        f"Coordinated cancellation pattern detected across related instruments: "
                        f"{symbol} and {related_symbols}. {synchronized_pairs} synchronized "
                        f"cancellation pairs within {self._cross_market_time_window_seconds}s"
                    ),
                    evidence={
                        "primary_symbol": symbol,
                        "related_symbols": related_symbols,
                        "primary_cancels": len(primary_cancels),
                        "related_cancels": len(related_cancels),
                        "synchronized_pairs": synchronized_pairs,
                        "time_window_seconds": self._cross_market_time_window_seconds,
                    },
                    orders_involved=order_ids,
                    recommended_action="Review cross-market activity for MAR Article 12(1)(c) violation",
                    symbols_involved=all_symbols,
                )

        # Check for coordinated directional trading
        primary_buys = sum(1 for o in primary_orders if o.side == "buy" and o.status == "filled")
        primary_sells = sum(1 for o in primary_orders if o.side == "sell" and o.status == "filled")
        related_buys = sum(1 for o in related_orders if o.side == "buy" and o.status == "filled")
        related_sells = sum(1 for o in related_orders if o.side == "sell" and o.status == "filled")

        # Detect coordinated one-sided trading
        if (primary_buys > primary_sells * 2 and related_buys > related_sells * 2):
            if primary_buys >= 5 and related_buys >= 5:
                self._generate_alert(
                    alert_type=SurveillanceAlertType.COORDINATED_TRADING,
                    severity=AlertSeverity.MEDIUM,
                    symbol=symbol,
                    description=(
                        f"Coordinated buying detected across {symbol} and related instruments: "
                        f"Primary buys={primary_buys}, Related buys={related_buys}"
                    ),
                    evidence={
                        "primary_symbol": symbol,
                        "related_symbols": related_symbols,
                        "primary_buys": primary_buys,
                        "primary_sells": primary_sells,
                        "related_buys": related_buys,
                        "related_sells": related_sells,
                    },
                    recommended_action="Review for potential coordinated manipulation",
                    symbols_involved=[symbol] + related_symbols,
                )

    def set_symbol_correlations(self, correlations: dict[str, list[str]]) -> None:
        """
        Set symbol correlation map for cross-market surveillance (P2).

        Args:
            correlations: Dict mapping symbol to list of related symbols
        """
        self._symbol_correlations = correlations
        logger.info(f"SurveillanceAgent: Updated symbol correlations for {len(correlations)} symbols")

    # =========================================================================
    # P2: ADVANCED PATTERN RECOGNITION
    # =========================================================================

    async def _detect_momentum_ignition(self, symbol: str) -> None:
        """
        Detect potential momentum ignition (P2).

        Momentum ignition: A strategy that involves entering orders or trades
        with the intent of starting or exacerbating a trend, and then trading
        out at a profit.

        MAR Article 12(2)(a)(i) - manipulative orders to start a trend.

        Detection:
        1. Rapid price movement (>2%) following aggressive orders
        2. Followed by reversal trades from same account
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=5)

        recent_orders = [
            o for o in self._order_history
            if o.symbol == symbol and o.timestamp >= window_start
        ]

        if len(recent_orders) < 10:
            return

        # Group by minute to detect rapid activity
        filled_orders = [o for o in recent_orders if o.status == "filled" and o.fill_price]
        if len(filled_orders) < 5:
            return

        # Check for price movement pattern
        filled_orders.sort(key=lambda o: o.timestamp)

        # Get early and late prices
        early_prices = [o.fill_price for o in filled_orders[:3] if o.fill_price]
        late_prices = [o.fill_price for o in filled_orders[-3:] if o.fill_price]

        if not early_prices or not late_prices:
            return

        avg_early = sum(early_prices) / len(early_prices)
        avg_late = sum(late_prices) / len(late_prices)

        if avg_early <= 0:
            return

        price_change_pct = (avg_late - avg_early) / avg_early

        # Check for significant price movement
        if abs(price_change_pct) >= self._momentum_ignition_threshold:
            # Check for reversal - early orders one direction, late orders opposite
            early_buys = sum(1 for o in filled_orders[:5] if o.side == "buy")
            early_sells = sum(1 for o in filled_orders[:5] if o.side == "sell")
            late_buys = sum(1 for o in filled_orders[-5:] if o.side == "buy")
            late_sells = sum(1 for o in filled_orders[-5:] if o.side == "sell")

            # Pattern: heavy buying then heavy selling (or vice versa)
            if (early_buys > early_sells * 2 and late_sells > late_buys * 2):
                self._generate_alert(
                    alert_type=SurveillanceAlertType.MOMENTUM_IGNITION,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    description=(
                        f"Potential momentum ignition: {symbol} moved {price_change_pct:.2%} "
                        f"with early aggressive buying ({early_buys} buys) followed by "
                        f"selling reversal ({late_sells} sells)"
                    ),
                    evidence={
                        "price_change_pct": price_change_pct,
                        "early_avg_price": avg_early,
                        "late_avg_price": avg_late,
                        "early_buys": early_buys,
                        "early_sells": early_sells,
                        "late_buys": late_buys,
                        "late_sells": late_sells,
                        "total_orders": len(filled_orders),
                    },
                    orders_involved=[o.order_id for o in filled_orders],
                    recommended_action="Investigate for momentum ignition - MAR Article 12(2)(a)(i)",
                )
            elif (early_sells > early_buys * 2 and late_buys > late_sells * 2):
                self._generate_alert(
                    alert_type=SurveillanceAlertType.MOMENTUM_IGNITION,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    description=(
                        f"Potential momentum ignition: {symbol} moved {price_change_pct:.2%} "
                        f"with early aggressive selling ({early_sells} sells) followed by "
                        f"buying reversal ({late_buys} buys)"
                    ),
                    evidence={
                        "price_change_pct": price_change_pct,
                        "early_avg_price": avg_early,
                        "late_avg_price": avg_late,
                        "early_buys": early_buys,
                        "early_sells": early_sells,
                        "late_buys": late_buys,
                        "late_sells": late_sells,
                        "total_orders": len(filled_orders),
                    },
                    orders_involved=[o.order_id for o in filled_orders],
                    recommended_action="Investigate for momentum ignition - MAR Article 12(2)(a)(i)",
                )

    async def _detect_painting_the_tape(self, symbol: str) -> None:
        """
        Detect potential painting the tape (P2).

        Painting the tape: A series of transactions designed to create
        the appearance of active trading or to move the price, typically
        involving small trades at increasing/decreasing prices.

        MAR Article 12(1)(a)(i) - fictitious devices or deception.

        Detection:
        1. Many small trades in rapid succession
        2. Consistent price direction (stair-stepping)
        3. Similar trade sizes
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=10)

        filled_orders = [
            o for o in self._order_history
            if o.symbol == symbol
            and o.timestamp >= window_start
            and o.status == "filled"
            and o.fill_price
        ]

        if len(filled_orders) < 10:
            return

        # Sort by timestamp
        filled_orders.sort(key=lambda o: o.timestamp)

        # Check for stair-stepping pattern
        prices = [o.fill_price for o in filled_orders if o.fill_price]
        quantities = [o.fill_quantity for o in filled_orders]

        if len(prices) < 10:
            return

        # Count consecutive price increases/decreases
        increases = 0
        decreases = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                increases += 1
            elif prices[i] < prices[i - 1]:
                decreases += 1

        total_moves = increases + decreases
        if total_moves == 0:
            return

        # Pattern: >70% moves in one direction
        direction_ratio = max(increases, decreases) / total_moves

        # Check for similar trade sizes (coefficient of variation < 0.5)
        avg_qty = sum(quantities) / len(quantities)
        if avg_qty <= 0:
            return
        qty_std = (sum((q - avg_qty) ** 2 for q in quantities) / len(quantities)) ** 0.5
        cv = qty_std / avg_qty

        # Suspicious if direction is consistent AND trade sizes are similar AND trades are small
        max_qty = max(quantities)
        if direction_ratio > 0.7 and cv < 0.5 and max_qty < 500:
            direction = "up" if increases > decreases else "down"
            price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0

            self._generate_alert(
                alert_type=SurveillanceAlertType.PAINTING_THE_TAPE,
                severity=AlertSeverity.MEDIUM,
                symbol=symbol,
                description=(
                    f"Potential painting the tape: {len(filled_orders)} small trades "
                    f"({avg_qty:.0f} avg size) moving price {direction} "
                    f"({price_change:.2%} total move, {direction_ratio:.0%} directional)"
                ),
                evidence={
                    "trade_count": len(filled_orders),
                    "avg_quantity": avg_qty,
                    "quantity_cv": cv,
                    "direction": direction,
                    "direction_ratio": direction_ratio,
                    "price_increases": increases,
                    "price_decreases": decreases,
                    "total_price_change_pct": price_change,
                    "start_price": prices[0],
                    "end_price": prices[-1],
                },
                orders_involved=[o.order_id for o in filled_orders],
                recommended_action="Review for painting the tape - MAR Article 12(1)(a)(i)",
            )

    def _should_generate_stor(self, severity: AlertSeverity) -> bool:
        """Check if severity meets threshold for STOR generation."""
        severity_order = [AlertSeverity.INFO, AlertSeverity.LOW, AlertSeverity.MEDIUM,
                        AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        return severity_order.index(severity) >= severity_order.index(self._stor_severity_threshold)

    def _create_stor_from_alert(self, alert: SurveillanceAlert) -> STORReport:
        """
        Create a STOR from a surveillance alert (#C2).

        Per MAR Article 16, firms must report suspicious transactions
        "without delay" to the competent authority.
        """
        self._stor_counter += 1
        stor_id = f"STOR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._stor_counter:04d}"

        # Map alert type to STOR activity type
        activity_type_map = {
            SurveillanceAlertType.WASH_TRADING: "Wash trading (MAR Art.12(1)(a))",
            SurveillanceAlertType.SPOOFING: "Spoofing/Layering (MAR Art.12(2)(a))",
            SurveillanceAlertType.QUOTE_STUFFING: "Quote stuffing (MAR Art.12(2)(c))",
            SurveillanceAlertType.LAYERING: "Layering (MAR Art.12(2)(a))",
            SurveillanceAlertType.UNUSUAL_VOLUME: "Unusual trading volume",
            SurveillanceAlertType.PRICE_MANIPULATION: "Price manipulation (MAR Art.12(1)(a))",
        }

        # Get instrument details
        isin = self._isin_map.get(alert.symbol, "")
        venue = self._venue_map.get(alert.symbol, "XPAR")  # Default Paris

        # Calculate involved values
        total_volume = 0
        total_value = 0.0
        for order_id in alert.orders_involved:
            if order_id in self._orders:
                order = self._orders[order_id]
                total_volume += order.quantity
                if order.fill_price:
                    total_value += order.quantity * order.fill_price

        stor = STORReport(
            stor_id=stor_id,
            created_timestamp=datetime.now(timezone.utc),
            status=STORStatus.DRAFT,

            # Reporting entity
            reporting_entity_lei=self._firm_lei,
            reporting_entity_name=self._firm_name,
            reporting_entity_country=self._firm_country,
            contact_person_name=self._compliance_contact_name,
            contact_person_email=self._compliance_contact_email,
            contact_person_phone=self._compliance_contact_phone,

            # Instrument
            instrument_isin=isin,
            instrument_name=alert.symbol,
            trading_venue_mic=venue,

            # Suspicious activity
            description_of_suspicion=alert.description,
            type_of_suspicious_activity=activity_type_map.get(
                alert.alert_type, alert.alert_type.value
            ),
            start_date=alert.timestamp,
            end_date=alert.timestamp,

            # Involved orders/transactions
            orders_involved=list(alert.orders_involved),
            total_volume_involved=total_volume,
            total_value_involved=total_value,

            # Evidence
            evidence_summary=f"Evidence: {alert.evidence}",
            analysis_methodology="Automated surveillance detection algorithms",
            related_alerts=[alert.alert_id],
        )

        self._stor_reports.append(stor)

        # Log STOR creation
        self._audit_logger.log_agent_event(
            agent_name=self.name,
            event_type="stor_created",
            details={
                "stor_id": stor_id,
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
            },
        )

        logger.info(f"STOR created: {stor_id} from alert {alert.alert_id}")

        return stor

    def submit_stor(self, stor_id: str) -> bool:
        """
        Submit a STOR to the National Competent Authority (#C2).

        Per MAR Article 16(1), submission must be made "without delay"
        to the competent authority of the most relevant market.

        In production, this would submit to:
        - AMF (France): https://stor.amf-france.org/
        - FCA (UK): https://stor.fca.org.uk/
        - BaFin (Germany), etc.

        Returns:
            True if submission successful
        """
        stor = self.get_stor_by_id(stor_id)
        if not stor:
            logger.error(f"STOR {stor_id} not found")
            return False

        if stor.status == STORStatus.SUBMITTED:
            logger.warning(f"STOR {stor_id} already submitted")
            return True

        # Validate STOR before submission
        validation_errors = self._validate_stor(stor)
        if validation_errors:
            logger.error(f"STOR validation failed: {validation_errors}")
            return False

        # Mark as submitted (in production, would send to NCA API)
        stor.status = STORStatus.SUBMITTED
        stor.submitted_timestamp = datetime.now(timezone.utc)

        # Generate mock NCA reference
        stor.nca_reference = f"AMF-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # Log submission
        self._audit_logger.log_agent_event(
            agent_name=self.name,
            event_type="stor_submitted",
            details={
                "stor_id": stor_id,
                "nca_reference": stor.nca_reference,
                "submitted_timestamp": stor.submitted_timestamp.isoformat(),
            },
        )

        logger.info(f"STOR {stor_id} submitted to NCA (ref: {stor.nca_reference})")

        return True

    def _validate_stor(self, stor: STORReport) -> list[str]:
        """
        Validate STOR before submission per MAR Article 16 requirements.

        MAR Article 16 required fields:
        1. Reporting entity identification (LEI, name, country)
        2. Contact person details
        3. Instrument identification (ISIN or name + venue)
        4. Description and type of suspicious activity
        5. Dates of suspicious activity
        6. Evidence and supporting documentation
        """
        errors = []

        # 1. Reporting entity identification (mandatory)
        if not stor.reporting_entity_lei:
            errors.append("MAR Art.16(1): Missing reporting entity LEI")
        elif len(stor.reporting_entity_lei) != 20:
            errors.append("MAR Art.16(1): Invalid LEI format (must be 20 characters)")

        if not stor.reporting_entity_name:
            errors.append("MAR Art.16(1): Missing reporting entity name")

        if not stor.reporting_entity_country:
            errors.append("MAR Art.16(1): Missing reporting entity country")

        # 2. Contact person details (mandatory for follow-up)
        if not stor.contact_person_name:
            errors.append("MAR Art.16(2): Missing contact person name")

        if not stor.contact_person_email and not stor.contact_person_phone:
            errors.append("MAR Art.16(2): Missing contact details (email or phone required)")

        # 3. Instrument identification (mandatory)
        if not stor.instrument_isin and not stor.instrument_name:
            errors.append("MAR Art.16(3): Missing instrument identification (ISIN or name required)")

        if not stor.trading_venue_mic:
            errors.append("MAR Art.16(3): Missing trading venue MIC code")

        # 4. Description and type of suspicious activity (mandatory)
        if not stor.description_of_suspicion:
            errors.append("MAR Art.16(4): Missing description of suspicion")
        elif len(stor.description_of_suspicion) < 50:
            errors.append("MAR Art.16(4): Description of suspicion too brief (minimum 50 characters)")

        if not stor.type_of_suspicious_activity:
            errors.append("MAR Art.16(4): Missing type of suspicious activity")

        # 5. Dates of suspicious activity (mandatory)
        if not stor.start_date:
            errors.append("MAR Art.16(5): Missing start date of suspicious activity")

        if stor.start_date and stor.end_date and stor.end_date < stor.start_date:
            errors.append("MAR Art.16(5): End date cannot be before start date")

        # 6. Evidence (mandatory)
        if not stor.orders_involved and not stor.transactions_involved:
            errors.append("MAR Art.16(6): No orders or transactions referenced")

        if not stor.evidence_summary:
            errors.append("MAR Art.16(6): Missing evidence summary")

        return errors

    def get_stor_by_id(self, stor_id: str) -> STORReport | None:
        """Get a STOR by its ID."""
        for stor in self._stor_reports:
            if stor.stor_id == stor_id:
                return stor
        return None

    def get_pending_stors(self) -> list[STORReport]:
        """Get STORs that need to be submitted."""
        return [
            s for s in self._stor_reports
            if s.status in [STORStatus.DRAFT, STORStatus.PENDING_REVIEW, STORStatus.APPROVED]
        ]

    def get_stor_summary(self) -> dict[str, Any]:
        """Get summary of STOR reports (#C2)."""
        return {
            "total_stors": len(self._stor_reports),
            "by_status": {
                status.value: len([s for s in self._stor_reports if s.status == status])
                for status in STORStatus
            },
            "pending_submission": len(self.get_pending_stors()),
            "submitted_today": len([
                s for s in self._stor_reports
                if s.submitted_timestamp and
                s.submitted_timestamp.date() == datetime.now(timezone.utc).date()
            ]),
        }

    def _trim_history(self) -> None:
        """Trim old order history to prevent memory growth."""
        max_history = 10000
        if len(self._order_history) > max_history:
            self._order_history = self._order_history[-max_history:]

        if len(self._fills) > max_history:
            self._fills = self._fills[-max_history:]

        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

    def get_alerts(
        self,
        hours: int = 24,
        alert_type: SurveillanceAlertType | None = None,
        severity: AlertSeverity | None = None
    ) -> list[SurveillanceAlert]:
        """
        Get recent surveillance alerts.

        Args:
            hours: Lookback period
            alert_type: Filter by alert type
            severity: Filter by severity

        Returns:
            List of alerts
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        alerts = [a for a in self._alerts if a.timestamp >= cutoff]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_symbol_metrics(self, symbol: str) -> dict[str, Any]:
        """Get surveillance metrics for a symbol."""
        return dict(self._symbol_metrics[symbol])

    def get_status(self) -> dict[str, Any]:
        """Get agent status for monitoring."""
        base_status = super().get_status()

        # STOR summary (#C2)
        stor_summary = self.get_stor_summary()

        # P2: Get top priority alerts
        top_priority = self.get_prioritized_alerts(hours=24, limit=5)

        base_status.update({
            "orders_tracked": len(self._orders),
            "fills_tracked": len(self._fills),
            "total_alerts": len(self._alerts),
            "recent_alerts_24h": len(self.get_alerts(24)),
            "high_severity_alerts": len([
                a for a in self._alerts
                if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            ]),
            "detection_enabled": {
                "wash_trading": self._wash_trading_detection,
                "spoofing": self._spoofing_detection,
                "quote_stuffing": self._quote_stuffing_detection,
                "layering": self._layering_detection,
                # P2: New detections
                "momentum_ignition": self._momentum_ignition_detection,
                "painting_tape": self._painting_tape_detection,
                "cross_market": self._cross_market_surveillance,
            },
            # STOR metrics (#C2 - MAR Article 16)
            "stor": {
                "total": stor_summary["total_stors"],
                "pending_submission": stor_summary["pending_submission"],
                "submitted_today": stor_summary["submitted_today"],
                "by_status": stor_summary["by_status"],
            },
            # P2: Alert prioritization
            "alert_prioritization": {
                "enabled": self._alert_priority_enabled,
                "top_priority_alerts": [
                    {
                        "alert_id": a.alert_id,
                        "type": a.alert_type.value,
                        "severity": a.severity.value,
                        "symbol": a.symbol,
                        "priority_score": a.priority_score,
                    }
                    for a in top_priority
                ],
            },
            # P2: Cross-market surveillance
            "cross_market": {
                "enabled": self._cross_market_surveillance,
                "symbols_monitored": len(self._symbol_correlations),
                "correlation_threshold": self._cross_market_correlation_threshold,
            },
        })

        return base_status

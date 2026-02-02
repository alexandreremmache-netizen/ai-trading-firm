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
        self._spoofing_cancel_threshold = self._surveillance_config.get(
            "spoofing_cancel_threshold", 0.8
        )  # 80% cancel rate
        self._quote_stuffing_rate_per_second = self._surveillance_config.get(
            "quote_stuffing_rate_per_second", 10
        )
        self._layering_level_threshold = self._surveillance_config.get(
            "layering_level_threshold", 3
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

        logger.info(
            f"SurveillanceAgent initialized: "
            f"wash={self._wash_trading_detection}, "
            f"spoof={self._spoofing_detection}, "
            f"quote_stuff={self._quote_stuffing_detection}, "
            f"layer={self._layering_detection}, "
            f"stor_auto={self._stor_auto_generate}"
        )

    async def initialize(self) -> None:
        """Initialize the agent."""
        logger.info("SurveillanceAgent initialized")

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

    async def _detect_wash_trading(self, symbol: str) -> None:
        """
        Detect potential wash trading.

        Wash trading: Simultaneous or near-simultaneous buy and sell
        orders in the same instrument that result in no change in
        beneficial ownership.
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self._wash_trading_window_seconds)

        # Get recent orders for this symbol
        recent_orders = [
            o for o in self._order_history
            if o.symbol == symbol and o.timestamp >= window_start
        ]

        if len(recent_orders) < 2:
            return

        # Look for offsetting trades
        buys = [o for o in recent_orders if o.side == "buy" and o.status == "filled"]
        sells = [o for o in recent_orders if o.side == "sell" and o.status == "filled"]

        for buy in buys:
            for sell in sells:
                # Check for similar quantity and timing
                time_diff = abs((buy.timestamp - sell.timestamp).total_seconds())

                if time_diff <= self._wash_trading_window_seconds:
                    # Check quantity similarity (within 10%)
                    qty_ratio = min(buy.quantity, sell.quantity) / max(buy.quantity, sell.quantity)

                    if qty_ratio >= 0.9:
                        # Check price similarity
                        if buy.fill_price and sell.fill_price:
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
        Detect potential spoofing.

        Spoofing: Entering orders with intent to cancel before execution
        to create false impression of demand/supply.
        """
        metrics = self._symbol_metrics[symbol]

        # Need minimum order count for meaningful analysis
        if metrics["orders_submitted"] < 10:
            return

        cancel_rate = metrics["cancel_rate"]

        if cancel_rate >= self._spoofing_cancel_threshold:
            # Check for pattern: large cancelled orders followed by executions
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(minutes=5)

            recent = [
                o for o in self._order_history
                if o.symbol == symbol and o.timestamp >= window_start
            ]

            cancelled = [o for o in recent if o.status == "cancelled"]
            filled = [o for o in recent if o.status == "filled"]

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
        Detect potential layering.

        Layering: Entering multiple orders at different price levels
        to create artificial depth, then cancelling after execution
        on the opposite side.
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

        # Check for layering pattern
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

    def _generate_alert(
        self,
        alert_type: SurveillanceAlertType,
        severity: AlertSeverity,
        symbol: str,
        description: str,
        evidence: dict[str, Any],
        orders_involved: list[str] | None = None,
        recommended_action: str = ""
    ) -> SurveillanceAlert:
        """Generate and store a surveillance alert."""
        self._alert_counter += 1
        alert_id = f"SURV-{self._alert_counter:06d}"

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
        )

        self._alerts.append(alert)

        # Log the alert
        self._audit_logger.log_agent_event(
            agent_name=self.name,
            event_type="surveillance_alert",
            details=alert.to_dict(),
        )

        logger.warning(f"Surveillance alert: [{severity.value}] {alert_type.value} - {description}")

        # Auto-generate STOR for HIGH/CRITICAL alerts (#C2)
        if self._stor_auto_generate and self._should_generate_stor(severity):
            self._create_stor_from_alert(alert)

        return alert

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
        stor.nca_reference = f"AMF-{datetime.now().strftime('%Y%m%d%H%M%S')}"

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
        """Validate STOR before submission."""
        errors = []

        if not stor.reporting_entity_lei:
            errors.append("Missing reporting entity LEI")

        if not stor.instrument_isin and not stor.instrument_name:
            errors.append("Missing instrument identification")

        if not stor.description_of_suspicion:
            errors.append("Missing description of suspicion")

        if not stor.type_of_suspicious_activity:
            errors.append("Missing type of suspicious activity")

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
            },
            # STOR metrics (#C2 - MAR Article 16)
            "stor": {
                "total": stor_summary["total_stors"],
                "pending_submission": stor_summary["pending_submission"],
                "submitted_today": stor_summary["submitted_today"],
                "by_status": stor_summary["by_status"],
            },
        })

        return base_status

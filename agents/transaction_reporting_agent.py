"""
Transaction Reporting Agent
===========================

Implements ESMA RTS 22/23 transaction reporting requirements.
Generates reports within 15 minutes of execution per MiFID II.

Features:
- LEI validation
- AMF BDIF format support
- 65 required field population
- Report queue management
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from core.agent_base import BaseAgent, AgentConfig
from core.events import Event, EventType, FillEvent


logger = logging.getLogger(__name__)


class ReportStatus(Enum):
    """Status of a transaction report."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class TransactionType(Enum):
    """Transaction type codes per ESMA RTS."""
    BUYI = "BUYI"  # Buy
    SELL = "SELL"  # Sell
    SECU = "SECU"  # Securities lending/borrowing


class BuySellIndicator(Enum):
    """Buy/Sell indicator."""
    BUYI = "BUYI"
    SELL = "SELL"


@dataclass
class LEIInfo:
    """Legal Entity Identifier information."""
    lei: str
    legal_name: str
    country: str
    is_valid: bool = True
    validation_date: datetime | None = None


@dataclass
class TransactionReport:
    """
    Transaction report per ESMA RTS 22/23.

    Contains the 65 required fields for MiFID II transaction reporting.
    """
    # Report metadata
    report_id: str
    reporting_timestamp: datetime
    status: ReportStatus = ReportStatus.PENDING

    # Field 1-5: Report identification
    transaction_reference_number: str = ""  # Field 1
    trading_venue_transaction_id: str = ""  # Field 2
    executing_entity_id: str = ""  # Field 3 (LEI)
    investment_firm_covered: bool = True  # Field 4
    submitting_entity_id: str = ""  # Field 5 (LEI)

    # Field 6-10: Buyer identification
    buyer_identification_code: str = ""  # Field 6
    buyer_country: str = ""  # Field 7
    buyer_first_name: str = ""  # Field 8
    buyer_surname: str = ""  # Field 9
    buyer_date_of_birth: str = ""  # Field 10

    # Field 11-15: Buyer decision maker
    buyer_decision_maker_code: str = ""  # Field 11
    buyer_decision_maker_country: str = ""  # Field 12
    buyer_decision_maker_first_name: str = ""  # Field 13
    buyer_decision_maker_surname: str = ""  # Field 14
    buyer_decision_maker_dob: str = ""  # Field 15

    # Field 16-20: Seller identification
    seller_identification_code: str = ""  # Field 16
    seller_country: str = ""  # Field 17
    seller_first_name: str = ""  # Field 18
    seller_surname: str = ""  # Field 19
    seller_date_of_birth: str = ""  # Field 20

    # Field 21-25: Seller decision maker
    seller_decision_maker_code: str = ""  # Field 21
    seller_decision_maker_country: str = ""  # Field 22
    seller_decision_maker_first_name: str = ""  # Field 23
    seller_decision_maker_surname: str = ""  # Field 24
    seller_decision_maker_dob: str = ""  # Field 25

    # Field 26-30: Transmission
    transmission_indicator: bool = False  # Field 26
    transmitting_firm_buyer: str = ""  # Field 27
    transmitting_firm_seller: str = ""  # Field 28
    trading_date_time: datetime | None = None  # Field 29
    trading_capacity: str = "DEAL"  # Field 30 (DEAL/MTCH/AOTC/RPTD)

    # Field 31-35: Quantity
    quantity: float = 0.0  # Field 31
    quantity_currency: str = ""  # Field 32
    derivative_notional_increase: float = 0.0  # Field 33
    derivative_notional_decrease: float = 0.0  # Field 34
    price: float = 0.0  # Field 35

    # Field 36-40: Price details
    price_currency: str = ""  # Field 36
    net_amount: float = 0.0  # Field 37
    venue: str = ""  # Field 38 (MIC)
    country_of_branch: str = ""  # Field 39
    upfront_payment: float = 0.0  # Field 40

    # Field 41-45: Instrument identification
    upfront_payment_currency: str = ""  # Field 41
    instrument_identification_code: str = ""  # Field 42 (ISIN)
    instrument_full_name: str = ""  # Field 43
    instrument_classification: str = ""  # Field 44 (CFI)
    notional_currency_1: str = ""  # Field 45

    # Field 46-50: Derivative details
    notional_currency_2: str = ""  # Field 46
    price_multiplier: float = 1.0  # Field 47
    underlying_instrument_code: str = ""  # Field 48
    underlying_index_name: str = ""  # Field 49
    underlying_index_term: str = ""  # Field 50

    # Field 51-55: Option details
    option_type: str = ""  # Field 51 (CALL/PUTO)
    strike_price: float = 0.0  # Field 52
    strike_price_currency: str = ""  # Field 53
    option_exercise_style: str = ""  # Field 54 (EURO/AMER)
    maturity_date: str = ""  # Field 55

    # Field 56-60: Derivative dates
    expiry_date: str = ""  # Field 56
    delivery_type: str = ""  # Field 57 (CASH/PHYS)
    investment_decision_within_firm: str = ""  # Field 58
    country_of_investment_decision_branch: str = ""  # Field 59
    execution_within_firm: str = ""  # Field 60

    # Field 61-65: Execution details
    country_of_execution_branch: str = ""  # Field 61
    waiver_indicator: str = ""  # Field 62
    short_selling_indicator: str = ""  # Field 63
    otc_post_trade_indicator: str = ""  # Field 64
    commodity_derivative_indicator: bool = False  # Field 65

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "reporting_timestamp": self.reporting_timestamp.isoformat(),
            "status": self.status.value,
            "transaction_reference_number": self.transaction_reference_number,
            "executing_entity_id": self.executing_entity_id,
            "trading_date_time": self.trading_date_time.isoformat() if self.trading_date_time else None,
            "instrument_identification_code": self.instrument_identification_code,
            "quantity": self.quantity,
            "price": self.price,
            "price_currency": self.price_currency,
            "venue": self.venue,
            "buyer_identification_code": self.buyer_identification_code,
            "seller_identification_code": self.seller_identification_code,
        }

    def to_bdif_format(self) -> dict[str, Any]:
        """
        Convert to AMF BDIF (Banque de France Data Input Format).

        Returns structured format for French regulator submission.
        """
        return {
            "Header": {
                "MessageType": "NEWT",
                "ReportingEntityId": self.executing_entity_id,
                "ReportDate": self.reporting_timestamp.strftime("%Y-%m-%d"),
            },
            "Transaction": {
                "TxnRefNo": self.transaction_reference_number,
                "TradingVenueTxnId": self.trading_venue_transaction_id,
                "ExecutingEntityId": self.executing_entity_id,
                "SubmittingEntityId": self.submitting_entity_id,
                "BuyerDetails": {
                    "BuyerId": self.buyer_identification_code,
                    "BuyerCountry": self.buyer_country,
                },
                "SellerDetails": {
                    "SellerId": self.seller_identification_code,
                    "SellerCountry": self.seller_country,
                },
                "TradingDateTime": self.trading_date_time.isoformat() if self.trading_date_time else "",
                "TradingCapacity": self.trading_capacity,
                "Quantity": self.quantity,
                "Price": self.price,
                "PriceCurrency": self.price_currency,
                "NetAmount": self.net_amount,
                "Venue": self.venue,
                "InstrumentId": self.instrument_identification_code,
                "InstrumentName": self.instrument_full_name,
                "InstrumentClassification": self.instrument_classification,
            },
        }


class TransactionReportingAgent(BaseAgent):
    """
    Transaction reporting agent for MiFID II compliance.

    Generates and submits transaction reports within 15 minutes
    of execution as required by ESMA RTS 22/23.
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus,
        audit_logger,
        reporting_config: dict[str, Any] | None = None
    ):
        """
        Initialize transaction reporting agent.

        Args:
            config: Agent configuration
            event_bus: Event bus
            audit_logger: Audit logger
            reporting_config: Reporting-specific configuration
        """
        super().__init__(config, event_bus, audit_logger)

        self._reporting_config = reporting_config or {}

        # Configuration
        self._enabled = self._reporting_config.get("enabled", True)
        self._deadline_minutes = self._reporting_config.get("reporting_deadline_minutes", 15)
        self._firm_lei = self._reporting_config.get("firm_lei", "")
        self._firm_country = self._reporting_config.get("firm_country", "FR")
        self._default_venue = self._reporting_config.get("default_venue", "XPAR")

        # Report queue
        self._pending_reports: deque[TransactionReport] = deque()
        self._submitted_reports: list[TransactionReport] = []
        self._report_counter = 0

        # Deadline tracking (#C4)
        self._deadline_breaches = 0
        self._deadline_warnings = 0

        # LEI cache
        self._lei_cache: dict[str, LEIInfo] = {}

        # Instrument mapping
        self._isin_map: dict[str, str] = self._reporting_config.get("isin_map", {})

        # Submission task
        self._submission_task: asyncio.Task | None = None

        logger.info(
            f"TransactionReportingAgent initialized: "
            f"enabled={self._enabled}, deadline={self._deadline_minutes}min"
        )

    async def initialize(self) -> None:
        """Initialize the agent."""
        # Validate firm LEI at startup (Issue #I3)
        if self._enabled:
            if not self._firm_lei:
                logger.critical(
                    "CRITICAL: firm_lei not configured! "
                    "Transaction reporting will fail. "
                    "Set transaction_reporting.firm_lei in config.yaml"
                )
            elif not self._validate_lei(self._firm_lei):
                logger.critical(
                    f"CRITICAL: Invalid or placeholder LEI detected: {self._firm_lei}. "
                    "Replace with a valid LEI before production! "
                    "Get your LEI from a Local Operating Unit (LOU) "
                    "registered with GLEIF: https://www.gleif.org/"
                )
                # Disable reporting if LEI is invalid
                self._enabled = False
                logger.warning(
                    "Transaction reporting DISABLED due to invalid LEI. "
                    "Reports will not be generated until a valid LEI is configured."
                )
            else:
                logger.info(f"Firm LEI validated: {self._firm_lei[:4]}...{self._firm_lei[-4:]}")

        if self._enabled:
            # Start submission loop
            self._submission_task = asyncio.create_task(self._submission_loop())
        logger.info(f"TransactionReportingAgent initialized (enabled={self._enabled})")

    async def stop(self) -> None:
        """Stop the agent."""
        if self._submission_task:
            self._submission_task.cancel()
            try:
                await self._submission_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    def get_subscribed_events(self) -> list[EventType]:
        """Subscribe to fill events."""
        return [EventType.FILL]

    async def process_event(self, event: Event) -> None:
        """Process incoming events."""
        if event.event_type == EventType.FILL:
            await self._process_fill(event)

    async def _process_fill(self, event: FillEvent) -> None:
        """Process a fill event and generate transaction report."""
        if not self._enabled:
            return

        # Generate report
        report = self._create_report_from_fill(event)

        # Add to queue
        self._pending_reports.append(report)

        logger.info(
            f"Transaction report queued: {report.report_id} - "
            f"{event.symbol} {event.side.value} {event.filled_quantity} @ {event.fill_price}"
        )

    def _create_report_from_fill(self, fill: FillEvent) -> TransactionReport:
        """Create a transaction report from a fill event."""
        self._report_counter += 1
        report_id = f"TXN-{self._report_counter:08d}"

        # Determine buyer/seller based on side
        if fill.side.value == "buy":
            buyer_id = self._firm_lei
            seller_id = ""  # Market counterparty
        else:
            buyer_id = ""  # Market counterparty
            seller_id = self._firm_lei

        # Get ISIN if available
        isin = self._isin_map.get(fill.symbol, "")

        report = TransactionReport(
            report_id=report_id,
            reporting_timestamp=datetime.now(timezone.utc),

            # Identification
            transaction_reference_number=report_id,
            trading_venue_transaction_id=str(fill.broker_order_id),
            executing_entity_id=self._firm_lei,
            submitting_entity_id=self._firm_lei,

            # Buyer/Seller
            buyer_identification_code=buyer_id,
            buyer_country=self._firm_country if buyer_id else "",
            seller_identification_code=seller_id,
            seller_country=self._firm_country if seller_id else "",

            # Trading details
            trading_date_time=fill.timestamp,
            trading_capacity="DEAL",

            # Quantity and price
            quantity=float(fill.filled_quantity),
            price=fill.fill_price,
            price_currency="USD",  # Assuming USD, should come from config
            net_amount=fill.filled_quantity * fill.fill_price,

            # Venue
            venue=fill.exchange or self._default_venue,
            country_of_branch=self._firm_country,

            # Instrument
            instrument_identification_code=isin,
            instrument_full_name=fill.symbol,

            # Execution
            execution_within_firm="ALGO",  # Algorithm execution
            country_of_execution_branch=self._firm_country,
        )

        return report

    async def _submission_loop(self) -> None:
        """
        Background loop to submit pending reports.

        Checks every 15 seconds (not 60) to ensure timely reporting (#C4).
        """
        while True:
            try:
                await self._process_pending_reports()
                # Check frequently to meet 15-minute deadline (#C4)
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in submission loop: {e}")
                await asyncio.sleep(15)

    async def _process_pending_reports(self) -> None:
        """
        Process and submit pending reports with deadline enforcement (#C4).

        Implements:
        - Warning at 10 minutes (2/3 of deadline)
        - Critical alert at deadline
        - Deadline breach tracking for audit
        """
        now = datetime.now(timezone.utc)
        deadline = timedelta(minutes=self._deadline_minutes)
        warning_threshold = timedelta(minutes=self._deadline_minutes * 2 / 3)  # 10 min for 15 min deadline

        # Sort by age (oldest first) to prioritize urgent reports
        pending_list = list(self._pending_reports)
        pending_list.sort(key=lambda r: r.reporting_timestamp)
        self._pending_reports = deque(pending_list)

        while self._pending_reports:
            report = self._pending_reports[0]
            age = now - report.reporting_timestamp

            # Deadline enforcement (#C4)
            if age >= deadline:
                # CRITICAL: Deadline breached
                self._deadline_breaches += 1
                logger.critical(
                    f"COMPLIANCE VIOLATION: Report {report.report_id} EXCEEDED DEADLINE! "
                    f"Age: {age.total_seconds() / 60:.1f} min (deadline: {self._deadline_minutes} min). "
                    f"This is a MiFID II violation (RTS 22/23)."
                )
                # Log compliance breach for audit
                self._audit_logger.log_agent_event(
                    agent_name=self.name,
                    event_type="transaction_report_deadline_breach",
                    details={
                        "report_id": report.report_id,
                        "age_minutes": age.total_seconds() / 60,
                        "deadline_minutes": self._deadline_minutes,
                        "transaction_time": report.trading_date_time.isoformat() if report.trading_date_time else None,
                        "breach_time": now.isoformat(),
                        "total_breaches": self._deadline_breaches,
                    },
                )
            elif age >= warning_threshold:
                # Warning: approaching deadline
                remaining = deadline - age
                logger.warning(
                    f"URGENT: Report {report.report_id} approaching deadline! "
                    f"Age: {age.total_seconds() / 60:.1f} min, "
                    f"Remaining: {remaining.total_seconds() / 60:.1f} min"
                )

            # Submit report with priority
            success = await self._submit_report(report)

            if success:
                self._pending_reports.popleft()
                report.status = ReportStatus.SUBMITTED
                self._submitted_reports.append(report)

                # Log submission timing
                submission_latency = age.total_seconds() / 60
                if submission_latency < self._deadline_minutes:
                    logger.info(
                        f"Report {report.report_id} submitted within deadline "
                        f"({submission_latency:.1f} min < {self._deadline_minutes} min)"
                    )
            else:
                # Keep in queue for retry
                report.status = ReportStatus.FAILED
                # Don't break - try next report if this one failed
                self._pending_reports.popleft()
                self._pending_reports.append(report)  # Move to end
                break  # Avoid tight loop on repeated failures

    async def _submit_report(self, report: TransactionReport) -> bool:
        """
        Submit report to regulator.

        In production, this would send to the NCA (National Competent Authority).
        For now, we simulate and log.
        """
        try:
            # Validate report
            validation_errors = self._validate_report(report)
            if validation_errors:
                logger.error(f"Report validation failed: {validation_errors}")
                report.status = ReportStatus.REJECTED
                return False

            # Log submission (in production, send to ARM/APA)
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="transaction_report_submitted",
                details=report.to_dict(),
            )

            logger.info(f"Transaction report submitted: {report.report_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit report {report.report_id}: {e}")
            return False

    def _validate_report(self, report: TransactionReport) -> list[str]:
        """
        Validate transaction report.

        Returns list of validation errors, empty if valid.
        """
        errors = []

        # Check required fields
        if not report.transaction_reference_number:
            errors.append("Missing transaction reference number")

        if not report.executing_entity_id:
            errors.append("Missing executing entity ID (LEI)")
        elif not self._validate_lei(report.executing_entity_id):
            errors.append(f"Invalid LEI: {report.executing_entity_id}")

        if not report.trading_date_time:
            errors.append("Missing trading date/time")

        if report.quantity <= 0:
            errors.append("Invalid quantity")

        if report.price <= 0:
            errors.append("Invalid price")

        if not report.venue:
            errors.append("Missing venue (MIC)")

        # Check deadline
        age = datetime.now(timezone.utc) - report.reporting_timestamp
        if age > timedelta(minutes=self._deadline_minutes):
            errors.append(f"Report deadline exceeded: {age.total_seconds() / 60:.1f} min")

        return errors

    # Known placeholder/test LEI patterns to reject
    PLACEHOLDER_LEI_PATTERNS = {
        "PLACEHOLDER",
        "TEST",
        "DUMMY",
        "EXAMPLE",
        "XXXX",
        "0000000000",
        "X" * 20,  # All X's
        "0" * 20,  # All zeros
    }

    def _validate_lei(self, lei: str) -> bool:
        """
        Validate Legal Entity Identifier format.

        LEI is 20 alphanumeric characters.
        Rejects known placeholder patterns per Issue #I3.
        """
        if not lei:
            return False

        lei = lei.upper()

        # Check format: 20 alphanumeric characters
        if not re.match(r"^[A-Z0-9]{20}$", lei):
            return False

        # Check for placeholder patterns (Issue #I3)
        for pattern in self.PLACEHOLDER_LEI_PATTERNS:
            if pattern in lei:
                logger.warning(f"Rejected placeholder LEI pattern: {pattern}")
                return False

        # Check for all same character (another placeholder indicator)
        if len(set(lei)) <= 2:
            logger.warning(f"Rejected LEI with low entropy (likely placeholder)")
            return False

        # In production, would also check against GLEIF database
        return True

    def set_firm_lei(self, lei: str) -> bool:
        """
        Set the firm's LEI.

        Args:
            lei: Legal Entity Identifier

        Returns:
            True if valid and set
        """
        if self._validate_lei(lei):
            self._firm_lei = lei
            return True
        return False

    def add_isin_mapping(self, symbol: str, isin: str) -> None:
        """Add symbol to ISIN mapping."""
        self._isin_map[symbol] = isin

    def get_pending_count(self) -> int:
        """Get number of pending reports."""
        return len(self._pending_reports)

    def get_submitted_reports(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> list[TransactionReport]:
        """Get recently submitted reports."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        reports = [
            r for r in self._submitted_reports
            if r.reporting_timestamp >= cutoff
        ]

        return reports[-limit:]

    def get_report_by_id(self, report_id: str) -> TransactionReport | None:
        """Get a specific report by ID."""
        for report in self._submitted_reports:
            if report.report_id == report_id:
                return report
        for report in self._pending_reports:
            if report.report_id == report_id:
                return report
        return None

    def generate_daily_summary(self, report_date: datetime | None = None) -> dict[str, Any]:
        """
        Generate daily reporting summary.

        Args:
            report_date: Date to summarize (default: today)

        Returns:
            Summary dictionary
        """
        if report_date is None:
            report_date = datetime.now(timezone.utc)

        date_start = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)

        # Filter reports for the day
        day_reports = [
            r for r in self._submitted_reports
            if date_start <= r.reporting_timestamp < date_end
        ]

        # Calculate statistics
        total_reports = len(day_reports)
        total_volume = sum(r.quantity for r in day_reports)
        total_notional = sum(r.net_amount for r in day_reports)

        accepted = sum(1 for r in day_reports if r.status == ReportStatus.ACCEPTED)
        rejected = sum(1 for r in day_reports if r.status == ReportStatus.REJECTED)

        return {
            "report_date": report_date.strftime("%Y-%m-%d"),
            "total_reports": total_reports,
            "accepted": accepted,
            "rejected": rejected,
            "pending": len(self._pending_reports),
            "total_volume": total_volume,
            "total_notional": total_notional,
            "reporting_entity": self._firm_lei,
            "jurisdiction": self._firm_country,
        }

    def get_status(self) -> dict[str, Any]:
        """Get agent status for monitoring."""
        base_status = super().get_status()

        # Calculate oldest pending report age
        oldest_report_age_min = 0.0
        if self._pending_reports:
            oldest = min(r.reporting_timestamp for r in self._pending_reports)
            oldest_report_age_min = (datetime.now(timezone.utc) - oldest).total_seconds() / 60

        base_status.update({
            "enabled": self._enabled,
            "firm_lei": self._firm_lei[:10] + "..." if self._firm_lei else "NOT_SET",
            "deadline_minutes": self._deadline_minutes,
            "pending_reports": len(self._pending_reports),
            "submitted_reports": len(self._submitted_reports),
            "isin_mappings": len(self._isin_map),
            # Deadline enforcement metrics (#C4)
            "deadline_breaches": self._deadline_breaches,
            "oldest_pending_age_min": round(oldest_report_age_min, 1),
            "deadline_at_risk": oldest_report_age_min > self._deadline_minutes * 0.66,
        })

        return base_status

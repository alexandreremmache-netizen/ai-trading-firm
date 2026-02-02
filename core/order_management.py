"""
Order Management Module
=======================

Order amendment support (Issue #E26).
Broker error code mapping (Issue #E27).
Order rejection reason parsing (Issue #E29).
Fill notification latency tracking (Issue #E28).

Features:
- Order amendment/modification workflow
- Comprehensive IB error code mapping
- Rejection reason categorization
- Latency tracking for fills
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable
from collections import deque

logger = logging.getLogger(__name__)


class AmendmentType(str, Enum):
    """Type of order amendment."""
    PRICE_CHANGE = "price_change"
    QUANTITY_CHANGE = "quantity_change"
    PRICE_AND_QUANTITY = "price_and_quantity"
    TIME_IN_FORCE = "time_in_force"
    CANCEL_REPLACE = "cancel_replace"


class AmendmentStatus(str, Enum):
    """Status of amendment request."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"  # Original partially filled before amendment


class RejectionCategory(str, Enum):
    """Categories of order rejection reasons."""
    PRICE = "price"  # Price-related (outside limits, invalid tick)
    QUANTITY = "quantity"  # Size-related (minimum, maximum, lot size)
    MARGIN = "margin"  # Insufficient margin
    POSITION = "position"  # Position limit exceeded
    MARKET = "market"  # Market closed, halted
    REGULATORY = "regulatory"  # Compliance/regulatory block
    TECHNICAL = "technical"  # System/connectivity issues
    SYMBOL = "symbol"  # Invalid symbol, not tradeable
    PERMISSIONS = "permissions"  # No permissions
    UNKNOWN = "unknown"


@dataclass
class OrderAmendment:
    """Order amendment request."""
    amendment_id: str
    order_id: str
    amendment_type: AmendmentType

    # Original values
    original_price: float | None = None
    original_quantity: int | None = None
    original_tif: str | None = None

    # New values
    new_price: float | None = None
    new_quantity: int | None = None
    new_tif: str | None = None

    # Status
    status: AmendmentStatus = AmendmentStatus.PENDING
    submitted_at: datetime | None = None
    completed_at: datetime | None = None

    # Result
    rejection_reason: str = ""
    filled_qty_before_amendment: int = 0

    def to_dict(self) -> dict:
        return {
            'amendment_id': self.amendment_id,
            'order_id': self.order_id,
            'amendment_type': self.amendment_type.value,
            'original_price': self.original_price,
            'original_quantity': self.original_quantity,
            'new_price': self.new_price,
            'new_quantity': self.new_quantity,
            'status': self.status.value,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'rejection_reason': self.rejection_reason,
        }


@dataclass
class FillLatency:
    """Fill notification latency tracking (#E28)."""
    order_id: str
    fill_id: str

    # Timestamps
    order_submitted_at: datetime
    fill_executed_at: datetime  # When fill actually happened (exchange time)
    fill_reported_at: datetime  # When we received notification

    # Latencies in milliseconds
    execution_latency_ms: float  # Order submission to fill
    reporting_latency_ms: float  # Fill to notification

    venue: str

    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'fill_id': self.fill_id,
            'execution_latency_ms': self.execution_latency_ms,
            'reporting_latency_ms': self.reporting_latency_ms,
            'venue': self.venue,
        }


@dataclass
class RejectionAnalysis:
    """Analysis of order rejection (#E29)."""
    order_id: str
    error_code: int
    error_message: str

    # Parsed information
    category: RejectionCategory
    is_recoverable: bool
    suggested_action: str

    # Context
    symbol: str
    venue: str
    order_type: str

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'category': self.category.value,
            'is_recoverable': self.is_recoverable,
            'suggested_action': self.suggested_action,
            'symbol': self.symbol,
            'venue': self.venue,
            'order_type': self.order_type,
            'timestamp': self.timestamp.isoformat(),
        }


class IBErrorCodeMapper:
    """
    Maps Interactive Brokers error codes to categories (#E27).

    Reference: https://interactivebrokers.github.io/tws-api/message_codes.html
    """

    # Error code to category and recovery info
    ERROR_MAP = {
        # Connection/System (Technical)
        100: (RejectionCategory.TECHNICAL, True, "Max number of tickers reached - reduce subscriptions"),
        102: (RejectionCategory.TECHNICAL, True, "Duplicate ticker ID - use unique ID"),
        103: (RejectionCategory.TECHNICAL, True, "Duplicate order ID - use unique ID"),
        104: (RejectionCategory.TECHNICAL, False, "Cannot modify filled order"),
        105: (RejectionCategory.TECHNICAL, True, "Order modification failed - retry or cancel/replace"),
        106: (RejectionCategory.TECHNICAL, True, "Cannot transmit order - check permissions"),
        107: (RejectionCategory.TECHNICAL, False, "Cannot transmit incomplete order"),
        109: (RejectionCategory.TECHNICAL, False, "Price out of range"),
        110: (RejectionCategory.PRICE, True, "Price does not conform to tick size"),
        111: (RejectionCategory.TECHNICAL, False, "TIF/order type unsupported"),
        113: (RejectionCategory.TECHNICAL, False, "Transmit flag not set"),

        # Order Issues
        135: (RejectionCategory.TECHNICAL, False, "Cannot cancel already cancelled order"),
        136: (RejectionCategory.TECHNICAL, False, "Cannot cancel already filled order"),
        161: (RejectionCategory.TECHNICAL, True, "Cancel rejected - order already pending cancel"),
        162: (RejectionCategory.MARKET, False, "Historical data farm offline"),
        165: (RejectionCategory.MARKET, True, "Historical data not subscribed - wait and retry"),
        200: (RejectionCategory.SYMBOL, False, "No security definition found"),
        201: (RejectionCategory.TECHNICAL, True, "Order rejected - reason provided"),
        202: (RejectionCategory.TECHNICAL, True, "Order cancelled"),
        203: (RejectionCategory.PERMISSIONS, False, "Security not allowed"),
        300: (RejectionCategory.SYMBOL, False, "Invalid symbol"),
        309: (RejectionCategory.PRICE, True, "Max price less than min price"),
        312: (RejectionCategory.QUANTITY, False, "Cannot create zero quantity order"),
        313: (RejectionCategory.QUANTITY, True, "Quantity exceeds display size"),
        321: (RejectionCategory.TECHNICAL, True, "Server error - retry"),
        322: (RejectionCategory.TECHNICAL, True, "Server error - retry"),
        324: (RejectionCategory.SYMBOL, False, "No valid news articles"),

        # Market Data
        354: (RejectionCategory.PERMISSIONS, False, "Not subscribed to market data"),
        357: (RejectionCategory.PERMISSIONS, False, "No market data permissions"),

        # Position/Margin
        380: (RejectionCategory.SYMBOL, False, "Order rejected - check symbol/exchange"),
        382: (RejectionCategory.PRICE, True, "Order rejected - price too far from market"),
        399: (RejectionCategory.TECHNICAL, True, "Order not found"),
        400: (RejectionCategory.MARGIN, False, "Algo order error"),
        401: (RejectionCategory.QUANTITY, True, "Length restriction - reduce size"),
        402: (RejectionCategory.POSITION, False, "Total position limit exceeded"),
        404: (RejectionCategory.MARGIN, False, "Order modification would exceed maintenance margin"),
        405: (RejectionCategory.POSITION, False, "Position value exceeds limit"),
        410: (RejectionCategory.MARGIN, True, "Order rejected - insufficient margin"),
        411: (RejectionCategory.QUANTITY, True, "Order size exceeds maximum"),
        414: (RejectionCategory.TECHNICAL, False, "Order size/price combination invalid"),

        # Regulatory
        422: (RejectionCategory.REGULATORY, False, "Cannot route to exchange"),
        425: (RejectionCategory.REGULATORY, False, "Price exceeds allowable range"),
        427: (RejectionCategory.REGULATORY, False, "Cannot short sell"),
        434: (RejectionCategory.REGULATORY, False, "Stock is halted"),
        436: (RejectionCategory.POSITION, False, "Position at max limit"),
        437: (RejectionCategory.REGULATORY, False, "Pattern day trader restriction"),
        460: (RejectionCategory.PERMISSIONS, False, "Commission markup exceeds limit"),

        # Exchange
        501: (RejectionCategory.MARKET, False, "Already connected"),
        502: (RejectionCategory.TECHNICAL, True, "Could not connect - check TWS"),
        503: (RejectionCategory.TECHNICAL, True, "TWS not connected - reconnect"),
        504: (RejectionCategory.TECHNICAL, True, "Not connected - reconnect first"),
        505: (RejectionCategory.TECHNICAL, False, "Fatal error"),
        506: (RejectionCategory.TECHNICAL, True, "Unknown order type"),
        507: (RejectionCategory.TECHNICAL, False, "No FA API support"),
        508: (RejectionCategory.TECHNICAL, True, "Duplicate client ID"),
        510: (RejectionCategory.TECHNICAL, True, "Request sending failed - retry"),
        511: (RejectionCategory.TECHNICAL, True, "Cannot cancel native order"),
        517: (RejectionCategory.TECHNICAL, True, "Error processing request"),
        519: (RejectionCategory.POSITION, False, "Position exceeds limit"),
        520: (RejectionCategory.TECHNICAL, True, "Request not allowed in current state"),

        # Market State
        1100: (RejectionCategory.TECHNICAL, True, "Connectivity lost - reconnecting"),
        1101: (RejectionCategory.TECHNICAL, True, "Connectivity restored - data lost"),
        1102: (RejectionCategory.TECHNICAL, True, "Connectivity restored - data maintained"),
        1300: (RejectionCategory.TECHNICAL, True, "Socket dropped - reconnect"),

        # Informational (not rejections)
        2100: (RejectionCategory.MARKET, True, "API connection interrupted"),
        2101: (RejectionCategory.MARKET, True, "Market data farm disconnected"),
        2102: (RejectionCategory.MARKET, True, "Market data farm connected"),
        2103: (RejectionCategory.MARKET, True, "Historical data farm disconnected"),
        2104: (RejectionCategory.MARKET, True, "Historical data farm connected"),
        2105: (RejectionCategory.MARKET, True, "Historical data farm inactive"),
        2106: (RejectionCategory.MARKET, True, "Historical data farm active"),
        2107: (RejectionCategory.MARKET, True, "Historical data farm inactive"),
        2108: (RejectionCategory.MARKET, True, "Market data farm inactive"),
        2109: (RejectionCategory.MARKET, True, "Order being routed"),
        2110: (RejectionCategory.MARKET, True, "Connectivity restored"),
    }

    @classmethod
    def get_error_info(cls, error_code: int) -> tuple[RejectionCategory, bool, str]:
        """
        Get error category, recoverability, and suggested action.

        Returns:
            Tuple of (category, is_recoverable, suggested_action)
        """
        if error_code in cls.ERROR_MAP:
            return cls.ERROR_MAP[error_code]

        # Default for unknown codes
        return (RejectionCategory.UNKNOWN, True, "Unknown error - review and retry")

    @classmethod
    def is_fatal_error(cls, error_code: int) -> bool:
        """Check if error is fatal (non-recoverable)."""
        info = cls.get_error_info(error_code)
        return not info[1]

    @classmethod
    def is_margin_error(cls, error_code: int) -> bool:
        """Check if error is margin-related."""
        info = cls.get_error_info(error_code)
        return info[0] == RejectionCategory.MARGIN

    @classmethod
    def is_market_error(cls, error_code: int) -> bool:
        """Check if error is market-related (halts, closures)."""
        info = cls.get_error_info(error_code)
        return info[0] == RejectionCategory.MARKET


class OrderAmendmentManager:
    """
    Manages order amendments/modifications (#E26).

    Handles the workflow of modifying existing orders.
    """

    def __init__(
        self,
        submit_amendment: Callable[[str, dict], bool] | None = None,
        cancel_order: Callable[[str], bool] | None = None,
    ):
        self._submit_amendment = submit_amendment
        self._cancel_order = cancel_order

        # Amendment tracking
        self._amendments: dict[str, OrderAmendment] = {}
        self._amendment_counter = 0

        # Order state cache
        self._order_state: dict[str, dict] = {}  # order_id -> {price, qty, status, filled_qty}

    def update_order_state(
        self,
        order_id: str,
        price: float,
        quantity: int,
        status: str,
        filled_qty: int = 0,
    ) -> None:
        """Update cached order state."""
        self._order_state[order_id] = {
            'price': price,
            'quantity': quantity,
            'status': status,
            'filled_qty': filled_qty,
        }

    def can_amend(self, order_id: str) -> tuple[bool, str]:
        """
        Check if order can be amended.

        Returns:
            Tuple of (can_amend, reason)
        """
        state = self._order_state.get(order_id)

        if state is None:
            return False, "Order not found"

        status = state['status'].lower()

        if status in ['filled', 'cancelled', 'inactive', 'error']:
            return False, f"Cannot amend order with status: {status}"

        if status == 'pending_cancel':
            return False, "Order is pending cancellation"

        # Check for pending amendments
        for amend in self._amendments.values():
            if amend.order_id == order_id and amend.status == AmendmentStatus.PENDING:
                return False, "Amendment already pending for this order"

        return True, "OK"

    def create_amendment(
        self,
        order_id: str,
        new_price: float | None = None,
        new_quantity: int | None = None,
        new_tif: str | None = None,
    ) -> OrderAmendment | None:
        """
        Create an amendment request.

        Returns:
            OrderAmendment or None if cannot amend
        """
        can_amend, reason = self.can_amend(order_id)
        if not can_amend:
            logger.warning(f"Cannot amend order {order_id}: {reason}")
            return None

        state = self._order_state.get(order_id, {})

        # Determine amendment type
        if new_price is not None and new_quantity is not None:
            amend_type = AmendmentType.PRICE_AND_QUANTITY
        elif new_price is not None:
            amend_type = AmendmentType.PRICE_CHANGE
        elif new_quantity is not None:
            amend_type = AmendmentType.QUANTITY_CHANGE
        elif new_tif is not None:
            amend_type = AmendmentType.TIME_IN_FORCE
        else:
            logger.warning("No changes specified for amendment")
            return None

        self._amendment_counter += 1
        amendment_id = f"AMEND_{order_id}_{self._amendment_counter}"

        amendment = OrderAmendment(
            amendment_id=amendment_id,
            order_id=order_id,
            amendment_type=amend_type,
            original_price=state.get('price'),
            original_quantity=state.get('quantity'),
            new_price=new_price,
            new_quantity=new_quantity,
            new_tif=new_tif,
            filled_qty_before_amendment=state.get('filled_qty', 0),
        )

        self._amendments[amendment_id] = amendment
        return amendment

    def submit_amendment(self, amendment_id: str) -> bool:
        """
        Submit amendment to broker.

        Returns:
            True if submitted successfully
        """
        amendment = self._amendments.get(amendment_id)
        if amendment is None:
            logger.error(f"Amendment {amendment_id} not found")
            return False

        if amendment.status != AmendmentStatus.PENDING:
            logger.warning(f"Amendment {amendment_id} already submitted")
            return False

        amendment.submitted_at = datetime.now(timezone.utc)
        amendment.status = AmendmentStatus.SUBMITTED

        if self._submit_amendment:
            changes = {}
            if amendment.new_price is not None:
                changes['price'] = amendment.new_price
            if amendment.new_quantity is not None:
                changes['quantity'] = amendment.new_quantity
            if amendment.new_tif is not None:
                changes['tif'] = amendment.new_tif

            success = self._submit_amendment(amendment.order_id, changes)
            if not success:
                amendment.status = AmendmentStatus.REJECTED
                amendment.rejection_reason = "Broker rejected amendment submission"
                return False

        logger.info(f"Submitted amendment {amendment_id} for order {amendment.order_id}")
        return True

    def handle_amendment_response(
        self,
        order_id: str,
        accepted: bool,
        rejection_reason: str = "",
    ) -> None:
        """Handle broker response to amendment."""
        # Find pending amendment for this order
        for amendment in self._amendments.values():
            if amendment.order_id == order_id and amendment.status == AmendmentStatus.SUBMITTED:
                amendment.completed_at = datetime.now(timezone.utc)
                if accepted:
                    amendment.status = AmendmentStatus.ACCEPTED
                    # Update order state
                    if amendment.new_price is not None:
                        self._order_state[order_id]['price'] = amendment.new_price
                    if amendment.new_quantity is not None:
                        self._order_state[order_id]['quantity'] = amendment.new_quantity
                    logger.info(f"Amendment {amendment.amendment_id} accepted")
                else:
                    amendment.status = AmendmentStatus.REJECTED
                    amendment.rejection_reason = rejection_reason
                    logger.warning(f"Amendment {amendment.amendment_id} rejected: {rejection_reason}")
                break

    def cancel_and_replace(
        self,
        order_id: str,
        new_price: float,
        new_quantity: int,
    ) -> str | None:
        """
        Cancel order and create replacement (for venues that don't support modify).

        Returns:
            New amendment ID or None
        """
        state = self._order_state.get(order_id)
        if state is None:
            return None

        amendment = self.create_amendment(order_id, new_price, new_quantity)
        if amendment is None:
            return None

        amendment.amendment_type = AmendmentType.CANCEL_REPLACE

        # Cancel original
        if self._cancel_order:
            self._cancel_order(order_id)

        return amendment.amendment_id

    def get_amendment(self, amendment_id: str) -> OrderAmendment | None:
        """Get amendment by ID."""
        return self._amendments.get(amendment_id)

    def get_amendments_for_order(self, order_id: str) -> list[OrderAmendment]:
        """Get all amendments for an order."""
        return [a for a in self._amendments.values() if a.order_id == order_id]

    def get_pending_amendments(self) -> list[OrderAmendment]:
        """Get all pending amendments."""
        return [
            a for a in self._amendments.values()
            if a.status in [AmendmentStatus.PENDING, AmendmentStatus.SUBMITTED]
        ]


class RejectionAnalyzer:
    """
    Analyzes order rejections and provides actionable insights (#E29).
    """

    def __init__(self):
        # History for pattern detection
        self._rejection_history: deque = deque(maxlen=1000)
        self._rejection_counts: dict[RejectionCategory, int] = {}

    def analyze_rejection(
        self,
        order_id: str,
        error_code: int,
        error_message: str,
        symbol: str,
        venue: str,
        order_type: str,
    ) -> RejectionAnalysis:
        """
        Analyze an order rejection.

        Returns:
            RejectionAnalysis with categorization and suggestions
        """
        category, is_recoverable, suggested_action = IBErrorCodeMapper.get_error_info(error_code)

        # Enhance suggestion based on message content
        enhanced_action = self._enhance_suggestion(
            error_code, error_message, category, suggested_action
        )

        analysis = RejectionAnalysis(
            order_id=order_id,
            error_code=error_code,
            error_message=error_message,
            category=category,
            is_recoverable=is_recoverable,
            suggested_action=enhanced_action,
            symbol=symbol,
            venue=venue,
            order_type=order_type,
        )

        # Track for pattern detection
        self._rejection_history.append(analysis)
        self._rejection_counts[category] = self._rejection_counts.get(category, 0) + 1

        logger.info(
            f"Rejection analyzed for {order_id}: {category.value} "
            f"(recoverable={is_recoverable})"
        )

        return analysis

    def _enhance_suggestion(
        self,
        error_code: int,
        message: str,
        category: RejectionCategory,
        base_suggestion: str,
    ) -> str:
        """Enhance suggestion based on message content."""
        msg_lower = message.lower()

        # Price-related enhancements
        if category == RejectionCategory.PRICE:
            if 'tick' in msg_lower:
                return "Adjust price to valid tick increment"
            if 'limit' in msg_lower and 'price' in msg_lower:
                return "Price outside acceptable range - move closer to market"
            if 'stale' in msg_lower:
                return "Update price with current market data"

        # Margin enhancements
        if category == RejectionCategory.MARGIN:
            if 'maintenance' in msg_lower:
                return "Reduce position size or deposit additional funds"
            if 'buying power' in msg_lower:
                return "Insufficient buying power - reduce order size"

        # Quantity enhancements
        if category == RejectionCategory.QUANTITY:
            if 'minimum' in msg_lower:
                return "Increase order size to meet minimum"
            if 'maximum' in msg_lower:
                return "Reduce order size below exchange maximum"
            if 'lot' in msg_lower:
                return "Adjust quantity to round lot"

        # Market enhancements
        if category == RejectionCategory.MARKET:
            if 'halt' in msg_lower:
                return "Wait for trading to resume"
            if 'closed' in msg_lower:
                return "Submit during market hours"

        return base_suggestion

    def get_rejection_patterns(self) -> dict:
        """Identify patterns in rejections."""
        recent = list(self._rejection_history)[-100:]  # Last 100

        if not recent:
            return {'has_patterns': False}

        # Count by category
        category_counts = {}
        symbol_counts = {}
        venue_counts = {}

        for r in recent:
            category_counts[r.category.value] = category_counts.get(r.category.value, 0) + 1
            symbol_counts[r.symbol] = symbol_counts.get(r.symbol, 0) + 1
            venue_counts[r.venue] = venue_counts.get(r.venue, 0) + 1

        # Find problematic areas
        problematic_categories = [
            c for c, count in category_counts.items()
            if count > len(recent) * 0.2  # >20% of rejections
        ]
        problematic_symbols = [
            s for s, count in symbol_counts.items()
            if count > len(recent) * 0.3  # >30% of rejections
        ]

        return {
            'has_patterns': len(problematic_categories) > 0 or len(problematic_symbols) > 0,
            'total_rejections': len(recent),
            'category_distribution': category_counts,
            'problematic_categories': problematic_categories,
            'problematic_symbols': problematic_symbols,
            'venue_distribution': venue_counts,
            'recovery_rate': sum(1 for r in recent if r.is_recoverable) / len(recent) if recent else 0,
        }

    def get_statistics(self) -> dict:
        """Get rejection statistics."""
        return {
            'total_rejections': len(self._rejection_history),
            'by_category': dict(self._rejection_counts),
            'recent_patterns': self.get_rejection_patterns(),
        }


class FillLatencyTracker:
    """
    Tracks fill notification latency (#E28).

    Monitors time from order submission to fill notification.
    """

    def __init__(
        self,
        alert_threshold_ms: float = 500.0,  # Alert if latency > 500ms
    ):
        self.alert_threshold_ms = alert_threshold_ms

        # Tracking by venue
        self._latencies: dict[str, deque] = {}  # venue -> latencies

        # Order submission times
        self._order_submitted: dict[str, datetime] = {}

        # Alerts
        self._latency_alerts: list[dict] = []

    def record_order_submitted(
        self,
        order_id: str,
        submitted_at: datetime | None = None,
    ) -> None:
        """Record when order was submitted."""
        self._order_submitted[order_id] = submitted_at or datetime.now(timezone.utc)

    def record_fill(
        self,
        order_id: str,
        fill_id: str,
        fill_executed_at: datetime,
        fill_reported_at: datetime | None = None,
        venue: str = "UNKNOWN",
    ) -> FillLatency | None:
        """
        Record fill notification and calculate latencies.

        Returns:
            FillLatency with calculated metrics
        """
        reported_at = fill_reported_at or datetime.now(timezone.utc)
        submitted_at = self._order_submitted.get(order_id)

        if submitted_at is None:
            logger.warning(f"No submission time for order {order_id}")
            return None

        # Calculate latencies
        execution_latency_ms = (fill_executed_at - submitted_at).total_seconds() * 1000
        reporting_latency_ms = (reported_at - fill_executed_at).total_seconds() * 1000

        latency = FillLatency(
            order_id=order_id,
            fill_id=fill_id,
            order_submitted_at=submitted_at,
            fill_executed_at=fill_executed_at,
            fill_reported_at=reported_at,
            execution_latency_ms=execution_latency_ms,
            reporting_latency_ms=reporting_latency_ms,
            venue=venue,
        )

        # Track by venue
        if venue not in self._latencies:
            self._latencies[venue] = deque(maxlen=1000)
        self._latencies[venue].append(latency)

        # Check for alerts
        if reporting_latency_ms > self.alert_threshold_ms:
            alert = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'order_id': order_id,
                'fill_id': fill_id,
                'venue': venue,
                'reporting_latency_ms': reporting_latency_ms,
                'threshold_ms': self.alert_threshold_ms,
            }
            self._latency_alerts.append(alert)
            logger.warning(
                f"High fill latency for {order_id}: {reporting_latency_ms:.1f}ms "
                f"(threshold: {self.alert_threshold_ms}ms)"
            )

        return latency

    def get_venue_statistics(self, venue: str) -> dict | None:
        """Get latency statistics for a venue."""
        latencies = self._latencies.get(venue)
        if not latencies:
            return None

        execution_times = [l.execution_latency_ms for l in latencies]
        reporting_times = [l.reporting_latency_ms for l in latencies]

        def percentile(data: list, p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            'venue': venue,
            'sample_count': len(latencies),
            'execution': {
                'mean_ms': sum(execution_times) / len(execution_times),
                'min_ms': min(execution_times),
                'max_ms': max(execution_times),
                'p50_ms': percentile(execution_times, 50),
                'p95_ms': percentile(execution_times, 95),
                'p99_ms': percentile(execution_times, 99),
            },
            'reporting': {
                'mean_ms': sum(reporting_times) / len(reporting_times),
                'min_ms': min(reporting_times),
                'max_ms': max(reporting_times),
                'p50_ms': percentile(reporting_times, 50),
                'p95_ms': percentile(reporting_times, 95),
                'p99_ms': percentile(reporting_times, 99),
            },
            'alerts_triggered': sum(
                1 for l in latencies if l.reporting_latency_ms > self.alert_threshold_ms
            ),
        }

    def get_all_statistics(self) -> dict:
        """Get latency statistics for all venues."""
        return {
            venue: self.get_venue_statistics(venue)
            for venue in self._latencies.keys()
        }

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        """Get recent latency alerts."""
        return self._latency_alerts[-limit:]

    def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """Clean up old order submission records."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        old_orders = [
            oid for oid, submitted in self._order_submitted.items()
            if submitted < cutoff
        ]

        for oid in old_orders:
            del self._order_submitted[oid]

        return len(old_orders)

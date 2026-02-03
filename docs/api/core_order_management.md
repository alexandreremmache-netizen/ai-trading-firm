# order_management

**Path**: `C:\Users\Alexa\ai-trading-firm\core\order_management.py`

## Overview

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

## Classes

### AmendmentType

**Inherits from**: str, Enum

Type of order amendment.

### AmendmentStatus

**Inherits from**: str, Enum

Status of amendment request.

### RejectionCategory

**Inherits from**: str, Enum

Categories of order rejection reasons.

### OrderAmendment

Order amendment request.

#### Methods

##### `def to_dict(self) -> dict`

### FillLatency

Fill notification latency tracking (#E28).

#### Methods

##### `def to_dict(self) -> dict`

### RejectionAnalysis

Analysis of order rejection (#E29).

#### Methods

##### `def to_dict(self) -> dict`

### IBErrorCodeMapper

Maps Interactive Brokers error codes to categories (#E27).

Reference: https://interactivebrokers.github.io/tws-api/message_codes.html

#### Methods

##### `def get_error_info(cls, error_code: int) -> tuple[RejectionCategory, bool, str]`

Get error category, recoverability, and suggested action.

Returns:
    Tuple of (category, is_recoverable, suggested_action)

##### `def is_fatal_error(cls, error_code: int) -> bool`

Check if error is fatal (non-recoverable).

##### `def is_margin_error(cls, error_code: int) -> bool`

Check if error is margin-related.

##### `def is_market_error(cls, error_code: int) -> bool`

Check if error is market-related (halts, closures).

### OrderAmendmentManager

Manages order amendments/modifications (#E26).

Handles the workflow of modifying existing orders.

#### Methods

##### `def __init__(self, submit_amendment: , cancel_order: )`

##### `def update_order_state(self, order_id: str, price: float, quantity: int, status: str, filled_qty: int) -> None`

Update cached order state.

##### `def can_amend(self, order_id: str) -> tuple[bool, str]`

Check if order can be amended.

Returns:
    Tuple of (can_amend, reason)

##### `def create_amendment(self, order_id: str, new_price: , new_quantity: , new_tif: )`

Create an amendment request.

Returns:
    OrderAmendment or None if cannot amend

##### `def submit_amendment(self, amendment_id: str) -> bool`

Submit amendment to broker.

Returns:
    True if submitted successfully

##### `def handle_amendment_response(self, order_id: str, accepted: bool, rejection_reason: str) -> None`

Handle broker response to amendment.

##### `def cancel_and_replace(self, order_id: str, new_price: float, new_quantity: int)`

Cancel order and create replacement (for venues that don't support modify).

Returns:
    New amendment ID or None

##### `def get_amendment(self, amendment_id: str)`

Get amendment by ID.

##### `def get_amendments_for_order(self, order_id: str) -> list[OrderAmendment]`

Get all amendments for an order.

##### `def get_pending_amendments(self) -> list[OrderAmendment]`

Get all pending amendments.

### RejectionAnalyzer

Analyzes order rejections and provides actionable insights (#E29).

#### Methods

##### `def __init__(self)`

##### `def analyze_rejection(self, order_id: str, error_code: int, error_message: str, symbol: str, venue: str, order_type: str) -> RejectionAnalysis`

Analyze an order rejection.

Returns:
    RejectionAnalysis with categorization and suggestions

##### `def get_rejection_patterns(self) -> dict`

Identify patterns in rejections.

##### `def get_statistics(self) -> dict`

Get rejection statistics.

### FillLatencyTracker

Tracks fill notification latency (#E28).

Monitors time from order submission to fill notification.

#### Methods

##### `def __init__(self, alert_threshold_ms: float)`

##### `def record_order_submitted(self, order_id: str, submitted_at: ) -> None`

Record when order was submitted.

##### `def record_fill(self, order_id: str, fill_id: str, fill_executed_at: datetime, fill_reported_at: , venue: str)`

Record fill notification and calculate latencies.

Returns:
    FillLatency with calculated metrics

##### `def get_venue_statistics(self, venue: str)`

Get latency statistics for a venue.

##### `def get_all_statistics(self) -> dict`

Get latency statistics for all venues.

##### `def get_recent_alerts(self, limit: int) -> list[dict]`

Get recent latency alerts.

##### `def cleanup_old_orders(self, max_age_hours: int) -> int`

Clean up old order submission records.

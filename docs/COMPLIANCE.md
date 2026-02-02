# Compliance Documentation

## Overview

The AI Trading Firm is designed for full compliance with EU/AMF regulatory requirements. This document covers the regulatory framework and compliance features implemented in the system.

## Regulatory Framework

### Applicable Regulations

| Regulation | Full Name | Scope |
|------------|-----------|-------|
| MiFID II | Markets in Financial Instruments Directive | Investment services framework |
| MiFIR | Markets in Financial Instruments Regulation | Transparency, trading venues |
| MAR | Market Abuse Regulation | Market manipulation, insider trading |
| EMIR | European Market Infrastructure Regulation | OTC derivatives, CCPs |
| SFTR | Securities Financing Transactions Regulation | Repos, securities lending |

### Regulatory Technical Standards (RTS)

| RTS | Topic | Implementation |
|-----|-------|----------------|
| RTS 6 | Algorithmic trading | Kill switch, risk controls |
| RTS 22 | Transaction reporting | ESMA reporting |
| RTS 23 | Trading data | Instrument reference data |
| RTS 24 | Order recordkeeping | Order ID format |
| RTS 25 | Order records | 65-field order records |
| RTS 27 | Execution quality | Best execution reporting |
| RTS 28 | Venue analysis | Execution venue data |

---

## MiFID II Compliance

### Transaction Reporting (RTS 22/23)

All transactions must be reported to the regulator within 15 minutes.

#### Required Fields

```python
@dataclass
class TransactionReport:
    # Identification
    transaction_reference: str
    firm_lei: str
    client_identification: str

    # Instrument
    instrument_isin: str
    instrument_name: str

    # Trade details
    quantity: float
    price: float
    currency: str
    execution_venue: str  # MIC code

    # Timestamps
    trading_datetime: datetime
    reporting_datetime: datetime

    # Classification
    buyer_seller: str  # "BUYR", "SELLR"
    trading_capacity: str  # "DEAL", "MTCH", "AOTC"
```

#### LEI Validation

Legal Entity Identifiers must be validated per ISO 17442:

```python
def validate_lei(lei: str, strict: bool = True) -> tuple[bool, str]:
    """
    Validate LEI per ISO 17442 standard.

    LEI format:
    - 20 characters total
    - Characters 1-4: LOU prefix
    - Characters 5-6: Reserved ("00")
    - Characters 7-18: Entity-specific
    - Characters 19-20: Check digits (MOD 97-10)
    """
    # Check length
    if len(lei) != 20:
        return False, f"LEI must be 20 characters"

    # Check alphanumeric
    if not lei.isalnum():
        return False, "LEI must be alphanumeric"

    # MOD 97-10 checksum
    numeric_lei = ""
    for char in lei:
        if char.isdigit():
            numeric_lei += char
        else:
            numeric_lei += str(ord(char) - ord('A') + 10)

    checksum = int(numeric_lei) % 97
    if checksum != 1:
        return False, f"Invalid checksum"

    return True, ""
```

#### Configuration

```yaml
transaction_reporting:
  enabled: true
  reporting_deadline_minutes: 15
  firm_lei: ""        # REQUIRED: Valid 20-char LEI
  firm_country: "FR"  # AMF jurisdiction
  default_venue: "XPAR"  # Paris exchange MIC
```

### Order Recordkeeping (RTS 25)

All orders must be recorded with 65 mandatory fields:

```python
@dataclass
class RTS25OrderRecord:
    # Identification (1-10)
    order_id: str
    client_order_id: str
    trading_venue_order_id: str | None
    order_submission_timestamp: datetime
    sequence_number: int
    segment_mic: str
    trading_capacity: str

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
    order_type: str  # "LMTO", "MAKT", "STOP"
    limit_price: float | None
    stop_price: float | None
    quantity: float
    quantity_currency: str
    initial_quantity: float
    remaining_quantity: float
    executed_quantity: float
    time_in_force: str
    order_restriction: str | None

    # ... (45 more fields)
```

### Algorithmic Trading (RTS 6)

#### Pre-Trade Risk Controls

```python
class PreTradeRiskControls:
    """
    RTS 6 compliant pre-trade risk controls.
    """

    def check_order_limits(self, order: OrderEvent) -> bool:
        """Check order value limits."""
        pass

    def check_price_limits(self, order: OrderEvent) -> bool:
        """Check price collar limits."""
        pass

    def check_position_limits(self, order: OrderEvent) -> bool:
        """Check position limits."""
        pass

    def check_throttling(self, order: OrderEvent) -> bool:
        """Check order rate limits."""
        pass
```

#### Kill Switch

RTS 6 requires immediate ability to halt all trading:

```python
@dataclass
class KillSwitchEvent(Event):
    """
    Kill switch event per MiFID II RTS 6.
    Triggers immediate halt of all trading activity.
    """
    activated: bool = True
    reason: str = ""
    trigger_type: str = "manual"  # "manual", "automatic", "regulatory"
    affected_symbols: tuple[str, ...] = ()  # Empty = all
    cancel_pending_orders: bool = True
    close_positions: bool = False
```

---

## MAR Compliance

### Market Abuse Detection

The Surveillance Agent monitors for:

#### Wash Trading
Trading with yourself to create misleading volume:

```python
def detect_wash_trading(
    self,
    orders: list[OrderEvent],
    fills: list[FillEvent],
    window_seconds: int = 60
) -> list[SurveillanceAlert]:
    """
    Detect potential wash trading patterns:
    - Buy/sell same instrument in short time
    - Same or similar quantities
    - No economic benefit
    """
```

#### Spoofing
Placing orders with intent to cancel:

```python
def detect_spoofing(
    self,
    orders: list[OrderEvent],
    cancellations: list[str],
    cancel_threshold: float = 0.8
) -> list[SurveillanceAlert]:
    """
    Detect spoofing patterns:
    - High cancel rate (>80%)
    - Orders at prices unlikely to execute
    - Orders cancelled before execution
    """
```

#### Quote Stuffing
Excessive orders to slow down systems:

```python
def detect_quote_stuffing(
    self,
    orders: list[OrderEvent],
    rate_per_second: int = 10
) -> list[SurveillanceAlert]:
    """
    Detect quote stuffing:
    - Excessive order rate
    - Rapid order/cancel cycles
    - No intent to trade
    """
```

#### Layering
Multiple orders at different levels:

```python
def detect_layering(
    self,
    orders: list[OrderEvent],
    levels: int = 5
) -> list[SurveillanceAlert]:
    """
    Detect layering patterns:
    - Multiple orders at different prices
    - Orders on one side to move price
    - Orders cancelled after price moves
    """
```

### Configuration

```yaml
surveillance:
  wash_trading_detection: true
  spoofing_detection: true
  quote_stuffing_detection: true
  layering_detection: true
  wash_trading_window_seconds: 60
  spoofing_cancel_threshold: 0.8
  quote_stuffing_rate_per_second: 10
```

---

## Best Execution (RTS 27/28)

### Best Execution Policy

Orders must be executed to achieve the best possible result considering:

1. **Price** - Primary factor
2. **Costs** - Commissions, fees
3. **Speed** - Execution time
4. **Likelihood of execution** - Fill probability
5. **Settlement** - T+2 compliance
6. **Size** - Market impact
7. **Other relevant factors**

### Best Execution Analysis

```python
class BestExecutionAnalyzer:
    """
    Analyze execution quality per RTS 27/28.
    """

    def __init__(
        self,
        default_benchmark: str = "vwap",  # "arrival", "vwap", "twap"
        slippage_alert_bps: float = 50,
        report_retention_quarters: int = 8,
    ):

    def analyze_execution(
        self,
        order: OrderEvent,
        fills: list[FillEvent],
        market_data: MarketDataEvent
    ) -> ExecutionAnalysis:
        """
        Analyze execution against benchmark.

        Returns:
        - Slippage vs arrival price
        - Slippage vs VWAP
        - Market impact
        - Price improvement
        """
```

### Execution Venues

```python
# Venue analysis per RTS 28
EXECUTION_VENUES = {
    "XPAR": "Euronext Paris",
    "XAMS": "Euronext Amsterdam",
    "XBRU": "Euronext Brussels",
    "XLON": "London Stock Exchange",
    "XETR": "Xetra (Deutsche Borse)",
}
```

### Configuration

```yaml
best_execution:
  benchmark: "vwap"              # Default benchmark
  slippage_alert_bps: 50         # Alert threshold
  report_retention_quarters: 8   # Keep 8 quarters
```

---

## Compliance Agent

### Responsibilities

The Compliance Agent validates all decisions against:

1. **Blackout Periods** - No trading during earnings
2. **MNPI Detection** - No insider trading
3. **Restricted Instruments** - Banned securities
4. **Market Hours** - Trading hours only
5. **Short Sale Rules** - SSR compliance
6. **Data Source Validation** - Approved sources only

### Blackout Periods

```python
@dataclass
class BlackoutEvent:
    symbol: str
    blackout_type: BlackoutType  # EARNINGS, MA, CAPITAL_INCREASE
    event_date: datetime
    blackout_start: datetime
    blackout_end: datetime
    description: str
```

### Compliance Checks

```python
class ComplianceCheckResult:
    check_name: str
    passed: bool
    code: RejectionCode | None
    message: str
    details: dict
```

### Rejection Codes

| Code | Description |
|------|-------------|
| BLACKOUT_PERIOD | Trading during blackout |
| MNPI_DETECTED | Potential insider trading |
| RESTRICTED_INSTRUMENT | Banned security |
| MARKET_CLOSED | Outside trading hours |
| SSR_RESTRICTION | Short sale restriction |
| TRADING_SUSPENDED | Security suspended |
| THRESHOLD_BREACH | Regulatory threshold |
| UNAPPROVED_SOURCE | Invalid data source |
| NO_BORROW_AVAILABLE | Cannot locate shares |
| INVALID_LEI | Invalid entity identifier |
| INVALID_ISIN | Invalid security identifier |

---

## Audit Trail

### Requirements

- All decisions must be logged
- Full rationale required
- Data sources documented
- 7-year retention (MiFID II)

### Audit Log Structure

```python
class AuditLogger:
    """
    Compliance-grade audit logging.
    """

    def log_decision(
        self,
        agent_name: str,
        decision_id: str,
        symbol: str,
        action: str,
        quantity: int,
        rationale: str,
        data_sources: list[str],
        contributing_signals: list[str],
        conviction_score: float,
    ):
        """
        Log trading decision with full traceability.
        """

    def log_trade(
        self,
        order_id: str,
        fill_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        venue: str,
        timestamp: datetime,
    ):
        """
        Log executed trade.
        """
```

### Configuration

```yaml
compliance:
  jurisdiction: "EU"
  regulator: "AMF"
  require_rationale: true
  audit_retention_days: 2555  # 7 years

logging:
  audit_file: "logs/audit.jsonl"
  trade_file: "logs/trades.jsonl"
  decision_file: "logs/decisions.jsonl"
```

---

## Data Protection (GDPR)

### Personal Data Handling

- No personal data stored unnecessarily
- Access controls on sensitive data
- Data retention policies enforced
- Right to erasure supported (where regulatory requirements allow)

### Data Retention

| Data Type | Retention Period | Basis |
|-----------|-----------------|-------|
| Order records | 7 years | MiFID II RTS 25 |
| Transaction reports | 7 years | MiFID II RTS 22 |
| Audit logs | 7 years | MiFID II |
| Market data | 1 year | Operational |
| Performance metrics | 1 year | Operational |

---

## Clock Synchronization

### Requirements (RTS 25)

Timestamps must be synchronized to UTC with:
- Business clock accuracy: +/- 1 second
- HFT systems: +/- 1 millisecond (not applicable)

### Implementation

```python
from datetime import datetime, timezone

def get_synchronized_timestamp() -> datetime:
    """Get UTC timestamp for compliance logging."""
    return datetime.now(timezone.utc)
```

---

## Regulatory Reporting Schedule

### Daily Reports

| Report | Deadline | Recipient |
|--------|----------|-----------|
| Transaction reports | T+15 minutes | AMF/ESMA |
| Suspicious activity | Immediately | Compliance |

### Quarterly Reports

| Report | Deadline | Recipient |
|--------|----------|-----------|
| RTS 27 Execution quality | T+3 months | Public |
| RTS 28 Top 5 venues | T+3 months | Public |

### Annual Reports

| Report | Deadline | Recipient |
|--------|----------|-----------|
| Best execution policy | Annual | Public |
| Conflicts of interest | Annual | AMF |

---

## Compliance Checklist

### Pre-Production

- [ ] Valid LEI obtained from GLEIF-registered LOU
- [ ] Transaction reporting connection tested
- [ ] Surveillance systems operational
- [ ] Audit logging verified
- [ ] Clock synchronization confirmed
- [ ] Kill switch tested

### Ongoing

- [ ] Daily transaction report verification
- [ ] Weekly surveillance review
- [ ] Monthly compliance metrics review
- [ ] Quarterly execution quality report
- [ ] Annual policy review

# API Reference

## Overview

This document provides a reference for the core APIs, event types, and configuration schema used in the AI Trading Firm system.

---

## Event Types

### Base Event

All events inherit from the base `Event` class:

```python
@dataclass(frozen=True)
class Event:
    """
    Base event class. All events are immutable (frozen).
    """
    event_id: str              # Unique identifier (UUID)
    timestamp: datetime        # UTC timestamp
    event_type: EventType      # Type enumeration
    source_agent: str          # Originating agent

    def to_audit_dict(self) -> dict[str, Any]:
        """Convert to dictionary for audit logging."""
```

### Event Type Enumeration

```python
class EventType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    DECISION = "decision"
    VALIDATED_DECISION = "validated_decision"
    ORDER = "order"
    FILL = "fill"
    RISK_ALERT = "risk_alert"
    SYSTEM = "system"
    ROLL_SIGNAL = "roll_signal"
    ROLL_COMPLETE = "roll_complete"
    SURVEILLANCE_ALERT = "surveillance_alert"
    TRANSACTION_REPORT = "transaction_report"
    STRESS_TEST_RESULT = "stress_test_result"
    CORRELATION_ALERT = "correlation_alert"
    GREEKS_UPDATE = "greeks_update"
    KILL_SWITCH = "kill_switch"
    ORDER_STATE_CHANGE = "order_state_change"
```

---

### MarketDataEvent

Market data from Interactive Brokers:

```python
@dataclass(frozen=True)
class MarketDataEvent(Event):
    event_type: EventType = EventType.MARKET_DATA
    symbol: str = ""
    exchange: str = ""
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    bid_size: int = 0
    ask_size: int = 0
    high: float = 0.0
    low: float = 0.0
    open_price: float = 0.0
    close: float = 0.0

    @property
    def mid(self) -> float:
        """Calculate mid price."""

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
```

---

### SignalEvent

Trading signal from strategy agents:

```python
@dataclass(frozen=True)
class SignalEvent(Event):
    event_type: EventType = EventType.SIGNAL
    strategy_name: str = ""
    symbol: str = ""
    direction: SignalDirection = SignalDirection.FLAT
    strength: float = 0.0        # -1.0 to 1.0
    confidence: float = 0.0      # 0.0 to 1.0
    target_price: float | None = None
    stop_loss: float | None = None
    rationale: str = ""          # Required for compliance
    data_sources: tuple[str, ...] = ()  # Required for compliance
```

#### SignalDirection

```python
class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
```

---

### DecisionEvent

Trading decision from CIO agent:

```python
@dataclass(frozen=True)
class DecisionEvent(Event):
    event_type: EventType = EventType.DECISION
    symbol: str = ""
    action: OrderSide | None = None
    quantity: int = 0
    order_type: OrderType = OrderType.LIMIT
    limit_price: float | None = None
    stop_price: float | None = None
    rationale: str = ""              # Required
    contributing_signals: tuple[str, ...] = ()
    data_sources: tuple[str, ...] = ()
    conviction_score: float = 0.0
```

---

### ValidatedDecisionEvent

Decision approved by Risk/Compliance:

```python
@dataclass(frozen=True)
class ValidatedDecisionEvent(Event):
    event_type: EventType = EventType.VALIDATED_DECISION
    original_decision_id: str = ""
    approved: bool = False
    adjusted_quantity: int | None = None
    rejection_reason: str | None = None
    risk_metrics: dict[str, float] = field(default_factory=dict)
    compliance_checks: tuple[str, ...] = ()
```

---

### OrderEvent

Order sent to broker:

```python
@dataclass(frozen=True)
class OrderEvent(Event):
    event_type: EventType = EventType.ORDER
    decision_id: str = ""
    validation_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    order_type: OrderType = OrderType.LIMIT
    limit_price: float | None = None
    stop_price: float | None = None
    broker_order_id: int | None = None
    algo: str = "TWAP"
    time_in_force: TimeInForce = TimeInForce.DAY
```

#### OrderSide

```python
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
```

#### OrderType

```python
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
```

#### TimeInForce

```python
class TimeInForce(Enum):
    DAY = "DAY"      # Valid for trading day
    GTC = "GTC"      # Good Till Cancelled
    IOC = "IOC"      # Immediate or Cancel
    FOK = "FOK"      # Fill or Kill
    GTD = "GTD"      # Good Till Date
    OPG = "OPG"      # At the Opening
    MOC = "MOC"      # Market on Close
```

---

### FillEvent

Order fill from broker:

```python
@dataclass(frozen=True)
class FillEvent(Event):
    event_type: EventType = EventType.FILL
    order_id: str = ""
    broker_order_id: int = 0
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    filled_quantity: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    exchange: str = ""
```

---

### RiskAlertEvent

Risk alert from Risk Agent:

```python
@dataclass(frozen=True)
class RiskAlertEvent(Event):
    event_type: EventType = EventType.RISK_ALERT
    severity: RiskAlertSeverity = RiskAlertSeverity.INFO
    alert_type: str = ""
    message: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    affected_symbols: tuple[str, ...] = ()
    halt_trading: bool = False
```

#### RiskAlertSeverity

```python
class RiskAlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
```

---

### KillSwitchEvent

Emergency trading halt:

```python
@dataclass(frozen=True)
class KillSwitchEvent(Event):
    event_type: EventType = EventType.KILL_SWITCH
    activated: bool = True
    reason: str = ""
    trigger_type: str = "manual"  # "manual", "automatic", "regulatory"
    affected_symbols: tuple[str, ...] = ()
    cancel_pending_orders: bool = True
    close_positions: bool = False
```

---

### OrderStateChangeEvent

Order state machine transition:

```python
@dataclass(frozen=True)
class OrderStateChangeEvent(Event):
    event_type: EventType = EventType.ORDER_STATE_CHANGE
    order_id: str = ""
    broker_order_id: int | None = None
    symbol: str = ""
    previous_state: OrderState = OrderState.CREATED
    new_state: OrderState = OrderState.PENDING
    reason: str = ""
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
```

#### OrderState

```python
class OrderState(Enum):
    CREATED = "created"
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"
```

---

## Event Bus API

### EventBus Class

```python
class EventBus:
    """Central event bus for inter-agent communication."""

    def __init__(
        self,
        max_queue_size: int = 10000,
        signal_timeout: float = 5.0,
        barrier_timeout: float = 10.0,
    ):
        """
        Initialize event bus.

        Args:
            max_queue_size: Maximum events in queue
            signal_timeout: Timeout for signal processing
            barrier_timeout: Timeout for signal barrier
        """

    def register_signal_agent(self, agent_name: str) -> None:
        """Register agent as signal producer."""

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine],
    ) -> None:
        """Subscribe handler to event type."""

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine],
    ) -> None:
        """Unsubscribe handler from event type."""

    async def publish(self, event: Event, priority: bool = False) -> bool:
        """
        Publish event to bus.

        Args:
            event: Event to publish
            priority: High-priority event

        Returns:
            True if published, False if dropped
        """

    async def publish_signal(self, signal: SignalEvent) -> None:
        """Publish signal with barrier synchronization."""

    async def wait_for_signals(self) -> dict[str, SignalEvent]:
        """Wait for signal barrier (fan-in)."""

    async def start(self) -> None:
        """Start event processing loop."""

    async def stop(self) -> None:
        """Stop event bus gracefully."""

    def get_status(self) -> dict:
        """Get event bus status."""

    @property
    def queue_size(self) -> int:
        """Current queue size."""

    @property
    def backpressure_level(self) -> BackpressureLevel:
        """Current backpressure level."""
```

---

## Broker API

### IBBroker Class

```python
class IBBroker:
    """Interactive Brokers integration."""

    def __init__(self, config: BrokerConfig):
        """Initialize with configuration."""

    async def connect(self) -> bool:
        """Connect to TWS/Gateway. Returns True on success."""

    async def disconnect(self) -> None:
        """Disconnect from broker."""

    async def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""

    async def get_positions(self) -> dict[str, Position]:
        """Get all positions."""

    async def place_order(self, order: OrderEvent) -> int | None:
        """
        Place order with broker.

        Returns:
            Broker order ID or None on failure
        """

    async def cancel_order(self, broker_order_id: int) -> bool:
        """Cancel pending order."""

    async def cancel_all_orders(self) -> int:
        """Cancel all pending orders. Returns count cancelled."""

    async def request_market_data_type(self, data_type: int) -> None:
        """
        Set market data type.
        1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen
        """

    def on_fill(self, callback: Callable[[FillEvent], None]) -> None:
        """Register fill callback."""

    def on_market_data(self, callback: Callable[[MarketDataEvent], None]) -> None:
        """Register market data callback."""

    @property
    def is_connected(self) -> bool:
        """Check if connected."""

    @property
    def account_id(self) -> str | None:
        """Get account ID."""
```

### BrokerConfig

```python
@dataclass
class BrokerConfig:
    host: str = "127.0.0.1"
    port: int = 7497              # Paper trading
    client_id: int = 1
    timeout_seconds: float = 30.0
    readonly: bool = False
    account: str = ""
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
```

---

## Agent API

### BaseAgent Class

```python
class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        """Initialize agent."""

    @property
    def name(self) -> str:
        """Agent name."""

    @property
    def is_running(self) -> bool:
        """Check if running."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent (async setup)."""

    @abstractmethod
    async def process_event(self, event: Event) -> None:
        """Process incoming event."""

    @abstractmethod
    def get_subscribed_events(self) -> list[EventType]:
        """Return subscribed event types."""

    async def start(self) -> None:
        """Start agent."""

    async def stop(self, timeout: float | None = None) -> bool:
        """
        Stop agent gracefully.

        Returns:
            True if graceful shutdown
        """

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
```

### AgentConfig

```python
@dataclass
class AgentConfig:
    name: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    parameters: dict[str, Any] = field(default_factory=dict)
```

---

## Risk Calculator API

### VaRCalculator Class

```python
class VaRCalculator:
    """Value at Risk calculator."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize VaR calculator.

        Config options:
        - confidence_level: float (default 0.95)
        - horizon_days: int (default 1)
        - monte_carlo_simulations: int (default 10000)
        - decay_factor: float (default 0.94)
        """

    def calculate_parametric_var(
        self,
        portfolio_value: float,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence_level: float = 0.95,
    ) -> VaRResult:
        """Calculate parametric (variance-covariance) VaR."""

    def calculate_historical_var(
        self,
        portfolio_value: float,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence_level: float = 0.95,
    ) -> VaRResult:
        """Calculate historical simulation VaR."""

    def calculate_monte_carlo_var(
        self,
        portfolio_value: float,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
    ) -> VaRResult:
        """Calculate Monte Carlo VaR."""

    def calculate_all(
        self,
        portfolio_value: float,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> dict[str, VaRResult]:
        """Calculate VaR using all methods."""
```

### VaRResult

```python
@dataclass
class VaRResult:
    method: VaRMethod
    confidence_level: float
    horizon_days: int
    var_absolute: float      # In currency
    var_pct: float          # As percentage
    expected_shortfall: float | None
    timestamp: datetime
    details: dict[str, Any]
```

---

## Configuration Schema

### Top-Level Structure

```yaml
# Core settings
firm:
  name: string
  version: string
  mode: "paper" | "live"

# Broker connection
broker:
  host: string
  port: integer
  client_id: integer
  timeout_seconds: float
  readonly: boolean
  use_delayed_data: boolean

# Event bus
event_bus:
  max_queue_size: integer
  signal_timeout_seconds: float
  sync_barrier_timeout_seconds: float

# Risk management
risk:
  max_portfolio_var_pct: float
  max_position_size_pct: float
  max_sector_exposure_pct: float
  max_daily_loss_pct: float
  max_drawdown_pct: float
  max_leverage: float
  max_orders_per_minute: integer
  min_order_interval_ms: integer

# VaR settings
var:
  method: "parametric" | "historical" | "monte_carlo" | "all"
  confidence_level: float
  horizon_days: integer
  monte_carlo_simulations: integer

# Greeks limits
greeks:
  max_portfolio_delta: float
  max_portfolio_gamma: float
  max_portfolio_vega: float
  max_portfolio_theta: float

# Compliance
compliance:
  jurisdiction: string
  regulator: string
  require_rationale: boolean
  audit_retention_days: integer
  banned_instruments: list
  allowed_asset_classes: list

# Surveillance
surveillance:
  wash_trading_detection: boolean
  spoofing_detection: boolean
  wash_trading_window_seconds: integer
  spoofing_cancel_threshold: float

# Transaction reporting
transaction_reporting:
  enabled: boolean
  reporting_deadline_minutes: integer
  firm_lei: string
  firm_country: string
  default_venue: string

# Logging
logging:
  level: string
  audit_file: string
  trade_file: string
  decision_file: string

# Agent configuration
agents:
  macro:
    enabled: boolean
    indicators: list
  stat_arb:
    enabled: boolean
    lookback_days: integer
    zscore_entry_threshold: float
    pairs: list
  momentum:
    enabled: boolean
    fast_period: integer
    slow_period: integer
    rsi_period: integer
  cio:
    signal_weight_*: float
    min_conviction_threshold: float
    use_kelly_sizing: boolean
  execution:
    default_algo: string
    slice_interval_seconds: float

# Trading universe
universe:
  equities: list
  etfs: list
  futures: list
  forex: list
```

---

## Health Check API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | System health status |
| /ready | GET | Readiness check |
| /live | GET | Liveness check |

### Response Format

```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "timestamp": "ISO8601",
  "checks": {
    "broker": "connected" | "disconnected",
    "event_bus": "running" | "stopped",
    "agents": {
      "CIOAgent": "running",
      "RiskAgent": "running"
    }
  },
  "metrics": {
    "queue_size": 0,
    "events_processed": 1523,
    "uptime_seconds": 3600
  }
}
```

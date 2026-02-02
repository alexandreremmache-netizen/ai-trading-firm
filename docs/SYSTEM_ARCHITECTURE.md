# System Architecture Documentation

**Last Updated**: 2026-02-02
**Version**: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [System Components](#system-components)
4. [Agent Architecture](#agent-architecture)
5. [Event-Driven Design](#event-driven-design)
6. [Data Flow](#data-flow)
7. [Concurrency Model](#concurrency-model)
8. [Integration Points](#integration-points)
9. [Deployment Architecture](#deployment-architecture)
10. [Security Architecture](#security-architecture)

---

## Overview

The AI Trading Firm is a professional, institutional-grade multi-agent trading system designed to emulate a hedge fund's operations. The system follows strict architectural principles that prioritize:

- **Determinism**: All decisions are reproducible and auditable
- **Separation of Concerns**: Each agent has a single, well-defined responsibility
- **Event-Driven Execution**: No polling, no infinite loops
- **Regulatory Compliance**: EU/AMF compatible design with full audit trails

### High-Level Architecture Diagram

```
                                    +------------------+
                                    |   Market Data    |
                                    |   (IB Gateway)   |
                                    +--------+---------+
                                             |
                                             v
+------------------+              +----------+-----------+
|  External Events |------------->|      Event Bus       |<----- Scheduled Events
| (News, Economic) |              | (Fan-out/Fan-in Hub) |
+------------------+              +----------+-----------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
                    v                        v                        v
           +-------+--------+       +-------+--------+       +-------+--------+
           | Signal Agent 1 |       | Signal Agent 2 |       | Signal Agent N |
           | (Momentum)     |       | (Stat Arb)     |       | (Options Vol)  |
           +-------+--------+       +-------+--------+       +-------+--------+
                    |                        |                        |
                    +------------------------+------------------------+
                                             |
                                             v
                                   +---------+---------+
                                   |  Synchronization  |
                                   |     Barrier       |
                                   +---------+---------+
                                             |
                                             v
                                   +---------+---------+
                                   |    CIO Agent      |
                                   | (Decision Making) |
                                   +---------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
                    v                        v                        v
           +-------+--------+       +-------+--------+       +-------+--------+
           |   Risk Agent   |------>| Compliance     |------>|  Execution     |
           | (Validation)   |       |    Agent       |       |    Agent       |
           +----------------+       +----------------+       +-------+--------+
                                                                      |
                                                                      v
                                                            +---------+---------+
                                                            |   IB Broker       |
                                                            |   (Execution)     |
                                                            +-------------------+
```

---

## Architecture Principles

### 1. Multi-Agent Separation

Each agent has exactly ONE responsibility:

| Agent Type | Responsibility | State |
|------------|---------------|-------|
| Signal Agents | Generate trading signals | Stateless |
| CIO Agent | Make portfolio decisions | Stateless |
| Risk Agent | Validate risk constraints | Minimal state |
| Compliance Agent | Ensure regulatory compliance | Audit state |
| Execution Agent | Execute approved orders | Order state |

### 2. No Omniscient Agents

Agents cannot:
- Access other agents' internal state directly
- Make decisions outside their domain
- Self-modify their strategies
- Scale capital autonomously

### 3. Deterministic Behavior

All operations must be:
- Reproducible given the same inputs
- Fully logged with timestamps and sources
- Testable in isolation
- Observable via metrics

### 4. Event-Driven Execution

The system operates on three event types:

1. **Market Events**: Price updates, fills, order status
2. **Scheduled Events**: Timer ticks, scheduled rebalancing
3. **External Events**: News, economic releases

---

## System Components

### Core Infrastructure

```
core/
+-- event_bus.py          # Central event routing with fan-out/fan-in
+-- events.py             # Event type definitions (OrderEvent, MarketEvent, etc.)
+-- event_persistence.py  # SQLite-based event persistence for recovery
+-- agent_base.py         # Base class for all agents
+-- circuit_breaker.py    # Broker connection resilience
+-- health_check.py       # HTTP health endpoints (/health, /ready, /live)
```

### Market Data & Execution

```
core/
+-- broker.py             # IB Gateway integration
+-- smart_order_router.py # Multi-venue order routing
+-- best_execution.py     # Execution quality analysis
+-- slippage_estimator.py # Market impact modeling
```

### Risk Management

```
core/
+-- var_calculator.py     # VaR (Historical, Parametric, Monte Carlo)
+-- stress_tester.py      # Scenario analysis
+-- correlation_manager.py # Correlation tracking with stress modes
+-- risk_factors.py       # Factor decomposition
+-- margin_optimizer.py   # Cross-margin optimization
```

### Compliance

```
core/
+-- regulatory_compliance.py  # MiFID II, EMIR, MAR compliance
+-- data_retention.py         # 7-year retention enforcement
+-- regulatory_calendar.py    # Deadline tracking
```

---

## Agent Architecture

### Agent Lifecycle

```python
class AgentBase:
    """
    All agents inherit from this base class.

    Lifecycle:
    1. __init__() - Configuration only
    2. start() - Begin processing
    3. process_event() - Handle incoming events
    4. shutdown() - Graceful termination
    """

    async def start(self):
        """Initialize agent resources."""
        pass

    async def process_event(self, event: Event) -> Optional[Event]:
        """Process incoming event, optionally emit response."""
        pass

    async def shutdown(self, timeout: float = 30.0):
        """Graceful shutdown with timeout."""
        pass
```

### Agent Communication

Agents communicate ONLY through the Event Bus:

```python
# Signal Agent emits a signal
await event_bus.publish(SignalEvent(
    source="momentum_agent",
    symbol="AAPL",
    direction=SignalDirection.BUY,
    strength=0.75,
    timestamp=datetime.now()
))

# CIO Agent receives all signals via barrier
signals = await event_bus.wait_for_barrier("signal_collection")
```

---

## Event-Driven Design

### Event Types

```python
class EventType(Enum):
    # Market Events
    MARKET_DATA = "market_data"
    TICK = "tick"

    # Order Lifecycle
    ORDER_SUBMIT = "order_submit"
    ORDER_FILL = "order_fill"
    ORDER_CANCEL = "order_cancel"
    ORDER_REJECT = "order_reject"

    # Signals
    SIGNAL = "signal"

    # System
    SCHEDULED_TICK = "scheduled_tick"
    KILL_SWITCH = "kill_switch"
    SHUTDOWN = "shutdown"
```

### Event Flow

```
1. Market tick arrives from IB Gateway
2. Event Bus distributes to all Signal Agents (fan-out)
3. Signal Agents compute signals independently (parallel)
4. Barrier collects all signals (fan-in)
5. CIO Agent receives aggregated signals
6. CIO makes portfolio decision
7. Risk Agent validates decision
8. Compliance Agent checks regulatory constraints
9. Execution Agent sends orders to broker
```

---

## Data Flow

### Market Data Flow

```
IB Gateway
    |
    v
MarketDataEvent (symbol, bid, ask, last, volume)
    |
    v
Event Bus --> Signal Agents (parallel processing)
    |
    v
SignalEvent (symbol, direction, strength, rationale)
    |
    v
Barrier Synchronization
    |
    v
CIO Agent (decision aggregation)
```

### Order Flow

```
CIO Decision
    |
    v
OrderRequest (symbol, side, quantity, order_type)
    |
    v
Risk Agent --> Check position limits, VaR impact
    |
    v
Compliance Agent --> Check MAR, position limits, best execution
    |
    v
Execution Agent --> Order management, TWAP/VWAP execution
    |
    v
IB Broker --> Order submission
    |
    v
FillEvent --> Portfolio update, P&L tracking
```

---

## Concurrency Model

### Fan-Out/Fan-In Pattern

```python
class SignalBarrier:
    """
    Collects signals from multiple agents before CIO decision.

    1. Fan-out: All signal agents process market data in parallel
    2. Wait: Barrier waits for all expected signals (with timeout)
    3. Fan-in: Collected signals delivered to CIO
    """

    def __init__(self, expected_agents: List[str], timeout: float = 5.0):
        self.expected = set(expected_agents)
        self.collected = {}
        self.timeout = timeout
        self._lock = asyncio.Lock()

    async def submit(self, agent_id: str, signal: SignalEvent):
        async with self._lock:
            self.collected[agent_id] = signal

    async def wait_for_all(self) -> Dict[str, SignalEvent]:
        # Wait with timeout for all signals
        pass
```

### Sequential Processing

After CIO decision, processing is strictly sequential:

```
CIO Decision
    |
    v (must complete before next step)
Risk Validation
    |
    v (must complete before next step)
Compliance Check
    |
    v (must complete before next step)
Execution
```

---

## Integration Points

### Interactive Brokers Integration

```python
class IBBroker:
    """
    Exclusive broker interface for:
    - Market data subscriptions
    - Order execution
    - Portfolio state
    - Account information

    Features:
    - Circuit breaker for connection resilience
    - Order reconciliation on reconnect
    - Automatic reconnection with backoff
    """
```

### Supported IB Order Types

| Order Type | Support | Implementation |
|------------|---------|----------------|
| Market | Yes | Direct submission |
| Limit | Yes | With tick size enforcement |
| Stop | Yes | Server-side monitoring |
| Stop-Limit | Yes | Combined trigger |
| TWAP | Yes | Algorithmic execution |
| VWAP | Yes | Algorithmic execution |
| Iceberg | Yes | Display size management |

---

## Deployment Architecture

### Paper Trading (Default)

```
+-------------------+     +-------------------+
|   Trading System  |<--->|   IB Gateway      |
|   (Paper Mode)    |     |   (Paper Account) |
+-------------------+     +-------------------+
```

### Production Deployment

```
+-------------------+     +-------------------+     +------------------+
|   Load Balancer   |---->|   Trading System  |<--->|   IB Gateway     |
|   (Health Checks) |     |   (Live Mode)     |     |   (Live Account) |
+-------------------+     +--------+----------+     +------------------+
                                   |
                          +--------+----------+
                          |   Monitoring      |
                          |   (Prometheus)    |
                          +-------------------+
```

### Health Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| /health | Overall system health | JSON status |
| /live | Liveness probe (K8s) | 200 OK |
| /ready | Readiness probe (K8s) | 200 OK when ready |
| /metrics | Prometheus metrics | Metrics format |

---

## Security Architecture

### Authentication

- IB Gateway: Username/password with 2FA
- Internal APIs: Token-based authentication
- No secrets in code (environment variables)

### Data Protection

- All logs encrypted at rest
- Personal data minimized (LEI-based identification)
- GDPR compliance for EU data subjects

### Access Control

```python
class AccessControlLogger:
    """
    Logs all system access with:
    - User identification
    - Timestamp
    - Action performed
    - Success/failure status
    """
```

---

## Key Design Decisions

### Why Event-Driven?

1. **Determinism**: Same events produce same results
2. **Testability**: Events can be replayed for testing
3. **Auditability**: Full event history for compliance
4. **Scalability**: Agents can be distributed

### Why Single CIO Authority?

1. **Accountability**: One decision maker, clear responsibility
2. **Conflict Resolution**: No competing decisions
3. **Compliance**: Clear audit trail for regulators

### Why Stateless Agents?

1. **Resilience**: Agents can restart without data loss
2. **Testing**: Pure functions are easier to test
3. **Scaling**: Stateless agents can be replicated

---

## Appendix

### Configuration Schema

```yaml
system:
  mode: paper  # paper | live
  log_level: INFO

broker:
  gateway_host: 127.0.0.1
  gateway_port: 4002
  client_id: 1

risk:
  max_portfolio_var: 100000
  max_position_var: 25000
  max_leverage: 2.0

execution:
  default_algo: TWAP
  max_slippage_bps: 10
```

### Module Dependencies

```
agents/
  +-- cio_agent.py --> core/risk_budget.py, core/position_sizing.py
  +-- risk_agent.py --> core/var_calculator.py, core/stress_tester.py
  +-- execution_agent.py --> core/broker.py, core/smart_order_router.py

core/
  +-- event_bus.py --> core/events.py, core/event_persistence.py
  +-- broker.py --> core/circuit_breaker.py
```

---

*This document is auto-generated and updated as the system evolves.*

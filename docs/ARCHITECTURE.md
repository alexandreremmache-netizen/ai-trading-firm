# Architecture Documentation

## Overview

The AI Trading Firm implements a multi-agent architecture inspired by institutional hedge funds. The system follows strict design principles ensuring auditability, reproducibility, and regulatory compliance.

## Design Principles

### 1. Multi-Agent Architecture
Each agent has a single, well-defined responsibility:
- **Signal Agents**: Generate trading signals (advisory only)
- **Decision Agent**: Make trading decisions (CIO only)
- **Validation Agents**: Risk and compliance checks
- **Execution Agent**: Send orders to broker

### 2. Single Decision Authority
The CIO Agent is the ONLY entity authorized to make trading decisions. This prevents:
- Conflicting orders from multiple sources
- Unclear responsibility chains
- Audit trail gaps

### 3. Event-Driven Model
The system operates on events, not polling:
- Market data triggers signal generation
- Signals trigger decision-making
- Decisions trigger validation
- Validation triggers execution

### 4. Stateless Agents
Agents are stateless where possible:
- State is carried in events
- Enables horizontal scaling
- Simplifies testing and debugging

## System Architecture

### High-Level Flow

```
+------------------------------------------------------------------+
|                         ORCHESTRATOR                              |
|  - Loads configuration                                            |
|  - Initializes components                                         |
|  - Manages agent lifecycle                                        |
|  - Handles shutdown                                               |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                          EVENT BUS                                |
|  - Central message routing                                        |
|  - Signal barrier synchronization (fan-in)                        |
|  - Backpressure handling                                          |
|  - Event persistence for crash recovery                           |
+------------------------------------------------------------------+
                                |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        v           v           v           v           v
   +--------+  +--------+  +--------+  +--------+  +--------+
   | Signal |  | Signal |  | Signal |  | Signal |  | Signal |
   | Agent  |  | Agent  |  | Agent  |  | Agent  |  | Agent  |
   | Macro  |  |StatArb |  |Momentum|  |  MM    |  |Options |
   +--------+  +--------+  +--------+  +--------+  +--------+
        |           |           |           |           |
        +-----+-----+-----+-----+-----+-----+-----+-----+
              |           SIGNAL BARRIER            |
              +------------------+------------------+
                                 |
                                 v
                         +-------------+
                         |     CIO     |
                         |    Agent    |
                         +-------------+
                                 |
                                 v
                         +-------------+
                         |    Risk     |
                         |    Agent    |
                         +-------------+
                                 |
                                 v
                         +-------------+
                         | Compliance  |
                         |    Agent    |
                         +-------------+
                                 |
                                 v
                         +-------------+
                         | Execution   |
                         |    Agent    |
                         +-------------+
                                 |
                                 v
                         +-------------+
                         | Interactive |
                         |   Brokers   |
                         +-------------+
```

### Concurrency Model

#### Fan-Out (Parallel Signal Generation)
Signal agents execute in parallel when market data arrives:

```
MarketDataEvent
      |
      +---> MacroAgent      ---> SignalEvent
      |
      +---> StatArbAgent    ---> SignalEvent
      |
      +---> MomentumAgent   ---> SignalEvent
      |
      +---> MarketMakingAgent --> SignalEvent
      |
      +---> OptionsVolAgent ---> SignalEvent
```

#### Fan-In (Signal Barrier Synchronization)
CIO waits for all signals before making decisions:

```
SignalEvent (Macro)      -+
                          |
SignalEvent (StatArb)    -+---> Signal Barrier ---> CIO Decision
                          |     (timeout: 10s)
SignalEvent (Momentum)   -+
                          |
SignalEvent (MM)         -+
                          |
SignalEvent (OptionsVol) -+
```

#### Sequential Validation
After CIO decision, validation is sequential:

```
DecisionEvent --> Risk Agent --> ValidatedDecisionEvent
                                        |
                                        v
                              Compliance Agent --> ValidatedDecisionEvent
                                                          |
                                                          v
                                                  Execution Agent --> OrderEvent
```

## Component Architecture

### Event Bus

The Event Bus is the central nervous system:

```
+------------------------------------------------------------------+
|                          EVENT BUS                                |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+                     |
|  |   Event Queue    |    |   Subscribers    |                     |
|  |   (Bounded)      |    |   (By Type)      |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Signal Barrier   |    | Backpressure     |                     |
|  | (Fan-In Sync)    |    | Handler          |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Event History    |    | Persistence      |                     |
|  | (Audit Trail)    |    | (Crash Recovery) |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
```

### Broker Integration

Interactive Brokers integration with circuit breaker:

```
+------------------------------------------------------------------+
|                         IB BROKER                                 |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Connection       |    | Circuit Breaker  |                     |
|  | Manager          |    | (Fault Tolerance)|                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Market Data      |    | Order            |                     |
|  | Handler          |    | Management       |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Portfolio        |    | Reconnection     |                     |
|  | State            |    | Handler          |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
```

## Data Flow

### Signal to Order Flow

```
1. Market Data Received
   |
   v
2. MarketDataEvent Published
   |
   v
3. Signal Agents Process (Parallel)
   |
   v
4. SignalEvents Published
   |
   v
5. Signal Barrier Waits (Fan-In)
   |
   v
6. CIO Aggregates Signals
   |
   v
7. CIO Makes Decision (if conviction > threshold)
   |
   v
8. DecisionEvent Published
   |
   v
9. Risk Agent Validates
   |
   v
10. ValidatedDecisionEvent Published (if approved)
    |
    v
11. Compliance Agent Validates
    |
    v
12. ValidatedDecisionEvent Published (if approved)
    |
    v
13. Execution Agent Creates Order
    |
    v
14. OrderEvent Published
    |
    v
15. Broker Executes
    |
    v
16. FillEvent Published
```

### Event Types

| Event Type | Source | Destination | Purpose |
|------------|--------|-------------|---------|
| MarketDataEvent | Broker | Signal Agents | Price/volume updates |
| SignalEvent | Signal Agents | CIO Agent | Trading signals |
| DecisionEvent | CIO Agent | Risk Agent | Trading decisions |
| ValidatedDecisionEvent | Risk/Compliance | Next Validator | Approved decisions |
| OrderEvent | Execution Agent | Broker | Order to execute |
| FillEvent | Broker | System | Execution confirmation |
| RiskAlertEvent | Risk Agent | System | Risk warnings/halt |
| KillSwitchEvent | Risk Agent | System | Emergency halt |

## Infrastructure Components

### Core Components

| Component | Purpose |
|-----------|---------|
| EventBus | Central event routing and synchronization |
| IBBroker | Interactive Brokers connection management |
| AuditLogger | Compliance logging with 7-year retention |
| MonitoringSystem | Metrics, alerts, anomaly detection |

### Risk Infrastructure

| Component | Purpose |
|-----------|---------|
| VaRCalculator | Multi-method Value at Risk calculation |
| StressTester | Scenario-based stress testing |
| CorrelationManager | Cross-asset correlation monitoring |
| PositionSizer | Kelly criterion position sizing |

### Compliance Infrastructure

| Component | Purpose |
|-----------|---------|
| SurveillanceAgent | MAR market abuse detection |
| TransactionReportingAgent | ESMA RTS 22/23 reporting |
| RegulatoryCompliance | RTS 25, RTS 6, etc. |

## Configuration Architecture

```yaml
# Hierarchical configuration structure
firm:
  mode: "paper"              # Trading mode

broker:
  host: "127.0.0.1"         # IB connection
  port: 4002

event_bus:
  max_queue_size: 10000     # Backpressure limits

risk:
  max_portfolio_var_pct: 2.0  # Risk limits

compliance:
  jurisdiction: "EU"        # Regulatory framework

agents:
  macro:                    # Per-agent config
    enabled: true
  cio:
    min_conviction_threshold: 0.6

universe:
  equities: [...]           # Tradable instruments
  futures: [...]
```

## Fault Tolerance

### Circuit Breaker Pattern
The broker connection uses a circuit breaker:
- **Closed**: Normal operation
- **Open**: Requests fail immediately (after threshold breaches)
- **Half-Open**: Test if service recovered

### Event Persistence
Events are persisted for crash recovery:
- Events saved before processing
- On restart, pending events replayed
- Deduplication prevents double-processing

### Graceful Shutdown
Agents shut down gracefully:
1. Stop accepting new events
2. Complete pending tasks (with timeout)
3. Run cleanup handlers
4. Unsubscribe from events

## Security Considerations

### Data Protection
- No storage of credentials in code
- Audit logs encrypted at rest
- GDPR-compliant data handling

### Access Control
- Read-only mode available
- Kill-switch requires manual override
- Live trading requires explicit authorization

## Scalability Considerations

### Horizontal Scaling
- Signal agents can run on separate processes
- Event bus supports distributed deployment
- Stateless agents enable easy scaling

### Performance Limits
- Max 10,000 events in queue
- Max 10 orders per minute (anti-HFT)
- Min 100ms between orders

## Testing Architecture

### Unit Tests
- Each component independently tested
- Mock event bus for isolation
- Mock broker for deterministic testing

### Integration Tests
- End-to-end signal-to-order flow
- Event bus synchronization
- Broker connection handling

### Backtesting
- Historical data replay
- Strategy performance analysis
- Risk metrics validation

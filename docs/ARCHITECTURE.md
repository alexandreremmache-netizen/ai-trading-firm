# Architecture Documentation

## Overview

The AI Trading Firm implements a multi-agent architecture inspired by institutional hedge funds. The system follows strict design principles ensuring auditability, reproducibility, and regulatory compliance (EU/AMF framework).

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

---

## System Architecture

### High-Level Data Flow

```
+------------------------------------------------------------------+
|                     MARKET DATA (Interactive Brokers)             |
|   Real-time prices, volumes, quotes from IB TWS/Gateway           |
+------------------------------------------------------------------+
                              |
                              | MarketDataEvent
                              v
+------------------------------------------------------------------+
|              SIGNAL AGENTS (Parallel Fan-Out Execution)           |
|                                                                    |
|  +----------+ +----------+ +----------+ +----------+ +----------+ |
|  |  Macro   | | StatArb  | |Momentum  | |  Market  | | Options  | |
|  |  Agent   | |  Agent   | |  Agent   | |  Making  | |   Vol    | |
|  +----------+ +----------+ +----------+ +----------+ +----------+ |
|                                                                    |
|  +----------+ +----------+ +----------+ +----------+ +----------+ |
|  | Session  | |  Index   | |   TTM    | |  Event   | |  Mean    | |
|  |  Agent   | | Spread   | | Squeeze  | | Driven   | |Reversion | |
|  +----------+ +----------+ +----------+ +----------+ +----------+ |
|                                                                    |
|  +----------+ +----------+ +----------+                            |
|  |Sentiment | |  Chart   | |Forecast  |  (LLM Agents - Optional)  |
|  |  Agent   | | Analysis | |  Agent   |                            |
|  +----------+ +----------+ +----------+                            |
+------------------------------------------------------------------+
                              |
                              | SignalEvent (Barrier Synchronization)
                              v
+------------------------------------------------------------------+
|                    CIO AGENT (Chief Investment Officer)           |
|                                                                    |
|   - THE single decision-making authority                          |
|   - Aggregates signals with dynamic weights                       |
|   - Kelly criterion position sizing                               |
|   - Regime-aware weight adjustment                                |
|   - Autonomous position management                                |
+------------------------------------------------------------------+
                              |
                              | DecisionEvent
                              v
+------------------------------------------------------------------+
|                         RISK AGENT                                 |
|                                                                    |
|   - Position/sector/leverage limits                               |
|   - VaR (Parametric, Historical, Monte Carlo)                     |
|   - Kill-switch mechanism                                         |
|   - Tiered drawdown response                                      |
|   - Crash protection (velocity-aware)                             |
+------------------------------------------------------------------+
                              |
                              | ValidatedDecisionEvent
                              v
+------------------------------------------------------------------+
|                    COMPLIANCE AGENT (EU/AMF)                       |
|                                                                    |
|   - LEI validation (ISO 17442)                                    |
|   - Blackout period enforcement                                   |
|   - MNPI detection                                                |
|   - Restricted instrument checks                                  |
|   - Market hours validation                                       |
+------------------------------------------------------------------+
                              |
                              | ValidatedDecisionEvent
                              v
+------------------------------------------------------------------+
|                      EXECUTION AGENT                               |
|                                                                    |
|   - ONLY agent authorized to send orders                          |
|   - TWAP/VWAP algorithmic execution                               |
|   - Order book depth analysis                                     |
|   - Fill quality monitoring                                       |
|   - Best execution reporting                                      |
+------------------------------------------------------------------+
                              |
                              | OrderEvent
                              v
+------------------------------------------------------------------+
|                    INTERACTIVE BROKERS                             |
|   Paper Trading: Port 4002 (Gateway) / 7497 (TWS)                 |
|   Live Trading:  Port 4001 (Gateway) / 7496 (TWS)                 |
+------------------------------------------------------------------+
```

---

## Concurrency Model

### Fan-Out (Parallel Signal Generation)

Signal agents execute in parallel when market data arrives:

```
                    MarketDataEvent
                          |
         +-------+-------+-------+-------+-------+
         |       |       |       |       |       |
         v       v       v       v       v       v
      Macro  StatArb Momentum   MM   OptionsVol Session
         |       |       |       |       |       |
         v       v       v       v       v       v
    SignalEvent (all published to EventBus concurrently)
```

### Fan-In (Signal Barrier Synchronization)

CIO waits for all registered signal agents before making decisions:

```
SignalEvent (Macro)       --|
SignalEvent (StatArb)     --|
SignalEvent (Momentum)    --|
SignalEvent (MM)          --|--> Signal Barrier --> CIO Decision
SignalEvent (OptionsVol)  --|    (timeout: 10s)
SignalEvent (Session)     --|
SignalEvent (IndexSpread) --|
...                       --|
```

**Configuration**:
```yaml
event_bus:
  signal_timeout_seconds: 5.0
  sync_barrier_timeout_seconds: 10.0
```

### Sequential Validation Pipeline

After CIO decision, validation is strictly sequential:

```
DecisionEvent
      |
      v
Risk Agent ----[REJECTED]----> Decision Logged (no execution)
      |
      | [APPROVED]
      v
Compliance Agent ---[REJECTED]----> Decision Logged (no execution)
      |
      | [APPROVED]
      v
Execution Agent ----> OrderEvent ----> Interactive Brokers
```

---

## Component Architecture

### Event Bus

The Event Bus is the central nervous system of the trading system:

```
+------------------------------------------------------------------+
|                          EVENT BUS                                |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+                     |
|  |   Event Queue    |    |   Subscribers    |                     |
|  |   (Bounded:      |    |   (By EventType) |                     |
|  |    10,000 max)   |    +------------------+                     |
|  +------------------+                                             |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Signal Barrier   |    | Backpressure     |                     |
|  | (Fan-In Sync)    |    | Handler          |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Dead Letter      |    | Health Check     |                     |
|  | Queue            |    | (Auto-Recovery)  |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Event History    |    | Persistence      |                     |
|  | (Audit Trail)    |    | (Crash Recovery) |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
```

**Backpressure Levels**:
| Level | Queue Usage | Action |
|-------|-------------|--------|
| NORMAL | < 50% | Normal operation |
| WARNING | 50-75% | Log warnings |
| HIGH | 75-90% | Rate limiting |
| CRITICAL | > 90% | Drop low-priority events |

### Broker Integration

Interactive Brokers integration with circuit breaker pattern:

```
+------------------------------------------------------------------+
|                         IB BROKER                                 |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Connection       |    | Circuit Breaker  |                     |
|  | Manager          |    | Pattern          |                     |
|  | (Auto-Reconnect) |    | (Fault Tolerance)|                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Market Data      |    | Order            |                     |
|  | Handler          |    | Management       |                     |
|  | (Rate Limited)   |    | (State Machine)  |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Portfolio        |    | Contract         |                     |
|  | State            |    | Specifications   |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
```

**Rate Limits** (IB API):
- 60 requests per 10 minutes
- 15 seconds minimum between identical requests

---

## Event Types

| Event Type | Source | Destination | Purpose |
|------------|--------|-------------|---------|
| `MarketDataEvent` | Broker | Signal Agents | Price/volume updates |
| `SignalEvent` | Signal Agents | CIO Agent | Trading signals |
| `DecisionEvent` | CIO Agent | Risk Agent | Trading decisions |
| `ValidatedDecisionEvent` | Risk/Compliance | Next Validator | Approved decisions |
| `OrderEvent` | Execution Agent | Broker | Order to execute |
| `FillEvent` | Broker | System | Execution confirmation |
| `RiskAlertEvent` | Risk Agent | System | Risk warnings |
| `KillSwitchEvent` | Risk Agent | System | Emergency halt |
| `RollSignalEvent` | Futures Roll Manager | Execution | Futures contract roll |
| `SurveillanceAlertEvent` | Surveillance Agent | Compliance | Market abuse detection |
| `TransactionReportEvent` | Reporting Agent | ESMA | Regulatory reporting |
| `GreeksUpdateEvent` | Options Agent | Risk Agent | Portfolio Greeks |
| `StressTestResultEvent` | Stress Tester | Risk Agent | Scenario analysis |
| `CorrelationAlertEvent` | Correlation Manager | Risk Agent | Correlation breakdown |

---

## Infrastructure Components

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `EventBus` | `core/event_bus.py` | Central event routing and synchronization |
| `IBBroker` | `core/broker.py` | Interactive Brokers connection management |
| `AuditLogger` | `core/logger.py` | Compliance logging with 7-year retention |
| `MonitoringSystem` | `core/monitoring.py` | Metrics, alerts, anomaly detection |
| `HealthChecker` | `core/health_check.py` | System health monitoring endpoints |

### Risk Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| `VaRCalculator` | `core/var_calculator.py` | Multi-method Value at Risk |
| `StressTester` | `core/stress_tester.py` | Scenario-based stress testing |
| `CorrelationManager` | `core/correlation_manager.py` | Cross-asset correlation monitoring |
| `PositionSizer` | `core/position_sizing.py` | Kelly criterion position sizing |
| `CrashProtection` | `core/crash_protection.py` | Velocity-aware crash detection |
| `StopLossManager` | `core/stop_loss_manager.py` | Automatic stop-loss management |

### Advanced Analytics

| Component | File | Purpose |
|-----------|------|---------|
| `HMMRegimeDetector` | `core/hmm_regime.py` | Hidden Markov Model regime detection |
| `YieldCurveAnalyzer` | `core/yield_curve.py` | Yield curve analysis (2s10s spread) |
| `DXYAnalyzer` | `core/dxy_analyzer.py` | Dollar index correlation analysis |
| `VolumeIndicators` | `core/volume_indicators.py` | VWMA, VWAP, OBV, Volume Profile |
| `KalmanHedgeRatio` | `core/kalman_filter.py` | Dynamic hedge ratio estimation |
| `WalkForwardValidator` | `core/walk_forward.py` | Strategy validation framework |

### Compliance Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| `SurveillanceAgent` | `agents/surveillance_agent.py` | MAR market abuse detection |
| `TransactionReportingAgent` | `agents/transaction_reporting_agent.py` | ESMA RTS 22/23 reporting |
| `LEI Validator` | `agents/compliance_agent.py` | ISO 17442 LEI validation |

---

## Dashboard Architecture

The system includes a real-time monitoring dashboard accessible at `http://localhost:8081`.

```
+------------------------------------------------------------------+
|                     DASHBOARD SERVER (Port 8081)                  |
|                        FastAPI + WebSocket                        |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+                     |
|  | REST API         |    | WebSocket        |                     |
|  | /api/*           |    | /ws              |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Dashboard State  |    | Connection       |                     |
|  | Manager          |    | Manager          |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Advanced         |    | Broker           |                     |
|  | Analytics        |    | Sync             |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
          |                        |
          v                        v
+------------------+    +------------------+
|   Event Bus      |    |   Orchestrator   |
|   (Subscribe)    |    |   (Live Data)    |
+------------------+    +------------------+
```

### Dashboard Features

| Feature | Description |
|---------|-------------|
| Performance Metrics | Real-time P&L, Sharpe, drawdown, Rolling metrics |
| Position Monitoring | Open positions with live P&L, TP/SL levels |
| Closed Positions | Realized P&L history |
| Signal Stream | Live signals from all agents |
| Decision Tracking | CIO decisions with APPROVED/REJECTED status |
| Risk Visualization | Limit utilization gauges, VaR display |
| Correlation Heatmap | Real-time asset correlation visualization |
| Session Performance | Win rate by trading session |
| Signal Consensus | Multi-agent agreement visualization |
| Agent Toggle | Enable/disable agents from dashboard |

---

## Fault Tolerance

### Circuit Breaker Pattern

The broker connection uses a circuit breaker:

```
        +----------+
        |  CLOSED  |  <-- Normal operation
        +----------+
             |
             | Failures exceed threshold
             v
        +----------+
        |   OPEN   |  <-- Requests fail immediately
        +----------+
             |
             | After timeout
             v
        +----------+
        |HALF-OPEN |  <-- Test if service recovered
        +----------+
             |
             +---> Success: Return to CLOSED
             +---> Failure: Return to OPEN
```

### Event Persistence

Events are persisted for crash recovery:
1. Events saved before processing (write-ahead log)
2. On restart, pending events replayed
3. Deduplication prevents double-processing

### Dead Letter Queue

Failed events are moved to DLQ:
- Configurable retry count (default: 3)
- Exponential backoff between retries
- Alerts on DLQ growth

### Graceful Shutdown

Agents shut down gracefully with timeout protection:
1. Stop accepting new events
2. Complete pending tasks (with timeout)
3. Run cleanup handlers
4. Unsubscribe from events
5. Log final state

---

## Security Considerations

### Data Protection
- No storage of credentials in code
- API keys via environment variables
- Audit logs can be encrypted at rest
- GDPR-compliant data handling

### Access Control
- Read-only mode available
- Kill-switch requires manual override
- Live trading requires explicit confirmation
- Paper/Live mode protection

### Network Security
- Dashboard binds to configurable host
- WebSocket supports authentication
- Rate limiting on API endpoints

---

## Scalability Considerations

### Horizontal Scaling
- Signal agents can run on separate processes
- Event bus supports distributed deployment
- Stateless agents enable easy scaling

### Performance Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Max Queue Size | 10,000 events | Backpressure protection |
| Max Orders/Minute | 10 | Anti-HFT compliance |
| Min Order Interval | 100ms | Rate limiting |
| Signal Timeout | 5s | Prevent blocking |
| Barrier Timeout | 10s | CIO decision timing |

---

## Testing Architecture

### Unit Tests
- Each component independently tested
- Mock event bus for isolation
- Mock broker for deterministic testing
- 1,000+ tests total

### Integration Tests
- End-to-end signal-to-order flow
- Event bus synchronization
- Broker connection handling
- Dashboard WebSocket testing

### Walk-Forward Validation
- Out-of-sample strategy testing
- Rolling train/test windows
- Prevents overfitting

---

## Directory Structure

```
ai-trading-firm/
|-- main.py                     # Orchestrator entry point
|-- config.yaml                 # Full configuration
|-- config.simple.yaml          # Simplified configuration
|
|-- agents/                     # 22 Trading agents
|   |-- cio_agent.py            # Chief Investment Officer
|   |-- risk_agent.py           # Risk validation
|   |-- compliance_agent.py     # EU/AMF compliance
|   |-- execution_agent.py      # Order execution
|   |-- macro_agent.py          # Macro signals
|   |-- stat_arb_agent.py       # Statistical arbitrage
|   |-- momentum_agent.py       # Momentum/trend
|   |-- market_making_agent.py  # Market making
|   |-- options_vol_agent.py    # Options volatility
|   |-- session_agent.py        # Session-based trading
|   |-- index_spread_agent.py   # Index spread trading
|   |-- ttm_squeeze_agent.py    # TTM Squeeze signals
|   |-- event_driven_agent.py   # Economic events
|   |-- mean_reversion_agent.py # Mean reversion
|   |-- sentiment_agent.py      # LLM sentiment (optional)
|   |-- chart_analysis_agent.py # Claude Vision (optional)
|   |-- forecasting_agent.py    # LLM forecasting (optional)
|   |-- surveillance_agent.py   # MAR compliance
|   `-- transaction_reporting_agent.py  # ESMA reporting
|
|-- strategies/                 # 13 Strategy implementations
|   |-- momentum_strategy.py
|   |-- stat_arb_strategy.py
|   |-- macro_strategy.py
|   |-- market_making_strategy.py
|   |-- options_vol_strategy.py
|   |-- session_strategy.py
|   |-- index_spread_strategy.py
|   |-- ttm_squeeze_strategy.py
|   |-- event_driven_strategy.py
|   |-- mean_reversion_strategy.py
|   |-- ichimoku_strategy.py
|   `-- seasonality.py
|
|-- core/                       # 90+ Infrastructure modules
|   |-- event_bus.py            # Central event routing
|   |-- broker.py               # IB integration
|   |-- events.py               # Event definitions
|   |-- var_calculator.py       # VaR calculations
|   |-- position_sizing.py      # Kelly criterion
|   `-- ...
|
|-- dashboard/                  # Real-time monitoring
|   |-- server.py               # FastAPI server
|   |-- broker_sync.py          # IB portfolio sync
|   |-- templates/              # HTML templates
|   `-- components/             # Dashboard components
|
|-- data/                       # Data management
|   `-- market_data.py          # Market data manager
|
|-- tests/                      # Test suite (1000+ tests)
|
|-- logs/                       # Audit trail
|   |-- audit.jsonl             # Full audit log
|   |-- trades.jsonl            # Trade history
|   |-- decisions.jsonl         # Decision history
|   `-- system.log              # System logs
|
`-- docs/                       # Documentation
```

---

*Document Version: 2.0*
*Last Updated: February 2026*

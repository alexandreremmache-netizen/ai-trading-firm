# Agent Documentation

## Overview

The AI Trading Firm uses a multi-agent architecture where each agent has a specific, well-defined responsibility. This document describes each agent's purpose, behavior, and configuration.

## Agent Hierarchy

```
                    SIGNAL AGENTS
                    (Generate Signals)
                          |
        +--------+--------+--------+--------+
        |        |        |        |        |
      Macro   StatArb  Momentum   MM    OptionsVol
        |        |        |        |        |
        +--------+--------+--------+--------+
                          |
                    DECISION AGENT
                    (Make Decisions)
                          |
                        CIO
                          |
                   VALIDATION AGENTS
                   (Risk & Compliance)
                          |
                    +-----+-----+
                    |           |
                  Risk     Compliance
                    |           |
                    +-----+-----+
                          |
                   EXECUTION AGENT
                   (Send Orders)
                          |
                      Execution
                          |
                   SURVEILLANCE AGENTS
                   (Monitor Activity)
                          |
                    +-----+-----+
                    |           |
              Surveillance  Transaction
                            Reporting
```

## Base Agent Class

All agents inherit from `BaseAgent` which provides:

```python
class BaseAgent:
    """
    Common functionality:
    - Event subscription
    - Timeout handling
    - Error tracking
    - Graceful shutdown
    - Audit logging
    """

    def get_status(self) -> dict:
        """Returns agent status for monitoring"""
        return {
            "name": self.name,
            "running": self._running,
            "events_processed": self._event_count,
            "errors": self._error_count,
            "uptime_seconds": self._get_uptime_seconds(),
        }
```

---

## Signal Agents

Signal agents subscribe to market data and generate trading signals. They run in parallel (fan-out) and their signals are aggregated by the CIO agent.

### MacroAgent

**Purpose**: Generate signals based on macroeconomic indicators.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Indicators Used**:
- Yield curve (2y-10y spread)
- VIX (volatility index)
- DXY (dollar index)

**Configuration**:
```yaml
agents:
  macro:
    enabled: true
    indicators:
      - "yield_curve"
      - "vix"
      - "dxy"
    rebalance_frequency: "daily"
```

**Signal Logic**:
- Risk-off signal when VIX > 25
- Risk-off signal when yield curve inverts
- Dollar strength affects international exposure

---

### StatArbAgent

**Purpose**: Generate statistical arbitrage and pairs trading signals.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Strategies**:
- Pairs trading (cointegration-based)
- Commodity spreads (crack, crush, calendar)
- Mean reversion

**Configuration**:
```yaml
agents:
  stat_arb:
    enabled: true
    lookback_days: 60
    zscore_entry_threshold: 2.0
    zscore_exit_threshold: 0.5
    pairs:
      - ["AAPL", "MSFT"]
      - ["GOOGL", "META"]
      - ["ES", "NQ"]
      - ["GC", "SI"]
```

**Signal Logic**:
- Entry when z-score > 2.0 (short) or < -2.0 (long)
- Exit when z-score returns to 0.5 threshold
- Uses rolling cointegration tests

---

### MomentumAgent

**Purpose**: Generate trend-following and momentum signals.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Indicators**:
- Moving Average Crossovers (fast/slow)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Rate of Change

**Configuration**:
```yaml
agents:
  momentum:
    enabled: true
    fast_period: 10
    slow_period: 30
    signal_period: 9
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
```

**Signal Logic**:
- Long when fast MA > slow MA and RSI < 70
- Short when fast MA < slow MA and RSI > 30
- MACD histogram confirms direction

---

### MarketMakingAgent

**Purpose**: Generate market-making signals for spread capture.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Configuration**:
```yaml
agents:
  market_making:
    enabled: true
    spread_bps: 10
    max_inventory: 1000
    quote_refresh_ms: 1000
```

**Signal Logic**:
- Quote around mid-price with spread
- Inventory management to avoid directional exposure
- Widen spread in high volatility

---

### OptionsVolAgent

**Purpose**: Generate options volatility signals.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Strategies**:
- IV percentile trading
- Delta-targeted positions
- Volatility surface analysis

**Configuration**:
```yaml
agents:
  options_vol:
    enabled: true
    iv_percentile_threshold: 80
    min_dte: 7
    max_dte: 45
    delta_range: [0.20, 0.40]
```

**Signal Logic**:
- Sell volatility when IV percentile > 80
- Buy volatility when IV percentile < 20
- Target delta range for directional bias

---

## Decision Agent

### CIOAgent (Chief Investment Officer)

**Purpose**: THE SINGLE DECISION-MAKING AUTHORITY. Aggregates signals from all strategy agents and makes final trading decisions.

**Subscribed Events**: `SignalEvent` (via barrier), `ValidatedDecisionEvent`

**Emitted Events**: `DecisionEvent`

**Key Responsibilities**:
1. Wait for signal barrier synchronization (fan-in)
2. Aggregate signals with dynamic weights
3. Apply conviction threshold
4. Calculate position sizes (Kelly criterion)
5. Log decisions with full rationale

**Configuration**:
```yaml
agents:
  cio:
    signal_weight_macro: 0.15
    signal_weight_stat_arb: 0.25
    signal_weight_momentum: 0.25
    signal_weight_market_making: 0.15
    signal_weight_options_vol: 0.20
    min_conviction_threshold: 0.6
    max_concurrent_decisions: 5
    use_dynamic_weights: true
    use_kelly_sizing: true
    kelly_fraction: 0.5
```

**Decision Logic**:
```
1. Collect all signals from barrier
2. Group signals by symbol
3. Calculate weighted average strength/confidence
4. Determine consensus direction
5. If confidence > threshold:
   a. Calculate position size
   b. Create DecisionEvent with rationale
   c. Publish decision
```

**Dynamic Weight Adjustment**:
- Regime-based: Risk-on/off adjusts strategy weights
- Performance-based: Higher Sharpe increases weight
- Correlation-adjusted: Correlated signals get discounted

---

## Validation Agents

### RiskAgent

**Purpose**: Validate all trading decisions against risk limits. Implements kill-switch for emergency situations.

**Subscribed Events**: `DecisionEvent`

**Emitted Events**: `ValidatedDecisionEvent`, `RiskAlertEvent`

**Risk Checks**:

| Check | Limit | Action on Breach |
|-------|-------|------------------|
| Position Size | 5% of portfolio | Reject/Reduce |
| Sector Exposure | 20% of portfolio | Reject |
| Leverage | 2x max | Reject |
| VaR (95%) | 2% of portfolio | Reject |
| Daily Loss | -3% | Kill-switch |
| Drawdown | -10% | Kill-switch |
| Orders/Minute | 10 max | Reject |
| Order Interval | 100ms min | Reject |

**Configuration**:
```yaml
risk:
  max_portfolio_var_pct: 2.0
  max_position_size_pct: 5.0
  max_sector_exposure_pct: 20.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
  max_leverage: 2.0
  max_orders_per_minute: 10
  min_order_interval_ms: 100
```

**Kill-Switch Triggers**:
- Daily loss exceeds limit
- Maximum drawdown breached
- VaR limit breach
- Manual activation
- Connectivity loss
- Latency breach (MiFID II RTS 6)

---

### ComplianceAgent

**Purpose**: Validate all trading decisions against regulatory requirements (EU/AMF).

**Subscribed Events**: `ValidatedDecisionEvent` (from Risk)

**Emitted Events**: `ValidatedDecisionEvent`, `RiskAlertEvent`

**Compliance Checks**:

| Check | Rule | Action on Breach |
|-------|------|------------------|
| Blackout Period | No trading during earnings | Reject |
| MNPI Detection | No insider trading | Reject |
| Restricted List | Banned instruments | Reject |
| Market Hours | Trading hours only | Reject |
| SSR (Short Sale) | Uptick rule compliance | Reject |
| Data Sources | Approved sources only | Reject |
| LEI Validation | Valid Legal Entity ID | Reject |

**Configuration**:
```yaml
compliance:
  jurisdiction: "EU"
  regulator: "AMF"
  require_rationale: true
  audit_retention_days: 2555  # 7 years
  banned_instruments: []
  allowed_asset_classes:
    - "equity"
    - "etf"
    - "option"
    - "future"
    - "forex"
```

---

## Execution Agent

### ExecutionAgentImpl

**Purpose**: THE ONLY AGENT AUTHORIZED TO SEND ORDERS TO THE BROKER. Receives validated decisions and executes them using algorithmic execution.

**Subscribed Events**: `ValidatedDecisionEvent`, `FillEvent`, `KillSwitchEvent`

**Emitted Events**: `OrderEvent`, `OrderStateChangeEvent`

**Execution Algorithms**:
- **TWAP**: Time-weighted average price
- **VWAP**: Volume-weighted average price
- **Market**: Immediate execution

**Configuration**:
```yaml
agents:
  execution:
    default_algo: "TWAP"
    slice_interval_seconds: 60
    max_slippage_bps: 50
```

**Order State Machine**:
```
CREATED --> PENDING --> SUBMITTED --> ACKNOWLEDGED
                                          |
                    +----------+----------+----------+
                    |          |          |          |
                 PARTIAL    FILLED    CANCELLED   REJECTED
                    |
                    v
                 FILLED
```

**Features**:
- Slice-level fill tracking
- Price improvement analysis
- Market impact estimation
- Best execution reporting

---

## Surveillance Agents

### SurveillanceAgent

**Purpose**: Monitor for market abuse per MAR 2014/596/EU.

**Subscribed Events**: `OrderEvent`, `FillEvent`, `MarketDataEvent`

**Emitted Events**: `SurveillanceAlertEvent`

**Detection Capabilities**:
- Wash trading
- Spoofing
- Quote stuffing
- Layering

**Configuration**:
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

### TransactionReportingAgent

**Purpose**: Generate transaction reports per ESMA RTS 22/23.

**Subscribed Events**: `FillEvent`

**Emitted Events**: `TransactionReportEvent`

**Configuration**:
```yaml
transaction_reporting:
  enabled: true
  reporting_deadline_minutes: 15
  firm_lei: ""  # Must be valid LEI
  firm_country: "FR"
  default_venue: "XPAR"
```

---

## Agent Lifecycle

### Startup Sequence
```
1. Orchestrator loads configuration
2. Event bus initialized
3. Broker connection established
4. Agents initialized in order:
   a. Signal agents (parallel)
   b. CIO agent
   c. Risk agent
   d. Compliance agent
   e. Execution agent
   f. Surveillance agents
5. Agents subscribe to events
6. Market data streaming starts
```

### Shutdown Sequence
```
1. Shutdown signal received
2. Market data stopped
3. Surveillance agents stopped
4. Execution agent stopped (completes pending orders)
5. Compliance agent stopped
6. Risk agent stopped
7. CIO agent stopped
8. Signal agents stopped (parallel)
9. Event bus stopped
10. Broker disconnected
```

### Graceful Shutdown
Each agent implements graceful shutdown:
```python
async def stop(self, timeout: float = 10.0) -> bool:
    """
    1. Stop accepting new events
    2. Wait for pending tasks (with timeout)
    3. Run cleanup handlers
    4. Unsubscribe from events
    5. Log shutdown
    """
```

---

## Monitoring

All agents expose status information:

```python
{
    "name": "CIOAgent",
    "enabled": true,
    "running": true,
    "shutdown_state": "running",
    "pending_tasks": 0,
    "started_at": "2024-01-15T10:30:00Z",
    "last_heartbeat": "2024-01-15T10:35:00Z",
    "events_processed": 1523,
    "errors": 0,
    "uptime_seconds": 300.5,
    # Agent-specific fields...
}
```

Health check endpoint provides aggregate status:
```
GET /health
GET /ready
GET /live
```

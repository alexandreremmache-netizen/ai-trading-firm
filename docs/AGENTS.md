# Agent Documentation

## Overview

The AI Trading Firm uses a multi-agent architecture where each agent has a specific, well-defined responsibility. This document describes all 22 agents, their purposes, behaviors, and configurations.

## Agent Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Signal Agents | 13 | Generate trading signals |
| Decision Agent | 1 | Make trading decisions |
| Validation Agents | 2 | Risk and compliance checks |
| Execution Agent | 1 | Send orders to broker |
| Surveillance Agents | 2 | Monitor and report activity |
| Alternative Agents | 3 | Experimental implementations |

## Agent Hierarchy

```
                         SIGNAL AGENTS (13)
                        (Generate Signals)
                              |
    +--------+--------+--------+--------+--------+--------+
    |        |        |        |        |        |        |
  Macro  StatArb  Momentum   MM    OptionsVol Session  IndexSpread
    |        |        |        |        |        |        |
  TTMSqueeze EventDriven MeanReversion Sentiment* Chart* Forecast*
    |                                              (* = LLM agents)
    +------------------+------------------+------------------+
                              |
                        DECISION AGENT (1)
                        (Make Decisions)
                              |
                            CIO
                              |
                   VALIDATION AGENTS (2)
                   (Risk & Compliance)
                              |
                    +-----+-----+
                    |           |
                  Risk     Compliance
                    |           |
                    +-----+-----+
                              |
                   EXECUTION AGENT (1)
                   (Send Orders)
                              |
                      Execution
                              |
                   SURVEILLANCE AGENTS (2)
                   (Monitor Activity)
                              |
                    +-----+-----+
                    |           |
              Surveillance  Transaction
                            Reporting
```

---

## Base Agent Classes

All agents inherit from specialized base classes defined in `core/agent_base.py`:

```python
class BaseAgent:
    """Common functionality for all agents."""
    name: str
    config: AgentConfig
    event_bus: EventBus
    audit_logger: AuditLogger

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def get_status(self) -> dict: ...

class SignalAgent(BaseAgent):
    """Base for signal-generating agents."""
    async def process_market_data(self, event: MarketDataEvent) -> SignalEvent | None: ...

class DecisionAgent(BaseAgent):
    """Base for decision-making agents."""
    async def process_signals(self, signals: list[SignalEvent]) -> DecisionEvent | None: ...

class ValidationAgent(BaseAgent):
    """Base for validation agents."""
    async def validate(self, decision: DecisionEvent) -> ValidatedDecisionEvent | None: ...

class ExecutionAgent(BaseAgent):
    """Base for execution agents."""
    async def execute(self, decision: ValidatedDecisionEvent) -> OrderEvent | None: ...
```

---

## Signal Agents (13 Agents)

Signal agents subscribe to market data and generate trading signals. They run in parallel (fan-out) and their signals are aggregated by the CIO agent.

### 1. MacroAgent

**File**: `agents/macro_agent.py`

**Purpose**: Generate signals based on macroeconomic indicators and regime detection.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- Yield curve analysis (2s10s spread)
- VIX-based risk-on/risk-off signals
- DXY (Dollar Index) correlation
- HMM regime detection integration

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

### 2. StatArbAgent

**File**: `agents/stat_arb_agent.py`

**Purpose**: Generate statistical arbitrage and pairs trading signals.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- Pairs trading (cointegration-based)
- Johansen cointegration test
- Kalman filter for dynamic hedge ratios
- Commodity spreads (crack, crush, calendar)
- Dollar-neutral spread sizing

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
    divergence_enabled: true
    divergence_threshold: 2.0
```

**Signal Logic**:
- Entry when z-score > 2.0 (short spread) or < -2.0 (long spread)
- Exit when z-score returns to 0.5 threshold
- Uses rolling cointegration tests
- Half-life calculation for mean reversion timing

---

### 3. MomentumAgent

**File**: `agents/momentum_agent.py`

**Purpose**: Generate trend-following and momentum signals.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- Moving Average Crossovers (fast/slow)
- RSI with asset-specific thresholds
- MACD with timeframe-specific settings
- ADX trend strength filter
- 52-week high/low signal
- Dual Momentum (Antonacci)
- Multi-timeframe analysis
- Volume confirmation

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
    adx_period: 14
    adx_strong_threshold: 25
    use_triple_confirmation: true
    demand_zones_enabled: true
```

**Signal Logic**:
- Long when fast MA > slow MA and RSI < overbought
- Short when fast MA < slow MA and RSI > oversold
- Triple confirmation (RSI, MACD, Stochastic) for high-conviction signals
- ADX filter for trend strength

---

### 4. MarketMakingAgent

**File**: `agents/market_making_agent.py`

**Purpose**: Generate market-making signals for spread capture using Avellaneda-Stoikov model.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- Optimal market making (Avellaneda-Stoikov)
- Inventory management
- Volatility-adjusted spreads
- Quote refresh management

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
- Skew quotes based on inventory (lean toward reducing inventory)
- Widen spread in high volatility
- Reduce size near inventory limits

---

### 5. OptionsVolAgent

**File**: `agents/options_vol_agent.py`

**Purpose**: Generate options volatility signals.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- IV percentile trading
- Delta-targeted positions
- Volatility surface analysis
- Term structure analysis
- Greeks monitoring

**Configuration**:
```yaml
agents:
  options_vol:
    enabled: false  # Currently disabled
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

### 6. SessionAgent

**File**: `agents/session_agent.py`

**Purpose**: Generate signals based on session-specific patterns and Opening Range Breakout (ORB).

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- Opening Range Breakout strategy
- Session momentum tracking
- Time-of-day patterns
- Session-specific volatility

**Configuration**:
```yaml
agents:
  session:
    enabled: true
    opening_range_minutes: 30
    breakout_threshold_atr: 0.5
    session_momentum_enabled: true
    symbols:
      - "ES"
      - "NQ"
      - "SPY"
      - "QQQ"
```

**Signal Logic**:
- Track first 30 minutes high/low (opening range)
- Long on breakout above range + ATR threshold
- Short on breakdown below range - ATR threshold
- Session momentum confirms direction

---

### 7. IndexSpreadAgent

**File**: `agents/index_spread_agent.py`

**Purpose**: Generate signals for index spread trading (MES/MNQ pairs).

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- Z-score based spread trading
- Micro futures focus (MES, MNQ)
- Mean reversion on index ratio

**Configuration**:
```yaml
agents:
  index_spread:
    enabled: true
    spread_pairs:
      - ["MES", "MNQ"]
      - ["ES", "NQ"]
    zscore_entry: 2.0
    zscore_exit: 0.5
    lookback_periods: 20
```

**Signal Logic**:
- Calculate spread ratio between indices
- Entry when z-score exceeds threshold
- Exit when z-score returns to mean

---

### 8. TTMSqueezeAgent

**File**: `agents/ttm_squeeze_agent.py`

**Purpose**: Generate signals based on TTM Squeeze (Bollinger Bands inside Keltner Channels).

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- BB inside KC detection (squeeze)
- Momentum histogram direction
- Squeeze release detection

**Configuration**:
```yaml
agents:
  ttm_squeeze:
    enabled: true
    bb_length: 20
    bb_mult: 2.0
    kc_length: 20
    kc_mult: 1.5
    momentum_length: 12
    symbols:
      - "ES"
      - "NQ"
      - "SPY"
      - "AAPL"
```

**Signal Logic**:
- Squeeze ON: BB bands inside KC bands (low volatility)
- Squeeze OFF: BB bands outside KC (volatility expansion)
- Entry on squeeze release with momentum confirmation

---

### 9. EventDrivenAgent

**File**: `agents/event_driven_agent.py`

**Purpose**: Generate signals around major economic events (FOMC, NFP, CPI).

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- Economic calendar integration
- Pre/post event analysis
- Volatility spike trading
- Event-specific strategies

**Configuration**:
```yaml
agents:
  event_driven:
    enabled: true
    events:
      - "FOMC"
      - "NFP"
      - "CPI"
    pre_event_hours: 24
    post_event_hours: 4
    min_volatility_increase: 1.5
```

**Signal Logic**:
- Monitor volatility leading up to events
- Position for expected moves based on historical patterns
- Trade volatility spikes post-event

---

### 10. MeanReversionAgent

**File**: `agents/mean_reversion_agent.py`

**Purpose**: Generate mean reversion signals using RSI extremes, Bollinger Bands, and z-scores.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**Features**:
- RSI extreme detection
- Bollinger Band touches
- Z-score mean reversion
- Multiple confirmation modes

**Configuration**:
```yaml
agents:
  mean_reversion:
    enabled: true
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
    bb_period: 20
    bb_std: 2.0
    zscore_threshold: 2.0
    symbols:
      - "SPY"
      - "QQQ"
      - "AAPL"
```

**Signal Logic**:
- Long when RSI < 30 and price at lower BB
- Short when RSI > 70 and price at upper BB
- Z-score confirmation for entry timing

---

### 11. SentimentAgent (LLM)

**File**: `agents/sentiment_agent.py`

**Purpose**: Generate signals based on news sentiment analysis using LLM (Claude/GPT).

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**API Tokens**: Yes (Claude/OpenAI) - Disabled by default

**Features**:
- RSS feed integration
- LLM-powered sentiment extraction
- Confidence-weighted signal generation
- Recency-weighted aggregation

**Configuration**:
```yaml
agents:
  sentiment:
    enabled: false  # Disabled by default - uses API tokens
    llm_provider: "anthropic"
    model: "claude-sonnet-4-5-20250929"
    news_sources:
      - "https://www.investing.com/rss/news.rss"
      - "https://feeds.marketwatch.com/marketwatch/topstories/"
    analysis_interval_seconds: 300
    max_news_age_hours: 24
    min_confidence: 0.5
```

**Signal Logic**:
- Fetch news from RSS feeds
- LLM analyzes sentiment for each symbol
- Aggregate sentiment with recency weighting
- Generate signal when sentiment exceeds threshold

---

### 12. ChartAnalysisAgent (LLM Vision)

**File**: `agents/chart_analysis_agent.py`

**Purpose**: Generate signals based on visual chart pattern analysis using Claude Vision.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**API Tokens**: Yes (Claude Vision) - Disabled by default

**Features**:
- Candlestick pattern detection
- Support/resistance identification
- Trend line analysis
- Chart formations (head & shoulders, triangles, etc.)

**Configuration**:
```yaml
agents:
  chart_analysis:
    enabled: false  # Disabled by default - uses API tokens
    model: "claude-sonnet-4-5-20250929"
    symbols:
      - "SPY"
      - "QQQ"
      - "AAPL"
    analysis_interval_seconds: 300
    candle_count: 50
    chart_timeframe: "15min"
    min_confidence: 0.5
```

**Signal Logic**:
- Generate candlestick chart images
- Send to Claude Vision for pattern analysis
- Returns direction, confidence, and detected patterns
- Includes support/resistance levels

---

### 13. ForecastingAgent (LLM)

**File**: `agents/forecasting_agent.py`

**Purpose**: Generate signals based on LLM-powered price forecasting with confidence intervals.

**Subscribed Events**: `MarketDataEvent`

**Emitted Events**: `SignalEvent`

**API Tokens**: Yes (Claude/OpenAI) - Disabled by default

**Features**:
- Multi-horizon forecasts (1h, 4h, 1d)
- Confidence intervals
- Optional TFT (Temporal Fusion Transformer) backend
- Price history analysis

**Configuration**:
```yaml
agents:
  forecasting:
    enabled: false  # Disabled by default - uses API tokens
    forecast_horizons:
      - "1h"
      - "4h"
    min_confidence: 0.6
    symbols:
      - "SPY"
      - "ES"
      - "NQ"
    llm:
      llm_provider: "anthropic"
      model: "claude-sonnet-4-5-20250929"
```

**Signal Logic**:
- Analyze price history and calculate momentum
- Generate price forecast with confidence bounds
- Direction based on predicted vs current price
- Tighter confidence interval = higher conviction

---

## Decision Agent (1 Agent)

### CIOAgent (Chief Investment Officer)

**File**: `agents/cio_agent.py`

**Purpose**: THE SINGLE DECISION-MAKING AUTHORITY. Aggregates signals from all strategy agents and makes final trading decisions.

**Subscribed Events**: `SignalEvent` (via barrier), `ValidatedDecisionEvent`

**Emitted Events**: `DecisionEvent`

**Key Responsibilities**:
1. Wait for signal barrier synchronization (fan-in)
2. Aggregate signals with dynamic weights
3. Apply conviction threshold
4. Calculate position sizes (Kelly criterion)
5. Autonomous position management
6. Log decisions with full rationale

**Configuration**:
```yaml
agents:
  cio:
    signal_weight_macro: 0.10
    signal_weight_stat_arb: 0.12
    signal_weight_momentum: 0.15
    signal_weight_market_making: 0.08
    signal_weight_options_vol: 0.0
    signal_weight_sentiment: 0.13
    signal_weight_chart_analysis: 0.15
    signal_weight_forecasting: 0.15
    min_conviction_threshold: 0.6
    max_concurrent_decisions: 5
    use_dynamic_weights: true
    use_kelly_sizing: true
    kelly_fraction: 0.5

cio:
  position_management_enabled: true
  position_review_interval_seconds: 60
  position_management:
    max_loss_pct: 5.0
    profit_target_pct: 15.0
    conviction_drop_threshold: 0.3
    max_holding_days: 30.0
```

**Decision Actions**:
| Action | Description |
|--------|-------------|
| `BUY` | Open or increase long position |
| `SELL` | Open or increase short position |
| `CLOSE_LOSER` | Close losing position (full exit) |
| `TAKE_PROFIT` | Close profitable position |
| `REDUCE_POSITION` | Reduce position size |
| `INCREASE_POSITION` | Increase position size |
| `HOLD` | No action needed |

**Dynamic Weight Adjustment**:
- Regime-based: Risk-on/off adjusts strategy weights
- Performance-based: Higher Sharpe increases weight
- Correlation-adjusted: Correlated signals get discounted
- VIX regime weights

---

### DRLCIOAgent (Alternative - Experimental)

**File**: `agents/drl_cio_agent.py`

**Purpose**: Alternative CIO implementation using Deep Reinforcement Learning (PPO/A2C) for signal aggregation.

**Features**:
- PPO/A2C from stable-baselines3
- Learns optimal signal weights from experience
- Adapts to market regimes automatically
- Falls back to heuristic CIO if model unavailable

**Note**: Experimental - requires separate training.

---

## Validation Agents (2 Agents)

### RiskAgent

**File**: `agents/risk_agent.py`

**Purpose**: Validate all trading decisions against risk limits. Implements kill-switch and tiered drawdown response.

**Subscribed Events**: `DecisionEvent`, `GreeksUpdateEvent`

**Emitted Events**: `ValidatedDecisionEvent`, `RiskAlertEvent`

**Risk Checks**:

| Check | Default Limit | Action on Breach |
|-------|---------------|------------------|
| Position Size | 2.5% of portfolio | Reject/Reduce |
| Sector Exposure | 20% of portfolio | Reject |
| Leverage | 1.5x max | Reject |
| VaR (95%) | 2% of portfolio | Reject |
| Daily Loss | -3% | Tiered response |
| Drawdown | -10% | Tiered response |
| Orders/Minute | 10 max | Reject |
| Order Interval | 100ms min | Reject |

**Tiered Drawdown Response**:
| Level | Threshold | Action |
|-------|-----------|--------|
| NORMAL | < 3% | Normal trading |
| WARNING | 3-5% | Reduce new position sizes 50% |
| CRITICAL | 5-8% | Close worst 20%, no new longs |
| SEVERE | 8-10% | Close worst 50%, defensive mode |
| MAXIMUM | >= 10% | Close all positions |

**Configuration**:
```yaml
risk:
  max_portfolio_var_pct: 2.0
  max_position_size_pct: 2.5
  max_sector_exposure_pct: 20.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
  max_leverage: 1.5
  max_orders_per_minute: 10
  min_order_interval_ms: 100
  drawdown:
    warning_pct: 3.0
    critical_pct: 5.0
    severe_pct: 8.0
    maximum_pct: 10.0
```

**Kill-Switch Reasons**:
- `DAILY_LOSS_LIMIT` - Daily loss exceeds limit
- `MAX_DRAWDOWN` - Maximum drawdown breached
- `VAR_BREACH` - VaR limit breach
- `MANUAL` - Manual activation
- `CONNECTIVITY_LOSS` - Broker connection lost
- `LATENCY_BREACH` - MiFID II RTS 6 latency
- `MARKET_DISRUPTION` - Unusual market conditions

---

### ComplianceAgent

**File**: `agents/compliance_agent.py`

**Purpose**: Validate all trading decisions against regulatory requirements (EU/AMF).

**Subscribed Events**: `ValidatedDecisionEvent` (from Risk)

**Emitted Events**: `ValidatedDecisionEvent`, `RiskAlertEvent`

**Compliance Checks**:

| Check | Rule | Action on Breach |
|-------|------|------------------|
| LEI Validation | ISO 17442 | Reject |
| Blackout Period | No trading during earnings | Reject |
| MNPI Detection | No insider trading | Reject |
| Restricted List | Banned instruments | Reject |
| Market Hours | Trading hours only | Reject |
| SSR (Short Sale) | Uptick rule compliance | Reject |
| Data Sources | Approved sources only | Reject |
| Asset Class | Allowed asset classes | Reject |

**LEI Validation**:
- 20 characters alphanumeric
- Valid LOU (Local Operating Unit) prefix
- MOD 97-10 checksum verification
- GLEIF database verification recommended for production

**Configuration**:
```yaml
compliance:
  jurisdiction: "EU"
  regulator: "AMF"
  firm_lei: "529900T8BM49AURSDO55"
  require_rationale: true
  audit_retention_days: 2555  # 7 years
  banned_instruments: []
  allowed_asset_classes:
    - "equity"
    - "future"
    - "forex"
    - "commodity"
```

---

## Execution Agent (1 Agent)

### ExecutionAgentImpl

**File**: `agents/execution_agent.py`

**Purpose**: THE ONLY AGENT AUTHORIZED TO SEND ORDERS TO THE BROKER. Receives validated decisions and executes them using algorithmic execution.

**Subscribed Events**: `ValidatedDecisionEvent`, `FillEvent`, `KillSwitchEvent`

**Emitted Events**: `OrderEvent`, `OrderStateChangeEvent`

**Execution Algorithms**:
- **TWAP**: Time-weighted average price (slices order over time)
- **VWAP**: Volume-weighted average price
- **Market**: Immediate execution
- **Adaptive TWAP**: Volatility-adjusted slicing

**Features**:
- Order book depth analysis
- Fill quality monitoring
- Slippage tracking
- Best execution reporting
- Stop-loss integration

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

**Configuration**:
```yaml
agents:
  execution:
    default_algo: "TWAP"
    slice_interval_seconds: 60
    max_slippage_bps: 50

execution:
  max_spread_atr_pct: 0.10
  spread_limits:
    ES: 0.05
    MES: 0.05
    NG: 0.15
```

---

## Surveillance Agents (2 Agents)

### SurveillanceAgent

**File**: `agents/surveillance_agent.py`

**Purpose**: Monitor for market abuse per MAR 2014/596/EU.

**Subscribed Events**: `OrderEvent`, `FillEvent`, `MarketDataEvent`

**Emitted Events**: `SurveillanceAlertEvent`

**Detection Capabilities**:
| Abuse Type | Detection Method |
|------------|------------------|
| Wash Trading | Same-side trades within time window |
| Spoofing | High cancel rate > 80% |
| Quote Stuffing | > 10 quotes/second |
| Layering | Multi-level order manipulation |

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

**File**: `agents/transaction_reporting_agent.py`

**Purpose**: Generate transaction reports per ESMA RTS 22/23.

**Subscribed Events**: `FillEvent`

**Emitted Events**: `TransactionReportEvent`

**Report Fields**:
- Trade execution details
- Firm LEI
- Counterparty information
- Venue identifier
- Timestamp (microsecond precision)

**Configuration**:
```yaml
transaction_reporting:
  enabled: true
  reporting_deadline_minutes: 15
  firm_lei: "529900T8BM49AURSDO55"
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
3. Stop-loss manager stopped
4. Surveillance agents stopped
5. Transaction reporting agent stopped
6. Execution agent stopped (completes pending orders)
7. Compliance agent stopped
8. Risk agent stopped
9. CIO agent stopped
10. Signal agents stopped (parallel)
11. Event bus stopped
12. Broker disconnected
```

### Graceful Shutdown

Each agent implements graceful shutdown with timeout:
```python
async def stop(self, timeout: float = 10.0) -> bool:
    """
    1. Stop accepting new events
    2. Wait for pending tasks (with timeout)
    3. Run cleanup handlers
    4. Unsubscribe from events
    5. Log shutdown state
    """
```

---

## Agent Status Monitoring

All agents expose status information via `get_status()`:

```python
{
    "name": "CIOAgent",
    "enabled": true,
    "running": true,
    "shutdown_state": "running",
    "pending_tasks": 0,
    "started_at": "2026-02-04T10:30:00Z",
    "last_heartbeat": "2026-02-04T10:35:00Z",
    "events_processed": 1523,
    "errors": 0,
    "uptime_seconds": 300.5,
    "latency_ms": 12.5,
    "health_score": 100.0
}
```

### Dashboard Integration

Agents are displayed in the dashboard with:
- Status indicator (Active/Idle/Error/Stopped)
- Latency tracking (rolling average)
- Event count
- Health score (0-100)
- Toggle on/off (signal agents only)
- LLM badge for API token-consuming agents

---

## Creating Custom Agents

To create a custom signal agent:

```python
from core.agent_base import SignalAgent, AgentConfig
from core.events import SignalEvent, SignalDirection, MarketDataEvent

class CustomAgent(SignalAgent):
    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)
        # Initialize custom parameters
        self._threshold = config.parameters.get("threshold", 0.5)

    async def process_market_data(
        self,
        event: MarketDataEvent
    ) -> SignalEvent | None:
        # Implement signal generation logic
        signal_strength = self._calculate_signal(event)

        if abs(signal_strength) > self._threshold:
            return SignalEvent(
                agent_name=self.name,
                symbol=event.symbol,
                direction=SignalDirection.LONG if signal_strength > 0 else SignalDirection.SHORT,
                strength=abs(signal_strength),
                confidence=0.7,
                rationale="Custom signal triggered",
            )
        return None
```

Register the agent in `main.py` or use `AgentFactory`:

```python
custom_agent = CustomAgent(
    config=AgentConfig(name="CustomAgent", enabled=True, parameters={}),
    event_bus=self._event_bus,
    audit_logger=self._audit_logger,
)
self._signal_agents.append(custom_agent)
self._event_bus.register_signal_agent("CustomAgent")
```

---

*Document Version: 2.0*
*Last Updated: February 2026*

# cio_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\cio_agent.py`

## Overview

CIO (Chief Investment Officer) Agent
====================================

THE SINGLE DECISION-MAKING AUTHORITY.

This agent is the ONLY one authorized to make trading decisions.
It aggregates signals from all strategy agents and decides whether to trade.

Per the constitution:
- One and only one decision authority
- Decisions must include rationale and data sources
- All decisions are logged for compliance

Enhanced features:
- Kelly criterion position sizing
- Dynamic signal weights (regime-dependent, performance-weighted)
- Correlation-adjusted sizing
- Performance attribution integration

## Classes

### MarketRegime

**Inherits from**: Enum

Market regime classification for weight adjustment.

### SignalAggregation

Aggregated signals for decision making.

### StrategyPerformance

Performance metrics for a strategy used in dynamic weighting.

### CIOAgent

**Inherits from**: DecisionAgent

Chief Investment Officer Agent.

THE ONLY DECISION-MAKING AUTHORITY IN THE SYSTEM.

Responsibilities:
1. Wait for signal barrier synchronization (fan-in)
2. Aggregate signals from all strategy agents
3. Apply portfolio-level constraints
4. Make final trading decisions
5. Log decisions with full rationale

This agent does NOT:
- Generate signals (that's strategy agents' job)
- Execute orders (that's execution agent's job)
- Validate risk/compliance (that's risk agent's job)

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `async def initialize(self) -> None`

Initialize CIO agent.

##### `def get_subscribed_events(self) -> list[EventType]`

CIO subscribes to validated decisions only - signals come via barrier.

##### `async def start(self) -> None`

Start CIO agent with barrier monitoring loop.

##### `async def stop(self) -> None`

Stop CIO agent and cancel barrier monitoring.

##### `async def process_event(self, event: Event) -> None`

Process validated decision events.

Signal processing is handled by the barrier monitoring loop,
not by individual event subscription (per CLAUDE.md fan-in).

##### `def get_signal_correlations(self) -> dict[str, Any]`

Get signal correlation information for monitoring (#Q5).

Returns:
    Dictionary with correlation matrix and statistics

##### `def set_position_sizer(self, position_sizer) -> None`

Set position sizer for Kelly criterion sizing.

##### `def set_attribution(self, attribution) -> None`

Set performance attribution for tracking.

##### `def set_correlation_manager(self, correlation_manager) -> None`

Set correlation manager for correlation-adjusted sizing.

##### `def set_risk_budget_manager(self, risk_budget_manager) -> None`

Set cross-strategy risk budget manager (#P3).

Enables:
- Risk budget allocation across strategies
- Position rejection if strategy is over budget
- Rebalancing trigger monitoring

##### `def set_portfolio_value(self, value: float) -> None`

Update portfolio value for position sizing.

##### `def update_price(self, symbol: str, price: float) -> None`

Update price cache for a symbol.

Called by orchestrator when market data is received.
Required for accurate Kelly position sizing.

##### `def update_prices(self, prices: dict[str, float]) -> None`

Bulk update price cache.

Called by orchestrator with latest market prices.

##### `def set_market_regime(self, regime: MarketRegime) -> None`

Set current market regime.

This triggers recalculation of dynamic weights.

##### `def update_strategy_performance(self, strategy: str, rolling_sharpe: float, win_rate: float, recent_pnl: float, signal_accuracy: float, avg_win: float, avg_loss: float, total_trades: int) -> None`

Update performance metrics for a strategy.

Called by the orchestrator or attribution system.

Args:
    strategy: Strategy name
    rolling_sharpe: Rolling Sharpe ratio
    win_rate: Probability of winning trade (0-1)
    recent_pnl: Recent P&L in dollars
    signal_accuracy: Signal accuracy rate (0-1)
    avg_win: Average profit on winning trades (dollars)
    avg_loss: Average loss on losing trades (positive dollars)
    total_trades: Total number of trades for statistical significance

##### `def get_current_weights(self) -> dict[str, float]`

Get current effective signal weights.

##### `def get_base_weights(self) -> dict[str, float]`

Get base (unadjusted) signal weights.

##### `def get_status(self) -> dict[str, Any]`

Get agent status for monitoring.

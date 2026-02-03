# stat_arb_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\stat_arb_agent.py`

## Overview

Statistical Arbitrage Agent
===========================

Generates signals based on statistical relationships between instruments.
Implements pairs trading and mean reversion strategies.

Responsibility: Statistical arbitrage signal generation ONLY.
Does NOT make trading decisions or send orders.

## Classes

### PairState

State for a trading pair.

### StatArbAgent

**Inherits from**: SignalAgent

Statistical Arbitrage Agent.

Implements pairs trading strategy based on cointegration.

Methodology:
1. Identify cointegrated pairs
2. Calculate hedge ratio via OLS or Kalman filter
3. Compute spread z-score
4. Generate mean reversion signals

Signal output:
- Long/short pair signals when z-score exceeds threshold
- Exit signals when z-score reverts

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `async def initialize(self) -> None`

Initialize pairs state.

##### `async def process_event(self, event: Event) -> None`

Process market data and generate stat arb signals.

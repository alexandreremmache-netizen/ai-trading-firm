# momentum_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\momentum_agent.py`

## Overview

Momentum / Trend Following Agent
================================

Generates signals based on price momentum and trend indicators.
Implements moving average crossovers, RSI, and breakout detection.

Responsibility: Momentum/trend signal generation ONLY.
Does NOT make trading decisions or send orders.

## Classes

### MomentumState

State for tracking momentum indicators per symbol.

### MomentumAgent

**Inherits from**: SignalAgent

Momentum / Trend Following Agent.

Implements multiple momentum indicators:
1. Moving Average Crossover (fast/slow MA)
2. RSI (Relative Strength Index)
3. MACD (Moving Average Convergence Divergence)

Signal output:
- Trend direction (long/short/flat)
- Signal strength based on indicator confluence

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `async def initialize(self) -> None`

Initialize momentum tracking.

##### `async def process_event(self, event: Event) -> None`

Process market data and generate momentum signals.

# market_making_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\market_making_agent.py`

## Overview

Market Making Agent
===================

Generates signals for market making strategy (IB-compatible latency).
Provides liquidity by quoting bid/ask spreads.

NOTE: This is NOT High-Frequency Trading (HFT).
- Quote refresh rate is IB-compatible (>100ms)
- No nanosecond-level strategies
- Focus on spread capture, not speed

Responsibility: Market making signal generation ONLY.
Does NOT make trading decisions or send orders.

## Classes

### MarketMakingState

State for market making per symbol.

### MarketMakingAgent

**Inherits from**: SignalAgent

Market Making Agent.

Implements passive market making strategy:
1. Calculate fair value from order book
2. Estimate short-term volatility
3. Set bid/ask quotes around fair value
4. Manage inventory risk

NOT HFT - operates at IB-compatible latencies (100ms+).

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `async def initialize(self) -> None`

Initialize market making state.

##### `async def process_event(self, event: Event) -> None`

Process market data and generate market making signals.

##### `def update_inventory(self, symbol: str, quantity_change: int) -> None`

Update inventory after a fill.

Called by execution agent to track positions.

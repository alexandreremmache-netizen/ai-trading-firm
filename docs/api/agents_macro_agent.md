# macro_agent

**Path**: `C:\Users\Alexa\ai-trading-firm\agents\macro_agent.py`

## Overview

Macro Strategy Agent
====================

Generates signals based on macroeconomic indicators.
Monitors yield curves, VIX, DXY, and other macro factors.

Responsibility: Macro signal generation ONLY.
Does NOT make trading decisions or send orders.

## Classes

### MacroAgent

**Inherits from**: SignalAgent

Macro Strategy Agent.

Analyzes macroeconomic indicators to generate regime-based signals.

Indicators monitored:
- Yield curve (2s10s spread)
- VIX (volatility index)
- DXY (dollar index)
- Credit spreads

Signal output:
- Risk-on / Risk-off regime
- Sector rotation signals
- Duration signals

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `async def initialize(self) -> None`

Initialize macro data feeds.

##### `async def process_event(self, event: Event) -> None`

Process market data and generate macro signals.

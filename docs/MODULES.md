# Module Reference

Complete reference of all modules in the AI Trading Firm system.

## Core Modules (`core/`)

### Event System

| Module | Description |
|--------|-------------|
| `event_bus.py` | Async publish/subscribe event bus with topic routing |
| `events.py` | Event type definitions (MarketData, Signal, Decision, Order, Fill, etc.) |
| `event_persistence.py` | Event replay and persistence for audit |

### Broker Integration

| Module | Description |
|--------|-------------|
| `broker.py` | Interactive Brokers interface (ib_insync wrapper) |
| `best_execution.py` | Best execution analysis (RTS 27/28 compliance) |
| `smart_order_router.py` | Intelligent order routing logic |
| `slippage_estimator.py` | Slippage prediction for execution planning |

### Risk Management

| Module | Description |
|--------|-------------|
| `var_calculator.py` | Value at Risk (Parametric, Historical, Monte Carlo) |
| `risk_factors.py` | Risk factor models and calculations |
| `risk_budget.py` | Cross-strategy risk budget allocation |
| `stress_tester.py` | Historical scenario stress testing |
| `correlation_manager.py` | Correlation tracking and regime detection |
| `greeks_analytics.py` | Options Greeks calculation and limits |
| `scenario_analysis.py` | What-if scenario analysis |
| `time_based_risk.py` | Time-of-day and calendar risk adjustments |
| `risk_reports.py` | Risk reporting and visualization |

### Position Management

| Module | Description |
|--------|-------------|
| `position_sizing.py` | Position sizing (Kelly, Vol Target, Fixed Fractional) |
| `position_netting.py` | Position aggregation and netting |
| `portfolio_snapshots.py` | Periodic portfolio state capture |
| `portfolio_construction.py` | Portfolio optimization and construction |
| `margin_optimizer.py` | Margin efficiency optimization |

### Order Management

| Module | Description |
|--------|-------------|
| `order_management.py` | Order amendment, rejection parsing, latency tracking |
| `order_throttling.py` | Rate limiting and anti-HFT controls |
| `circuit_breaker.py` | System circuit breaker for emergencies |

### Compliance & Regulatory

| Module | Description |
|--------|-------------|
| `regulatory_compliance.py` | EU/AMF compliance rules engine |
| `regulatory_calendar.py` | Regulatory deadlines and reminders |
| `compliance_control.py` | Pre-trade compliance checks |

### Signal & Strategy

| Module | Description |
|--------|-------------|
| `signal_decay.py` | Signal strength decay over time |
| `signal_validation.py` | Signal quality and consistency checks |
| `regime_detector.py` | Market regime detection (volatility, trend, correlation) |
| `technical_indicators.py` | Technical analysis indicators |
| `strategy_parameters.py` | Strategy parameter management |

### Infrastructure

| Module | Description |
|--------|-------------|
| `agent_base.py` | Base class for all agents |
| `logger.py` | Audit logging with retention |
| `structured_logging.py` | JSON structured logging |
| `logging_config.py` | Logging configuration |
| `monitoring.py` | System monitoring and metrics |
| `health_check.py` | Health check endpoints |
| `config_validator.py` | Configuration validation |
| `cache_manager.py` | In-memory caching |
| `data_retention.py` | Data retention and cleanup policies |
| `system_infrastructure.py` | System utilities |

### Market Data

| Module | Description |
|--------|-------------|
| `contract_specs.py` | Contract specifications and margin requirements |
| `futures_roll_manager.py` | Futures contract roll management |
| `fx_sessions.py` | Forex session times and liquidity windows |

### Analytics & Reporting

| Module | Description |
|--------|-------------|
| `attribution.py` | Performance attribution analysis |
| `metrics_export.py` | Metrics export (Prometheus format) |
| `dashboard_metrics.py` | Dashboard data generation |
| `custom_reports.py` | Custom report generation |

### Testing & Simulation

| Module | Description |
|--------|-------------|
| `backtest.py` | Comprehensive backtesting framework |
| `notifications.py` | Multi-channel alert notifications |

---

## Agents (`agents/`)

### Signal Agents (Parallel Execution)

| Agent | Description |
|-------|-------------|
| `macro_agent.py` | Macro-economic signal generation (VIX, yield curve, DXY) |
| `stat_arb_agent.py` | Statistical arbitrage and pairs trading |
| `momentum_agent.py` | Momentum and trend-following signals |
| `market_making_agent.py` | Market making and liquidity provision |
| `options_vol_agent.py` | Options volatility strategies |

### Decision Agent

| Agent | Description |
|-------|-------------|
| `cio_agent.py` | Chief Investment Officer - single decision authority |

### Validation Agents (Sequential)

| Agent | Description |
|-------|-------------|
| `risk_agent.py` | Risk validation, position limits, kill-switch |
| `compliance_agent.py` | Regulatory compliance validation |
| `risk_compliance_agent.py` | Combined risk and compliance (legacy) |

### Execution

| Agent | Description |
|-------|-------------|
| `execution_agent.py` | Order execution with TWAP/VWAP algorithms |

### Surveillance (EU/AMF)

| Agent | Description |
|-------|-------------|
| `surveillance_agent.py` | Market abuse detection (MAR compliance) |
| `transaction_reporting_agent.py` | Transaction reporting (MiFIR) |

---

## Strategies (`strategies/`)

| Strategy | Description |
|----------|-------------|
| `macro_strategy.py` | Macro-economic trading strategy |
| `stat_arb_strategy.py` | Statistical arbitrage implementation |
| `momentum_strategy.py` | Momentum trading strategy |
| `market_making_strategy.py` | Market making strategy |
| `options_vol_strategy.py` | Options volatility strategy |
| `seasonality.py` | Calendar seasonality patterns |

---

## Data (`data/`)

| Module | Description |
|--------|-------------|
| `market_data.py` | Market data management and subscription |

---

## Module Dependencies

```
main.py
  ├── core/event_bus.py
  ├── core/broker.py
  ├── core/logger.py
  ├── core/monitoring.py
  │
  ├── agents/macro_agent.py ────┐
  ├── agents/stat_arb_agent.py ─┼──> core/agent_base.py
  ├── agents/momentum_agent.py ─┤
  ├── agents/market_making_agent.py
  ├── agents/options_vol_agent.py
  │
  ├── agents/cio_agent.py ──────> core/position_sizing.py
  │                               core/risk_budget.py
  │
  ├── agents/risk_agent.py ─────> core/var_calculator.py
  │                               core/stress_tester.py
  │                               core/circuit_breaker.py
  │
  ├── agents/compliance_agent.py > core/regulatory_compliance.py
  │                                core/regulatory_calendar.py
  │
  └── agents/execution_agent.py ─> core/order_management.py
                                   core/best_execution.py
                                   core/slippage_estimator.py
```

---

## Adding New Modules

When adding new modules, follow these guidelines:

1. **Naming**: Use snake_case for module names
2. **Location**: Place in appropriate directory (core/, agents/, strategies/)
3. **Imports**: Use relative imports within packages
4. **Logging**: Use `logging.getLogger(__name__)`
5. **Types**: Include type hints for all public functions
6. **Docs**: Include module-level docstring with description

Example:

```python
"""
New Module Name
===============

Brief description of the module's purpose.

Features:
- Feature 1
- Feature 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MyClass:
    """Class description."""
    field: str

    def method(self) -> str:
        """Method description."""
        return self.field
```

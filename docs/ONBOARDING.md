# Onboarding Documentation

**Last Updated**: 2026-02-02
**Audience**: New developers, operators, and traders

---

## Table of Contents

1. [Welcome](#welcome)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Project Structure](#project-structure)
5. [Running the System](#running-the-system)
6. [Key Concepts](#key-concepts)
7. [Development Workflow](#development-workflow)
8. [Testing](#testing)
9. [Common Tasks](#common-tasks)
10. [Resources](#resources)

---

## Welcome

Welcome to the AI Trading Firm codebase! This document will guide you through setting up your development environment and understanding the system architecture.

### What is this system?

This is a **professional, institutional-grade multi-agent trading system** designed to:

- Execute quantitative trading strategies across multiple asset classes
- Manage risk in real-time with VaR and stress testing
- Ensure EU/AMF regulatory compliance (MiFID II, EMIR, MAR)
- Provide full auditability of all trading decisions

### What this system is NOT

- Not a toy or demo project
- Not a single autonomous agent
- Not high-frequency trading (HFT)
- Not self-modifying or auto-scaling

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Core runtime |
| Poetry | 1.4+ | Dependency management |
| Git | 2.30+ | Version control |
| IB Gateway | Latest | Broker connection |

### Required Knowledge

- Python async/await programming
- Basic trading concepts (orders, positions, P&L)
- Understanding of event-driven architectures
- Familiarity with financial regulations (helpful)

### Optional but Recommended

- Docker for containerized deployment
- PostgreSQL for production database
- Grafana/Prometheus for monitoring

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/ai-trading-firm.git
cd ai-trading-firm
```

### Step 2: Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=4002
IB_CLIENT_ID=1
TRADING_MODE=paper

# Compliance (required for EU trading)
FIRM_LEI=YOUR_LEI_HERE
FIRM_MIC=XPAR

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Step 4: Install IB Gateway

1. Download IB Gateway from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Install and configure with your account
3. Enable API connections in Gateway settings
4. For paper trading, use port 4002

### Step 5: Verify Installation

```bash
# Run system checks
python -m pytest tests/test_setup.py -v

# Check broker connection
python scripts/check_ib_connection.py
```

---

## Project Structure

```
ai-trading-firm/
|
+-- agents/                    # Trading agents
|   +-- cio_agent.py          # Chief Investment Officer - decision making
|   +-- risk_agent.py         # Risk management and validation
|   +-- execution_agent.py    # Order execution
|   +-- compliance_agent.py   # Regulatory compliance
|   +-- surveillance_agent.py # Market abuse detection
|
+-- strategies/               # Trading strategies
|   +-- momentum_strategy.py  # Trend following
|   +-- stat_arb_strategy.py  # Statistical arbitrage
|   +-- options_vol_strategy.py # Volatility trading
|
+-- core/                     # Core infrastructure
|   +-- event_bus.py         # Event routing
|   +-- events.py            # Event definitions
|   +-- broker.py            # IB integration
|   +-- var_calculator.py    # Risk calculations
|   +-- technical_indicators.py # TA indicators
|
+-- config/                   # Configuration
|   +-- config.yaml          # Main config
|   +-- strategies.yaml      # Strategy parameters
|
+-- tests/                    # Test suite
|   +-- unit/                # Unit tests
|   +-- integration/         # Integration tests
|   +-- backtest/            # Strategy backtests
|
+-- docs/                     # Documentation
|   +-- ARCHITECTURE.md      # System design
|   +-- ONBOARDING.md        # This file
|   +-- RUNBOOKS.md          # Operational guides
|
+-- scripts/                  # Utility scripts
|   +-- run_backtest.py      # Backtest runner
|   +-- check_ib_connection.py # Connection test
```

---

## Running the System

### Paper Trading Mode (Default)

```bash
# Start the trading system in paper mode
python main.py --mode paper

# Or with configuration override
python main.py --config config/paper_trading.yaml
```

### Backtest Mode

```bash
# Run a strategy backtest
python scripts/run_backtest.py \
    --strategy momentum \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --symbols AAPL,MSFT,GOOGL
```

### Health Check

```bash
# Verify system health
curl http://localhost:8080/health

# Expected response:
{
    "status": "healthy",
    "components": {
        "event_bus": "ok",
        "broker": "connected",
        "risk_agent": "ok"
    }
}
```

---

## Key Concepts

### 1. Multi-Agent Architecture

The system uses multiple specialized agents:

```
Signal Agents (Parallel)
    |
    v
CIO Agent (Decision)
    |
    v
Risk Agent (Validation)
    |
    v
Compliance Agent (Regulatory)
    |
    v
Execution Agent (Orders)
```

### 2. Event-Driven Design

All communication happens through events:

```python
# Emitting an event
await event_bus.publish(SignalEvent(
    symbol="AAPL",
    direction=SignalDirection.BUY,
    strength=0.8
))

# Subscribing to events
@event_bus.subscribe(EventType.SIGNAL)
async def handle_signal(event: SignalEvent):
    # Process the signal
    pass
```

### 3. Risk Management

Risk is validated at multiple levels:

- **Position level**: Max position size per symbol
- **Portfolio level**: Total VaR, leverage limits
- **Strategy level**: Risk budget allocation

### 4. Compliance Framework

All trades must pass compliance checks:

- Pre-trade risk controls (RTS 6)
- Best execution requirements (RTS 27/28)
- Transaction reporting (EMIR)
- Market abuse monitoring (MAR)

---

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes with tests**
   ```bash
   # Write your code
   # Add unit tests
   ```

3. **Run tests locally**
   ```bash
   pytest tests/ -v
   ```

4. **Check code quality**
   ```bash
   # Type checking
   mypy agents/ core/

   # Linting
   ruff check .

   # Formatting
   black .
   ```

5. **Submit for review**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request
   ```

### Code Standards

- Type hints on all public functions
- Docstrings for all classes and public methods
- Unit tests for all new functionality
- No hardcoded values (use configuration)

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires IB)
pytest tests/integration/ -v --ib-connected

# With coverage
pytest tests/ --cov=core --cov=agents
```

### Writing Tests

```python
# tests/unit/test_var_calculator.py

import pytest
from core.var_calculator import VaRCalculator

class TestVaRCalculator:
    def test_historical_var(self):
        calc = VaRCalculator()
        returns = [-0.02, 0.01, -0.03, 0.02, -0.01]

        var_95 = calc.calculate_historical_var(returns, confidence=0.95)

        assert var_95 < 0  # VaR should be negative (loss)
        assert abs(var_95) < 0.05  # Reasonable magnitude
```

---

## Common Tasks

### Adding a New Strategy

1. Create strategy file in `strategies/`
2. Inherit from `StrategyBase`
3. Implement required methods
4. Register in configuration
5. Add unit tests

```python
# strategies/my_strategy.py
from strategies.base import StrategyBase

class MyStrategy(StrategyBase):
    def __init__(self, config):
        super().__init__(config)

    async def generate_signals(self, market_data):
        # Your signal logic here
        pass
```

### Adding a New Risk Check

1. Add check method in `risk_agent.py`
2. Register in validation pipeline
3. Add configuration parameters
4. Add tests

### Modifying Event Types

1. Add to `EventType` enum in `events.py`
2. Create corresponding dataclass
3. Update event bus handlers
4. Update documentation

---

## Resources

### Internal Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design details
- [RUNBOOKS.md](./RUNBOOKS.md) - Operational procedures
- [API_REFERENCE.md](./API_REFERENCE.md) - API documentation
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues

### External Resources

- [Interactive Brokers API](https://interactivebrokers.github.io/tws-api/)
- [MiFID II Overview](https://www.esma.europa.eu/policy-rules/mifid-ii-and-mifir)
- [EMIR Reporting](https://www.esma.europa.eu/policy-rules/post-trading/trade-reporting)

### Getting Help

- **Slack**: #trading-system
- **Email**: trading-support@company.com
- **Wiki**: Internal knowledge base

---

## Checklist

Before you start developing, ensure you have:

- [ ] Python 3.10+ installed
- [ ] Poetry installed and dependencies installed
- [ ] IB Gateway installed and configured
- [ ] `.env` file created with required variables
- [ ] Tests passing locally
- [ ] Access to Slack channel
- [ ] Read ARCHITECTURE.md

---

*Welcome to the team! If you have questions, don't hesitate to ask.*

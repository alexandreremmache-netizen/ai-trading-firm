# AI Trading Firm - Documentation

## Overview

The AI Trading Firm is an institutional-grade, multi-agent trading system designed for EU/AMF regulatory compliance. It emulates a professional hedge fund structure with strict separation of concerns between signal generation, decision-making, risk validation, compliance checking, and order execution.

## System Description

This system implements a multi-strategy trading platform with the following key characteristics:

- **Multi-Agent Architecture**: Specialized agents handle distinct responsibilities (signals, decisions, risk, compliance, execution)
- **Single Decision Authority**: The CIO (Chief Investment Officer) Agent is the sole decision-making entity
- **Event-Driven Execution**: All operations are triggered by events, with no infinite loops or continuous polling
- **EU/AMF Compliance**: Full regulatory compliance with MiFID II, EMIR, and MAR requirements
- **Interactive Brokers Integration**: Exclusive broker for market data and execution (paper trading default)

## Key Features

### Trading Strategies
- **Momentum/Trend Following**: Moving averages, RSI, MACD-based signals
- **Statistical Arbitrage**: Pairs trading, cointegration, commodity spreads
- **Options Volatility**: IV percentile trading, delta-targeted strategies
- **Market Making**: Spread capture with inventory management
- **Macro**: Yield curve, VIX, and dollar index indicators

### Risk Management
- Multi-methodology VaR (Parametric, Historical, Monte Carlo)
- Real-time position and leverage limits
- Greeks monitoring for options portfolios
- Stress testing with predefined scenarios
- Kill-switch for emergency situations

### Compliance Features
- MiFID II transaction reporting
- MAR market abuse surveillance
- RTS 25 order record keeping
- Best execution analysis (RTS 27/28)
- Full audit trail with 7-year retention

## Quick Start

### Prerequisites
- Python 3.11+
- Interactive Brokers TWS or IB Gateway
- Required packages (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-trading-firm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy and customize the configuration:
```bash
cp config.yaml config.local.yaml
```

2. Configure Interactive Brokers connection:
```yaml
broker:
  host: "127.0.0.1"
  port: 4002        # IB Gateway Paper Trading
  client_id: 1
  use_delayed_data: true
```

3. Set paper trading mode (default):
```yaml
firm:
  mode: "paper"     # NEVER change to "live" without authorization
```

### Running the System

```bash
# Start with default configuration
python main.py

# The system will:
# 1. Initialize all agents
# 2. Connect to Interactive Brokers
# 3. Start processing market data
# 4. Generate signals and make decisions
```

## Architecture Overview

```
                    MARKET DATA (Interactive Brokers)
                              |
                              v
    +--------------------------------------------------+
    |         SIGNAL AGENTS (Parallel Fan-Out)         |
    |  [Macro] [StatArb] [Momentum] [MM] [OptionsVol]  |
    +--------------------------------------------------+
                              |
                    SignalEvent (Barrier Sync)
                              v
    +--------------------------------------------------+
    |              CIO AGENT (Single Authority)         |
    |         THE decision-making authority             |
    +--------------------------------------------------+
                              |
                        DecisionEvent
                              v
    +--------------------------------------------------+
    |                    RISK AGENT                     |
    |       Kill-switch, VaR, Position Limits          |
    +--------------------------------------------------+
                              |
                    ValidatedDecisionEvent
                              v
    +--------------------------------------------------+
    |             COMPLIANCE AGENT (EU/AMF)             |
    |         Blackout, MNPI, Restricted List          |
    +--------------------------------------------------+
                              |
                    ValidatedDecisionEvent
                              v
    +--------------------------------------------------+
    |                EXECUTION AGENT                    |
    |     TWAP/VWAP Algorithms - ONLY order sender     |
    +--------------------------------------------------+
                              |
                          OrderEvent
                              v
    +--------------------------------------------------+
    |               INTERACTIVE BROKERS                 |
    +--------------------------------------------------+
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detailed system architecture and design |
| [AGENTS.md](AGENTS.md) | Agent documentation and responsibilities |
| [STRATEGIES.md](STRATEGIES.md) | Trading strategy implementations |
| [RISK_MANAGEMENT.md](RISK_MANAGEMENT.md) | Risk management and VaR calculations |
| [COMPLIANCE.md](COMPLIANCE.md) | EU/AMF regulatory compliance |
| [API_REFERENCE.md](API_REFERENCE.md) | Core API and event types |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment and configuration guide |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |

## Important Constraints

### What This System Does NOT Do
- **No HFT**: Minimum 100ms between orders, max 10 orders/minute
- **No Market Manipulation**: Surveillance for wash trading, spoofing, layering
- **No Autonomous Strategy Evolution**: All strategy changes require human validation
- **No Black Box Decisions**: Full rationale and data sources logged for every decision

### Safety Features
- Paper trading is the default and recommended mode
- Kill-switch activates on daily loss > 3% or drawdown > 10%
- All decisions require compliance approval before execution
- 7-year audit log retention per MiFID II

## License

See LICENSE file for details.

## Disclaimer

This system is designed for paper trading and educational purposes. Use in live trading requires proper authorization, licensing, and regulatory approval. The authors are not responsible for any financial losses incurred through the use of this software.

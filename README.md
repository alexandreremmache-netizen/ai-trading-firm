# AI Trading Firm

A professional, institutional-grade multi-agent trading system designed for EU/AMF regulatory compliance.

## Overview

The AI Trading Firm is a comprehensive algorithmic trading platform that emulates a hedge fund structure with strict separation of concerns between signal generation, decision-making, risk validation, compliance checking, and order execution. The system is built on an event-driven architecture with a single decision authority (CIO Agent) and full audit trail capabilities.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for signals, decisions, risk, compliance, and execution
- **Single Decision Authority**: CIO Agent as the sole decision-making entity
- **Event-Driven Execution**: No polling, no infinite loops - pure event-driven design
- **EU/AMF Compliance**: MiFID II, EMIR, MAR regulatory compliance built-in
- **Interactive Brokers Integration**: Exclusive broker for market data and execution
- **Paper Trading Default**: Safe by design - paper trading is the default mode

## Trading Strategies

| Strategy | Description |
|----------|-------------|
| Momentum | Trend-following with MA crossovers, RSI, MACD |
| Statistical Arbitrage | Pairs trading, cointegration, commodity spreads |
| Options Volatility | IV percentile trading, delta-targeted strategies |
| Market Making | Spread capture with inventory management |
| Macro | Yield curve, VIX, and dollar index indicators |

## Risk Management

- Multi-methodology VaR (Parametric, Historical, Monte Carlo)
- Real-time position and leverage limits
- Greeks monitoring for options portfolios
- Stress testing with predefined scenarios
- Kill-switch for emergency situations (MiFID II RTS 6 compliant)

## Quick Start

### Prerequisites

- Python 3.11+
- Interactive Brokers TWS or IB Gateway
- Paper trading account

### Installation

```bash
# Clone repository
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

```bash
# Edit configuration
# Ensure mode is "paper" and broker settings are correct
vi config.yaml
```

### Running

```bash
# Start IB Gateway/TWS first, then:
python main.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation overview |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/AGENTS.md](docs/AGENTS.md) | Agent documentation |
| [docs/STRATEGIES.md](docs/STRATEGIES.md) | Trading strategies |
| [docs/RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md) | Risk management |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | EU/AMF compliance |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | API reference |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Troubleshooting |

## Architecture

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
    +--------------------------------------------------+
                              |
                        DecisionEvent
                              v
    +--------------------------------------------------+
    |      RISK AGENT --> COMPLIANCE AGENT             |
    +--------------------------------------------------+
                              |
                    ValidatedDecisionEvent
                              v
    +--------------------------------------------------+
    |                EXECUTION AGENT                    |
    +--------------------------------------------------+
                              |
                          OrderEvent
                              v
    +--------------------------------------------------+
    |               INTERACTIVE BROKERS                 |
    +--------------------------------------------------+
```

## Safety Features

- Paper trading mode is the default
- Kill-switch activates on daily loss > 3% or drawdown > 10%
- All decisions require compliance approval
- 7-year audit log retention per MiFID II
- Anti-HFT rate limits (max 10 orders/minute)

## Regulatory Compliance

- **MiFID II**: Transaction reporting (RTS 22/23), order records (RTS 25)
- **MAR**: Market abuse surveillance, wash trading detection
- **RTS 6**: Algorithmic trading controls, kill switch
- **RTS 27/28**: Best execution reporting

## Project Structure

```
ai-trading-firm/
|-- agents/          # Agent implementations
|-- core/            # Core infrastructure
|-- data/            # Market data handling
|-- strategies/      # Strategy implementations
|-- tests/           # Test suite
|-- docs/            # Documentation
|-- config.yaml      # Configuration
|-- main.py          # Entry point
```

## Configuration

Key configuration sections in `config.yaml`:

```yaml
firm:
  mode: "paper"        # NEVER change to "live" without authorization

broker:
  port: 4002           # IB Gateway Paper Trading

risk:
  max_portfolio_var_pct: 2.0
  max_position_size_pct: 5.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
```

## Contributing

This system is designed with specific architectural constraints defined in `CLAUDE.md`. Please review the constitution before making changes.

## License

See LICENSE file for details.

## Disclaimer

This system is designed for paper trading and educational purposes. Use in live trading requires proper authorization, licensing, and regulatory approval. The authors are not responsible for any financial losses incurred through the use of this software.

---

**Important**: Always run in paper trading mode unless explicitly authorized for live trading. The system is configured for paper trading by default - this is intentional and should not be changed without proper risk assessment and regulatory compliance verification.

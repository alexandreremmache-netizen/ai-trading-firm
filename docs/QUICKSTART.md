# Quick Start Guide

Get the AI Trading Firm system running in paper trading mode in under 15 minutes.

## Prerequisites

- Python 3.11 or higher
- Interactive Brokers account (paper trading account is sufficient)
- IB Gateway or Trader Workstation (TWS) installed

## Step 1: Install Dependencies

```bash
# Clone and enter the project
cd ai-trading-firm

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Step 2: Configure IB Gateway

1. Start IB Gateway (or TWS)
2. Login with your paper trading credentials
3. Ensure API connections are enabled:
   - File -> Global Configuration -> API -> Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set Socket port to 4002 (Gateway) or 7497 (TWS)
   - Add 127.0.0.1 to trusted IPs

## Step 3: Configure the System

Edit `config.yaml`:

```yaml
# Minimum required configuration
firm:
  mode: "paper"  # ALWAYS start with paper trading

broker:
  host: "127.0.0.1"
  port: 4002      # 4002 for IB Gateway, 7497 for TWS
  client_id: 1
```

## Step 4: Verify Connection

Test your IB connection:

```bash
python test_ib_connection.py
```

Expected output:
```
Connected to IB Gateway
Account: DU1234567 (Paper Trading)
Net Liquidation: $1,000,000.00
Connection successful!
```

## Step 5: Run the System

```bash
python main.py
```

The system will:
1. Connect to Interactive Brokers
2. Initialize all agents
3. Subscribe to market data
4. Begin the event-driven trading loop

## Understanding the Output

```
2024-01-15 14:30:00 | main | INFO | AI Trading Firm starting...
2024-01-15 14:30:01 | broker | INFO | Connected to IB (Paper Trading)
2024-01-15 14:30:02 | event_bus | INFO | Event bus initialized
2024-01-15 14:30:03 | orchestrator | INFO | Signal agents started (5)
2024-01-15 14:30:03 | orchestrator | INFO | CIO Agent ready
2024-01-15 14:30:03 | orchestrator | INFO | Risk Agent ready
2024-01-15 14:30:03 | orchestrator | INFO | Compliance Agent ready
2024-01-15 14:30:03 | orchestrator | INFO | Execution Agent ready
2024-01-15 14:30:04 | orchestrator | INFO | Market data subscription active
2024-01-15 14:30:04 | orchestrator | INFO | System ready - Event loop started
```

## Monitoring

### Log Files
- `logs/audit.jsonl` - All events and decisions
- `logs/decisions.jsonl` - CIO decisions with rationale
- `logs/trades.jsonl` - Executed trades

### Health Check
```bash
curl http://localhost:8080/health
```

## Common First Steps

### 1. View Active Signals
Check `logs/audit.jsonl` for SignalEvent entries from each agent.

### 2. Review Decisions
Check `logs/decisions.jsonl` to see CIO decision rationale.

### 3. Check Risk Status
The Risk Agent logs position limits and VaR calculations.

### 4. Verify No Live Trading
Confirm in config.yaml:
```yaml
firm:
  mode: "paper"
```

## Stopping the System

Press `Ctrl+C` for graceful shutdown. The system will:
1. Stop accepting new market data
2. Cancel pending orders
3. Log final portfolio state
4. Disconnect from IB

## Next Steps

1. **Customize Universe**: Edit `config.yaml` -> `universe` section
2. **Adjust Risk Parameters**: Edit `config.yaml` -> `risk` section
3. **Run Backtest**: See [backtest-paper.html](backtest-paper.html)
4. **Review Architecture**: See [architecture.html](architecture.html)

## Troubleshooting

### Cannot Connect to IB
- Verify IB Gateway/TWS is running
- Check port number matches config
- Ensure API is enabled in IB settings

### No Market Data
- Check market hours
- Verify symbols in universe are correct
- Enable delayed data if no live data subscription

### System Exits Immediately
- Check logs/audit.jsonl for errors
- Verify Python 3.11+ is installed
- Check all dependencies installed

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more solutions.

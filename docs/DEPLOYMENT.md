# Deployment Guide

## Overview

This guide covers the deployment of the AI Trading Firm system for paper trading with Interactive Brokers. The system is designed to run in paper trading mode by default.

---

## Prerequisites

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Runtime |
| Interactive Brokers TWS or IB Gateway | Latest | Broker connection |
| Git | Any | Version control |

### Python Dependencies

Core dependencies (from requirements.txt):
```
ib-insync>=0.9.86      # Interactive Brokers API
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Scientific computing
pandas>=2.0.0          # Data analysis
pyyaml>=6.0            # Configuration
aiohttp>=3.8.0         # Async HTTP
nest-asyncio>=1.5.0    # Async compatibility
```

### Interactive Brokers Setup

1. **Create IB Account**
   - Sign up at [Interactive Brokers](https://www.interactivebrokers.com)
   - Paper trading account is sufficient for testing

2. **Download and Install**
   - Download TWS or IB Gateway
   - TWS: Full trading platform with GUI
   - IB Gateway: Lightweight headless connection

3. **Enable API Access**
   - In TWS/Gateway: Edit > Global Configuration > API > Settings
   - Check "Enable ActiveX and Socket Clients"
   - Set Socket Port (default: 7497 for TWS Paper, 4002 for Gateway Paper)
   - Add localhost to "Trusted IPs"

---

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ai-trading-firm
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "from ib_insync import IB; print('IB-insync installed')"
python -c "import numpy; print('NumPy installed')"
python -c "import scipy; print('SciPy installed')"
```

---

## Configuration

### Step 1: Copy Configuration Template

```bash
cp config.yaml config.local.yaml
```

### Step 2: Configure Broker Connection

Edit `config.yaml`:

```yaml
broker:
  host: "127.0.0.1"
  port: 4002              # IB Gateway Paper: 4002, TWS Paper: 7497
  client_id: 1            # Unique per connection
  timeout_seconds: 30
  readonly: false
  use_delayed_data: true  # Free delayed data (no subscription needed)
```

### Step 3: Verify Paper Trading Mode

Ensure paper trading is enabled:

```yaml
firm:
  name: "AI Trading Firm"
  version: "1.0.0"
  mode: "paper"           # NEVER change without authorization
```

### Step 4: Configure Risk Limits

Set appropriate risk limits for testing:

```yaml
risk:
  max_portfolio_var_pct: 2.0
  max_position_size_pct: 5.0
  max_sector_exposure_pct: 20.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
  max_leverage: 2.0
```

### Step 5: Configure Trading Universe

Define instruments to trade:

```yaml
universe:
  equities:
    - symbol: "AAPL"
      exchange: "SMART"
      currency: "USD"
    - symbol: "MSFT"
      exchange: "SMART"
      currency: "USD"

  etfs:
    - symbol: "SPY"
      exchange: "SMART"
      currency: "USD"

  # Futures require specific contract months
  futures:
    - symbol: "ES"
      exchange: "CME"
      currency: "USD"
```

---

## Paper Trading Setup

### Starting IB Gateway

1. Launch IB Gateway
2. Log in with paper trading credentials
3. Verify API is enabled (Configuration > API > Settings)
4. Note the port number (default 4002)

### Starting TWS (Alternative)

1. Launch Trader Workstation
2. Log in with paper trading credentials
3. Enable API (Edit > Global Configuration > API)
4. Note the port number (default 7497)

### Verifying Connection

Test the connection before starting the system:

```bash
python test_ib_connection.py
```

Expected output:
```
Connecting to IB at 127.0.0.1:4002...
Connected successfully!
Account: DU1234567
Net Liquidation: $1,000,000.00
```

---

## Running the System

### Start Command

```bash
python main.py
```

### Expected Startup Output

```
============================================================
     AI TRADING FIRM - Multi-Agent Trading System
============================================================

2024-01-15 10:30:00 | INFO | Loaded configuration from config.yaml
2024-01-15 10:30:00 | INFO | Running in PAPER mode
2024-01-15 10:30:01 | INFO | Connecting to IB at 127.0.0.1:4002...
2024-01-15 10:30:02 | INFO | Connected to IB - Account: DU1234567
2024-01-15 10:30:02 | INFO | Using delayed market data (free)
2024-01-15 10:30:03 | INFO | Initialized 5 signal agents
2024-01-15 10:30:03 | INFO | All agents initialized
============================================================
TRADING SYSTEM STARTED
  Broker: CONNECTED
  Mode: PAPER
  Waiting for market data events...
============================================================
```

### Graceful Shutdown

Press `Ctrl+C` to initiate graceful shutdown:

```
2024-01-15 11:30:00 | INFO | Keyboard interrupt received
============================================================
AI TRADING FIRM - STOPPING
============================================================
2024-01-15 11:30:01 | INFO | Stopping market data...
2024-01-15 11:30:02 | INFO | Stopping agents...
2024-01-15 11:30:03 | INFO | Disconnecting broker...
2024-01-15 11:30:03 | INFO | Trading system stopped
```

---

## Directory Structure

After deployment, the directory structure should be:

```
ai-trading-firm/
+-- agents/                 # Agent implementations
+-- core/                   # Core infrastructure
+-- data/                   # Market data handling
+-- strategies/             # Strategy implementations
+-- tests/                  # Test suite
+-- docs/                   # Documentation
+-- logs/                   # Log files (created at runtime)
�   +-- audit.jsonl        # Audit log
�   +-- trades.jsonl       # Trade log
�   +-- decisions.jsonl    # Decision log
�   +-- system.log         # System log
�   +-- agents/            # Per-agent logs
+-- config.yaml            # Configuration
+-- main.py                # Entry point
+-- requirements.txt       # Dependencies
+-- CLAUDE.md              # System constitution
```

---

## Environment Variables

Optional environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `IB_HOST` | IB Gateway/TWS host | 127.0.0.1 |
| `IB_PORT` | IB Gateway/TWS port | 4002 |
| `IB_CLIENT_ID` | Client ID | 1 |
| `LOG_LEVEL` | Logging level | INFO |

---

## Monitoring

### Log Files

- `logs/system.log`: General system logs
- `logs/audit.jsonl`: Compliance audit trail
- `logs/trades.jsonl`: Trade execution log
- `logs/decisions.jsonl`: Decision log

### Health Check Endpoint

If health check server is enabled:

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "broker": "connected",
  "agents": {"CIOAgent": "running", "RiskAgent": "running"},
  "queue_size": 0
}
```

### System Status

The orchestrator provides status via `get_status()`:

```python
status = orchestrator.get_status()
print(f"Mode: {status['mode']}")
print(f"Broker: {'Connected' if status['broker']['connected'] else 'Disconnected'}")
print(f"Queue Size: {status['event_bus_queue_size']}")
```

---

## Troubleshooting Deployment

### Connection Refused

**Symptom**: Cannot connect to IB Gateway/TWS

**Solutions**:
1. Verify IB Gateway/TWS is running
2. Check port number matches configuration
3. Verify API is enabled in IB settings
4. Check localhost is in trusted IPs

### No Market Data

**Symptom**: Connected but no market data received

**Solutions**:
1. Enable delayed data if no market data subscription
2. Verify instruments are valid (check symbol, exchange)
3. Check market hours (some instruments trade limited hours)

### Permission Denied

**Symptom**: Orders rejected with permission error

**Solutions**:
1. Verify paper trading account has permissions
2. Check instrument is tradeable in paper account
3. Verify order type is supported

---

## Production Considerations

### Security

- Store credentials securely (not in config files)
- Use environment variables for sensitive data
- Enable TLS for API connections
- Restrict network access

### High Availability

- Monitor system health continuously
- Implement alerting for failures
- Plan for broker connection failures
- Regular backup of audit logs

### Compliance

- Verify LEI is valid before production
- Test transaction reporting
- Validate audit log format
- Review surveillance alerts

### Performance

- Monitor event queue depth
- Track processing latency
- Size hardware appropriately
- Plan for market data volume

---

## Checklist

### Pre-Deployment

- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] IB Gateway/TWS installed
- [ ] Paper trading account created
- [ ] API enabled in IB settings
- [ ] Configuration file customized

### Deployment

- [ ] Connection test passed
- [ ] System starts without errors
- [ ] Agents initialize correctly
- [ ] Market data received
- [ ] Signals generated
- [ ] Decisions logged

### Post-Deployment

- [ ] Monitor log files
- [ ] Verify audit trail
- [ ] Check risk limits respected
- [ ] Review system performance
- [ ] Plan regular reviews

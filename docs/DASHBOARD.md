# Dashboard Documentation

## Overview

The AI Trading Firm includes a real-time monitoring dashboard built with FastAPI and WebSocket for live updates. The dashboard provides comprehensive visibility into the trading system's operation, including positions, signals, decisions, risk metrics, and agent health.

## Access

| Endpoint | URL | Description |
|----------|-----|-------------|
| Dashboard | `http://localhost:8081` | Main dashboard UI |
| Health Check | `http://localhost:8080/health` | System health endpoint |
| REST API | `http://localhost:8081/api/*` | REST API endpoints |
| WebSocket | `ws://localhost:8081/ws` | Real-time streaming |

---

## Architecture

```
+------------------------------------------------------------------+
|                     DASHBOARD SERVER (Port 8081)                  |
|                        FastAPI + WebSocket                        |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+                     |
|  | REST API         |    | WebSocket        |                     |
|  | /api/agents      |    | /ws              |                     |
|  | /api/positions   |    | Real-time events |                     |
|  | /api/signals     |    | Update: 500ms    |                     |
|  | /api/decisions   |    |                  |                     |
|  | /api/metrics     |    |                  |                     |
|  | /api/risk        |    |                  |                     |
|  | /api/analytics/* |    |                  |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Dashboard State  |    | Connection       |                     |
|  | (Event Tracking) |    | Manager          |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
          |                        |
          v                        v
+------------------+    +------------------+
|   Event Bus      |    |   Orchestrator   |
|   (Subscribe)    |    |   (Live Data)    |
+------------------+    +------------------+
          |                        |
          v                        v
+------------------+    +------------------+
|   All Agents     |    |   IB Broker      |
|   (Events)       |    |   (Positions)    |
+------------------+    +------------------+
```

---

## Dashboard Panels

### 1. Navigation Bar

Located at the top of the dashboard:

- **System Status**: Shows if system is running normally
- **Kill Switch Button**: Emergency halt trading (red button)
- **Alerts Badge**: Count of recent alerts
- **Time**: Current system time

### 2. Performance Panel

Displays key portfolio metrics in real-time:

| Metric | Description |
|--------|-------------|
| Total P&L | Realized + Unrealized P&L |
| Today's P&L | Today's running P&L |
| Unrealized P&L | P&L from open positions |
| Realized P&L | P&L from all closed positions |
| Sharpe Ratio | Risk-adjusted return metric |
| Win Rate | Percentage of winning trades |
| Drawdown | Current drawdown from peak |
| Position Count | Number of open positions |
| Total Trades | Cumulative trade count |

**Color Coding:**
- Green: Positive values
- Red: Negative values
- Gray: Neutral/zero values

### 3. Equity Curve Chart

Real-time chart showing portfolio equity over time:

- Updates every 2 seconds via WebSocket
- Displays time-filtered data (local timezone)
- Persistent storage in `logs/equity_history.json`
- Supports filtering by time range

### 4. Positions Panel (Open Positions)

Shows all currently open positions from IB broker:

| Column | Description |
|--------|-------------|
| Symbol | Instrument symbol |
| Qty | Position size (positive=long, negative=short) |
| Side | LONG (green badge) or SHORT (red badge) |
| Entry | Average entry price |
| Current | Current market price |
| P&L ($) | Unrealized profit/loss in dollars |
| P&L (%) | Unrealized profit/loss percentage |
| Entry Time | When position was opened |
| Conviction | CIO conviction score at entry (0-100%) |
| Signal | Source agent and direction |
| TP | Take profit target price |
| SL | Stop loss price |

**Features:**
- Real-time price updates via WebSocket
- Color-coded P&L (green=profit, red=loss)
- Position metadata from audit logs

### 5. Closed Positions Panel

Shows recently closed positions with realized P&L:

| Column | Description |
|--------|-------------|
| Symbol | Instrument symbol |
| Side | LONG or SHORT |
| Signal | Source agent and direction |
| Qty | Position size |
| Entry | Entry price |
| Exit | Exit price |
| P&L ($) | Realized profit/loss |
| P&L (%) | Realized profit/loss percentage |
| Closed At | When position was closed |

**Data Sources:**
- Real-time: Detected from broker position changes
- Historical: Loaded from `logs/trades.jsonl`

### 6. Signals Panel

Displays current signals from all strategy agents:

| Column | Description |
|--------|-------------|
| Agent | Source agent name |
| Symbol | Instrument symbol |
| Direction | LONG, SHORT, or FLAT |
| Confidence | Signal confidence (0-100%) |
| Strength | Signal strength (-1 to +1) |
| Time | Signal timestamp |
| Rationale | Human-readable explanation |

**Direction Badges:**
- LONG: Green badge
- SHORT: Red badge
- FLAT: Gray badge

### 7. Decisions Panel (CIO Decisions)

Shows CIO decisions with validation status:

| Column | Description |
|--------|-------------|
| Decision ID | Unique identifier |
| Symbol | Instrument symbol |
| Action | BUY, SELL, or HOLD |
| Quantity | Proposed position size |
| Conviction | CIO conviction score |
| Status | PENDING, APPROVED, or REJECTED |
| Rationale | Decision explanation |
| Est. Value | Estimated trade value |

**Status Badges:**
- APPROVED: Green badge (passed risk/compliance)
- REJECTED: Red badge (failed validation)
- PENDING: Yellow badge (awaiting validation)

### 8. Agent Status Panel

Displays health and status of all 22 agents:

| Column | Description |
|--------|-------------|
| Name | Agent name |
| Type | Signal, Decision, Validation, Execution |
| Status | active, idle, error, stopped |
| Events | Total events processed |
| Latency | Rolling average latency (ms) |
| Health | Health score (0-100%) |
| Enabled | Toggle switch for LLM agents |

**Agent Types:**
- Decision: CIOAgent (blue badge)
- Signal: Strategy agents (green badge)
- Validation: Risk, Compliance (yellow badge)
- Execution: ExecutionAgent (purple badge)

**LLM Agent Indicators:**
- Agents using LLM APIs show a cost indicator
- Can be enabled/disabled at runtime

### 9. Risk Monitor Panel

Displays risk limit utilization:

| Limit | Description |
|-------|-------------|
| Position Size | Largest position as % of portfolio |
| Leverage | Current portfolio leverage |
| Daily Loss | Today's loss as % of portfolio |
| Drawdown | Current drawdown from peak |
| VaR 95% | Value at Risk at 95% confidence |

**Status Indicators:**
- OK (green): Usage < 75%
- Warning (yellow): Usage 75-100%
- Breach (red): Usage > 100%

**Progress Bars:**
- Visual representation of limit usage
- Color changes based on status

### 10. Alerts Panel

Recent system alerts and notifications:

| Field | Description |
|-------|-------------|
| Title | Alert type/category |
| Message | Alert description |
| Severity | info, warning, critical |
| Time | When alert occurred |

**Severity Badges:**
- Info (blue): Informational messages
- Warning (yellow): Requires attention
- Critical (red): Immediate action needed

### 11. Event Stream

Live feed of all system events:

| Field | Description |
|-------|-------------|
| Type | Event type (SIGNAL, DECISION, FILL, etc.) |
| Source | Source agent name |
| Time | Event timestamp |
| Summary | Human-readable summary |

**Event Types:**
- SIGNAL: Strategy signal generated
- DECISION: CIO decision made
- VALIDATED_DECISION: Risk/Compliance result
- FILL: Order filled
- RISK_ALERT: Risk limit warning
- MARKET_DATA: Price update

---

## Real-Time Updates

The dashboard uses WebSocket for real-time updates:

| Data Type | Update Interval |
|-----------|-----------------|
| Positions | 500ms |
| Metrics | 500ms |
| Agents | 500ms |
| Signals | 500ms |
| Risk | 500ms |
| Decisions | 1s |
| Closed Positions | 2s |
| Equity Curve | 2s |
| Events | On occurrence |
| Alerts | On occurrence |

### Initial Connection

When a client connects, they receive an `initial` message with full state:

```json
{
  "type": "initial",
  "payload": {
    "metrics": { ... },
    "agents": [ ... ],
    "positions": [ ... ],
    "closed_positions": [ ... ],
    "signals": [ ... ],
    "decisions": [ ... ],
    "alerts": [ ... ],
    "risk": { "limits": [ ... ] },
    "equity": { ... }
  }
}
```

---

## Configuration

Dashboard configuration in `config.yaml`:

```yaml
dashboard:
  enabled: true                       # Enable/disable dashboard server
  host: "0.0.0.0"                     # Host to bind (0.0.0.0 for all interfaces)
  port: 8081                          # Dashboard port (different from health check)
```

### Security Considerations

For production deployments:

```yaml
dashboard:
  host: "127.0.0.1"  # Bind to localhost only
  port: 8081
```

Consider adding:
- Reverse proxy (nginx) with HTTPS
- Authentication middleware
- Rate limiting
- CORS configuration

---

## Components

The dashboard is organized into modular components:

| Component | File | Purpose |
|-----------|------|---------|
| Server | `dashboard/server.py` | Main FastAPI server and state management |
| Agent Status | `dashboard/components/agent_status.py` | Comprehensive agent health tracking |
| Signal Aggregation | `dashboard/components/signal_aggregation.py` | Signal consensus calculation |
| Performance Metrics | `dashboard/components/performance_metrics.py` | Portfolio metrics calculation |
| Position View | `dashboard/components/position_view.py` | Position display helpers |
| Risk Monitor | `dashboard/components/risk_monitor.py` | Risk visualization |
| Alerts | `dashboard/components/alerts.py` | Alert management |
| LLM Analytics | `dashboard/components/llm_analytics.py` | LLM agent metrics |
| Advanced Analytics | `dashboard/components/advanced_analytics.py` | Rolling metrics, session tracking |

### Advanced Analytics Components (Phase 8)

| Component | Purpose |
|-----------|---------|
| `RollingMetricsCalculator` | Rolling Sharpe/Sortino across time periods |
| `SessionPerformanceTracker` | Performance by trading session |
| `StrategyComparisonTracker` | Strategy comparison and ranking |
| `RiskHeatmapGenerator` | Position risk visualization |
| `TradeJournal` | Trade journal with quality stats |
| `SignalConsensusTracker` | Signal agreement tracking |

---

## Integration with Trading System

The dashboard integrates with the trading system via:

### 1. EventBus Subscription

Subscribes to all event types for real-time updates:

```python
event_types = [
    EventType.MARKET_DATA,
    EventType.SIGNAL,
    EventType.DECISION,
    EventType.VALIDATED_DECISION,
    EventType.ORDER,
    EventType.FILL,
    EventType.RISK_ALERT,
    EventType.SYSTEM,
    EventType.KILL_SWITCH,
]
```

### 2. Orchestrator Connection

Pulls live data from trading system components:

```python
from dashboard.server import create_dashboard_server

server = create_dashboard_server(
    event_bus=event_bus,
    orchestrator=orchestrator,
    port=8081,
)

# Connect to trading system
server.set_orchestrator(orchestrator)
```

### 3. Broker Integration

Direct access to broker for positions and P&L:

- Real-time positions from IB
- Unrealized P&L calculation
- Closed position detection

---

## Running Standalone

The dashboard can run standalone for testing:

```bash
# Run dashboard server only
python -m dashboard.server

# Access at http://localhost:8081
```

### With Mock Data

For development without a live trading system:

```python
from dashboard.server import DashboardServer

server = DashboardServer(
    event_bus=None,  # No event bus
    host="127.0.0.1",
    port=8081,
)

# Add mock data
server.state._positions["AAPL"] = PositionInfo(
    symbol="AAPL",
    quantity=100,
    entry_price=175.00,
    current_price=178.50,
)

# Run with uvicorn
import uvicorn
uvicorn.run(server.app, host="127.0.0.1", port=8081)
```

---

## Kill Switch

The dashboard includes an emergency kill switch:

### Activation Methods

1. **Dashboard Button**: Click the red "Kill Switch" button in the navbar
2. **WebSocket Command**: Send `{"action": "kill_switch", "active": true}`
3. **REST API**: POST to `/api/kill-switch?active=true`

### Kill Switch Behavior

When activated:
- Prevents all new order submissions
- Broadcasts critical alert to all clients
- Updates kill switch status indicator

**Note:** The kill switch does not automatically close positions. Use the Risk Agent's tiered drawdown response for automatic position management.

---

## Agent Control

### Enable/Disable Agents at Runtime

Toggle agents via the dashboard:

1. **Dashboard UI**: Click the toggle switch next to agent name
2. **REST API**: POST to `/api/agent/toggle?agent_name=SentimentAgent&enabled=false`
3. **WebSocket**: Send `{"action": "toggle_agent", "agent_name": "...", "enabled": false}`

### LLM Agent Cost Warning

LLM agents (Sentiment, ChartAnalysis, Forecasting) are marked with a cost indicator. Disabling these agents stops API token consumption.

---

## Troubleshooting

### Dashboard not loading

1. Verify port 8081 is not in use:
   ```bash
   netstat -an | findstr 8081  # Windows
   lsof -i :8081               # Linux/Mac
   ```

2. Check dashboard is enabled:
   ```yaml
   dashboard:
     enabled: true
   ```

3. Verify uvicorn is installed:
   ```bash
   pip install uvicorn
   ```

### WebSocket not connecting

1. Check browser console for errors
2. Verify CORS is enabled (default: all origins allowed)
3. Check for firewall blocking WebSocket connections
4. Try connecting with wscat:
   ```bash
   wscat -c ws://localhost:8081/ws
   ```

### No data displayed

1. Verify IB Gateway/TWS is connected and running
2. Check that orchestrator is passed to dashboard server
3. Verify event bus subscription in logs:
   ```
   INFO | Subscribed to 9 event types
   ```

### Positions not showing

1. Check broker connection in logs
2. Verify positions exist in IB
3. Check for errors in `get_positions_async`:
   ```
   WARNING | Error getting positions from broker: ...
   ```

### Latency showing 0ms

1. Ensure events are being processed
2. Check agent `last_event_time` is being updated
3. Verify timezone handling is correct

### Equity curve not persisting

1. Check `logs/` directory exists
2. Verify write permissions
3. Look for errors in `_save_equity_history`

---

## API Reference

See [API.md](./API.md) for complete REST API and WebSocket documentation.

---

## Browser Compatibility

The dashboard is tested with:

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | Supported |
| Firefox | 85+ | Supported |
| Safari | 14+ | Supported |
| Edge | 90+ | Supported |

**Requirements:**
- JavaScript enabled
- WebSocket support
- Modern CSS (flexbox, grid)

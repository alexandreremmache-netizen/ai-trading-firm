# API Reference

Complete reference for the AI Trading Firm Dashboard REST API and WebSocket interface.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
   - [Health & Status](#health--status)
   - [Trading Data](#trading-data)
   - [Analytics](#analytics)
   - [Control](#control)
4. [WebSocket Interface](#websocket-interface)
   - [Connection](#connection)
   - [Message Types](#message-types)
   - [Client Commands](#client-commands)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Code Examples](#code-examples)

---

## Overview

The AI Trading Firm provides two interfaces for monitoring and control:

| Interface | URL | Purpose |
|-----------|-----|---------|
| REST API | `http://localhost:8081/api/*` | Request/response data retrieval |
| WebSocket | `ws://localhost:8081/ws` | Real-time streaming updates |
| Health Check | `http://localhost:8080/health` | System health monitoring |

### Architecture

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
|  | /api/signals     |    | Broadcast every  |                     |
|  | /api/decisions   |    | 500ms            |                     |
|  | /api/metrics     |    |                  |                     |
|  | /api/risk        |    |                  |                     |
|  | /api/analytics/* |    |                  |                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
          |                        |
          v                        v
+------------------+    +------------------+
|   Event Bus      |    |   Orchestrator   |
|   (Subscribe)    |    |   (Live Data)    |
+------------------+    +------------------+
```

---

## Authentication

The dashboard API currently does not require authentication. For production deployments, consider:

- Adding API key authentication
- Using HTTPS with a reverse proxy
- Binding to `127.0.0.1` instead of `0.0.0.0`
- Implementing rate limiting

---

## REST API Endpoints

### Health & Status

#### GET /health

Health check endpoint returning system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-04T10:30:00.000Z",
  "websocket_connections": 3,
  "kill_switch_active": false
}
```

#### GET /api/status

Full dashboard status with all data in a single request.

**Response:**
```json
{
  "timestamp": "2026-02-04T10:30:00.000Z",
  "metrics": { ... },
  "agents": [ ... ],
  "positions": [ ... ],
  "signals": [ ... ],
  "risk": {
    "limits": [ ... ],
    "kill_switch_active": false
  }
}
```

---

### Trading Data

#### GET /api/agents

Get all agent statuses including health metrics.

**Response:**
```json
{
  "agents": [
    {
      "name": "CIOAgent",
      "type": "Decision",
      "status": "active",
      "last_event_time": "2026-02-04T10:35:00Z",
      "event_count": 1523,
      "latency_ms": 12.5,
      "error_message": null,
      "error_count": 0,
      "uptime_seconds": 3600.0,
      "health_score": 100.0,
      "enabled": true,
      "uses_llm": false
    },
    {
      "name": "SentimentAgent",
      "type": "Signal",
      "status": "active",
      "event_count": 234,
      "latency_ms": 1250.5,
      "health_score": 85.0,
      "enabled": true,
      "uses_llm": true
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Agent name (e.g., "CIOAgent") |
| `type` | string | Agent type: Signal, Decision, Validation, Execution, Surveillance, Reporting |
| `status` | string | Current status: active, idle, error, stopped |
| `last_event_time` | string | ISO 8601 timestamp of last event |
| `event_count` | int | Total events processed |
| `latency_ms` | float | Rolling average latency in milliseconds |
| `error_count` | int | Number of errors encountered |
| `uptime_seconds` | float | Time since agent started |
| `health_score` | float | Health score 0-100 |
| `enabled` | bool | Whether agent is enabled |
| `uses_llm` | bool | Whether agent uses LLM API (costs tokens) |

---

#### GET /api/positions

Get all open positions from the broker.

**Response:**
```json
{
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "entry_price": 175.50,
      "current_price": 178.25,
      "pnl": 275.00,
      "pnl_pct": 1.57,
      "market_value": 17825.00,
      "entry_time": "2026-02-04T09:35:00Z",
      "conviction": 0.75,
      "signal": "Momentum: LONG",
      "target_price": 185.00,
      "stop_loss": 170.00
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Instrument symbol |
| `quantity` | int | Position size (negative for short) |
| `entry_price` | float | Average entry price |
| `current_price` | float | Current market price |
| `pnl` | float | Unrealized P&L in dollars |
| `pnl_pct` | float | Unrealized P&L percentage |
| `market_value` | float | Current market value |
| `entry_time` | string | ISO 8601 entry timestamp |
| `conviction` | float | CIO conviction score (0-1) |
| `signal` | string | Source agent and direction |
| `target_price` | float | Take profit target |
| `stop_loss` | float | Stop loss level |

---

#### GET /api/closed_positions

Get recently closed positions with realized P&L.

**Response:**
```json
{
  "closed_positions": [
    {
      "symbol": "MSFT",
      "side": "LONG",
      "signal": "Momentum: LONG",
      "quantity": 50,
      "entry_price": 380.00,
      "exit_price": 395.50,
      "pnl": 775.00,
      "pnl_pct": 4.08,
      "entry_time": "2026-02-03T10:00:00Z",
      "closed_at": "2026-02-04T14:30:00Z",
      "conviction": 0.68,
      "target_price": 400.00,
      "stop_loss": 365.00
    }
  ]
}
```

---

#### GET /api/signals

Get current signals from all strategy agents.

**Response:**
```json
{
  "signals": [
    {
      "agent": "MomentumAgent",
      "symbol": "AAPL",
      "direction": "LONG",
      "confidence": 0.85,
      "strength": 0.72,
      "timestamp": "2026-02-04T10:30:00Z",
      "rationale": "MA crossover confirmed, RSI bullish"
    },
    {
      "agent": "StatArbAgent",
      "symbol": "MSFT",
      "direction": "SHORT",
      "confidence": 0.65,
      "strength": -0.45,
      "timestamp": "2026-02-04T10:30:00Z",
      "rationale": "Pair spread z-score > 2.0"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `agent` | string | Source agent name |
| `symbol` | string | Instrument symbol |
| `direction` | string | LONG, SHORT, or FLAT |
| `confidence` | float | Signal confidence (0-1) |
| `strength` | float | Signal strength (-1 to +1) |
| `timestamp` | string | ISO 8601 timestamp |
| `rationale` | string | Human-readable explanation |

---

#### GET /api/decisions

Get recent CIO decisions with validation status.

**Response:**
```json
{
  "decisions": [
    {
      "decision_id": "dec_abc123",
      "symbol": "NVDA",
      "direction": "BUY",
      "quantity": 25,
      "conviction": 0.78,
      "timestamp": "2026-02-04T10:30:00Z",
      "rationale": "Strong momentum signal with macro confirmation",
      "pnl": 125.50,
      "status": "APPROVED",
      "rejection_reason": "",
      "estimated_value": 14500.00
    },
    {
      "decision_id": "dec_def456",
      "symbol": "TSLA",
      "direction": "SELL",
      "quantity": 10,
      "conviction": 0.55,
      "timestamp": "2026-02-04T10:25:00Z",
      "rationale": "Mean reversion signal",
      "pnl": 0,
      "status": "REJECTED",
      "rejection_reason": "Below minimum conviction threshold",
      "estimated_value": 2500.00
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `decision_id` | string | Unique decision identifier |
| `symbol` | string | Instrument symbol |
| `direction` | string | BUY, SELL, or HOLD |
| `quantity` | int | Proposed position size |
| `conviction` | float | CIO conviction score (0-1) |
| `timestamp` | string | ISO 8601 timestamp |
| `rationale` | string | Decision explanation |
| `pnl` | float | Realized P&L if closed |
| `status` | string | PENDING, APPROVED, or REJECTED |
| `rejection_reason` | string | Reason for rejection |
| `estimated_value` | float | Estimated trade value |

---

#### GET /api/metrics

Get portfolio metrics including P&L from broker.

**Response:**
```json
{
  "metrics": {
    "total_pnl": 2450.75,
    "today_pnl": 875.25,
    "unrealized_pnl": 1250.50,
    "realized_pnl": 1200.25,
    "today_realized_pnl": 450.00,
    "sharpe_ratio": 1.85,
    "win_rate": 0.62,
    "drawdown": 0.0234,
    "position_count": 5,
    "total_trades": 47,
    "avg_latency_ms": 15.5
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_pnl` | float | Total P&L (realized + unrealized) |
| `today_pnl` | float | Today's P&L |
| `unrealized_pnl` | float | Unrealized P&L from open positions |
| `realized_pnl` | float | Realized P&L from all closed positions |
| `today_realized_pnl` | float | Realized P&L from today's closed positions |
| `sharpe_ratio` | float | Rolling Sharpe ratio |
| `win_rate` | float | Win rate (0-1) |
| `drawdown` | float | Current drawdown from peak |
| `position_count` | int | Number of open positions |
| `total_trades` | int | Total trades executed |
| `avg_latency_ms` | float | Average system latency |

---

#### GET /api/risk

Get risk limit statuses and kill switch state.

**Response:**
```json
{
  "limits": [
    {
      "name": "Position Size",
      "current": 2.5,
      "limit": 5.0,
      "usage": 50.0,
      "status": "ok"
    },
    {
      "name": "Leverage",
      "current": 1.2,
      "limit": 2.0,
      "usage": 60.0,
      "status": "ok"
    },
    {
      "name": "Daily Loss",
      "current": 0.8,
      "limit": 3.0,
      "usage": 26.7,
      "status": "ok"
    },
    {
      "name": "Drawdown",
      "current": 2.5,
      "limit": 10.0,
      "usage": 25.0,
      "status": "ok"
    },
    {
      "name": "VaR 95%",
      "current": 1.2,
      "limit": 2.0,
      "usage": 60.0,
      "status": "ok"
    }
  ],
  "kill_switch_active": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Risk metric name |
| `current` | float | Current value |
| `limit` | float | Configured limit |
| `usage` | float | Usage percentage (0-100) |
| `status` | string | ok, warning (>75%), or breach (>100%) |

---

#### GET /api/events

Get recent event stream.

**Response:**
```json
{
  "events": [
    {
      "type": "SIGNAL",
      "source": "MomentumAgent",
      "time": "10:35:00",
      "summary": "AAPL LONG (85%)"
    },
    {
      "type": "DECISION",
      "source": "CIOAgent",
      "time": "10:35:01",
      "summary": "AAPL BUY x100"
    },
    {
      "type": "VALIDATED_DECISION",
      "source": "RiskAgent",
      "time": "10:35:02",
      "summary": "AAPL APPROVED by RiskAgent"
    }
  ]
}
```

---

#### GET /api/alerts

Get recent alerts (risk warnings, compliance alerts).

**Response:**
```json
{
  "alerts": [
    {
      "title": "Risk Warning",
      "message": "Drawdown approaching warning threshold: 4.5%",
      "severity": "warning",
      "time": "10:30:00"
    },
    {
      "title": "System Started",
      "message": "AI Trading Firm dashboard connected and monitoring",
      "severity": "info",
      "time": "09:30:00"
    }
  ]
}
```

| Severity | Description |
|----------|-------------|
| `info` | Informational message |
| `warning` | Requires attention |
| `critical` | Immediate action needed |

---

#### GET /api/equity

Get equity curve data for charting.

**Response:**
```json
{
  "equity": {
    "labels": ["09:30", "09:35", "09:40", "09:45"],
    "values": [100000, 100250, 100500, 100375],
    "timestamps": [1707040200000, 1707040500000, 1707040800000, 1707041100000]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `labels` | array | Display labels (HH:MM format, local time) |
| `values` | array | Equity values in USD |
| `timestamps` | array | Unix timestamps in milliseconds (for filtering) |

---

### Analytics

#### GET /api/analytics/rolling-metrics

Get rolling Sharpe/Sortino ratios across time periods.

**Response:**
```json
{
  "rolling_metrics": {
    "1h": { "sharpe": 1.25, "sortino": 1.85, "trades": 5 },
    "4h": { "sharpe": 1.42, "sortino": 2.10, "trades": 18 },
    "1d": { "sharpe": 1.65, "sortino": 2.35, "trades": 47 },
    "1w": { "sharpe": 1.85, "sortino": 2.55, "trades": 215 }
  }
}
```

---

#### GET /api/analytics/session-performance

Get performance breakdown by trading session.

**Response:**
```json
{
  "session_performance": {
    "pre_market": { "pnl": 250.00, "trades": 5, "win_rate": 0.60 },
    "market_open": { "pnl": 1500.00, "trades": 25, "win_rate": 0.68 },
    "lunch_hour": { "pnl": -150.00, "trades": 8, "win_rate": 0.38 },
    "market_close": { "pnl": 650.00, "trades": 12, "win_rate": 0.58 }
  }
}
```

---

#### GET /api/analytics/strategy-comparison

Get strategy performance comparison and rankings.

**Response:**
```json
{
  "strategies": {
    "MomentumAgent": {
      "total_pnl": 2500.00,
      "trades": 45,
      "win_rate": 0.64,
      "sharpe": 1.85,
      "max_drawdown": 0.05
    },
    "StatArbAgent": {
      "total_pnl": 1800.00,
      "trades": 32,
      "win_rate": 0.72,
      "sharpe": 2.10,
      "max_drawdown": 0.03
    }
  },
  "ranking": [
    { "strategy": "MomentumAgent", "pnl": 2500.00, "rank": 1 },
    { "strategy": "StatArbAgent", "pnl": 1800.00, "rank": 2 }
  ]
}
```

---

#### GET /api/analytics/risk-heatmap

Get risk heatmap data for positions.

**Response:**
```json
{
  "risk_heatmap": [
    {
      "symbol": "AAPL",
      "value": 17500.00,
      "weight": 0.175,
      "var_contribution": 0.35,
      "risk_score": 0.65
    }
  ]
}
```

---

#### GET /api/analytics/signal-consensus

Get signal consensus and disagreement data.

**Response:**
```json
{
  "consensus": {
    "AAPL": {
      "direction": "LONG",
      "agreement": 0.85,
      "agents": ["MomentumAgent", "ChartAnalysisAgent", "ForecastingAgent"]
    }
  },
  "high_disagreement": [
    {
      "symbol": "TSLA",
      "agreement": 0.40,
      "bullish": 2,
      "bearish": 3
    }
  ]
}
```

---

#### GET /api/analytics/trade-journal

Get trade journal entries with quality statistics.

**Response:**
```json
{
  "entries": [
    {
      "timestamp": "2026-02-04T10:30:00Z",
      "symbol": "AAPL",
      "strategy": "MomentumAgent",
      "pnl": 275.00,
      "entry_price": 175.50,
      "exit_price": 178.25,
      "hold_time_minutes": 45,
      "conviction_at_entry": 0.78
    }
  ],
  "quality_stats": {
    "avg_hold_time_minutes": 62,
    "avg_pnl_per_trade": 52.50,
    "best_trade": 450.00,
    "worst_trade": -180.00
  }
}
```

---

### Control

#### POST /api/kill-switch

Toggle the emergency kill switch.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `active` | bool | No | True to activate, False to deactivate (default: True) |

**Response:**
```json
{
  "status": "acknowledged",
  "kill_switch_active": true
}
```

---

#### POST /api/agent/toggle

Enable or disable an agent at runtime.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_name` | string | Yes | Name of agent to toggle |
| `enabled` | bool | Yes | True to enable, False to disable |

**Example:**
```bash
curl -X POST "http://localhost:8081/api/agent/toggle?agent_name=SentimentAgent&enabled=false"
```

**Response:**
```json
{
  "agent_name": "SentimentAgent",
  "enabled": false,
  "success": true,
  "message": "Agent SentimentAgent disabled",
  "uses_llm": true
}
```

---

#### GET /api/agent/states

Get all agent enabled/disabled states.

**Response:**
```json
{
  "states": {
    "MomentumAgent": true,
    "StatArbAgent": true,
    "SentimentAgent": false,
    "ChartAnalysisAgent": false
  },
  "llm_agents": ["SentimentAgent", "ChartAnalysisAgent", "ForecastingAgent"]
}
```

---

## WebSocket Interface

### Connection

Connect to the WebSocket at `ws://localhost:8081/ws`

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8081/ws');

ws.onopen = () => {
  console.log('Connected to dashboard');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(`Received ${message.type}:`, message.payload);
};

ws.onclose = () => {
  console.log('Disconnected from dashboard');
};
```

**Python Example:**
```python
import asyncio
import websockets
import json

async def connect():
    async with websockets.connect('ws://localhost:8081/ws') as ws:
        async for message in ws:
            data = json.loads(message)
            print(f"Received {data['type']}: {data['payload']}")

asyncio.run(connect())
```

---

### Message Types

Messages are JSON objects with `type` and `payload` fields.

| Type | Description | Update Frequency |
|------|-------------|------------------|
| `initial` | Full state on connection | Once on connect |
| `metrics` | Portfolio metrics | Every 500ms |
| `agents` | Agent statuses | Every 500ms |
| `positions` | Open positions | Every 500ms |
| `closed_positions` | Closed positions | Every 2s |
| `signals` | Signal updates | Every 500ms |
| `decisions` | CIO decisions | Every 1s |
| `risk` | Risk limits | Every 500ms |
| `equity` | Equity curve | Every 2s |
| `event` | Latest event | On event |
| `alert` | Risk/compliance alerts | On alert |
| `agent_toggle` | Agent state change | On toggle |

**Message Format:**
```json
{
  "type": "positions",
  "payload": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "pnl": 275.00
    }
  ]
}
```

**Initial Message Payload:**
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
    "risk": {
      "limits": [ ... ]
    },
    "equity": { ... }
  }
}
```

---

### Client Commands

Clients can send commands via WebSocket.

#### Kill Switch Command
```json
{
  "action": "kill_switch",
  "active": true
}
```

#### Toggle Agent Command
```json
{
  "action": "toggle_agent",
  "agent_name": "SentimentAgent",
  "enabled": false
}
```

---

## Data Models

### AgentStatus Enum

```
ACTIVE   - Agent is running and processing events
IDLE     - Agent is running but no recent events
ERROR    - Agent has encountered errors
STOPPED  - Agent is not running
```

### AgentType Enum

```
Signal       - Generates trading signals
Decision     - Makes trading decisions (CIO)
Validation   - Validates decisions (Risk, Compliance)
Execution    - Executes orders
Surveillance - Market surveillance
Reporting    - Transaction reporting
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Resource not found |
| 500 | Internal server error |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Code Examples

### Python - Fetch Positions

```python
import requests

def get_positions():
    response = requests.get('http://localhost:8081/api/positions')
    response.raise_for_status()
    return response.json()['positions']

positions = get_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['quantity']} shares, P&L: ${pos['pnl']:.2f}")
```

### Python - Monitor Metrics via WebSocket

```python
import asyncio
import websockets
import json

async def monitor_metrics():
    async with websockets.connect('ws://localhost:8081/ws') as ws:
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'metrics':
                metrics = data['payload']
                print(f"P&L: ${metrics['total_pnl']:.2f} | "
                      f"Positions: {metrics['position_count']} | "
                      f"Win Rate: {metrics['win_rate']:.1%}")

asyncio.run(monitor_metrics())
```

### JavaScript - Real-time Position Updates

```javascript
const ws = new WebSocket('ws://localhost:8081/ws');

ws.onmessage = (event) => {
  const { type, payload } = JSON.parse(event.data);

  if (type === 'positions') {
    updatePositionTable(payload);
  } else if (type === 'alert') {
    showNotification(payload);
  }
};

function updatePositionTable(positions) {
  const table = document.getElementById('positions-table');
  table.innerHTML = positions.map(pos => `
    <tr class="${pos.pnl >= 0 ? 'profit' : 'loss'}">
      <td>${pos.symbol}</td>
      <td>${pos.quantity}</td>
      <td>$${pos.pnl.toFixed(2)}</td>
      <td>${pos.pnl_pct.toFixed(2)}%</td>
    </tr>
  `).join('');
}
```

### cURL - Toggle Agent

```bash
# Disable SentimentAgent (LLM agent)
curl -X POST "http://localhost:8081/api/agent/toggle?agent_name=SentimentAgent&enabled=false"

# Re-enable SentimentAgent
curl -X POST "http://localhost:8081/api/agent/toggle?agent_name=SentimentAgent&enabled=true"

# Activate kill switch
curl -X POST "http://localhost:8081/api/kill-switch?active=true"
```

---

## Rate Limiting

The API does not currently implement rate limiting. The WebSocket pushes updates at the following intervals:

| Data Type | Interval |
|-----------|----------|
| Positions, Metrics, Agents, Signals, Risk | 500ms |
| Decisions | 1s |
| Closed Positions, Equity | 2s |

For high-frequency polling via REST API, consider using the WebSocket interface instead.

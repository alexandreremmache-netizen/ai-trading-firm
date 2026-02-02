# Configuration Reference

Complete reference for `config.yaml` configuration options.

## Table of Contents

1. [Firm Settings](#firm-settings)
2. [Broker Configuration](#broker-configuration)
3. [Event Bus](#event-bus)
4. [Risk Management](#risk-management)
5. [Position Sizing](#position-sizing)
6. [Risk Budget](#risk-budget)
7. [VaR Settings](#var-settings)
8. [Stress Testing](#stress-testing)
9. [Compliance](#compliance)
10. [Surveillance](#surveillance)
11. [Logging](#logging)
12. [Monitoring](#monitoring)
13. [Universe](#universe)
14. [Agent Configuration](#agent-configuration)

---

## Firm Settings

```yaml
firm:
  name: "AI Trading Firm"
  version: "1.0.0"
  mode: "paper"  # paper | live
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "AI Trading Firm" | System identifier |
| `version` | string | "1.0.0" | Configuration version |
| `mode` | string | "paper" | Execution mode. **NEVER change to "live" without authorization** |

---

## Broker Configuration

```yaml
broker:
  host: "127.0.0.1"
  port: 4002
  client_id: 1
  timeout_seconds: 30
  readonly: false
  use_delayed_data: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | "127.0.0.1" | IB Gateway/TWS host |
| `port` | int | 4002 | Port (4002=Gateway Paper, 4001=Gateway Live, 7497=TWS Paper, 7496=TWS Live) |
| `client_id` | int | 1 | Unique client identifier for IB |
| `timeout_seconds` | int | 30 | Connection timeout |
| `readonly` | bool | false | Read-only mode (no order submission) |
| `use_delayed_data` | bool | true | Use free delayed data (15 min) |

---

## Event Bus

```yaml
event_bus:
  max_queue_size: 10000
  signal_timeout_seconds: 5.0
  sync_barrier_timeout_seconds: 10.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_queue_size` | int | 10000 | Maximum events in queue |
| `signal_timeout_seconds` | float | 5.0 | Per-agent signal generation timeout |
| `sync_barrier_timeout_seconds` | float | 10.0 | Barrier wait timeout before CIO decision |

---

## Risk Management

```yaml
risk:
  max_portfolio_var_pct: 2.0
  max_position_size_pct: 5.0
  max_sector_exposure_pct: 20.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
  max_leverage: 2.0
  max_orders_per_minute: 10
  min_order_interval_ms: 100
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_portfolio_var_pct` | float | 2.0 | Maximum VaR as % of portfolio |
| `max_position_size_pct` | float | 5.0 | Maximum single position as % of portfolio |
| `max_sector_exposure_pct` | float | 20.0 | Maximum sector concentration |
| `max_daily_loss_pct` | float | 3.0 | Daily loss limit (triggers halt) |
| `max_drawdown_pct` | float | 10.0 | Maximum drawdown (triggers halt) |
| `max_leverage` | float | 2.0 | Maximum portfolio leverage |
| `max_orders_per_minute` | int | 10 | Rate limit (anti-HFT) |
| `min_order_interval_ms` | int | 100 | Minimum ms between orders |

---

## Position Sizing

```yaml
position_sizing:
  method: "kelly"
  use_half_kelly: true
  max_position_pct: 10.0
  min_position_pct: 1.0
  vol_target: 0.15
  correlation_discount: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | "kelly" | Sizing method: kelly, vol_target, fixed_fractional, equal |
| `use_half_kelly` | bool | true | Use half-Kelly for safety margin |
| `max_position_pct` | float | 10.0 | Maximum position size |
| `min_position_pct` | float | 1.0 | Minimum position size |
| `vol_target` | float | 0.15 | Target annualized volatility (15%) |
| `correlation_discount` | bool | true | Reduce size for correlated positions |

---

## Risk Budget

```yaml
risk_budget:
  enabled: true
  allocation_method: "risk_parity"
  total_risk_budget_pct: 2.0
  rebalance_drift_threshold: 0.20
  strategy_max_drawdown: 0.10
  sharpe_freeze_threshold: -0.5
  vol_scaling_enabled: true
  high_vol_threshold: 0.30
  high_vol_reduction: 0.50
  fixed_allocations:
    MacroAgent: 0.15
    StatArbAgent: 0.25
    MomentumAgent: 0.25
    MarketMakingAgent: 0.15
    OptionsVolAgent: 0.20
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable risk budgeting |
| `allocation_method` | string | "risk_parity" | Method: equal, risk_parity, performance_weighted, fixed, drawdown_adjusted |
| `total_risk_budget_pct` | float | 2.0 | Total risk budget as % of NAV |
| `rebalance_drift_threshold` | float | 0.20 | Rebalance when drift exceeds 20% |
| `strategy_max_drawdown` | float | 0.10 | Freeze strategy at 10% drawdown |
| `sharpe_freeze_threshold` | float | -0.5 | Freeze if rolling Sharpe below -0.5 |
| `vol_scaling_enabled` | bool | true | Scale down in high volatility |
| `high_vol_threshold` | float | 0.30 | 30% vol is "high" |
| `high_vol_reduction` | float | 0.50 | Reduce budget by 50% in high vol |

---

## VaR Settings

```yaml
var:
  method: "all"
  confidence_level: 0.95
  horizon_days: 1
  monte_carlo_simulations: 10000
  ewma_decay_factor: 0.94
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | "all" | VaR method: parametric, historical, monte_carlo, all |
| `confidence_level` | float | 0.95 | Confidence level (95%) |
| `horizon_days` | int | 1 | VaR horizon in days |
| `monte_carlo_simulations` | int | 10000 | Number of MC simulations |
| `ewma_decay_factor` | float | 0.94 | EWMA decay for volatility |

---

## Stress Testing

```yaml
stress_testing:
  enabled: true
  max_scenario_loss_pct: 25.0
  margin_buffer_pct: 20.0
  run_on_startup: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable stress testing |
| `max_scenario_loss_pct` | float | 25.0 | Maximum acceptable scenario loss |
| `margin_buffer_pct` | float | 20.0 | Margin buffer requirement |
| `run_on_startup` | bool | false | Run stress tests at startup |

---

## Compliance

```yaml
compliance:
  jurisdiction: "EU"
  regulator: "AMF"
  require_rationale: true
  audit_retention_days: 2555
  banned_instruments: []
  allowed_asset_classes:
    - "equity"
    - "etf"
    - "option"
    - "future"
    - "forex"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jurisdiction` | string | "EU" | Regulatory jurisdiction |
| `regulator` | string | "AMF" | Primary regulator |
| `require_rationale` | bool | true | Require rationale for all decisions |
| `audit_retention_days` | int | 2555 | Retention period (7 years for MiFID II) |
| `banned_instruments` | list | [] | Instruments that cannot be traded |
| `allowed_asset_classes` | list | [...] | Permitted asset classes |

---

## Transaction Reporting

```yaml
transaction_reporting:
  enabled: true
  reporting_deadline_minutes: 15
  firm_lei: ""
  firm_country: "FR"
  default_venue: "XPAR"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable transaction reporting |
| `reporting_deadline_minutes` | int | 15 | Report within N minutes |
| `firm_lei` | string | "" | Legal Entity Identifier (required for production) |
| `firm_country` | string | "FR" | Firm's country code |
| `default_venue` | string | "XPAR" | Default venue MIC code |

---

## Surveillance

```yaml
surveillance:
  wash_trading_detection: true
  spoofing_detection: true
  quote_stuffing_detection: true
  layering_detection: true
  wash_trading_window_seconds: 60
  spoofing_cancel_threshold: 0.8
  quote_stuffing_rate_per_second: 10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wash_trading_detection` | bool | true | Detect wash trades |
| `spoofing_detection` | bool | true | Detect spoofing |
| `quote_stuffing_detection` | bool | true | Detect quote stuffing |
| `layering_detection` | bool | true | Detect layering |
| `wash_trading_window_seconds` | int | 60 | Window for wash trade detection |
| `spoofing_cancel_threshold` | float | 0.8 | Cancel rate threshold for spoofing |
| `quote_stuffing_rate_per_second` | int | 10 | Rate threshold for quote stuffing |

---

## Logging

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
  audit_file: "logs/audit.jsonl"
  trade_file: "logs/trades.jsonl"
  decision_file: "logs/decisions.jsonl"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | string | "INFO" | Log level (DEBUG, INFO, WARNING, ERROR) |
| `format` | string | ... | Log message format |
| `audit_file` | string | "logs/audit.jsonl" | Audit log file |
| `trade_file` | string | "logs/trades.jsonl" | Trade log file |
| `decision_file` | string | "logs/decisions.jsonl" | Decision log file |

---

## Monitoring

```yaml
monitoring:
  log_dir: "logs/agents"
  metrics_history_size: 10000
  alert_thresholds:
    daily_pnl_pct:
      warning: -2.0
      critical: -3.0
    drawdown_pct:
      warning: 5.0
      critical: 10.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | string | "logs/agents" | Agent log directory |
| `metrics_history_size` | int | 10000 | Metrics retention count |
| `alert_thresholds` | object | ... | Alert threshold configuration |

---

## Universe

```yaml
universe:
  equities:
    - symbol: "AAPL"
      exchange: "SMART"
      currency: "USD"
  etfs:
    - symbol: "SPY"
      exchange: "SMART"
      currency: "USD"
  futures:
    - symbol: "ES"
      exchange: "CME"
      currency: "USD"
  forex:
    - symbol: "EUR"
      exchange: "IDEALPRO"
      currency: "USD"
```

Each instrument requires:
- `symbol`: IB symbol
- `exchange`: Exchange code
- `currency`: Base currency

---

## Agent Configuration

### CIO Agent

```yaml
agents:
  cio:
    signal_weight_macro: 0.15
    signal_weight_stat_arb: 0.25
    signal_weight_momentum: 0.25
    signal_weight_market_making: 0.15
    signal_weight_options_vol: 0.20
    min_conviction_threshold: 0.6
    use_dynamic_weights: true
    use_kelly_sizing: true
    kelly_fraction: 0.5
```

### Signal Agents

```yaml
agents:
  macro:
    enabled: true
    indicators: ["yield_curve", "vix", "dxy"]
    rebalance_frequency: "daily"

  stat_arb:
    enabled: true
    lookback_days: 60
    zscore_entry_threshold: 2.0
    zscore_exit_threshold: 0.5
    pairs:
      - ["AAPL", "MSFT"]
      - ["ES", "NQ"]

  momentum:
    enabled: true
    fast_period: 10
    slow_period: 30
    rsi_period: 14

  market_making:
    enabled: true
    spread_bps: 10
    max_inventory: 1000
    quote_refresh_ms: 1000

  options_vol:
    enabled: true
    iv_percentile_threshold: 80
    min_dte: 7
    max_dte: 45
```

### Execution Agent

```yaml
agents:
  execution:
    default_algo: "TWAP"
    slice_interval_seconds: 60
    max_slippage_bps: 50
```

---

## Environment Variables

The following environment variables can override config values:

| Variable | Description |
|----------|-------------|
| `IB_HOST` | Interactive Brokers host |
| `IB_PORT` | Interactive Brokers port |
| `IB_CLIENT_ID` | Client ID |
| `TRADING_MODE` | "paper" or "live" |
| `LOG_LEVEL` | Logging level |

---

## Configuration Validation

The system validates configuration at startup. Common validation errors:

1. **Invalid port**: Port must be one of 4001, 4002, 7496, 7497
2. **Missing LEI**: Transaction reporting requires firm_lei
3. **Invalid weights**: CIO weights must sum to 1.0
4. **Risk limits**: max_position_size_pct must be <= 100

Run validation manually:
```bash
python -c "from core.config_validator import validate_config; validate_config('config.yaml')"
```

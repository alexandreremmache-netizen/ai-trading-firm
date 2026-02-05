# Configuration Reference

Complete reference for `config.yaml` configuration options.

## Table of Contents

1. [Configuration Files](#configuration-files)
2. [Firm Settings](#firm-settings)
3. [Broker Configuration](#broker-configuration)
4. [Event Bus](#event-bus)
5. [Dashboard](#dashboard)
6. [Risk Management](#risk-management)
7. [Position Sizing](#position-sizing)
8. [Risk Budget](#risk-budget)
9. [VaR Settings](#var-settings)
10. [Stress Testing](#stress-testing)
11. [Compliance](#compliance)
12. [Surveillance](#surveillance)
13. [Logging](#logging)
14. [Monitoring](#monitoring)
15. [Universe](#universe)
16. [Agent Configuration](#agent-configuration)
17. [Trading Sessions](#trading-sessions)
18. [Stop-Loss Scaling](#stop-loss-scaling)
19. [CIO Position Management](#cio-position-management)
20. [Environment Variables](#environment-variables)
21. [Configuration Validation](#configuration-validation)

---

## Configuration Files

The system supports multiple configuration files:

| File | Purpose |
|------|---------|
| `config.yaml` | Full configuration with all options (~993 lines) |
| `config.simple.yaml` | Simplified configuration for quick start (~50 lines) |

### Configuration Sections

The configuration is organized into safety categories:

```yaml
# === SAFE TO MODIFY === Basic setup, logging, universe
# === REVIEW CAREFULLY === Risk management, agent parameters
# === ADVANCED === Event system, contract specs
# === DANGEROUS === Regulatory compliance settings
```

### Required Settings for First-Time Setup

Only 3 settings are **required** for initial setup:

1. `broker.port` - Set to 4002 for paper trading
2. `firm.mode` - Keep as "paper" until authorized
3. `transaction_reporting.firm_lei` - Get from gleif.org for production

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

**WARNING:** The `mode` parameter controls whether real money is at risk. Always verify this setting before starting the system.

---

## Broker Configuration

```yaml
broker:
  host: "127.0.0.1"
  port: 4002
  client_id: 2
  timeout_seconds: 30
  readonly: false
  use_delayed_data: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | "127.0.0.1" | IB Gateway/TWS host |
| `port` | int | 4002 | API port (see port mapping below) |
| `client_id` | int | 2 | Unique client ID (1-32, each app needs different ID) |
| `timeout_seconds` | int | 30 | Connection timeout (10-120) |
| `readonly` | bool | false | Read-only mode (no order submission) |
| `use_delayed_data` | bool | true | Use free delayed data (15-20 min delay) |

### Port Mapping

| Port | Platform | Mode | Risk Level |
|------|----------|------|------------|
| 4002 | IB Gateway | Paper Trading | Safe (RECOMMENDED) |
| 4001 | IB Gateway | Live Trading | DANGEROUS |
| 7497 | TWS | Paper Trading | Safe |
| 7496 | TWS | Live Trading | DANGEROUS |

**Best Practice:** Always use port 4002 (Gateway Paper) for development and testing.

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
| `max_queue_size` | int | 10000 | Maximum events in queue before backpressure |
| `signal_timeout_seconds` | float | 5.0 | Per-agent signal generation timeout |
| `sync_barrier_timeout_seconds` | float | 10.0 | Barrier wait timeout before CIO decision |

### Orchestrator Settings

```yaml
orchestrator:
  market_data_interval_seconds: 1.0
  heartbeat_interval_seconds: 30.0
  shutdown_timeout_seconds: 60.0
```

---

## Dashboard

```yaml
dashboard:
  enabled: true
  host: "0.0.0.0"
  port: 8081
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable/disable dashboard server |
| `host` | string | "0.0.0.0" | Host to bind (0.0.0.0 for all interfaces) |
| `port` | int | 8081 | Dashboard port (different from health check port 8080) |

**Security Note:** For production, consider binding to `127.0.0.1` only.

---

## Risk Management

```yaml
risk:
  max_portfolio_var_pct: 2.0
  max_position_pct: 2.5
  max_position_size_pct: 2.5
  max_sector_pct: 20.0
  max_sector_exposure_pct: 20.0
  max_gross_exposure_pct: 50.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
  max_position_daily_loss_pct: 1.0
  max_leverage: 1.5
  max_orders_per_minute: 10
  min_order_interval_ms: 100

  drawdown:
    warning_pct: 3.0
    critical_pct: 5.0
    severe_pct: 8.0
    maximum_pct: 10.0
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `max_portfolio_var_pct` | float | 0.5-10.0 | Maximum VaR as % of portfolio |
| `max_position_pct` | float | 1.0-20.0 | Max single position % |
| `max_sector_pct` | float | 5.0-50.0 | Max sector exposure % |
| `max_gross_exposure_pct` | float | 25.0-200.0 | Max total exposure % |
| `max_daily_loss_pct` | float | 1.0-10.0 | Daily loss limit % |
| `max_drawdown_pct` | float | 5.0-25.0 | Max drawdown from peak % |
| `max_position_daily_loss_pct` | float | 0.5-3.0 | Max loss per position per day % |
| `max_leverage` | float | 1.0-4.0 | Maximum leverage ratio |
| `max_orders_per_minute` | int | 1-100 | Orders per minute limit (anti-HFT) |
| `min_order_interval_ms` | int | 50-5000 | Minimum ms between orders |

### Tiered Drawdown Response

The system implements autonomous risk management through tiered drawdown response:

| Threshold | Action |
|-----------|--------|
| `warning_pct` (3%) | Reduce position sizes by 50% |
| `critical_pct` (5%) | Close worst 20%, no new longs |
| `severe_pct` (8%) | Close worst 50%, defensive mode |
| `maximum_pct` (10%) | Close all positions (no shutdown) |

---

## Position Sizing

```yaml
position_sizing:
  method: "kelly"
  kelly_fraction: 0.25
  max_kelly_raw: 0.15
  max_position_pct: 3.0
  min_position_pct: 0.5
  max_total_exposure_pct: 50.0
  vol_target: 0.15
  correlation_discount: true
  use_turn_of_month_boost: true
  tom_multiplier: 1.25
  tom_asset_classes: ["equity", "future"]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | "kelly" | Sizing method: kelly, vol_target, fixed_fractional, equal |
| `kelly_fraction` | float | 0.25 | Fractional Kelly multiplier (0.25 = quarter Kelly) |
| `max_kelly_raw` | float | 0.15 | Max raw Kelly before fraction applied |
| `max_position_pct` | float | 3.0 | Maximum position as % of portfolio |
| `min_position_pct` | float | 0.5 | Minimum position as % of portfolio |
| `max_total_exposure_pct` | float | 50.0 | Maximum total portfolio exposure % |
| `vol_target` | float | 0.15 | Target annualized volatility (15%) |
| `correlation_discount` | bool | true | Reduce size for correlated positions |
| `use_turn_of_month_boost` | bool | true | Apply TOM effect boost |
| `tom_multiplier` | float | 1.25 | Position size multiplier during TOM |

---

## Risk Budget

```yaml
risk_budget:
  enabled: true
  allocation_method: "risk_parity"
  total_risk_budget_pct: 2.0
  rebalance_drift_threshold: 0.20
  rebalance_time_interval_hours: 24
  strategy_max_drawdown: 0.10
  sharpe_freeze_threshold: -0.5
  vol_scaling_enabled: true
  high_vol_threshold: 0.30
  high_vol_reduction: 0.50
  fixed_allocations:
    MacroAgent: 0.10
    StatArbAgent: 0.15
    MomentumAgent: 0.15
    MarketMakingAgent: 0.10
    OptionsVolAgent: 0.0
    SentimentAgent: 0.15
    ChartAnalysisAgent: 0.20
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable risk budgeting |
| `allocation_method` | string | "risk_parity" | Method (see below) |
| `total_risk_budget_pct` | float | 2.0 | Total risk budget as % of NAV |
| `rebalance_drift_threshold` | float | 0.20 | Rebalance when drift exceeds 20% |
| `strategy_max_drawdown` | float | 0.10 | Freeze strategy at 10% drawdown |
| `sharpe_freeze_threshold` | float | -0.5 | Freeze if rolling Sharpe below -0.5 |
| `vol_scaling_enabled` | bool | true | Scale down budget in high volatility |
| `high_vol_threshold` | float | 0.30 | 30% volatility is "high" |
| `high_vol_reduction` | float | 0.50 | Reduce budget by 50% in high vol |

### Allocation Methods

| Method | Description |
|--------|-------------|
| `equal` | Equal allocation to all strategies |
| `risk_parity` | Allocate based on inverse volatility |
| `performance_weighted` | Allocate based on recent performance |
| `fixed` | Use `fixed_allocations` mapping |
| `drawdown_adjusted` | Reduce allocation for strategies in drawdown |

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
  firm_lei: "529900T8BM49AURSDO55"
  require_rationale: true
  audit_retention_days: 2555
  banned_instruments: []
  allowed_asset_classes:
    - "equity"
    # - "etf"  # Disabled
    - "option"
    - "future"
    - "forex"
    - "commodity"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jurisdiction` | string | "EU" | Regulatory jurisdiction |
| `regulator` | string | "AMF" | Primary regulator |
| `firm_lei` | string | "" | Legal Entity Identifier (20 chars, ISO 17442) |
| `require_rationale` | bool | true | Require rationale for all decisions |
| `audit_retention_days` | int | 2555 | Retention period (7 years for MiFID II) |
| `banned_instruments` | list | [] | Instruments that cannot be traded |
| `allowed_asset_classes` | list | [...] | Permitted asset classes |

### Transaction Reporting

```yaml
transaction_reporting:
  enabled: true
  reporting_deadline_minutes: 15
  firm_lei: "529900T8BM49AURSDO55"
  firm_country: "FR"
  default_venue: "XPAR"
```

**IMPORTANT:** You must obtain a valid LEI from [GLEIF](https://www.gleif.org/) for production use.

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
| `spoofing_detection` | bool | true | Detect spoofing patterns |
| `quote_stuffing_detection` | bool | true | Detect quote stuffing |
| `layering_detection` | bool | true | Detect layering |
| `wash_trading_window_seconds` | int | 60 | Window for wash trade detection |
| `spoofing_cancel_threshold` | float | 0.8 | Cancel rate threshold (80%) |
| `quote_stuffing_rate_per_second` | int | 10 | Rate threshold |

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
  metrics_retention_hours: 24
  anomaly_zscore_threshold: 3.0
  anomaly_min_samples: 30
  alert_thresholds:
    daily_pnl_pct:
      warning: -2.0
      critical: -3.0
    drawdown_pct:
      warning: 5.0
      critical: 10.0
    latency_ms:
      warning: 500
      critical: 1000
    var_95_pct:
      warning: 1.5
      critical: 2.0
```

### Health Check Server

```yaml
health_check:
  enabled: true
  host: "127.0.0.1"
  port: 8080
  max_event_queue_pct: 90.0
  max_latency_ms: 1000.0
  min_active_agents: 3
  broker_required: false
```

---

## Universe

### Equities

```yaml
universe:
  equities:
    - symbol: "AAPL"
      exchange: "SMART"
      currency: "USD"
    - symbol: "MSFT"
      exchange: "SMART"
      currency: "USD"
    # ... more equities
```

### ETFs

```yaml
  etfs:
    - symbol: "SPY"
      exchange: "SMART"
      currency: "USD"
    # ... more ETFs
```

### Futures

```yaml
  futures:
    # Index Futures - E-mini
    - symbol: "ES"
      exchange: "CME"
      currency: "USD"
    - symbol: "NQ"
      exchange: "CME"
      currency: "USD"

    # Micro Futures
    - symbol: "MES"
      exchange: "CME"
      currency: "USD"

    # Energy
    - symbol: "CL"
      exchange: "NYMEX"
      currency: "USD"

    # Precious Metals
    - symbol: "GC"
      exchange: "COMEX"
      currency: "USD"

    # Agriculture
    - symbol: "ZC"
      exchange: "CBOT"
      currency: "USD"
```

### Forex

```yaml
  forex:
    - symbol: "EUR"
      exchange: "IDEALPRO"
      currency: "USD"
    - symbol: "GBP"
      exchange: "IDEALPRO"
      currency: "USD"
```

Each instrument requires:
- `symbol`: IB symbol
- `exchange`: Exchange code
- `currency`: Base currency

---

## Agent Configuration

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
    divergence_enabled: true
    divergence_threshold: 2.0

  momentum:
    enabled: true
    fast_period: 10
    slow_period: 30
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    adx_period: 14
    adx_strong_threshold: 25
    demand_zones_enabled: true

  market_making:
    enabled: true
    spread_bps: 10
    max_inventory: 1000
    quote_refresh_ms: 1000

  options_vol:
    enabled: false  # Disabled
    iv_percentile_threshold: 80
    min_dte: 7
    max_dte: 45
```

### LLM Agents

```yaml
agents:
  sentiment:
    enabled: false  # Uses API tokens
    llm_provider: "anthropic"
    model: "claude-sonnet-4-5-20250929"
    news_sources:
      - "https://www.investing.com/rss/news.rss"
    analysis_interval_seconds: 300
    min_confidence: 0.5

  chart_analysis:
    enabled: false  # Uses API tokens
    model: "claude-sonnet-4-5-20250929"
    symbols: ["SPY", "QQQ", "AAPL"]
    analysis_interval_seconds: 300
    chart_timeframe: "15min"

  forecasting:
    enabled: false  # Uses API tokens
    forecast_horizons: ["1h", "4h"]
    min_confidence: 0.6
```

### CIO Agent

```yaml
agents:
  cio:
    signal_weight_macro: 0.10
    signal_weight_stat_arb: 0.12
    signal_weight_momentum: 0.15
    signal_weight_market_making: 0.08
    signal_weight_options_vol: 0.0
    signal_weight_sentiment: 0.13
    signal_weight_chart_analysis: 0.15
    signal_weight_forecasting: 0.15
    min_conviction_threshold: 0.6
    use_dynamic_weights: true
    use_kelly_sizing: true
    kelly_fraction: 0.5
```

### Execution Agent

```yaml
agents:
  execution:
    default_algo: "TWAP"
    slice_interval_seconds: 60
    max_slippage_bps: 50
```

### Exit Rules

```yaml
agents:
  exit_rules:
    enabled: true
    use_atr_stops: true
    stop_loss_atr_multiplier: 2.5
    default_stop_loss_pct: 5.0
    default_take_profit_pct: 10.0
    trailing_take_profit: true
    trailing_enabled: true
    trailing_activation_pct: 2.0
    trailing_distance_pct: 3.0
```

---

## Trading Sessions

```yaml
trading_sessions:
  forex:
    eurusd:
      active_sessions: ["london", "ny_overlap"]
      london_start: "08:00"
      london_end: "17:00"
      avoid: ["asian"]

  futures:
    es_mes:
      optimal_hours: ["09:30-11:30", "14:00-16:00"]
      avoid_hours: ["12:00-14:00"]

  commodities:
    cl_mcl:
      optimal_hours: ["09:00-14:30"]
```

---

## Stop-Loss Scaling

```yaml
stop_loss_scaling:
  base_atr_multiplier: 2.5
  high_vol_threshold: 1.5
  low_vol_threshold: 0.7
  high_vol_multiplier: 1.3
  low_vol_multiplier: 0.8
  atr_overrides:
    MES: 1.5
    MNQ: 2.0
    NG: 3.0
    GC: 2.0
```

---

## CIO Position Management

```yaml
cio:
  position_management_enabled: true
  position_review_interval_seconds: 60

  position_management:
    max_loss_pct: 5.0
    extended_loss_pct: 8.0
    loss_time_threshold_hours: 48.0
    profit_target_pct: 15.0
    trailing_profit_pct: 3.0
    partial_profit_pct: 50.0
    conviction_drop_threshold: 0.3
    min_conviction_to_hold: 0.4
    max_holding_days: 30.0
    stale_signal_hours: 24.0
    volatile_regime_loss_pct: 3.0
    trending_regime_profit_mult: 1.5
```

---

## Environment Variables

Environment variables can override config values:

| Variable | Description |
|----------|-------------|
| `IB_HOST` | Interactive Brokers host |
| `IB_PORT` | Interactive Brokers port |
| `IB_CLIENT_ID` | Client ID |
| `TRADING_MODE` | "paper" or "live" |
| `LOG_LEVEL` | Logging level |
| `ANTHROPIC_API_KEY` | API key for Claude (LLM agents) |
| `OPENAI_API_KEY` | Alternative API key for OpenAI |

---

## Configuration Validation

The system validates configuration at startup. Common validation errors:

### Port Validation
```
Error: Invalid port. Must be one of 4001, 4002, 7496, 7497
```
**Fix:** Set `broker.port` to a valid IB port.

### LEI Validation
```
Error: Transaction reporting requires firm_lei
```
**Fix:** Obtain a valid 20-character LEI from [GLEIF](https://www.gleif.org/).

### Weight Validation
```
Warning: CIO signal weights do not sum to 1.0
```
**Note:** Weights are normalized automatically, but should sum to 1.0 for clarity.

### Risk Limits Validation
```
Error: max_position_size_pct must be <= 100
```
**Fix:** Ensure all percentage values are within valid ranges.

### Manual Validation

Run validation manually:
```bash
python -c "from core.config_validator import validate_config; validate_config('config.yaml')"
```

---

## Example Configuration

### Minimal Paper Trading Setup

```yaml
firm:
  mode: "paper"

broker:
  port: 4002
  client_id: 1

risk:
  max_position_pct: 5.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0

universe:
  futures:
    - symbol: "MES"
      exchange: "CME"
      currency: "USD"

agents:
  momentum:
    enabled: true
  stat_arb:
    enabled: true
```

### Full Production Setup

See `config.yaml` for a complete production configuration with all options enabled.

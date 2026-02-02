# Risk Management Documentation

## Overview

The AI Trading Firm implements institutional-grade risk management with multiple layers of protection. The Risk Agent validates all trading decisions before execution, with kill-switch capability for emergency situations.

## Risk Management Architecture

```
+------------------------------------------------------------------+
|                     RISK MANAGEMENT LAYER                         |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+                     |
|  | VaR Calculator   |    | Stress Tester    |                     |
|  | - Parametric     |    | - Market Crash   |                     |
|  | - Historical     |    | - Vol Spike      |                     |
|  | - Monte Carlo    |    | - Correlation    |                     |
|  +------------------+    +------------------+                     |
|           |                      |                                |
|           +----------+-----------+                                |
|                      |                                            |
|                      v                                            |
|  +--------------------------------------------------+            |
|  |                  RISK AGENT                       |            |
|  |  - Position Limits                                |            |
|  |  - Leverage Limits                                |            |
|  |  - VaR Limits                                     |            |
|  |  - Daily Loss Limits                              |            |
|  |  - Kill Switch                                    |            |
|  +--------------------------------------------------+            |
|                      |                                            |
|           +----------+-----------+                                |
|           |                      |                                |
|  +------------------+    +------------------+                     |
|  | Correlation      |    | Position Sizer   |                     |
|  | Manager          |    | (Kelly Criterion)|                     |
|  +------------------+    +------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Risk Limits

### Position Limits

| Limit | Value | Description |
|-------|-------|-------------|
| Max Position Size | 5% | Maximum single position as % of portfolio |
| Max Sector Exposure | 20% | Maximum exposure to any sector |
| Max Single Name | 10% | Maximum exposure to single security |

### Portfolio Limits

| Limit | Value | Description |
|-------|-------|-------------|
| Max Leverage | 2.0x | Maximum gross leverage |
| Max Net Exposure | 100% | Maximum net market exposure |
| Max Gross Exposure | 200% | Maximum gross market exposure |

### Risk Metric Limits

| Limit | Value | Description |
|-------|-------|-------------|
| Max VaR (95%) | 2% | Maximum 1-day VaR at 95% confidence |
| Max VaR (99%) | 4% | Maximum 1-day VaR at 99% confidence |
| Max Expected Shortfall | 3% | Maximum CVaR/ES |

### Loss Limits

| Limit | Value | Action |
|-------|-------|--------|
| Daily Loss | -3% | Kill-switch activated |
| Drawdown | -10% | Kill-switch activated |
| Intraday Loss | -2% | Warning alert |

### Rate Limits (Anti-HFT)

| Limit | Value | Description |
|-------|-------|-------------|
| Orders per Minute | 10 | Maximum order rate |
| Min Order Interval | 100ms | Minimum time between orders |

---

## Value at Risk (VaR)

### Overview

VaR measures the potential loss over a specific time horizon at a given confidence level. The system calculates VaR using three methodologies.

### Parametric VaR (Variance-Covariance)

Assumes returns are normally distributed:

```
VaR = Portfolio Value * z_alpha * sigma * sqrt(t)

Where:
- z_alpha = quantile of normal distribution (1.645 for 95%)
- sigma = portfolio volatility
- t = time horizon in days
```

**Implementation**:
```python
def calculate_parametric_var(
    self,
    portfolio_value: float,
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence_level: float = 0.95,
) -> VaRResult:
    """
    Calculate parametric VaR using variance-covariance method.
    """
    # Calculate portfolio volatility
    cov_matrix = np.cov(returns_matrix, rowvar=False)
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Calculate VaR
    z_score = stats.norm.ppf(confidence_level)
    var = portfolio_value * z_score * portfolio_volatility

    return VaRResult(
        method=VaRMethod.PARAMETRIC,
        confidence_level=confidence_level,
        var_absolute=var,
        var_pct=var / portfolio_value,
    )
```

### Historical VaR

Uses actual historical returns distribution:

```
VaR = Percentile of historical portfolio returns

Where:
- Sort historical portfolio returns
- VaR = return at (1 - confidence) percentile
```

**Implementation**:
```python
def calculate_historical_var(
    self,
    portfolio_value: float,
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence_level: float = 0.95,
) -> VaRResult:
    """
    Calculate historical VaR using actual return distribution.
    """
    # Calculate portfolio returns
    portfolio_returns = returns_matrix @ weights

    # Find percentile
    var_percentile = (1 - confidence_level) * 100
    var_return = np.percentile(portfolio_returns, var_percentile)
    var = portfolio_value * abs(var_return)

    return VaRResult(
        method=VaRMethod.HISTORICAL,
        confidence_level=confidence_level,
        var_absolute=var,
        var_pct=abs(var_return),
    )
```

### Monte Carlo VaR

Simulates potential future scenarios:

```
1. Generate N random return scenarios
2. Calculate portfolio value for each scenario
3. VaR = percentile of simulated portfolio values
```

**Implementation**:
```python
def calculate_monte_carlo_var(
    self,
    portfolio_value: float,
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
) -> VaRResult:
    """
    Calculate Monte Carlo VaR using simulated returns.
    """
    # Fit multivariate normal distribution
    mean_returns = np.mean(returns_matrix, axis=0)
    cov_matrix = np.cov(returns_matrix, rowvar=False)

    # Simulate returns
    simulated_returns = np.random.multivariate_normal(
        mean_returns, cov_matrix, n_simulations
    )

    # Calculate portfolio returns
    portfolio_returns = simulated_returns @ weights

    # Calculate VaR
    var_percentile = (1 - confidence_level) * 100
    var_return = np.percentile(portfolio_returns, var_percentile)
    var = portfolio_value * abs(var_return)

    return VaRResult(
        method=VaRMethod.MONTE_CARLO,
        confidence_level=confidence_level,
        var_absolute=var,
        var_pct=abs(var_return),
    )
```

### Expected Shortfall (CVaR)

Expected Shortfall measures the average loss beyond VaR:

```
ES = E[Loss | Loss > VaR]
```

This is more conservative than VaR as it captures tail risk.

### Liquidity-Adjusted VaR

Accounts for liquidation costs:

```python
@dataclass
class LiquidityProfile:
    symbol: str
    average_daily_volume: float
    bid_ask_spread_bps: float
    volatility_daily: float
    position_size: float

    @property
    def days_to_liquidate(self) -> float:
        """Estimate days to liquidate at 10% participation."""
        participation_rate = 0.10
        max_daily = self.average_daily_volume * participation_rate
        return abs(self.position_size) / max(max_daily, 1)

    @property
    def market_impact_bps(self) -> float:
        """Square-root market impact model."""
        participation = abs(self.position_size) / self.average_daily_volume
        impact = 0.6 * np.sqrt(participation) * self.volatility_daily * 10000
        return min(impact, 500.0)  # Cap at 5%
```

### Configuration

```yaml
var:
  method: "all"                       # Calculate all methods
  confidence_level: 0.95              # 95% confidence
  horizon_days: 1                     # 1-day VaR
  monte_carlo_simulations: 10000      # MC iterations
  ewma_decay_factor: 0.94             # EWMA for volatility
```

---

## Greeks Monitoring

For options portfolios, Greeks are monitored against limits:

### Portfolio Greeks Limits

| Greek | Limit | Description |
|-------|-------|-------------|
| Delta | 500 | Maximum portfolio delta |
| Gamma | 100 | Maximum portfolio gamma |
| Vega | 50,000 | Maximum portfolio vega |
| Theta | -10,000 | Maximum negative theta |

### Greeks State Tracking

```python
@dataclass
class GreeksState:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    last_update: datetime

    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if Greeks data is stale."""
        age = (datetime.now(timezone.utc) - self.last_update).total_seconds()
        return age > max_age_seconds
```

### Configuration

```yaml
greeks:
  max_portfolio_delta: 500
  max_portfolio_gamma: 100
  max_portfolio_vega: 50000
  max_portfolio_theta: -10000
```

---

## Stress Testing

### Overview

Stress testing evaluates portfolio performance under adverse scenarios. The system includes predefined scenarios and supports custom scenarios.

### Predefined Scenarios

#### Market Crash (-15%)
```python
StressScenario(
    name="Market Crash (-15%)",
    scenario_type=ScenarioType.MARKET_CRASH,
    price_shocks={
        "AAPL": -0.15, "MSFT": -0.15, "GOOGL": -0.15,
        "SPY": -0.15, "QQQ": -0.17,
        "ES": -0.15, "NQ": -0.17,
        "GC": 0.05,   # Gold rallies
        "ZB": 0.03,   # Bonds rally
    },
    volatility_multiplier=2.5,
    correlation_override=0.85,  # Correlations spike
    liquidity_haircut=0.30,
    severity=5,
)
```

#### Volatility Spike
```python
StressScenario(
    name="VIX Spike",
    scenario_type=ScenarioType.VOLATILITY_SPIKE,
    volatility_multiplier=3.0,
    price_shocks={"SPY": -0.08, "QQQ": -0.10},
    severity=4,
)
```

#### Flash Crash
```python
StressScenario(
    name="Flash Crash",
    scenario_type=ScenarioType.FLASH_CRASH,
    price_shocks={
        "SPY": -0.08, "QQQ": -0.10,
        "ES": -0.08, "NQ": -0.10,
    },
    liquidity_haircut=0.70,  # Severe liquidity impact
    horizon_days=1,
    severity=5,
)
```

#### Correlation Breakdown
```python
StressScenario(
    name="Correlation Breakdown",
    scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
    correlation_override=0.95,  # All correlations go to 1
    volatility_multiplier=1.5,
    severity=4,
)
```

### Stress Test Results

```python
@dataclass
class StressTestResult:
    scenario: StressScenario
    portfolio_value_before: float
    portfolio_value_after: float
    pnl_impact: float
    pnl_impact_pct: float
    positions_impacted: dict[str, float]
    margin_impact: float
    liquidity_impact: float
    passes_limit: bool
    limit_breaches: list[str]
```

### Configuration

```yaml
stress_testing:
  enabled: true
  max_scenario_loss_pct: 25.0    # Max acceptable loss
  margin_buffer_pct: 20.0        # Required margin buffer
  run_on_startup: false          # Run tests at startup
```

---

## Kill Switch

### Overview

The kill switch immediately halts all trading activity when critical conditions are met. This is required by MiFID II RTS 6.

### Trigger Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Daily Loss | > 3% | Activate kill switch |
| Max Drawdown | > 10% | Activate kill switch |
| VaR Breach | > limit | Activate kill switch |
| Connectivity Loss | Extended | Activate kill switch |
| Manual Trigger | Operator request | Activate kill switch |
| Latency Breach | > threshold | Activate kill switch |

### Kill Switch Actions

```python
class KillSwitchAction(Enum):
    HALT_NEW_ORDERS = "halt_new_orders"    # Stop new orders
    CANCEL_PENDING = "cancel_pending"       # Cancel pending orders
    CLOSE_POSITIONS = "close_positions"     # Liquidate positions
    FULL_SHUTDOWN = "full_shutdown"         # Complete shutdown
```

### Kill Switch Event

```python
@dataclass
class KillSwitchEvent(Event):
    activated: bool = True
    reason: str = ""
    trigger_type: str = "manual"  # "manual", "automatic", "regulatory"
    affected_symbols: tuple[str, ...] = ()  # Empty = all
    cancel_pending_orders: bool = True
    close_positions: bool = False
```

---

## Drawdown Management

### Tiered Response

```python
class DrawdownLevel(Enum):
    NORMAL = "normal"       # < 5%: Normal trading
    WARNING = "warning"     # 5-7.5%: Warning, log alerts
    REDUCE = "reduce"       # 7.5-10%: Reduce position sizes
    HALT = "halt"           # > 10%: Kill switch, halt trading
```

### Drawdown Recovery Tracking

```python
@dataclass
class DrawdownRecoveryState:
    drawdown_start_time: datetime | None
    drawdown_start_equity: float
    drawdown_trough_equity: float
    drawdown_trough_time: datetime | None
    is_recovering: bool
    recovery_times_days: list[float]
    avg_recovery_time_days: float
    max_recovery_time_days: float
```

---

## Margin Monitoring

### Intraday Margin Tracking

```python
@dataclass
class MarginState:
    initial_margin: float
    maintenance_margin: float
    available_margin: float
    margin_utilization_pct: float
    margin_excess: float

    # Thresholds
    warning_utilization_pct: float = 70.0
    critical_utilization_pct: float = 85.0
    margin_call_pct: float = 100.0

    def is_warning(self) -> bool:
        return self.margin_utilization_pct >= self.warning_utilization_pct

    def is_critical(self) -> bool:
        return self.margin_utilization_pct >= self.critical_utilization_pct
```

---

## Correlation Management

### Correlation Monitoring

```python
class CorrelationManager:
    """
    Monitors cross-asset correlations and detects regime changes.
    """

    def __init__(
        self,
        lookback_days: int = 60,
        max_pairwise_correlation: float = 0.85,
        min_history_days: int = 20,
        regime_change_threshold: float = 0.15,
    ):
```

### Correlation Alerts

| Alert Type | Threshold | Description |
|------------|-----------|-------------|
| High Correlation | > 0.85 | Pair correlation too high |
| Regime Change | > 0.15 change | Correlation pattern shift |
| Concentration | > 0.70 average | Portfolio too correlated |

### Configuration

```yaml
correlation:
  lookback_days: 60
  max_pairwise_correlation: 0.85
  min_history_days: 20
  regime_change_threshold: 0.15
```

---

## Position Sizing

### Kelly Criterion

Optimal position sizing based on edge and win rate:

```
f* = (bp - q) / b

Where:
- b = avg_win / avg_loss (odds)
- p = win probability
- q = 1 - p
```

### Half-Kelly

For safety, use half of Kelly:
```
f = f* * 0.5
```

### Implementation

```python
class PositionSizer:
    def __init__(
        self,
        method: str = "kelly",
        use_half_kelly: bool = True,
        max_position_pct: float = 10.0,
        min_position_pct: float = 1.0,
        vol_target: float = 0.15,
        correlation_discount: bool = True,
    ):
```

### Configuration

```yaml
position_sizing:
  method: "kelly"            # "kelly", "vol_target", "fixed", "equal"
  use_half_kelly: true       # Use half-Kelly for safety
  max_position_pct: 10.0     # Maximum position size
  min_position_pct: 1.0      # Minimum position size
  vol_target: 0.15           # Target volatility (15% annual)
  correlation_discount: true # Reduce size for correlated positions
```

---

## Risk Budget Management

### Cross-Strategy Risk Budget

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
```

### Strategy Allocation

| Strategy | Allocation |
|----------|------------|
| Macro | 15% |
| Statistical Arbitrage | 25% |
| Momentum | 25% |
| Market Making | 15% |
| Options Volatility | 20% |

### Strategy Freeze Conditions

A strategy is frozen (no new positions) when:
- Strategy drawdown > 10%
- Strategy Sharpe ratio < -0.5
- Risk budget exhausted

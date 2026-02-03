# greeks_analytics

**Path**: `C:\Users\Alexa\ai-trading-firm\core\greeks_analytics.py`

## Overview

Greeks Analytics Module
=======================

Comprehensive Greeks sensitivity analysis and risk exposure trending.

Issues Addressed:
- #R15: Greeks sensitivity analysis not comprehensive
- #R20: No risk exposure trending/forecasting
- #R21: Missing P&L attribution by risk factor
- #R24: No scenario-specific position limits

## Classes

### SensitivityType

**Inherits from**: str, Enum

Types of sensitivity analysis.

### GreeksSensitivity

Result of Greeks sensitivity analysis (#R15).

#### Methods

##### `def to_dict(self) -> dict`

### PortfolioGreeksSummary

Complete portfolio Greeks summary with sensitivities (#R15).

#### Methods

##### `def to_dict(self) -> dict`

### GreeksSensitivityAnalyzer

Comprehensive Greeks sensitivity analysis (#R15).

Analyzes:
- First, second, and third-order Greeks
- Greeks sensitivity to underlying factors
- Stress testing of Greeks under various scenarios
- Greeks term structure analysis

#### Methods

##### `def __init__(self, spot_shock_range: tuple[float, float], vol_shock_range: tuple[float, float], rate_shock_range: tuple[float, float], n_scenarios: int)`

##### `def update_position(self, symbol: str, position: dict) -> None`

Update position for Greeks analysis.

##### `def calculate_delta_sensitivity(self, portfolio_delta: float, portfolio_gamma: float, spot_price: float) -> GreeksSensitivity`

Calculate delta sensitivity to spot price changes (#R15).

Shows how delta changes across spot price scenarios.

##### `def calculate_gamma_sensitivity(self, portfolio_gamma: float, portfolio_speed: float, spot_price: float) -> GreeksSensitivity`

Calculate gamma sensitivity to spot price changes (#R15).

Speed = dGamma/dSpot

##### `def calculate_vega_sensitivity(self, portfolio_vega: float, portfolio_volga: float, current_vol: float) -> GreeksSensitivity`

Calculate vega sensitivity to volatility changes (#R15).

Volga = dVega/dVol

##### `def calculate_cross_greek_sensitivity(self, portfolio_delta: float, portfolio_vanna: float, spot_price: float, current_vol: float) -> dict`

Calculate cross-Greek sensitivities (delta-vol interaction) (#R15).

Vanna = dDelta/dVol = dVega/dSpot

##### `def run_full_sensitivity_analysis(self, portfolio_greeks: dict, spot_price: float, current_vol: float) -> dict`

Run comprehensive sensitivity analysis on all Greeks (#R15).

Returns full sensitivity report.

### RiskExposureForecast

Forecasted risk exposure (#R20).

#### Methods

##### `def to_dict(self) -> dict`

### RiskExposureTrending

Risk exposure trending and forecasting (#R20).

Tracks and forecasts:
- VaR trends
- Greeks evolution
- Exposure concentration
- Correlation regime

#### Methods

##### `def __init__(self, lookback_days: int, forecast_horizon: int)`

##### `def record_var(self, var_value: float) -> None`

Record daily VaR observation.

##### `def record_greeks(self, delta: float, vega: float) -> None`

Record daily Greeks snapshot.

##### `def record_concentration(self, hhi: float) -> None`

Record portfolio concentration (HHI).

##### `def forecast_var(self) -> RiskExposureForecast`

Forecast VaR trend (#R20).

Returns VaR forecast with trend analysis.

##### `def get_trend_dashboard(self) -> dict`

Get complete trend dashboard (#R20).

Returns trends and forecasts for all risk metrics.

### PnLAttribution

P&L attribution by risk factor (#R21).

#### Methods

##### `def to_dict(self) -> dict`

### PnLAttributionEngine

P&L attribution by risk factor (#R21).

Decomposes P&L into:
- First-order Greeks (delta, vega, theta, rho)
- Second-order effects (gamma, vanna, volga)
- Cross-effects and residual

#### Methods

##### `def __init__(self)`

##### `def record_snapshot(self, greeks: dict, market_data: dict) -> None`

Record Greeks and market snapshot for attribution.

Args:
    greeks: Portfolio Greeks {delta, gamma, vega, theta, rho, vanna, volga}
    market_data: Market data {spot_prices, vols, rates}

##### `def attribute_pnl(self, total_pnl: float, start_greeks: dict, end_greeks: dict, start_market: dict, end_market: dict) -> PnLAttribution`

Attribute P&L to risk factors (#R21).

Uses Taylor series expansion for attribution.

##### `def get_attribution_report(self, lookback_days: int) -> dict`

Generate P&L attribution report (#R21).

Provides cumulative attribution over period.

### ScenarioType

**Inherits from**: str, Enum

Market scenario types for position limits (#R24).

### ScenarioPositionLimit

Position limit for a specific scenario (#R24).

### ScenarioLimitCheck

Result of scenario limit check (#R24).

### ScenarioPositionLimits

Scenario-specific position limits (#R24).

Different limits apply under different market conditions:
- Normal: Standard limits
- High Vol: Reduced limits when VIX elevated
- Crisis: Severely reduced limits during market stress
- Earnings: Reduced limits around earnings
- FOMC: Reduced limits around Fed meetings
- Expiration: Reduced limits near option expiry

#### Methods

##### `def __init__(self)`

##### `def set_base_limit(self, symbol: str, max_position: int, max_notional: float, max_delta: float, max_gamma: float, max_vega: float) -> None`

Set base position limits for a symbol.

##### `def set_scenario_limit(self, scenario: ScenarioType, symbol: str, max_position: int, max_notional: float, max_delta: float, max_gamma: float, max_vega: float, rationale: str) -> None`

Set custom limit for specific scenario (#R24).

##### `def detect_scenario(self, vix_level: float, is_earnings: bool, is_fomc: bool, days_to_expiry: ) -> ScenarioType`

Detect current market scenario (#R24).

Args:
    vix_level: Current VIX level
    is_earnings: Whether in earnings period
    is_fomc: Whether FOMC meeting today/tomorrow
    days_to_expiry: Days to option expiry

Returns:
    Detected scenario type

##### `def get_effective_limit(self, symbol: str, scenario: ) -> dict`

Get effective position limit for symbol under scenario (#R24).

Args:
    symbol: Symbol to get limit for
    scenario: Scenario (uses current if not specified)

Returns:
    Effective limits

##### `def check_limit(self, symbol: str, current_position: int, current_notional: float, current_delta: float, current_gamma: float, current_vega: float, scenario: ) -> ScenarioLimitCheck`

Check if position is within scenario limits (#R24).

Returns limit check result with utilization.

##### `def update_scenario(self, vix_level: float, is_earnings: bool, is_fomc: bool, days_to_expiry: ) -> ScenarioType`

Update current scenario based on market conditions.

##### `def get_status(self) -> dict`

Get position limits status.

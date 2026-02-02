"""
Greeks Analytics Module
=======================

Comprehensive Greeks sensitivity analysis and risk exposure trending.

Issues Addressed:
- #R15: Greeks sensitivity analysis not comprehensive
- #R20: No risk exposure trending/forecasting
- #R21: Missing P&L attribution by risk factor
- #R24: No scenario-specific position limits
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# GREEKS SENSITIVITY ANALYSIS (#R15)
# =============================================================================

class SensitivityType(str, Enum):
    """Types of sensitivity analysis."""
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    RHO = "rho"
    VANNA = "vanna"  # dDelta/dVol
    VOLGA = "volga"  # dVega/dVol
    CHARM = "charm"  # dDelta/dTime
    VETA = "veta"    # dVega/dTime
    SPEED = "speed"  # dGamma/dSpot
    COLOR = "color"  # dGamma/dTime
    ULTIMA = "ultima"  # dVolga/dVol


@dataclass
class GreeksSensitivity:
    """Result of Greeks sensitivity analysis (#R15)."""
    greek: str
    base_value: float
    sensitivities: dict[str, float]  # factor -> sensitivity
    stress_values: dict[str, float]  # scenario -> stressed value
    confidence_interval: tuple[float, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            'greek': self.greek,
            'base_value': self.base_value,
            'sensitivities': self.sensitivities,
            'stress_values': self.stress_values,
            'confidence_interval': self.confidence_interval,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class PortfolioGreeksSummary:
    """Complete portfolio Greeks summary with sensitivities (#R15)."""
    timestamp: datetime

    # First-order Greeks
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    # Second-order Greeks
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    veta: float = 0.0

    # Third-order Greeks
    speed: float = 0.0
    color: float = 0.0
    ultima: float = 0.0

    # Sensitivities
    delta_1pct_spot: float = 0.0  # P&L for 1% spot move
    gamma_1pct_spot: float = 0.0  # Delta change for 1% spot move
    vega_1vol_point: float = 0.0  # P&L for 1 vol point
    theta_1day: float = 0.0  # Daily theta decay

    # By underlying
    by_underlying: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'first_order': {
                'delta': self.delta,
                'gamma': self.gamma,
                'vega': self.vega,
                'theta': self.theta,
                'rho': self.rho,
            },
            'second_order': {
                'vanna': self.vanna,
                'volga': self.volga,
                'charm': self.charm,
                'veta': self.veta,
            },
            'third_order': {
                'speed': self.speed,
                'color': self.color,
                'ultima': self.ultima,
            },
            'sensitivities': {
                'delta_1pct_spot': self.delta_1pct_spot,
                'gamma_1pct_spot': self.gamma_1pct_spot,
                'vega_1vol_point': self.vega_1vol_point,
                'theta_1day': self.theta_1day,
            },
            'by_underlying': self.by_underlying,
        }


class GreeksSensitivityAnalyzer:
    """
    Comprehensive Greeks sensitivity analysis (#R15).

    Analyzes:
    - First, second, and third-order Greeks
    - Greeks sensitivity to underlying factors
    - Stress testing of Greeks under various scenarios
    - Greeks term structure analysis
    """

    def __init__(
        self,
        spot_shock_range: tuple[float, float] = (-0.20, 0.20),
        vol_shock_range: tuple[float, float] = (-0.50, 0.50),
        rate_shock_range: tuple[float, float] = (-0.02, 0.02),
        n_scenarios: int = 21,
    ):
        self.spot_shock_range = spot_shock_range
        self.vol_shock_range = vol_shock_range
        self.rate_shock_range = rate_shock_range
        self.n_scenarios = n_scenarios

        # Position data
        self._positions: dict[str, dict] = {}  # symbol -> position data

    def update_position(
        self,
        symbol: str,
        position: dict,
    ) -> None:
        """Update position for Greeks analysis."""
        self._positions[symbol] = position

    def calculate_delta_sensitivity(
        self,
        portfolio_delta: float,
        portfolio_gamma: float,
        spot_price: float,
    ) -> GreeksSensitivity:
        """
        Calculate delta sensitivity to spot price changes (#R15).

        Shows how delta changes across spot price scenarios.
        """
        spot_shocks = np.linspace(
            self.spot_shock_range[0],
            self.spot_shock_range[1],
            self.n_scenarios
        )

        sensitivities = {}
        stress_values = {}

        for shock in spot_shocks:
            shocked_spot = spot_price * (1 + shock)
            # Delta change = Gamma * dS
            delta_change = portfolio_gamma * spot_price * shock
            new_delta = portfolio_delta + delta_change

            label = f"spot_{shock*100:+.0f}pct"
            sensitivities[label] = delta_change
            stress_values[label] = new_delta

        # 95% confidence interval (approximate)
        delta_std = abs(portfolio_gamma * spot_price * 0.02)  # 2% daily move
        ci = (portfolio_delta - 1.96 * delta_std, portfolio_delta + 1.96 * delta_std)

        return GreeksSensitivity(
            greek="delta",
            base_value=portfolio_delta,
            sensitivities=sensitivities,
            stress_values=stress_values,
            confidence_interval=ci,
        )

    def calculate_gamma_sensitivity(
        self,
        portfolio_gamma: float,
        portfolio_speed: float,
        spot_price: float,
    ) -> GreeksSensitivity:
        """
        Calculate gamma sensitivity to spot price changes (#R15).

        Speed = dGamma/dSpot
        """
        spot_shocks = np.linspace(
            self.spot_shock_range[0],
            self.spot_shock_range[1],
            self.n_scenarios
        )

        sensitivities = {}
        stress_values = {}

        for shock in spot_shocks:
            # Gamma change = Speed * dS
            gamma_change = portfolio_speed * spot_price * shock
            new_gamma = portfolio_gamma + gamma_change

            label = f"spot_{shock*100:+.0f}pct"
            sensitivities[label] = gamma_change
            stress_values[label] = new_gamma

        return GreeksSensitivity(
            greek="gamma",
            base_value=portfolio_gamma,
            sensitivities=sensitivities,
            stress_values=stress_values,
            confidence_interval=(portfolio_gamma * 0.8, portfolio_gamma * 1.2),
        )

    def calculate_vega_sensitivity(
        self,
        portfolio_vega: float,
        portfolio_volga: float,
        current_vol: float,
    ) -> GreeksSensitivity:
        """
        Calculate vega sensitivity to volatility changes (#R15).

        Volga = dVega/dVol
        """
        vol_shocks = np.linspace(
            self.vol_shock_range[0],
            self.vol_shock_range[1],
            self.n_scenarios
        )

        sensitivities = {}
        stress_values = {}

        for shock in vol_shocks:
            # Vega change = Volga * dVol
            vega_change = portfolio_volga * shock
            new_vega = portfolio_vega + vega_change

            label = f"vol_{shock*100:+.0f}pct"
            sensitivities[label] = vega_change
            stress_values[label] = new_vega

        return GreeksSensitivity(
            greek="vega",
            base_value=portfolio_vega,
            sensitivities=sensitivities,
            stress_values=stress_values,
            confidence_interval=(portfolio_vega * 0.5, portfolio_vega * 1.5),
        )

    def calculate_cross_greek_sensitivity(
        self,
        portfolio_delta: float,
        portfolio_vanna: float,
        spot_price: float,
        current_vol: float,
    ) -> dict:
        """
        Calculate cross-Greek sensitivities (delta-vol interaction) (#R15).

        Vanna = dDelta/dVol = dVega/dSpot
        """
        spot_shocks = np.linspace(-0.10, 0.10, 5)
        vol_shocks = np.linspace(-0.20, 0.20, 5)

        surface = {}

        for spot_shock in spot_shocks:
            for vol_shock in vol_shocks:
                # Combined effect through vanna
                delta_from_vol = portfolio_vanna * vol_shock
                new_delta = portfolio_delta + delta_from_vol

                key = f"spot_{spot_shock*100:+.0f}_vol_{vol_shock*100:+.0f}"
                surface[key] = {
                    'delta': new_delta,
                    'delta_change': delta_from_vol,
                    'spot_shock': spot_shock,
                    'vol_shock': vol_shock,
                }

        return {
            'cross_sensitivity': 'delta_vol',
            'vanna': portfolio_vanna,
            'surface': surface,
        }

    def run_full_sensitivity_analysis(
        self,
        portfolio_greeks: dict,
        spot_price: float,
        current_vol: float,
    ) -> dict:
        """
        Run comprehensive sensitivity analysis on all Greeks (#R15).

        Returns full sensitivity report.
        """
        delta = portfolio_greeks.get('delta', 0)
        gamma = portfolio_greeks.get('gamma', 0)
        vega = portfolio_greeks.get('vega', 0)
        theta = portfolio_greeks.get('theta', 0)
        rho = portfolio_greeks.get('rho', 0)
        vanna = portfolio_greeks.get('vanna', 0)
        volga = portfolio_greeks.get('volga', 0)
        speed = portfolio_greeks.get('speed', 0)
        charm = portfolio_greeks.get('charm', 0)

        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'base_greeks': portfolio_greeks,
            'spot_price': spot_price,
            'current_vol': current_vol,
            'sensitivities': {},
        }

        # Delta sensitivity
        delta_sens = self.calculate_delta_sensitivity(delta, gamma, spot_price)
        results['sensitivities']['delta'] = delta_sens.to_dict()

        # Gamma sensitivity
        gamma_sens = self.calculate_gamma_sensitivity(gamma, speed, spot_price)
        results['sensitivities']['gamma'] = gamma_sens.to_dict()

        # Vega sensitivity
        vega_sens = self.calculate_vega_sensitivity(vega, volga, current_vol)
        results['sensitivities']['vega'] = vega_sens.to_dict()

        # Cross sensitivities
        cross = self.calculate_cross_greek_sensitivity(delta, vanna, spot_price, current_vol)
        results['sensitivities']['cross_delta_vol'] = cross

        # Summary metrics
        results['summary'] = {
            'max_delta_at_risk': max(abs(v) for v in delta_sens.stress_values.values()),
            'max_gamma_at_risk': max(abs(v) for v in gamma_sens.stress_values.values()),
            'vega_to_vol_ratio': vega / current_vol if current_vol > 0 else 0,
            'theta_to_gamma_ratio': theta / gamma if gamma != 0 else 0,
        }

        return results


# =============================================================================
# RISK EXPOSURE TRENDING/FORECASTING (#R20)
# =============================================================================

@dataclass
class RiskExposureForecast:
    """Forecasted risk exposure (#R20)."""
    metric: str
    current_value: float
    forecast_1d: float
    forecast_5d: float
    forecast_10d: float
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float
    factors: dict[str, float]  # Contributing factors

    def to_dict(self) -> dict:
        return {
            'metric': self.metric,
            'current_value': self.current_value,
            'forecast_1d': self.forecast_1d,
            'forecast_5d': self.forecast_5d,
            'forecast_10d': self.forecast_10d,
            'trend': self.trend,
            'confidence': self.confidence,
            'factors': self.factors,
        }


class RiskExposureTrending:
    """
    Risk exposure trending and forecasting (#R20).

    Tracks and forecasts:
    - VaR trends
    - Greeks evolution
    - Exposure concentration
    - Correlation regime
    """

    def __init__(
        self,
        lookback_days: int = 60,
        forecast_horizon: int = 10,
    ):
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon

        # Historical data
        self._var_history: list[tuple[datetime, float]] = []
        self._delta_history: list[tuple[datetime, float]] = []
        self._vega_history: list[tuple[datetime, float]] = []
        self._concentration_history: list[tuple[datetime, float]] = []

    def record_var(self, var_value: float) -> None:
        """Record daily VaR observation."""
        self._var_history.append((datetime.now(timezone.utc), var_value))
        self._var_history = self._var_history[-self.lookback_days:]

    def record_greeks(self, delta: float, vega: float) -> None:
        """Record daily Greeks snapshot."""
        now = datetime.now(timezone.utc)
        self._delta_history.append((now, delta))
        self._vega_history.append((now, vega))
        self._delta_history = self._delta_history[-self.lookback_days:]
        self._vega_history = self._vega_history[-self.lookback_days:]

    def record_concentration(self, hhi: float) -> None:
        """Record portfolio concentration (HHI)."""
        self._concentration_history.append((datetime.now(timezone.utc), hhi))
        self._concentration_history = self._concentration_history[-self.lookback_days:]

    def _fit_trend(self, data: list[tuple[datetime, float]]) -> tuple[float, float, float]:
        """
        Fit linear trend to time series.

        Returns: (slope, intercept, r_squared)
        """
        if len(data) < 5:
            return 0.0, 0.0, 0.0

        values = [v for _, v in data]
        x = np.arange(len(values))

        slope, intercept, r_value, _, _ = stats.linregress(x, values)
        return slope, intercept, r_value ** 2

    def _forecast_value(
        self,
        data: list[tuple[datetime, float]],
        horizon_days: int,
    ) -> tuple[float, float]:
        """
        Forecast value using exponential smoothing.

        Returns: (forecast, confidence)
        """
        if len(data) < 3:
            return data[-1][1] if data else 0.0, 0.0

        values = [v for _, v in data]

        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        smoothed = values[0]
        for v in values[1:]:
            smoothed = alpha * v + (1 - alpha) * smoothed

        # Trend component
        slope, _, r2 = self._fit_trend(data)

        # Forecast with trend
        forecast = smoothed + slope * horizon_days

        # Confidence based on R-squared and data length
        confidence = min(0.95, r2 * min(1.0, len(data) / 30))

        return forecast, confidence

    def forecast_var(self) -> RiskExposureForecast:
        """
        Forecast VaR trend (#R20).

        Returns VaR forecast with trend analysis.
        """
        if not self._var_history:
            return RiskExposureForecast(
                metric="var",
                current_value=0,
                forecast_1d=0,
                forecast_5d=0,
                forecast_10d=0,
                trend="unknown",
                confidence=0,
                factors={},
            )

        current = self._var_history[-1][1]

        f1d, c1 = self._forecast_value(self._var_history, 1)
        f5d, c5 = self._forecast_value(self._var_history, 5)
        f10d, c10 = self._forecast_value(self._var_history, 10)

        slope, _, r2 = self._fit_trend(self._var_history)

        if slope > 0.01 * current:
            trend = "increasing"
        elif slope < -0.01 * current:
            trend = "decreasing"
        else:
            trend = "stable"

        # Identify contributing factors
        factors = {}
        if len(self._delta_history) >= 5:
            delta_slope, _, _ = self._fit_trend(self._delta_history)
            factors['delta_trend'] = delta_slope

        if len(self._vega_history) >= 5:
            vega_slope, _, _ = self._fit_trend(self._vega_history)
            factors['vega_trend'] = vega_slope

        if len(self._concentration_history) >= 5:
            conc_slope, _, _ = self._fit_trend(self._concentration_history)
            factors['concentration_trend'] = conc_slope

        return RiskExposureForecast(
            metric="var",
            current_value=current,
            forecast_1d=f1d,
            forecast_5d=f5d,
            forecast_10d=f10d,
            trend=trend,
            confidence=(c1 + c5 + c10) / 3,
            factors=factors,
        )

    def get_trend_dashboard(self) -> dict:
        """
        Get complete trend dashboard (#R20).

        Returns trends and forecasts for all risk metrics.
        """
        dashboard = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': {},
        }

        # VaR forecast
        var_forecast = self.forecast_var()
        dashboard['metrics']['var'] = var_forecast.to_dict()

        # Delta trend
        if self._delta_history:
            slope, _, r2 = self._fit_trend(self._delta_history)
            dashboard['metrics']['delta'] = {
                'current': self._delta_history[-1][1],
                'trend_slope': slope,
                'trend_strength': r2,
            }

        # Vega trend
        if self._vega_history:
            slope, _, r2 = self._fit_trend(self._vega_history)
            dashboard['metrics']['vega'] = {
                'current': self._vega_history[-1][1],
                'trend_slope': slope,
                'trend_strength': r2,
            }

        # Concentration trend
        if self._concentration_history:
            slope, _, r2 = self._fit_trend(self._concentration_history)
            dashboard['metrics']['concentration'] = {
                'current': self._concentration_history[-1][1],
                'trend_slope': slope,
                'trend_strength': r2,
            }

        # Overall risk assessment
        alerts = []
        if dashboard['metrics'].get('var', {}).get('trend') == 'increasing':
            alerts.append("VaR trend increasing - review risk exposure")
        if dashboard['metrics'].get('concentration', {}).get('trend_slope', 0) > 0.01:
            alerts.append("Portfolio concentration increasing")

        dashboard['alerts'] = alerts

        return dashboard


# =============================================================================
# P&L ATTRIBUTION BY RISK FACTOR (#R21)
# =============================================================================

@dataclass
class PnLAttribution:
    """P&L attribution by risk factor (#R21)."""
    period_start: datetime
    period_end: datetime
    total_pnl: float

    # Factor contributions
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    vega_pnl: float = 0.0
    theta_pnl: float = 0.0
    rho_pnl: float = 0.0

    # Cross-effects
    cross_gamma_pnl: float = 0.0  # Higher-order delta
    vanna_pnl: float = 0.0  # Delta-vol cross
    volga_pnl: float = 0.0  # Vega-vol cross

    # Residual
    residual_pnl: float = 0.0

    # By underlying
    by_underlying: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        attributed = (
            self.delta_pnl + self.gamma_pnl + self.vega_pnl +
            self.theta_pnl + self.rho_pnl + self.cross_gamma_pnl +
            self.vanna_pnl + self.volga_pnl
        )
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_pnl': self.total_pnl,
            'factors': {
                'delta': self.delta_pnl,
                'gamma': self.gamma_pnl,
                'vega': self.vega_pnl,
                'theta': self.theta_pnl,
                'rho': self.rho_pnl,
                'cross_gamma': self.cross_gamma_pnl,
                'vanna': self.vanna_pnl,
                'volga': self.volga_pnl,
            },
            'residual': self.residual_pnl,
            'attribution_pct': {
                'delta': (self.delta_pnl / self.total_pnl * 100) if self.total_pnl != 0 else 0,
                'gamma': (self.gamma_pnl / self.total_pnl * 100) if self.total_pnl != 0 else 0,
                'vega': (self.vega_pnl / self.total_pnl * 100) if self.total_pnl != 0 else 0,
                'theta': (self.theta_pnl / self.total_pnl * 100) if self.total_pnl != 0 else 0,
                'residual': (self.residual_pnl / self.total_pnl * 100) if self.total_pnl != 0 else 0,
            },
            'explained_pct': (attributed / self.total_pnl * 100) if self.total_pnl != 0 else 0,
            'by_underlying': self.by_underlying,
        }


class PnLAttributionEngine:
    """
    P&L attribution by risk factor (#R21).

    Decomposes P&L into:
    - First-order Greeks (delta, vega, theta, rho)
    - Second-order effects (gamma, vanna, volga)
    - Cross-effects and residual
    """

    def __init__(self):
        self._greeks_snapshots: list[tuple[datetime, dict]] = []
        self._market_snapshots: list[tuple[datetime, dict]] = []

    def record_snapshot(
        self,
        greeks: dict,
        market_data: dict,
    ) -> None:
        """
        Record Greeks and market snapshot for attribution.

        Args:
            greeks: Portfolio Greeks {delta, gamma, vega, theta, rho, vanna, volga}
            market_data: Market data {spot_prices, vols, rates}
        """
        now = datetime.now(timezone.utc)
        self._greeks_snapshots.append((now, greeks))
        self._market_snapshots.append((now, market_data))

        # Keep last 100 snapshots
        self._greeks_snapshots = self._greeks_snapshots[-100:]
        self._market_snapshots = self._market_snapshots[-100:]

    def attribute_pnl(
        self,
        total_pnl: float,
        start_greeks: dict,
        end_greeks: dict,
        start_market: dict,
        end_market: dict,
    ) -> PnLAttribution:
        """
        Attribute P&L to risk factors (#R21).

        Uses Taylor series expansion for attribution.
        """
        period_start = datetime.now(timezone.utc) - timedelta(days=1)
        period_end = datetime.now(timezone.utc)

        # Extract data
        delta = start_greeks.get('delta', 0)
        gamma = start_greeks.get('gamma', 0)
        vega = start_greeks.get('vega', 0)
        theta = start_greeks.get('theta', 0)
        rho = start_greeks.get('rho', 0)
        vanna = start_greeks.get('vanna', 0)
        volga = start_greeks.get('volga', 0)

        # Calculate market changes
        spot_change = end_market.get('spot', 100) - start_market.get('spot', 100)
        spot_change_pct = spot_change / start_market.get('spot', 100)
        vol_change = end_market.get('vol', 0.20) - start_market.get('vol', 0.20)
        rate_change = end_market.get('rate', 0.05) - start_market.get('rate', 0.05)
        time_decay = 1 / 365  # Daily

        # First-order attribution
        delta_pnl = delta * spot_change
        vega_pnl = vega * vol_change * 100  # Vega is per 1% vol
        theta_pnl = theta * time_decay * 365  # Theta is daily
        rho_pnl = rho * rate_change * 100  # Rho is per 1% rate

        # Second-order attribution
        gamma_pnl = 0.5 * gamma * spot_change ** 2
        cross_gamma_pnl = gamma_pnl  # Already captured in gamma

        # Cross-effects
        vanna_pnl = vanna * spot_change * vol_change
        volga_pnl = 0.5 * volga * vol_change ** 2

        # Residual
        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl + vanna_pnl + volga_pnl
        residual_pnl = total_pnl - explained

        return PnLAttribution(
            period_start=period_start,
            period_end=period_end,
            total_pnl=total_pnl,
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            vega_pnl=vega_pnl,
            theta_pnl=theta_pnl,
            rho_pnl=rho_pnl,
            cross_gamma_pnl=0,  # Included in gamma_pnl
            vanna_pnl=vanna_pnl,
            volga_pnl=volga_pnl,
            residual_pnl=residual_pnl,
        )

    def get_attribution_report(
        self,
        lookback_days: int = 5,
    ) -> dict:
        """
        Generate P&L attribution report (#R21).

        Provides cumulative attribution over period.
        """
        if len(self._greeks_snapshots) < 2:
            return {'error': 'insufficient_data'}

        report = {
            'lookback_days': lookback_days,
            'daily_attributions': [],
            'cumulative': {
                'delta_pnl': 0,
                'gamma_pnl': 0,
                'vega_pnl': 0,
                'theta_pnl': 0,
                'residual_pnl': 0,
            },
        }

        # Process pairs of snapshots
        for i in range(1, min(len(self._greeks_snapshots), lookback_days + 1)):
            start_greeks = self._greeks_snapshots[-(i+1)][1]
            end_greeks = self._greeks_snapshots[-i][1]
            start_market = self._market_snapshots[-(i+1)][1]
            end_market = self._market_snapshots[-i][1]

            # Estimate daily P&L (would come from actual P&L in production)
            daily_pnl = end_greeks.get('portfolio_value', 0) - start_greeks.get('portfolio_value', 0)

            attr = self.attribute_pnl(
                daily_pnl,
                start_greeks,
                end_greeks,
                start_market,
                end_market,
            )

            report['daily_attributions'].append(attr.to_dict())

            # Accumulate
            report['cumulative']['delta_pnl'] += attr.delta_pnl
            report['cumulative']['gamma_pnl'] += attr.gamma_pnl
            report['cumulative']['vega_pnl'] += attr.vega_pnl
            report['cumulative']['theta_pnl'] += attr.theta_pnl
            report['cumulative']['residual_pnl'] += attr.residual_pnl

        return report


# =============================================================================
# SCENARIO-SPECIFIC POSITION LIMITS (#R24)
# =============================================================================

class ScenarioType(str, Enum):
    """Market scenario types for position limits (#R24)."""
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    EARNINGS = "earnings"
    FOMC = "fomc"
    EXPIRATION = "expiration"


@dataclass
class ScenarioPositionLimit:
    """Position limit for a specific scenario (#R24)."""
    scenario: ScenarioType
    symbol: str
    max_position: int
    max_notional: float
    max_delta: float
    max_gamma: float
    max_vega: float
    rationale: str


@dataclass
class ScenarioLimitCheck:
    """Result of scenario limit check (#R24)."""
    scenario: ScenarioType
    symbol: str
    current_position: int
    current_notional: float
    limit_position: int
    limit_notional: float
    is_within_limit: bool
    utilization_pct: float
    warning_message: str | None


class ScenarioPositionLimits:
    """
    Scenario-specific position limits (#R24).

    Different limits apply under different market conditions:
    - Normal: Standard limits
    - High Vol: Reduced limits when VIX elevated
    - Crisis: Severely reduced limits during market stress
    - Earnings: Reduced limits around earnings
    - FOMC: Reduced limits around Fed meetings
    - Expiration: Reduced limits near option expiry
    """

    # Default limit multipliers by scenario
    SCENARIO_MULTIPLIERS: dict[ScenarioType, float] = {
        ScenarioType.NORMAL: 1.0,
        ScenarioType.HIGH_VOL: 0.7,
        ScenarioType.CRISIS: 0.3,
        ScenarioType.EARNINGS: 0.5,
        ScenarioType.FOMC: 0.6,
        ScenarioType.EXPIRATION: 0.4,
    }

    def __init__(self):
        # Base limits by symbol
        self._base_limits: dict[str, dict] = {}

        # Custom scenario limits
        self._scenario_limits: dict[tuple[ScenarioType, str], ScenarioPositionLimit] = {}

        # Current scenario
        self._current_scenario = ScenarioType.NORMAL

        # Scenario detection thresholds
        self._vix_high_threshold = 25.0
        self._vix_crisis_threshold = 40.0

    def set_base_limit(
        self,
        symbol: str,
        max_position: int,
        max_notional: float,
        max_delta: float = float('inf'),
        max_gamma: float = float('inf'),
        max_vega: float = float('inf'),
    ) -> None:
        """Set base position limits for a symbol."""
        self._base_limits[symbol] = {
            'max_position': max_position,
            'max_notional': max_notional,
            'max_delta': max_delta,
            'max_gamma': max_gamma,
            'max_vega': max_vega,
        }

    def set_scenario_limit(
        self,
        scenario: ScenarioType,
        symbol: str,
        max_position: int,
        max_notional: float,
        max_delta: float = float('inf'),
        max_gamma: float = float('inf'),
        max_vega: float = float('inf'),
        rationale: str = "",
    ) -> None:
        """Set custom limit for specific scenario (#R24)."""
        self._scenario_limits[(scenario, symbol)] = ScenarioPositionLimit(
            scenario=scenario,
            symbol=symbol,
            max_position=max_position,
            max_notional=max_notional,
            max_delta=max_delta,
            max_gamma=max_gamma,
            max_vega=max_vega,
            rationale=rationale,
        )

    def detect_scenario(
        self,
        vix_level: float = 20.0,
        is_earnings: bool = False,
        is_fomc: bool = False,
        days_to_expiry: int | None = None,
    ) -> ScenarioType:
        """
        Detect current market scenario (#R24).

        Args:
            vix_level: Current VIX level
            is_earnings: Whether in earnings period
            is_fomc: Whether FOMC meeting today/tomorrow
            days_to_expiry: Days to option expiry

        Returns:
            Detected scenario type
        """
        # Priority: Crisis > High Vol > Expiration > FOMC > Earnings > Normal

        if vix_level >= self._vix_crisis_threshold:
            return ScenarioType.CRISIS

        if vix_level >= self._vix_high_threshold:
            return ScenarioType.HIGH_VOL

        if days_to_expiry is not None and days_to_expiry <= 2:
            return ScenarioType.EXPIRATION

        if is_fomc:
            return ScenarioType.FOMC

        if is_earnings:
            return ScenarioType.EARNINGS

        return ScenarioType.NORMAL

    def get_effective_limit(
        self,
        symbol: str,
        scenario: ScenarioType | None = None,
    ) -> dict:
        """
        Get effective position limit for symbol under scenario (#R24).

        Args:
            symbol: Symbol to get limit for
            scenario: Scenario (uses current if not specified)

        Returns:
            Effective limits
        """
        if scenario is None:
            scenario = self._current_scenario

        # Check for custom scenario limit
        custom_limit = self._scenario_limits.get((scenario, symbol))
        if custom_limit:
            return {
                'max_position': custom_limit.max_position,
                'max_notional': custom_limit.max_notional,
                'max_delta': custom_limit.max_delta,
                'max_gamma': custom_limit.max_gamma,
                'max_vega': custom_limit.max_vega,
                'source': 'custom',
                'scenario': scenario.value,
            }

        # Apply multiplier to base limits
        base = self._base_limits.get(symbol)
        if not base:
            return {
                'max_position': float('inf'),
                'max_notional': float('inf'),
                'max_delta': float('inf'),
                'max_gamma': float('inf'),
                'max_vega': float('inf'),
                'source': 'none',
                'scenario': scenario.value,
            }

        multiplier = self.SCENARIO_MULTIPLIERS.get(scenario, 1.0)

        return {
            'max_position': int(base['max_position'] * multiplier),
            'max_notional': base['max_notional'] * multiplier,
            'max_delta': base['max_delta'] * multiplier,
            'max_gamma': base['max_gamma'] * multiplier,
            'max_vega': base['max_vega'] * multiplier,
            'source': 'base_with_multiplier',
            'multiplier': multiplier,
            'scenario': scenario.value,
        }

    def check_limit(
        self,
        symbol: str,
        current_position: int,
        current_notional: float,
        current_delta: float = 0,
        current_gamma: float = 0,
        current_vega: float = 0,
        scenario: ScenarioType | None = None,
    ) -> ScenarioLimitCheck:
        """
        Check if position is within scenario limits (#R24).

        Returns limit check result with utilization.
        """
        limits = self.get_effective_limit(symbol, scenario)

        # Check each limit
        position_ok = abs(current_position) <= limits['max_position']
        notional_ok = abs(current_notional) <= limits['max_notional']
        delta_ok = abs(current_delta) <= limits['max_delta']
        gamma_ok = abs(current_gamma) <= limits['max_gamma']
        vega_ok = abs(current_vega) <= limits['max_vega']

        is_within_limit = position_ok and notional_ok and delta_ok and gamma_ok and vega_ok

        # Calculate utilization
        utilization = max(
            abs(current_position) / limits['max_position'] if limits['max_position'] != float('inf') else 0,
            abs(current_notional) / limits['max_notional'] if limits['max_notional'] != float('inf') else 0,
        )

        # Warning message
        warning = None
        if not is_within_limit:
            warnings = []
            if not position_ok:
                warnings.append(f"position {current_position} > limit {limits['max_position']}")
            if not notional_ok:
                warnings.append(f"notional ${current_notional:,.0f} > limit ${limits['max_notional']:,.0f}")
            if not delta_ok:
                warnings.append(f"delta {current_delta:.0f} > limit {limits['max_delta']:.0f}")
            warning = "; ".join(warnings)
        elif utilization > 0.8:
            warning = f"Utilization at {utilization*100:.0f}% - approaching limit"

        return ScenarioLimitCheck(
            scenario=scenario or self._current_scenario,
            symbol=symbol,
            current_position=current_position,
            current_notional=current_notional,
            limit_position=limits['max_position'],
            limit_notional=limits['max_notional'],
            is_within_limit=is_within_limit,
            utilization_pct=utilization * 100,
            warning_message=warning,
        )

    def update_scenario(
        self,
        vix_level: float,
        is_earnings: bool = False,
        is_fomc: bool = False,
        days_to_expiry: int | None = None,
    ) -> ScenarioType:
        """Update current scenario based on market conditions."""
        new_scenario = self.detect_scenario(vix_level, is_earnings, is_fomc, days_to_expiry)

        if new_scenario != self._current_scenario:
            logger.info(
                f"Scenario changed from {self._current_scenario.value} to {new_scenario.value}"
            )
            self._current_scenario = new_scenario

        return self._current_scenario

    def get_status(self) -> dict:
        """Get position limits status."""
        return {
            'current_scenario': self._current_scenario.value,
            'base_limits_configured': len(self._base_limits),
            'custom_scenario_limits': len(self._scenario_limits),
            'scenario_multipliers': {k.value: v for k, v in self.SCENARIO_MULTIPLIERS.items()},
        }

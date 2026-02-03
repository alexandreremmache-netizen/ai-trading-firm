"""
Monte Carlo Simulation Module
=============================

P2 Enhancement: Comprehensive Monte Carlo simulation for risk analysis.

Features:
- Path-dependent simulation
- Regime-switching models
- Correlation stress scenarios
- Portfolio path simulation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification."""
    LOW_VOL = "low_vol"  # VIX < 15
    NORMAL = "normal"  # VIX 15-20
    ELEVATED = "elevated"  # VIX 20-30
    HIGH_VOL = "high_vol"  # VIX 30-40
    CRISIS = "crisis"  # VIX > 40


@dataclass
class RegimeParameters:
    """Parameters for a market regime."""
    regime: MarketRegime
    drift_annual: float  # Annual drift
    volatility_annual: float  # Annual volatility
    correlation_adjustment: float  # Correlation shift from baseline
    jump_probability: float  # Daily probability of jump
    jump_mean: float  # Mean jump size
    jump_std: float  # Jump size std dev

    @classmethod
    def default_regimes(cls) -> dict[MarketRegime, "RegimeParameters"]:
        """Get default regime parameters calibrated to historical data."""
        return {
            MarketRegime.LOW_VOL: cls(
                regime=MarketRegime.LOW_VOL,
                drift_annual=0.12,
                volatility_annual=0.10,
                correlation_adjustment=-0.1,
                jump_probability=0.001,
                jump_mean=-0.01,
                jump_std=0.02,
            ),
            MarketRegime.NORMAL: cls(
                regime=MarketRegime.NORMAL,
                drift_annual=0.08,
                volatility_annual=0.16,
                correlation_adjustment=0.0,
                jump_probability=0.005,
                jump_mean=-0.015,
                jump_std=0.025,
            ),
            MarketRegime.ELEVATED: cls(
                regime=MarketRegime.ELEVATED,
                drift_annual=0.04,
                volatility_annual=0.22,
                correlation_adjustment=0.1,
                jump_probability=0.01,
                jump_mean=-0.02,
                jump_std=0.03,
            ),
            MarketRegime.HIGH_VOL: cls(
                regime=MarketRegime.HIGH_VOL,
                drift_annual=-0.05,
                volatility_annual=0.35,
                correlation_adjustment=0.2,
                jump_probability=0.02,
                jump_mean=-0.03,
                jump_std=0.04,
            ),
            MarketRegime.CRISIS: cls(
                regime=MarketRegime.CRISIS,
                drift_annual=-0.20,
                volatility_annual=0.60,
                correlation_adjustment=0.4,
                jump_probability=0.05,
                jump_mean=-0.05,
                jump_std=0.06,
            ),
        }


@dataclass
class TransitionMatrix:
    """Regime transition probabilities (daily)."""
    matrix: np.ndarray = field(default_factory=lambda: np.array([
        # LOW_VOL  NORMAL  ELEVATED  HIGH_VOL  CRISIS
        [0.95,     0.04,   0.01,     0.00,     0.00],  # From LOW_VOL
        [0.03,     0.90,   0.06,     0.01,     0.00],  # From NORMAL
        [0.01,     0.10,   0.80,     0.08,     0.01],  # From ELEVATED
        [0.00,     0.02,   0.15,     0.75,     0.08],  # From HIGH_VOL
        [0.00,     0.00,   0.05,     0.20,     0.75],  # From CRISIS
    ]))

    def get_next_regime(
        self,
        current_regime: MarketRegime,
        rng: np.random.Generator,
    ) -> MarketRegime:
        """Sample next regime based on transition probabilities."""
        regime_order = list(MarketRegime)
        current_idx = regime_order.index(current_regime)
        probs = self.matrix[current_idx]
        next_idx = rng.choice(len(regime_order), p=probs)
        return regime_order[next_idx]


@dataclass
class CorrelationStressScenario:
    """Correlation stress scenario definition."""
    name: str
    description: str
    correlation_multiplier: float  # Multiply all correlations
    correlation_floor: float  # Minimum correlation
    asset_specific_adjustments: dict[str, float] = field(default_factory=dict)

    @classmethod
    def standard_scenarios(cls) -> dict[str, "CorrelationStressScenario"]:
        """Get standard correlation stress scenarios."""
        return {
            "normal": cls(
                name="Normal",
                description="Normal market conditions",
                correlation_multiplier=1.0,
                correlation_floor=-1.0,
            ),
            "risk_on": cls(
                name="Risk On",
                description="Low correlations, diversification works",
                correlation_multiplier=0.5,
                correlation_floor=-0.5,
            ),
            "risk_off": cls(
                name="Risk Off",
                description="Flight to quality, correlations spike",
                correlation_multiplier=1.5,
                correlation_floor=0.3,
                asset_specific_adjustments={
                    "treasury": -0.3,  # Treasuries become negative correlated
                    "gold": -0.2,
                },
            ),
            "crisis": cls(
                name="Crisis",
                description="Correlations go to 1",
                correlation_multiplier=2.0,
                correlation_floor=0.6,
                asset_specific_adjustments={
                    "treasury": -0.5,
                    "gold": -0.3,
                },
            ),
            "decorrelation": cls(
                name="Decorrelation Shock",
                description="Historical correlations break down",
                correlation_multiplier=0.2,
                correlation_floor=-1.0,
            ),
        }


@dataclass
class SimulationPath:
    """Single simulation path result."""
    path_id: int
    timestamps: list[datetime]
    values: np.ndarray  # Portfolio values over time
    returns: np.ndarray  # Daily returns
    regimes: list[MarketRegime]  # Regime at each point

    # Summary statistics
    final_value: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0

    def calculate_statistics(self, risk_free_rate: float = 0.02) -> None:
        """Calculate summary statistics for this path."""
        self.final_value = self.values[-1]
        self.total_return = (self.final_value / self.values[0]) - 1

        # Max drawdown
        peak = self.values[0]
        max_dd = 0.0
        for val in self.values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        self.max_drawdown = max_dd

        # Annualized volatility
        if len(self.returns) > 1:
            self.volatility = float(np.std(self.returns) * np.sqrt(252))

        # Sharpe ratio
        if self.volatility > 0:
            excess_return = self.total_return - risk_free_rate * len(self.returns) / 252
            self.sharpe = excess_return / self.volatility

    def to_dict(self) -> dict:
        return {
            'path_id': self.path_id,
            'final_value': self.final_value,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe': self.sharpe,
            'num_days': len(self.values),
        }


@dataclass
class MonteCarloResult:
    """Aggregate results from Monte Carlo simulation."""
    num_paths: int
    num_days: int
    initial_value: float
    paths: list[SimulationPath]

    # Distribution statistics
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0
    var_95: float = 0.0  # 95% VaR (5th percentile)
    var_99: float = 0.0  # 99% VaR (1st percentile)
    cvar_95: float = 0.0  # Expected shortfall at 95%
    cvar_99: float = 0.0

    # Drawdown statistics
    mean_max_drawdown: float = 0.0
    median_max_drawdown: float = 0.0
    worst_max_drawdown: float = 0.0

    # Probability metrics
    prob_loss: float = 0.0  # P(return < 0)
    prob_loss_10pct: float = 0.0  # P(return < -10%)
    prob_gain_10pct: float = 0.0  # P(return > 10%)

    # Regime statistics
    regime_distribution: dict[MarketRegime, float] = field(default_factory=dict)

    def calculate_statistics(self) -> None:
        """Calculate aggregate statistics from paths."""
        if not self.paths:
            return

        returns = np.array([p.total_return for p in self.paths])
        drawdowns = np.array([p.max_drawdown for p in self.paths])

        # Return distribution
        self.mean_return = float(np.mean(returns))
        self.median_return = float(np.median(returns))
        self.std_return = float(np.std(returns))

        # VaR and CVaR
        sorted_returns = np.sort(returns)
        var_95_idx = int(0.05 * len(sorted_returns))
        var_99_idx = int(0.01 * len(sorted_returns))

        self.var_95 = -float(sorted_returns[var_95_idx])
        self.var_99 = -float(sorted_returns[var_99_idx])
        self.cvar_95 = -float(np.mean(sorted_returns[:var_95_idx + 1]))
        self.cvar_99 = -float(np.mean(sorted_returns[:var_99_idx + 1]))

        # Drawdown statistics
        self.mean_max_drawdown = float(np.mean(drawdowns))
        self.median_max_drawdown = float(np.median(drawdowns))
        self.worst_max_drawdown = float(np.max(drawdowns))

        # Probability metrics
        self.prob_loss = float(np.mean(returns < 0))
        self.prob_loss_10pct = float(np.mean(returns < -0.10))
        self.prob_gain_10pct = float(np.mean(returns > 0.10))

        # Regime distribution
        all_regimes = []
        for path in self.paths:
            all_regimes.extend(path.regimes)

        if all_regimes:
            for regime in MarketRegime:
                count = sum(1 for r in all_regimes if r == regime)
                self.regime_distribution[regime] = count / len(all_regimes)

    def to_dict(self) -> dict:
        return {
            'num_paths': self.num_paths,
            'num_days': self.num_days,
            'initial_value': self.initial_value,
            'returns': {
                'mean': self.mean_return,
                'median': self.median_return,
                'std': self.std_return,
            },
            'var': {
                'var_95': self.var_95,
                'var_99': self.var_99,
                'cvar_95': self.cvar_95,
                'cvar_99': self.cvar_99,
            },
            'drawdown': {
                'mean': self.mean_max_drawdown,
                'median': self.median_max_drawdown,
                'worst': self.worst_max_drawdown,
            },
            'probabilities': {
                'prob_loss': self.prob_loss,
                'prob_loss_10pct': self.prob_loss_10pct,
                'prob_gain_10pct': self.prob_gain_10pct,
            },
            'regime_distribution': {
                r.value: p for r, p in self.regime_distribution.items()
            },
        }


class PathDependentSimulator:
    """
    Path-dependent Monte Carlo simulator (P2 Enhancement).

    Simulates paths where future behavior depends on past path history.
    Supports:
    - Barrier conditions (stop-loss, take-profit)
    - Drawdown-based deleveraging
    - Path-dependent volatility (GARCH-like)
    """

    def __init__(
        self,
        initial_value: float = 100_000.0,
        base_drift: float = 0.08,  # 8% annual
        base_volatility: float = 0.16,  # 16% annual
        seed: int | None = None,
    ):
        self.initial_value = initial_value
        self.base_drift = base_drift
        self.base_volatility = base_volatility
        self.rng = np.random.default_rng(seed)

        # Path-dependent features
        self.stop_loss_pct: float | None = None  # Stop if drawdown exceeds
        self.take_profit_pct: float | None = None  # Exit if gain exceeds
        self.deleveraging_threshold: float | None = None  # Reduce exposure on DD
        self.deleveraging_factor: float = 0.5  # Reduce by this factor

        # GARCH-like vol persistence
        self.vol_persistence: float = 0.9  # How much yesterday's vol carries over
        self.vol_innovation_weight: float = 0.1  # Weight on new shocks

    def set_barrier_conditions(
        self,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        """Set barrier conditions for paths."""
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def set_deleveraging(
        self,
        threshold_pct: float,
        factor: float = 0.5,
    ) -> None:
        """Set drawdown-based deleveraging."""
        self.deleveraging_threshold = threshold_pct
        self.deleveraging_factor = factor

    def simulate_path(
        self,
        num_days: int,
        path_id: int = 0,
        start_date: datetime | None = None,
    ) -> SimulationPath:
        """
        Simulate a single path-dependent path.

        Args:
            num_days: Number of trading days to simulate
            path_id: Identifier for this path
            start_date: Starting date for timestamps

        Returns:
            SimulationPath with full path data
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc)

        # Initialize
        values = np.zeros(num_days + 1)
        returns = np.zeros(num_days)
        values[0] = self.initial_value
        timestamps = [start_date]

        # Path-dependent state
        peak_value = self.initial_value
        current_vol = self.base_volatility
        leverage = 1.0
        stopped_out = False

        # Daily simulation
        dt = 1 / 252  # Daily time step

        for day in range(num_days):
            if stopped_out:
                values[day + 1] = values[day]
                returns[day] = 0.0
                timestamps.append(start_date + datetime.timedelta(days=day + 1)
                                  if hasattr(datetime, 'timedelta') else start_date)
                continue

            # Update peak for drawdown calc
            peak_value = max(peak_value, values[day])
            current_drawdown = (peak_value - values[day]) / peak_value

            # Check deleveraging
            if self.deleveraging_threshold and current_drawdown >= self.deleveraging_threshold:
                leverage = self.deleveraging_factor
            else:
                leverage = 1.0

            # GARCH-like volatility update
            if day > 0:
                realized_vol = abs(returns[day - 1]) * np.sqrt(252)
                current_vol = (
                    self.vol_persistence * current_vol +
                    self.vol_innovation_weight * realized_vol +
                    (1 - self.vol_persistence - self.vol_innovation_weight) * self.base_volatility
                )

            # Generate return
            drift_daily = self.base_drift * dt * leverage
            vol_daily = current_vol * np.sqrt(dt) * leverage
            z = self.rng.standard_normal()
            daily_return = drift_daily + vol_daily * z

            returns[day] = daily_return
            values[day + 1] = values[day] * (1 + daily_return)

            # Timestamp
            try:
                from datetime import timedelta
                timestamps.append(start_date + timedelta(days=day + 1))
            except Exception:
                timestamps.append(start_date)

            # Check barrier conditions
            if self.stop_loss_pct:
                drawdown = (peak_value - values[day + 1]) / peak_value
                if drawdown >= self.stop_loss_pct:
                    stopped_out = True

            if self.take_profit_pct:
                gain = (values[day + 1] - self.initial_value) / self.initial_value
                if gain >= self.take_profit_pct:
                    stopped_out = True

        # Create path object
        path = SimulationPath(
            path_id=path_id,
            timestamps=timestamps,
            values=values,
            returns=returns,
            regimes=[MarketRegime.NORMAL] * (num_days + 1),  # No regime switching here
        )
        path.calculate_statistics()

        return path

    def simulate_paths(
        self,
        num_paths: int,
        num_days: int,
        start_date: datetime | None = None,
    ) -> MonteCarloResult:
        """Simulate multiple paths."""
        paths = []
        for i in range(num_paths):
            path = self.simulate_path(num_days, path_id=i, start_date=start_date)
            paths.append(path)

        result = MonteCarloResult(
            num_paths=num_paths,
            num_days=num_days,
            initial_value=self.initial_value,
            paths=paths,
        )
        result.calculate_statistics()

        return result


class RegimeSwitchingSimulator:
    """
    Regime-switching Monte Carlo simulator (P2 Enhancement).

    Simulates paths with regime-dependent parameters using Markov chain.
    """

    def __init__(
        self,
        initial_value: float = 100_000.0,
        regime_params: dict[MarketRegime, RegimeParameters] | None = None,
        transition_matrix: TransitionMatrix | None = None,
        seed: int | None = None,
    ):
        self.initial_value = initial_value
        self.regime_params = regime_params or RegimeParameters.default_regimes()
        self.transition_matrix = transition_matrix or TransitionMatrix()
        self.rng = np.random.default_rng(seed)

    def simulate_path(
        self,
        num_days: int,
        initial_regime: MarketRegime = MarketRegime.NORMAL,
        path_id: int = 0,
        start_date: datetime | None = None,
    ) -> SimulationPath:
        """
        Simulate a single path with regime switching.

        Args:
            num_days: Number of trading days
            initial_regime: Starting regime
            path_id: Path identifier
            start_date: Starting date

        Returns:
            SimulationPath with regime information
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc)

        values = np.zeros(num_days + 1)
        returns = np.zeros(num_days)
        regimes = [initial_regime]
        timestamps = [start_date]

        values[0] = self.initial_value
        current_regime = initial_regime
        dt = 1 / 252

        for day in range(num_days):
            # Get regime parameters
            params = self.regime_params[current_regime]

            # Daily drift and volatility
            drift = params.drift_annual * dt
            vol = params.volatility_annual * np.sqrt(dt)

            # Generate return with possible jump
            z = self.rng.standard_normal()
            daily_return = drift + vol * z

            # Add jump component
            if self.rng.random() < params.jump_probability:
                jump = params.jump_mean + params.jump_std * self.rng.standard_normal()
                daily_return += jump

            returns[day] = daily_return
            values[day + 1] = values[day] * (1 + daily_return)

            # Transition to next regime
            current_regime = self.transition_matrix.get_next_regime(
                current_regime, self.rng
            )
            regimes.append(current_regime)

            # Timestamp
            try:
                from datetime import timedelta
                timestamps.append(start_date + timedelta(days=day + 1))
            except Exception:
                timestamps.append(start_date)

        path = SimulationPath(
            path_id=path_id,
            timestamps=timestamps,
            values=values,
            returns=returns,
            regimes=regimes,
        )
        path.calculate_statistics()

        return path

    def simulate_paths(
        self,
        num_paths: int,
        num_days: int,
        initial_regime: MarketRegime = MarketRegime.NORMAL,
        start_date: datetime | None = None,
    ) -> MonteCarloResult:
        """Simulate multiple regime-switching paths."""
        paths = []
        for i in range(num_paths):
            path = self.simulate_path(
                num_days, initial_regime, path_id=i, start_date=start_date
            )
            paths.append(path)

        result = MonteCarloResult(
            num_paths=num_paths,
            num_days=num_days,
            initial_value=self.initial_value,
            paths=paths,
        )
        result.calculate_statistics()

        return result

    def simulate_conditional_paths(
        self,
        num_paths: int,
        num_days: int,
        condition_regime: MarketRegime,
        start_date: datetime | None = None,
    ) -> MonteCarloResult:
        """
        Simulate paths conditional on starting in specific regime.

        Useful for stress testing: "What if we start in a crisis?"
        """
        return self.simulate_paths(
            num_paths, num_days, initial_regime=condition_regime, start_date=start_date
        )


class CorrelationStressSimulator:
    """
    Correlation stress scenario simulator (P2 Enhancement).

    Simulates portfolio returns under different correlation assumptions.
    """

    def __init__(
        self,
        initial_value: float = 100_000.0,
        seed: int | None = None,
    ):
        self.initial_value = initial_value
        self.rng = np.random.default_rng(seed)

        # Asset definitions
        self._assets: list[str] = []
        self._weights: np.ndarray = np.array([])
        self._returns: np.ndarray = np.array([])  # Annual
        self._volatilities: np.ndarray = np.array([])  # Annual
        self._base_correlation: np.ndarray = np.array([])
        self._asset_classes: dict[str, str] = {}  # Asset to class mapping

    def set_portfolio(
        self,
        assets: list[str],
        weights: list[float],
        returns: list[float],
        volatilities: list[float],
        correlation_matrix: np.ndarray,
        asset_classes: dict[str, str] | None = None,
    ) -> None:
        """
        Set portfolio composition.

        Args:
            assets: Asset names
            weights: Portfolio weights
            returns: Expected annual returns
            volatilities: Annual volatilities
            correlation_matrix: Base correlation matrix
            asset_classes: Optional mapping of assets to classes
        """
        self._assets = assets
        self._weights = np.array(weights)
        self._returns = np.array(returns)
        self._volatilities = np.array(volatilities)
        self._base_correlation = correlation_matrix
        self._asset_classes = asset_classes or {a: "equity" for a in assets}

    def apply_stress_scenario(
        self,
        scenario: CorrelationStressScenario,
    ) -> np.ndarray:
        """
        Apply stress scenario to correlation matrix.

        Returns:
            Stressed correlation matrix
        """
        stressed = self._base_correlation.copy()

        # Apply multiplier and floor
        for i in range(len(self._assets)):
            for j in range(len(self._assets)):
                if i != j:
                    # Get asset class for adjustments
                    asset_i_class = self._asset_classes.get(self._assets[i], "equity")
                    asset_j_class = self._asset_classes.get(self._assets[j], "equity")

                    # Base stressed correlation
                    base_stressed = stressed[i, j] * scenario.correlation_multiplier

                    # Apply asset-specific adjustments
                    for asset_class, adjustment in scenario.asset_specific_adjustments.items():
                        if asset_i_class == asset_class or asset_j_class == asset_class:
                            base_stressed += adjustment

                    # Apply floor and cap
                    stressed[i, j] = max(
                        scenario.correlation_floor,
                        min(1.0, base_stressed)
                    )

        # Ensure matrix is positive semi-definite
        stressed = self._nearest_psd(stressed)

        return stressed

    def _nearest_psd(self, matrix: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix."""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Set negative eigenvalues to small positive
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Reconstruct
        psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Normalize to correlation matrix (diagonal = 1)
        d = np.sqrt(np.diag(psd))
        psd = psd / np.outer(d, d)
        np.fill_diagonal(psd, 1.0)

        return psd

    def simulate_scenario(
        self,
        scenario: CorrelationStressScenario,
        num_paths: int,
        num_days: int,
        start_date: datetime | None = None,
    ) -> MonteCarloResult:
        """
        Simulate portfolio paths under stressed correlations.

        Args:
            scenario: Correlation stress scenario
            num_paths: Number of simulation paths
            num_days: Number of trading days
            start_date: Starting date

        Returns:
            MonteCarloResult with simulation results
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc)

        # Apply stress to correlation matrix
        stressed_corr = self.apply_stress_scenario(scenario)

        # Build covariance matrix
        vol_matrix = np.diag(self._volatilities)
        cov_matrix = vol_matrix @ stressed_corr @ vol_matrix

        # Cholesky decomposition for correlated samples
        try:
            chol = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue method
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            chol = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # Daily parameters
        dt = 1 / 252
        daily_drift = self._returns * dt
        daily_vol_scale = np.sqrt(dt)

        paths = []

        for path_idx in range(num_paths):
            values = np.zeros(num_days + 1)
            returns = np.zeros(num_days)
            timestamps = [start_date]

            values[0] = self.initial_value

            for day in range(num_days):
                # Generate correlated random returns
                z = self.rng.standard_normal(len(self._assets))
                correlated_z = chol @ z

                # Asset returns
                asset_returns = daily_drift + daily_vol_scale * correlated_z

                # Portfolio return
                portfolio_return = float(np.sum(self._weights * asset_returns))

                returns[day] = portfolio_return
                values[day + 1] = values[day] * (1 + portfolio_return)

                # Timestamp
                try:
                    from datetime import timedelta
                    timestamps.append(start_date + timedelta(days=day + 1))
                except Exception:
                    timestamps.append(start_date)

            path = SimulationPath(
                path_id=path_idx,
                timestamps=timestamps,
                values=values,
                returns=returns,
                regimes=[MarketRegime.NORMAL] * (num_days + 1),
            )
            path.calculate_statistics()
            paths.append(path)

        result = MonteCarloResult(
            num_paths=num_paths,
            num_days=num_days,
            initial_value=self.initial_value,
            paths=paths,
        )
        result.calculate_statistics()

        return result

    def compare_scenarios(
        self,
        num_paths: int,
        num_days: int,
        scenarios: list[CorrelationStressScenario] | None = None,
    ) -> dict[str, MonteCarloResult]:
        """
        Compare portfolio behavior across correlation scenarios.

        Args:
            num_paths: Number of paths per scenario
            num_days: Simulation horizon
            scenarios: List of scenarios (uses standard if None)

        Returns:
            Dictionary mapping scenario name to result
        """
        if scenarios is None:
            scenarios = list(CorrelationStressScenario.standard_scenarios().values())

        results = {}
        for scenario in scenarios:
            result = self.simulate_scenario(scenario, num_paths, num_days)
            results[scenario.name] = result

        return results


class MonteCarloEngine:
    """
    Unified Monte Carlo simulation engine (P2 Enhancement).

    Provides a unified interface for all simulation types.
    """

    def __init__(
        self,
        initial_value: float = 100_000.0,
        seed: int | None = None,
    ):
        self.initial_value = initial_value
        self.seed = seed

        # Initialize simulators
        self.path_dependent = PathDependentSimulator(
            initial_value=initial_value, seed=seed
        )
        self.regime_switching = RegimeSwitchingSimulator(
            initial_value=initial_value, seed=seed
        )
        self.correlation_stress = CorrelationStressSimulator(
            initial_value=initial_value, seed=seed
        )

    def run_comprehensive_analysis(
        self,
        num_paths: int = 1000,
        num_days: int = 252,
    ) -> dict:
        """
        Run comprehensive Monte Carlo analysis with all methods.

        Returns:
            Dictionary with results from all simulation types
        """
        results = {}

        # Path-dependent simulation
        self.path_dependent.set_barrier_conditions(
            stop_loss_pct=0.20,  # 20% stop loss
            take_profit_pct=0.50,  # 50% take profit
        )
        results['path_dependent'] = self.path_dependent.simulate_paths(
            num_paths, num_days
        ).to_dict()

        # Regime switching - normal start
        results['regime_normal'] = self.regime_switching.simulate_paths(
            num_paths, num_days, MarketRegime.NORMAL
        ).to_dict()

        # Regime switching - crisis start
        results['regime_crisis'] = self.regime_switching.simulate_conditional_paths(
            num_paths, num_days, MarketRegime.CRISIS
        ).to_dict()

        return results

    def stress_test_portfolio(
        self,
        assets: list[str],
        weights: list[float],
        returns: list[float],
        volatilities: list[float],
        correlation_matrix: np.ndarray,
        num_paths: int = 1000,
        num_days: int = 252,
    ) -> dict[str, MonteCarloResult]:
        """
        Run correlation stress tests on a portfolio.

        Returns:
            Results for each correlation scenario
        """
        self.correlation_stress.set_portfolio(
            assets, weights, returns, volatilities, correlation_matrix
        )

        return self.correlation_stress.compare_scenarios(num_paths, num_days)
